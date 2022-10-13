# load dependencies
# 1a) add linear layer at end of model, fine tune just it and see how it does
# 1b) add softmax on end of model, fine tune just ffn and see how it does
# 1c) howd does using random weights for ffn effect 1a and 1b
# 2) SANITY CHECK verify that sparse preds give exact same results as before!
# 3) can we use the examples where entity typying is off during inference to create rules and then see how that affects final predictions ( ie, change those components )

import glob
import json
import numpy as np
import pandas as pd
import pickle
import random
import shutil
import sys
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup, AdamW

from sklearn import metrics, preprocessing

# Set PATHs
sys.path.insert(0, '../ier_model')
import run_et
import transformer_constant
import transformer_data_utils   
from models import TransformerModel, PriorTransformerModel, PriorTransformerModelwithFFN, TRANSFORMER_MODELS

import wandb

#convert to multi-label model that uses HOC data
# NEXT ADD optimizer, train stuff as per elc ( copy/paste ), etc ( continue experiments for bases)


## MODEL DEFS
def to_torch(batch, device):
  inputs_to_model = {k: v.to(device) for k, v in batch['inputs'].items()}
  targets = batch['targets'].to(device)
  return inputs_to_model, targets

class PTMFlinear(nn.Module):
  def __init__(self, model, num_labels, freeze_mod=True, rando_weights=False, has8layers=False):
    super(PTMFlinear, self).__init__()
    self.num_labels = num_labels
    self.sigmoid_fn = nn.Sigmoid()
    #self.loss_fn = nn.CrossEntropyLoss() 
    self.loss_fn = nn.BCEWithLogitsLoss()  #since its a multi-label task
    self.irl_model = model
    self.label_classifier = nn.Linear(model.transformer_config.hidden_size, num_labels)

    print("Number layers in PriorT FFN", model.num_ff_layers)

    if freeze_mod != "endtoend":
      print("FREEZING IER LAYERS")
      #IER weights are frozen if not endtoend!
      for param in self.irl_model.encoder.parameters():
        param.requires_grad = False
      for param in self.irl_model.classifier.parameters():
        param.requires_grad = False 

    if freeze_mod == True:
      print("FREEZING FFN LAYERS")
      #freeze ffn layers
      for param in self.irl_model.down_project.parameters():
        param.requires_grad = False
      for param in self.irl_model.decoder_l1.parameters():
        param.requires_grad = False

      if model.num_ff_layers >= 3:
        for param in self.irl_model.decoder_l2.parameters():
          param.requires_grad = False
        for param in self.irl_model.decoder_l3.parameters():
          param.requires_grad = False
      
        if model.num_ff_layers >= 5:
          for param in self.irl_model.decoder_l4.parameters():
            param.requires_grad = False
          for param in self.irl_model.decoder_l5.parameters():
            param.requires_grad = False

          if model.num_ff_layers >= 8:
            for param in self.irl_model.decoder_l6.parameters():
              param.requires_grad = False
            for param in self.irl_model.decoder_l7.parameters():
              param.requires_grad = False
            for param in self.irl_model.decoder_l8.parameters():
              param.requires_grad = False

    if rando_weights:
      print("RANDOMIZING FFN LAYER WEIGHTS")
      self.irl_model.init_weights(self.irl_model.down_project)
      self.irl_model.init_weights(self.irl_model.decoder_l1)

      if model.num_ff_layers >= 3:
        self.irl_model.init_weights(self.irl_model.decoder_l2)
        self.irl_model.init_weights(self.irl_model.decoder_l3)

    if not has8layers or rando_weights:
      if model.num_ff_layers >= 5:
        print("Randomize 4 and 5 weights")
        self.irl_model.init_weights(self.irl_model.decoder_l4)
        self.irl_model.init_weights(self.irl_model.decoder_l5)
  
        if model.num_ff_layers >= 8:
          print("Randomize 6,7,8 weights")
          self.irl_model.init_weights(self.irl_model.decoder_l6)
          self.irl_model.init_weights(self.irl_model.decoder_l7)
          self.irl_model.init_weights(self.irl_model.decoder_l8)
  

  #see if i need to add device here and call to_torch on inputs, targets
  def forward(self, inputs, targets=None, labels=None):
    #_, _, logits, ier_layer = self.irl_model(inputs, targets)
    _, _, logits, ier_layer = self.irl_model(inputs)
    sig_logits = self.sigmoid_fn(logits) 
    label_logits = self.label_classifier(sig_logits)
    if labels is not None:
      #loss = self.loss_fn(label_logits.view(-1, self.num_labels), labels.view(-1))
      loss = self.loss_fn(label_logits, labels)
    else:
      loss = None
    return loss, ier_layer, label_logits, sig_logits


## HOC DATA HELPERS

def encode_labels(train_df, dev_df, test_df):
  lb = preprocessing.MultiLabelBinarizer()
  train_labels =  [ str(a).split(",") for a in train_df.labels.values ]  #11 categories :  None + 0 - 9
  train_labels_enc = lb.fit_transform(train_labels)
  train_df['list'] = [ list(l) for l in train_labels_enc]

  print("TRAIN CLASSES:",lb.classes_)
  if dev_df != None:
    TODO=1

  test_labels = [ str(a).split(",") for a in test_df.labels.values ]
  test_labels_enc = lb.transform(test_labels)
  test_df['list'] = [ list(l) for l in test_labels_enc]

  return train_df, dev_df, test_df


def load_hoc_data(combine_train_dev=False, use_mini=False):
  base_path = '/home/diego/biomed_fall20/scibert/data/text_classification/hoc_sent/'
  if combine_train_dev == False and use_mini == False:
    print("Load Train and Dev sets seperately for validation tuning ( selection of epochs to run)")
    train_df = pd.read_csv(base_path + 'train.tsv', sep='\t')
    dev_df = pd.read_csv(base_path + 'dev.tsv', sep='\t')
  elif combine_train_dev and not use_mini:
    print("Load Combined Train and Dev sets into TRaining")
    #train_df = pd.read_csv(base_path + 'train_and_dev.tsv', sep='\t')
    train_df = pd.read_csv(base_path + 'new_blurb_hoc_traindev.tsv', sep='\t')
    dev_df = None
  elif use_mini:
    print("Load mini Training set")
    #train_df = pd.read_csv(base_path + 'train_mini.tsv', sep='\t')
    train_df = pd.read_csv(base_path + 'new_blurb_hoc_train_mini.tsv', sep='\t')
    dev_df = None

  #test_df = pd.read_csv(base_path + 'test.tsv', sep='\t')  #TODO: verify this order is same as train/dev
  test_df = pd.read_csv(base_path + 'new_blurb_hoc_test.tsv', sep='\t')  #TODO: verify this order is same as train/dev
  train_df, dev_df, test_df = encode_labels(train_df, dev_df, test_df)
  print("TRAIN DF HEAD: ", train_df.head(25))
  print("TEST DF HEAD: ", test_df.head(25))
  #import sys
  #sys.exit()
  print(train_df.shape, test_df.shape)

  return train_df, dev_df, test_df

def copy_mod_to(env, mod_to_copy, new_name, mod_path=None):
  if mod_path == None:
      mod_path = transformer_constant.get(env,'EXP_ROOT')
  if ".pt" not in new_name:
      new_name = args.model_id + ".pt"
  print("Copying ", mod_to_copy, " to ", new_name)
  shutil.copy(mod_path + mod_to_copy, mod_path + new_name)  

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.sent_text = dataframe.sentence
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.sent_text)

    def __getitem__(self, index):
        sent_text = str(self.sent_text[index])
        sent_text = " ".join(sent_text.split())    #what is this about?  
        inputs = self.tokenizer.encode_plus(
            sent_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

def get_dataloader(model, df, args):
  params = {'batch_size': 8, 'shuffle': False, 'num_workers': 0 }
  dset = CustomDataset(df, model.transformer_tokenizer, args.max_len)
  return DataLoader(dset, **params)


def train_model(model, optimizer, train_dataloader, dev_dataloader, run, device, epochs, args, test_vars, label_map):
  by_epoch = args.by_epoch
  if args.seed == 113:
    seed_val = 42  #change default 
  else:
    seed_val = args.seed
  
  total_steps = len(train_dataloader) * epochs
  scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
  gradient_accumulation_steps = 4
  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)
  loss_values = []
  predictions, labels, et_preds, multi_preds = [], [], [], []
  cur_best = -1
  for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()
    total_loss = 0
    model.train()

    cur_et_preds = []
    accs, f1mis, f1mas = [], [], []
    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch['ids'].to(device)
        b_input_mask = batch['mask'].to(device)
        b_labels = batch['targets'].to(device)
        b_toktypes = batch['token_type_ids'].to(device)
        #if args.debug_mode and total_loss == 0:
        #  print("Input sizes:", b_input_ids.shape, b_input_mask.shape,  b_labels.shape,     b_toktypes.shape)
          #      Input sizes: torch.Size([8, 512]) torch.Size([8, 512]) torch.Size([8, 11]) torch.Size([8, 512])
        
        torch.cuda.empty_cache()
        model.zero_grad()        
        inputs = {'input_ids': b_input_ids, 'token_type_ids': b_toktypes, 'attention_mask': b_input_mask}  #token_type_ids was None
        outputs = model(inputs, labels=b_labels)    #loss, ier_layer, label_logits, sig_logits
        
        loss = outputs[0]
        if args.do_grad_accum and gradient_accumulation_steps > 1:
          loss = loss / args.gradient_accumulation_steps

        total_loss += loss.item()
        loss.backward()
  
        #if args.save_types: 
        entity_type_logits = outputs[1]
        entity_types = entity_type_logits.detach().cpu().numpy()
        cur_et_preds.append(entity_types)

        train_logits = outputs[2]
        train_logits = train_logits.detach().cpu().numpy() 
        label_ids = b_labels.to('cpu').numpy()
        predictions.append(train_logits)
        labels.append(label_ids)
        
        train_outs = train_logits >= 0.5
        multi_preds.append(train_outs)

        if step % 100 == 0 and not step == 0:
            elapsed = int(round(time.time() - t0))
            loss_v = float(1.0 * loss.clone().item())

            """
            if args.debug_mode:
              print( label_ids.shape, train_logits.shape, train_outs.shape )
              print("Label_ids: ", label_ids)
              print("Train_logits:", train_logits)
              print("Train_outs: ", train_outs)
            """
            accuracy = metrics.accuracy_score(label_ids, train_outs)
            f1_micro = metrics.f1_score(label_ids, train_outs, average='micro')
            f1_macro = metrics.f1_score(label_ids, train_outs, average='macro')
            accs.append(accuracy)
            f1mis.append(f1_micro)
            f1mas.append(f1_macro)

            print('  Batch {:>5,}  of  {:>5,}.   Loss: {:} Elapsed: {:}.'.format(step, len(train_dataloader), loss_v, elapsed))
            print("  Acc: ", accuracy, "F1micro: ", f1_micro, "F1macro: ", f1_macro)
            if run != None:
              wandb.log({"train/loss": loss_v, "train/step": step * (epoch_i + 1), "train/acc": accuracy, "train/f1_micro": f1_micro, "train/f1_macro": f1_macro})

        if args.do_clipping:
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if args.do_grad_accum and step % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        else:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        #TODO 
        if dev_dataloader != None:
          TODO=1

    avg_train_loss = total_loss / len(train_dataloader)            
    loss_values.append(avg_train_loss)
    et_preds.append(cur_et_preds)

    t_all_predictions = np.concatenate(predictions, axis=0)
    t_all_true_labels = np.concatenate(labels, axis=0)
    #t_predicted_label_ids = np.argmax(t_all_predictions, axis=1)
    #t_predicted_label_probs = np.max(t_all_predictions, axis=1)
    t_predicted_label_ids = np.concatenate(multi_preds)
    t_predicted_label_probs = t_all_predictions
     

    #t_all_et_preds = np.concatenate(et_preds, axis=0)
    print(t_all_predictions.shape, t_all_true_labels.shape, t_predicted_label_ids.shape)  
    #train_acc = flat_accuracy(t_predicted_label_ids , t_all_true_labels) 
    train_acc = np.mean(accs)
    train_f1mi = np.mean(f1mis)
    train_f1ma = np.mean(f1mas)

    print("\n  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(int(round(time.time() - t0))))        
    #TODO: print("Train acc:", train_acc, "at epoch", epoch_i + 1)
    print("Train acc:", train_acc, "f1_micro:", train_f1mi, "f1_macro:", train_f1ma)

    id2label = { label_map[l]:l for l in list(label_map.keys()) } 
    true_ls = [ id2label[a] for a in list(t_all_true_labels.flatten())]
    pred_ls = [ id2label[a] for a in list(t_predicted_label_ids.flatten())]
    correct = [ v == pred_ls[i] for i,v in enumerate(true_ls)]
    
    if args.save_types and epoch_i+1 == epochs:
      print("save ET PREDS", len(et_preds), len(et_preds[0]))
      
      # we are only storing final run of train as opposed to overtime
      st = time.time()
      test_prediction_dataloader, model_id, label_map, run = test_vars
      t_all_et_preds = np.concatenate(cur_et_preds, axis=0)
      np.save("out/"+model_id+"_ep"+str(epochs)+"_train_full_predictions.npy", t_all_predictions)
      np.save("out/"+model_id+"_ep"+str(epochs)+"_train_true_labels.npy", t_all_true_labels)
      np.save("out/"+model_id+"_ep"+str(epochs)+"_train_single_predictions.npy", t_predicted_label_ids)
      np.save("out/"+model_id+"_ep"+str(epochs)+"_train_single_pred_probs.npy", t_predicted_label_probs)
      np.save("out/"+model_id+"_ep"+str(epochs)+"_entity_types.npy", t_all_et_preds)

    md = [true_ls, pred_ls, list(t_predicted_label_probs.flatten()),correct ]
    my_data = [ [md[0][i], md[1][i], md[2][i], md[3][i]] for i in range(len(true_ls)) ]
    df = pd.DataFrame(my_data, columns=["true label", "pred_label", "pred_prob", "correct?"])
    my_table = wandb.Table(data=df)

    if run != None:
      try:
        wandb.log({"train/train_examples", my_table})
      except Exception as e:
        print("Error saving table to wandb", e)
        print(len(my_data), len(my_data[0]))
        print([ (i,type(my_data[0][i])) for i in range(len(my_data[0]))])
        print(my_data[0])
  
      wandb.log({"train/avg_train_loss": avg_train_loss, "train/epoch": epoch_i+1, "train/avg_acc": train_acc, "train/avg_f1micro": train_f1mi, "train/avg_f1macro": train_f1ma})

    if by_epoch and epoch_i + 1 >= args.starting_at:
      #write out perf on train AND do test on model at this point
      print("Do Test Eval")
      model.eval()
      test_prediction_dataloader, model_id, label_map, run = test_vars
   
      outfile = "out/train_"+model_id+"_ep"+str(epoch_i+1)+"_res.pkl"
      with open(outfile, 'wb') as fp:
        pickle.dump(my_data, fp)
 
      test_acc = do_test(model, test_prediction_dataloader, model_id, device, label_map, run, args.save_types, epoch_i+1, epochs, cur_best)
      if test_acc > cur_best:
        print("Found better acc model at epoch ", str(epoch_i+1), " now: ", test_acc)
        cur_best = test_acc
        save_fname = '{0:s}/{1:s}_best.pt'.format(transformer_constant.get(env,'EXP_ROOT'), model_id)
        print("Saving best model to ", save_fname)
        torch.save(
          {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': args
          },
          save_fname
        )

      model.train()
      
    t0 = time.time()

  print("")
  print("Training complete!")   #20 mins per epoch on gpu  so 80 minutes for 4 epochs | 1,822 batches of size 8

  return model, optimizer



def flat_accuracy(preds, labels):
  pred_flat = preds.flatten()
  labels_flat = labels.flatten()
  return np.sum(pred_flat == labels_flat) / len(labels_flat)


def do_test(model, test_prediction_dataloader, model_id, device, label_map, run, save_types, cur_epoch=8, last_epoch=8, cur_best = -1):
  #Eval test 
  model.eval()
  print("Evaling ", model_id)
  test_predictions , test_true_labels, et_preds, multi_preds  = [], [], [], []
  
  for tbatch in test_prediction_dataloader:

    test_b_input_ids = tbatch['ids'].to(device)
    test_b_input_mask = tbatch['mask'].to(device)
    test_b_labels = tbatch['targets'].to(device)
    test_b_toktypes = tbatch['token_type_ids'].to(device)
    #tbatch = tuple(t.to(device) for t in tbatch)
    #test_b_input_ids, test_b_input_mask, test_b_labels = tbatch
  
    with torch.no_grad():
      torch.cuda.empty_cache()

      test_inputs = {'input_ids': test_b_input_ids, 'token_type_ids': test_b_toktypes, 'attention_mask': test_b_input_mask}
      test_outputs = model(test_inputs)      

      #if save_types:
      entity_type_logits = test_outputs[1]
      entity_types = entity_type_logits.detach().cpu().numpy()
      et_preds.append(entity_types)

      test_logits = test_outputs[2]   #this may need to be test_outputs[2] and not 0?
      test_logits = test_logits.detach().cpu().numpy()

      test_outs = test_logits >= 0.5
      multi_preds.append(test_outs)
  
      test_label_ids = test_b_labels.to('cpu').numpy()
      test_predictions.append(test_logits)
      test_true_labels.append(test_label_ids)


  t_all_predictions = np.concatenate(test_predictions, axis=0)
  t_all_true_labels = np.concatenate(test_true_labels, axis=0)
  #t_predicted_label_ids = np.argmax(t_all_predictions, axis=1)
  #t_predicted_label_probs = np.max(t_all_predictions, axis=1)
  t_predicted_label_ids = np.concatenate(multi_preds, axis=0)
  t_predicted_label_probs = t_all_predictions

  print(t_all_predictions.shape, t_all_true_labels.shape, t_predicted_label_ids.shape)  #(6955, 16) (6955,) (6955,), (8 x 63808 )
  #test_acc = flat_accuracy(t_predicted_label_ids , t_all_true_labels) 

  test_acc = metrics.accuracy_score(t_all_true_labels, t_predicted_label_ids)
  f1_micro = metrics.f1_score(t_all_true_labels, t_predicted_label_ids, average='micro')
  f1_macro = metrics.f1_score(t_all_true_labels, t_predicted_label_ids, average='macro')

  print("Test Acc: ", test_acc, "F1micro: ", f1_micro, "F1macro: ", f1_macro)
  print("Test acc:", test_acc, "at epoch", cur_epoch)
  if run != None:
    wandb.log({"test/acc": test_acc, "test/epoch": cur_epoch, "test/f1_micro": f1_micro, "test/f1_macro": f1_macro})

  #ADD TEXT OF EACH EXAMPLE,

  id2label = { label_map[l]:l for l in list(label_map.keys()) } 
  true_ls = [ id2label[a] for a in list(t_all_true_labels.flatten())]
  pred_ls = [ id2label[a] for a in list(t_predicted_label_ids.flatten())]
  correct = [ v == pred_ls[i] for i,v in enumerate(true_ls)]

  if save_types or ( cur_best != -1 and test_acc > cur_best):
    print("TEST ET PREDS")
    st = time.time()
    t_all_et_preds = np.concatenate(et_preds, axis=0)
    if cur_best != -1 and test_acc > cur_best:
      np.save("out/hoc_"+model_id+"_test_entity_types_best.npy", t_all_et_preds)  #this will get overwritten everytime a better model is found
    else:
      np.save("out/hoc_"+model_id+"_ep"+str(cur_epoch)+"_test_entity_types.npy", t_all_et_preds)

    """
    et_types = [ get_topk_types(p,k=-1) for p in et_preds ]
    print("elapsed:", time.time() - st)
    md = [true_ls, pred_ls, list(t_predicted_label_probs.flatten()),correct,et_types ]
    my_data = [ [md[0][i], md[1][i], md[2][i], md[3][i], md[4][i]] for i in range(len(true_ls)) ]
    my_table = wandb.Table(columns=["true label", "pred_label", "pred_prob", "correct?","types"], data=my_data)
    """

  md = [true_ls, pred_ls, list(t_predicted_label_probs.flatten()),correct ]
  my_data = [ [md[0][i], md[1][i], md[2][i], md[3][i]] for i in range(len(true_ls)) ]
  #my_table = wandb.Table(columns=["true label", "pred_label", "pred_prob", "correct?"], data=my_data)
  df = pd.DataFrame(my_data, columns=["true label", "pred_label", "pred_prob", "correct?"])
  my_table = wandb.Table(data=df)

  #how to look at preds within WandB.. don't save text its too long!
  #wandb.log({"test/test_examples", my_table})  #gives wandb TypeError: "unhashable type: 'Table'"  so try with run as per https://docs.wandb.ai/guides/data-vis/log-tables#save-tables
  if run != None:
    try:
      run.log({"test/test_examples", my_table})
    except Exception as e:
      print("Error saving TEST table to wandb", e)
      print(len(my_data), len(my_data[0]))
      print([ (i,type(my_data[0][i])) for i in range(len(my_data[0]))])
      print(my_data[0])

  # save out results
  outfile = "out/test_"+model_id+"_ep"+str(cur_epoch)+"_res.pkl"
  with open(outfile, 'wb') as fp:
    pickle.dump(my_data, fp)

  # save this out only for final round
  if cur_epoch == last_epoch:
    np.save("out/"+model_id+"_ep"+str(cur_epoch)+"_test_full_predictions.npy", t_all_predictions)
    np.save("out/"+model_id+"_ep"+str(cur_epoch)+"_test_true_labels.npy", t_all_true_labels)
    np.save("out/"+model_id+"_ep"+str(cur_epoch)+"_test_single_predictions.npy", t_predicted_label_ids)
    np.save("out/"+model_id+"_ep"+str(cur_epoch)+"_test_single_pred_probs.npy", t_predicted_label_probs)
    #np.save("out/"+model_id+"_ep"+str(cur_epoch)+"_test_single_entity_types.npy", t_all_et_preds)

  return test_acc

def load_loc_model(reload_model_name, save_dir, model_id, model,
               optimizer_enc=None, optimizer_cls=None, optimizer=None, scheduler_cls=None):
  if reload_model_name:
    model_file_name = '{0:s}/{1:s}.pt'.format(save_dir, reload_model_name)
  else:
    model_file_name = '{0:s}/{1:s}.pt'.format(save_dir, model_id)
  print("Loading ", model_file_name)
  checkpoint = torch.load(model_file_name)

  # for debug purposes
  try:
    for c in checkpoint.keys():
      print("----", c, type(checkpoint[c]))
      k = checkpoint[c]
      for e in k.keys():
        print("---- ___ ",e, type(k[e]))
  except Exception as e:
    print(e)

  model.load_state_dict(checkpoint['state_dict'], strict=False)

  if len(model.state_dict()) != len(checkpoint['state_dict']):
    print("Size expected vs state_dict size passed in ", len(model.state_dict()), len(checkpoint['state_dict']))


  if optimizer:
    print("load ffn optimizer")
    #TODO do i need to save the scheduler?
    optimizer.load_state_dict(checkpoint['optimizer'])

  if optimizer_enc and optimizer_cls:  # Continue training
    print("CHECKPOINT Optim Enc", checkpoint['optimizer_enc'].keys())
    optimizer_enc.load_state_dict(checkpoint['optimizer_enc'])
    optimizer_cls.load_state_dict(checkpoint['optimizer_cls'])
  else: 
    #otherwise don't load optimizer weights
    total_params = 0
    # Log params
    for k in checkpoint['state_dict']:
      elem = checkpoint['state_dict'][k]
      param_s = 1
      for size_dim in elem.size():
        param_s = size_dim * param_s
      #print(k, elem.size())
      total_params += param_s
    param_str = ('Number of total parameters..{0:d}'.format(total_params))
    print(param_str)

  print('Loading model from ... {0:s}'.format(model_file_name))

def load_optimizers(model, args):
  no_decay = ["bias", "LayerNorm.weight"]
  classifier_param_name = ["classifier.linear.weight"]
  encoder_parameters = [
    {
      "params": [p for n, p in model.named_parameters()
                 if not any(nd in n for nd in no_decay) and n not in classifier_param_name],
      "weight_decay": 0.0  #args.weight_decay,
    },
    {
      "params": [p for n, p in model.named_parameters()
                 if any(nd in n for nd in no_decay) and n not in classifier_param_name],
      "weight_decay": 0.0
    },
  ]
  classifier_parameters = [
    {
      "params": [p for n, p in model.named_parameters() if n in classifier_param_name],
      "weight_decay": 0.0
    },
  ]
  print(
    'Encoder {}, Classifier {}'.format(
      sum([len(p['params']) for p in encoder_parameters]),
      sum([len(p['params']) for p in classifier_parameters])
    )
  )
  optimizer_enc = AdamW(encoder_parameters, lr=args.learning_rate_enc, eps=args.adam_epsilon_enc)
  optimizer_cls = AdamW(classifier_parameters, lr=args.learning_rate_cls, eps=args.adam_epsilon_cls)

  return optimizer_enc, optimizer_cls
if __name__ == "__main__":

  run_et.parser.add_argument("--mtype", default=None, type=str, choices=['ffn_with_linear_head_to_tune', 'ffn_with_softmax','endtoend'], help="fine tuning method")
  run_et.parser.add_argument('--num_labels', type=int, default=11, help='num labels in task')
  run_et.parser.add_argument('--max_len', type=int, default=512, help='max seq len')
  run_et.parser.add_argument("--save_model", action='store_const', default=False, const=True)
  run_et.parser.add_argument("--debug_mode", action='store_const', default=False, const=True)
  run_et.parser.add_argument("--by_epoch", action='store_const', default=False, const=True)
  run_et.parser.add_argument("--save_types", action='store_const', default=False, const=True)
  run_et.parser.add_argument('--starting_at', type=int, default=0, help='if by_epoch is set, what epoch to start showing at')
  run_et.parser.add_argument("--do_clipping", action='store_const', default=False, const=True)
  run_et.parser.add_argument("--do_grad_accum", action='store_const', default=False, const=True)
  run_et.parser.add_argument("--combine_train_dev", action='store_const', default=True, const=True)
  run_et.parser.add_argument('--learning_rate_ffn', type=float, default=2e-5, help='learning_rate_for_ffn')
  run_et.parser.add_argument("--separate_opts", action='store_const', default=False, const=True)
  run_et.parser.add_argument("--pretrained_has_8layers", action='store_const', default=False, const=True)
  args = run_et.parser.parse_args()

  device = torch.device("cuda")

  label_map  = {'activating invasion and metastasis': 0, 'avoiding immune destruction': 1, 'cellular energetics': 2, 
             'enabling replicative immortality': 3, 'evading growth suppressors': 4, 'genomic instability and mutation': 5, 
             'inducing angiogenesis': 6, '': 7, 'resisting cell death': 8, 'sustaining proliferative signaling': 9,'tumor promoting inflammation': 10}


  if args.reload_model_name:
    mod = args.reload_model_name
  else:
    mod = "0504_prior_ier_frozen_with_ffn_best"

  if args.model_id:
    model_id = args.model_id
  else:
    model_id = "debug_elc_0506"

  if args.mtype:
    modtype = args.mtype
  else:
    modtype = "ffn_with_linear"  #ffn_with_linear_head_to_tune,  ffn_with_softmax, endtoend
  
  if args.env:
    env = args.env
  else:
    env = "0720_600k_full_orig"

  if args.num_epoch == 5000:
    num_epoch = 8  #change default of 5000 to smaller
  else:
    num_epoch = args.num_epoch

  use_mini = False
  if args.debug_mode:
    num_epoch = 3
    use_mini = True

  # randomly initialize weights to test importance of pretraining!
  set_init_weights = args.set_init_weights
  torch.cuda.empty_cache()

  wandb.login(key='289874e53b5f0d8a776da5968900653c84f30eb4')

  if args.debug_mode:
    # IN DEBUG MODE DON'T LOG TO WANDB ( to avoid clutter there )
    run = None
  else:
    run = wandb.init(project="new_biers_HOC_macro", entity="diegoolano", name=model_id, config=vars(args))

  #1. get hoc data
  train_df, dev_df, test_df = load_hoc_data(args.combine_train_dev, use_mini)  
  
  #2. Load OLD MODEL Architecture and get optimizers
  #I'm EXPECTING TO EXTEND AN OLD IER MODEL ( AND NOT A NEW ONE ) which I'm not allowing
  model = PriorTransformerModelwithFFN(args, transformer_constant.ANSWER_NUM_DICT[env])

  if args.separate_opts:
    optimizer_enc, optimizer_cls = load_optimizers(model, args)
    run_et.load_model(mod, transformer_constant.get(env,'EXP_ROOT'), model_id, model, optimizer_enc, optimizer_cls)
  else:
    run_et.load_model(mod, transformer_constant.get(env,'EXP_ROOT'), model_id, model)

  #print(model)
  print("MODEL:",type(model))

  train_dataloader = get_dataloader(model, train_df, args)
  if not args.combine_train_dev:
    dev_dataloader = get_dataloader(model, dev_df, args)
  else:
    dev_dataloader = None
  test_dataloader = get_dataloader(model, test_df, args)
  
  # in both cases, set_init_weights, randomizes the FFN values
  if modtype == "ffn_with_linear_head_to_tune":
    # add linear layer on top of model, freeze rest of model (IER and FFN) and update only linear layer
    print("Loading ", modtype, " which Freezes model and adds linear head.. Randomize FFN weights?", set_init_weights,"pretrained has 8 layers?", args.pretrained_has_8layers )
    pmodel = PTMFlinear(model, args.num_labels, True, set_init_weights, args.pretrained_has_8layers)
  elif modtype == "ffn_with_softmax":
    # add linear layer, freeze IER but update FFN AND soft max on top
    print("Loading ", modtype, " which Freezes IER, but updates FFN.. Randomize FFN weights?", set_init_weights,"pretrained has 8 layers?", args.pretrained_has_8layers )
    pmodel = PTMFlinear(model, args.num_labels, False, set_init_weights, args.pretrained_has_8layers)
  else:
    # tune end to end
    print("Loading End to End.. Randomize weights?", set_init_weights ,"pretrained has 8 layers?", args.pretrained_has_8layers)
    pmodel = PTMFlinear(model, args.num_labels, "endtoend", set_init_weights, args.pretrained_has_8layers)

  print("TRAINING ", modtype)
  pmodel.to(device)
 
  print("PMODEL:",type(pmodel))

  if args.separate_opts:
    no_decay = ["bias", "LayerNorm.weight"]
    classifier_param_name = ["classifier.linear.weight"]
    ffn_parameters = [
      { "params": [p for n, p in pmodel.named_parameters() if not any(nd in n for nd in no_decay) and n not in model.named_parameters()],
        "weight_decay": 0.0  #args.weight_decay,
      },
      { "params": [p for n, p in pmodel.named_parameters() if any(nd in n for nd in no_decay) and n not in model.named_parameters()],
        "weight_decay": 0.0
      },
    ]
  
    optimizer = AdamW(ffn_parameters, lr=args.learning_rate_ffn, eps=args.adam_epsilon_cls )  
  else:
    optimizer = AdamW(pmodel.parameters(), lr=args.learning_rate_ffn, eps=args.adam_epsilon_cls )  

  test_vars = [test_dataloader, model_id, label_map, run]
  pmodel, optimizer = train_model(pmodel, optimizer, train_dataloader, dev_dataloader, run, device, num_epoch, args, test_vars, label_map)

  if args.save_model:
    save_fname = '{0:s}/{1:s}_{2:s}_{3:s}.pt'.format(transformer_constant.get(env,'EXP_ROOT'), model_id, modtype, mod)
    print("Saving model to ", save_fname)
    torch.save(
      {
        'state_dict': pmodel.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': args
      },
      save_fname
    )

  if not args.by_epoch:
    pmodel.eval()
    test_acc = do_test(pmodel, test_prediction_dataloader, model_id, device, label_map, run, args.save_types)
