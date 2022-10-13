# load dependencies

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
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup, AdamW

# Set PATHs
sys.path.insert(0, '../ier_model')
import run_et
import transformer_constant
import transformer_data_utils   
from models import TransformerModel, PriorTransformerModel, PriorTransformerModelwithFFN, TRANSFORMER_MODELS

import wandb

## MODEL DEFS
def to_torch(batch, device):
  inputs_to_model = {k: v.to(device) for k, v in batch['inputs'].items()}
  targets = batch['targets'].to(device)
  return inputs_to_model, targets

class PTMFlinear(nn.Module):
  def __init__(self, model, num_labels, freeze_mod=True, rando_weights=False):
    super(PTMFlinear, self).__init__()
    self.num_labels = num_labels
    self.sigmoid_fn = nn.Sigmoid()
    self.loss_fn = nn.CrossEntropyLoss() 
    self.irl_model = model
    self.label_classifier = nn.Linear(model.transformer_config.hidden_size, num_labels)

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
      for param in self.irl_model.decoder_l2.parameters():
        param.requires_grad = False
      for param in self.irl_model.decoder_l3.parameters():
        param.requires_grad = False
      

    if rando_weights:
      print("RANDOMIZING FFN LAYER WEIGHTS")
      self.irl_model.init_weights(self.irl_model.down_project)
      self.irl_model.init_weights(self.irl_model.decoder_l1)
      self.irl_model.init_weights(self.irl_model.decoder_l2)
      self.irl_model.init_weights(self.irl_model.decoder_l3)


  #see if i need to add device here and call to_torch on inputs, targets
  def forward(self, inputs, targets=None, labels=None):
    _, _, logits, ier_layer = self.irl_model(inputs, targets)
    sig_logits = self.sigmoid_fn(logits) 
    label_logits = self.label_classifier(sig_logits)
    if labels is not None:
      loss = self.loss_fn(label_logits.view(-1, self.num_labels), labels.view(-1))
    else:
      loss = None
    return loss, ier_layer, label_logits, sig_logits


## ELC DATA HELPERS
def get_mentions(ids):
  mentions = {}
  for a1file in ids:
    pmid = a1file.split("/")[-1].split(".")[0]    
    pm_txtfile = "/".join(a1file.split("/")[0:-1]) + "/" + pmid + ".txt"
    with open(a1file) as f:
      lines = [ [ t.strip("\n") for t in l.split("\t")] for l in f.readlines() ]  
    with open(pm_txtfile) as f:
      title, abstract = f.readlines()    
    mentions[pmid] = [title.strip("\n"), abstract.strip("\n"), lines]
  return mentions

def load_elc_data():
  train_ids = glob.glob('./elc_data/BioNLP-ST_2013_CG_training_data/PMID-*.a1')  #300 
  dev_ids = glob.glob('./elc_data/BioNLP-ST_2013_CG_development_data/PMID-*.a1') #100
  test_ids = glob.glob('./elc_data/BioNLP-ST_2013_CG_test_data/PMID-*.a1')       #200

  train_data = get_mentions(train_ids)
  ents_per_file_train = [len(train_data[i][2]) for i in list(train_data.keys())]
  
  dev_data = get_mentions(dev_ids)
  ents_per_file_dev = [len(dev_data[i][2]) for i in list(dev_data.keys())]
  
  test_data = get_mentions(test_ids)
  ents_per_file_test = [len(test_data[i][2]) for i in list(test_data.keys())]

  return [train_ids, train_data, ents_per_file_train, dev_ids, dev_data, ents_per_file_dev, test_ids, test_data, ents_per_file_test]

def copy_mod_to(env, mod_to_copy, new_name, mod_path=None):
  if mod_path == None:
      mod_path = transformer_constant.get(env,'EXP_ROOT')
  if ".pt" not in new_name:
      new_name = args.model_id + ".pt"
  print("Copying ", mod_to_copy, " to ", new_name)
  shutil.copy(mod_path + mod_to_copy, mod_path + new_name)  


def get_train_data(model, train_ids, dev_ids, train_data, dev_data, debug_mode, label_map, max_len=512, batch_size=8):
  st = time.time()
  if debug_mode:
    pmids = train_ids[0:32]
  else:
    pmids = train_ids + dev_ids
  train_input_ids, train_attention_masks = [], []
  train_labels = []
  train_info = []
  
  for i, pmid in enumerate(pmids):
    tmp_pmid = "PMID" + pmid.split("PMID")[1]
    pmid = tmp_pmid.split(".")[0]
    if i < len(train_ids):
      exs = train_data[pmid] # title, abstract, mentions
    else:
      exs = dev_data[pmid] # title, abstract, mentions
  
    title = exs[0]
    abstract = exs[1]
    mentions = exs[2]
    for m in mentions:
      torch.cuda.empty_cache()
      mention = m[2]
      abstract_txt = abstract
      label, start, end = m[1].split(" ") 
      context_tokens = model.transformer_tokenizer.encode_plus(mention,abstract_txt)
      cur_len = len(context_tokens['input_ids'])
      encoded_dict = model.transformer_tokenizer.encode_plus(    
            mention,abstract_txt,
            add_special_tokens=True,
            max_length=max_len,
            truncation_strategy='only_second',
            pad_to_max_length=True,
            return_tensors='pt'
          )
      train_input_ids.append(encoded_dict['input_ids'][0])
      train_attention_masks.append(encoded_dict['attention_mask'][0]) 
      train_labels.append(label_map[label])
      train_info.append([pmid,m[0]])
  print("Elapsed.", time.time() - st)     #119
  
  # Convert the lists into PyTorch tensors.
  train_pt_input_ids = torch.stack(train_input_ids, dim=0)
  train_pt_attention_masks = torch.stack(train_attention_masks, dim=0)
  train_pt_labels = torch.tensor(train_labels, dtype=torch.long)  
  
  # Create the DataLoader for our training set.
  train_data_set = TensorDataset(train_pt_input_ids, train_pt_attention_masks, train_pt_labels)
  train_sampler = RandomSampler(train_data_set)
  train_dataloader = DataLoader(train_data_set, sampler=train_sampler, batch_size=batch_size)

  #return what
  return train_dataloader

def loc_sigmoid(x):
  sig = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
  return sig

def get_topk_types(prob,  k=100, threshhold = .001):
  #if you pass k=-1, then the method returns all types with prob above threshhold

  id2ans = transformer_constant.ID2ANS_MEDWIKI_DICT['0720_600k_full_orig']
  probs = [["" for i in range(100)] for i in range(100)]

  if prob.ndim > 1:
    for iv, vec in enumerate(prob):
      sig_vec = loc_sigmoid(vec)
      loc_prob = []
      for i, p in enumerate(sig_vec):
        if i != 0:
          try:
            val = id2ans[i]
            loc_prob.append((p, val))
          except Exception as e:
            print(i,p,e)
      sorted_prob = sorted( [(p[0],p[1]) for i, p in enumerate(loc_prob) if i != 0], key=lambda x: x[0], reverse=True)
      if k == -1:
        probs[iv] = [(v1, v2) for v1, v2 in sorted_prob if v1 > threshhold]
      else:
        probs[iv] = sorted_prob[:k] 
      #probs.append(sorted_prob[:k] + [(k, v) for k, v in sorted_prob if k <= 0. and v > threshhold])
    return probs
  else:
    sig_prob = loc_sigmoid(prob)
    sorted_prob = sorted( [(p, id2ans[i]) for i, p in enumerate(sig_prob) if i != 0], key=lambda x: x[0], reverse=True)  #instead of id2ans[i+1]
    if k == -1:
      return [(v1, v2) for v1, v2 in sorted_prob if v1 > threshhold]
    else:
      return sorted_prob[:k] + [(k, v) for k, v in sorted_prob if k <= 0. and v > threshhold]


def train_model(model, optimizer, train_dataloader, device, epochs, args, test_vars, label_map):
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
  predictions, labels, et_preds = [], [], []

  cur_best = -1
  for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()
    total_loss = 0
    model.train()

    cur_et_preds = []
    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        torch.cuda.empty_cache()
        model.zero_grad()        
        inputs = {'input_ids': b_input_ids, 'token_type_ids': None, 'attention_mask': b_input_mask}
        outputs = model(inputs, labels=b_labels)    #loss, ier_layer, label_logits, sig_logits
        
        loss = outputs[0]
        if args.do_grad_accum and gradient_accumulation_steps > 1:
          loss = loss / args.gradient_accumulation_steps

        total_loss += loss.item()
        loss.backward()
  
        entity_type_logits = outputs[1]
        entity_types = entity_type_logits.detach().cpu().numpy()
        cur_et_preds.append(entity_types)

        train_logits = outputs[2]
        train_logits = train_logits.detach().cpu().numpy() 
        label_ids = b_labels.to('cpu').numpy()
        predictions.append(train_logits)
        labels.append(label_ids)
        

        if step % 100 == 0 and not step == 0:
            elapsed = int(round(time.time() - t0))
            loss_v = float(1.0 * loss.clone().item())
            print('  Batch {:>5,}  of  {:>5,}.   Loss: {:} Elapsed: {:}.'.format(step, len(train_dataloader), loss_v, elapsed))
            wandb.log({"train/loss": loss_v, "train/step": step * (epoch_i + 1)})

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

    avg_train_loss = total_loss / len(train_dataloader)            
    loss_values.append(avg_train_loss)
    et_preds.append(cur_et_preds)

    t_all_predictions = np.concatenate(predictions, axis=0)
    t_all_true_labels = np.concatenate(labels, axis=0)
    t_predicted_label_ids = np.argmax(t_all_predictions, axis=1)
    t_predicted_label_probs = np.max(t_all_predictions, axis=1)
    print(t_all_predictions.shape, t_all_true_labels.shape, t_predicted_label_ids.shape)  
    train_acc = flat_accuracy(t_predicted_label_ids , t_all_true_labels) 

    print("\n  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(int(round(time.time() - t0))))        
    print("Train acc:", train_acc, "at epoch", epoch_i + 1)

    id2label = { label_map[l]:l for l in list(label_map.keys()) } 
    true_ls = [ id2label[a] for a in list(t_all_true_labels.flatten())]
    pred_ls = [ id2label[a] for a in list(t_predicted_label_ids.flatten())]
    correct = [ v == pred_ls[i] for i,v in enumerate(true_ls)]
    
    if args.save_types and epoch_i+1 == epochs:
      print("save ET PREDS", len(et_preds), len(et_preds[0]))
      
      #we are only storing final run of train as opposed to overtime
      st = time.time()
      test_prediction_dataloader, model_id, label_map, test_texts, run = test_vars
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

    try:
      wandb.log({"train/train_examples", my_table})
    except Exception as e:
      print("Error saving table to wandb", e)
      print(len(my_data), len(my_data[0]))
      print([ (i,type(my_data[0][i])) for i in range(len(my_data[0]))])
      print(my_data[0])

    wandb.log({"train/avg_train_loss": avg_train_loss, "train/epoch": epoch_i+1, "train/acc": train_acc})

    if by_epoch and epoch_i + 1 >= args.starting_at:
      print("Do Test Eval")
      model.eval()
      test_prediction_dataloader, model_id, label_map, test_texts, run = test_vars
   
      outfile = "out/train_"+model_id+"_ep"+str(epoch_i+1)+"_res.pkl"
      with open(outfile, 'wb') as fp:
        pickle.dump(my_data, fp)
 
      test_acc = do_test(model, test_prediction_dataloader, model_id, device, label_map, test_texts, run, args.save_types, epoch_i+1, epochs, cur_best)
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


def get_test_data(model, test_pmids, test_data, label_map, max_len=512):
  st = time.time()
  test_input_ids, test_attention_masks = [], []
  test_labels = []
  test_info = []
  test_texts = []
  
  for i, pmid in enumerate(test_pmids):
    tmp_pmid = "PMID" + pmid.split("PMID")[1]
    pmid = tmp_pmid.split(".")[0]
    exs = test_data[pmid]
    title = exs[0]
    abstract = exs[1]
    mentions = exs[2]
    for m in mentions:
      torch.cuda.empty_cache()
      mention = m[2]
      abstract_txt = abstract
      label, start, end = m[1].split(" ") 
      context_tokens = model.transformer_tokenizer.encode_plus(mention,abstract_txt)
      cur_len = len(context_tokens['input_ids'])
      encoded_dict = model.transformer_tokenizer.encode_plus(    
            mention,abstract_txt,
            add_special_tokens=True,
            max_length=max_len,
            truncation_strategy='only_second',
            pad_to_max_length=True,
            return_tensors='pt'
          )
      test_input_ids.append(encoded_dict['input_ids'][0])
      test_attention_masks.append(encoded_dict['attention_mask'][0]) 
      test_labels.append(label_map[label])
      test_info.append([pmid,m[0]])
      test_texts.append(mention + "<SEP>" + abstract_txt)
  
  
  # Convert the lists into PyTorch tensors.
  test_pt_input_ids = torch.stack(test_input_ids, dim=0)
  test_pt_attention_masks = torch.stack(test_attention_masks, dim=0)
  test_pt_labels = torch.tensor(test_labels, dtype=torch.long)  
  
  # Set the batch size and # Create the DataLoader for our testing set.
  test_batch_size = 1
  test_prediction_data = TensorDataset(test_pt_input_ids, test_pt_attention_masks, test_pt_labels)
  test_prediction_sampler = SequentialSampler(test_prediction_data)
  test_prediction_dataloader = DataLoader(test_prediction_data, sampler=test_prediction_sampler, batch_size=test_batch_size)
  print("Elapsed.", time.time() - st)   #50 seconds

  return test_prediction_dataloader, test_texts

def flat_accuracy(preds, labels):
  pred_flat = preds.flatten()
  labels_flat = labels.flatten()
  return np.sum(pred_flat == labels_flat) / len(labels_flat)


def do_test(model, test_prediction_dataloader, model_id, device, label_map, test_texts, run, save_types, cur_epoch=8, last_epoch=8, cur_best = -1):
  #Eval test 
  print("Evaling ", model_id)
  test_predictions , test_true_labels, et_preds  = [], [], []
  
  for tbatch in test_prediction_dataloader:
    tbatch = tuple(t.to(device) for t in tbatch)
    test_b_input_ids, test_b_input_mask, test_b_labels = tbatch
  
    with torch.no_grad():
      torch.cuda.empty_cache()

      test_inputs = {'input_ids': test_b_input_ids, 'token_type_ids': None, 'attention_mask': test_b_input_mask}
      test_outputs = model(test_inputs)      

      #if save_types:
      entity_type_logits = test_outputs[1]
      entity_types = entity_type_logits.detach().cpu().numpy()
      et_preds.append(entity_types)

      test_logits = test_outputs[2]   #this may need to be test_outputs[2] and not 0?
      test_logits = test_logits.detach().cpu().numpy()
  
      test_label_ids = test_b_labels.to('cpu').numpy()
      test_predictions.append(test_logits)
      test_true_labels.append(test_label_ids)

  t_all_predictions = np.concatenate(test_predictions, axis=0)
  t_all_true_labels = np.concatenate(test_true_labels, axis=0)
  t_predicted_label_ids = np.argmax(t_all_predictions, axis=1)
  t_predicted_label_probs = np.max(t_all_predictions, axis=1)
  print(t_all_predictions.shape, t_all_true_labels.shape, t_predicted_label_ids.shape)  #(6955, 16) (6955,) (6955,), (8 x 63808 )
  test_acc = flat_accuracy(t_predicted_label_ids , t_all_true_labels) 

  print("Test acc:", test_acc, "at epoch", cur_epoch)
  wandb.log({"test/acc": test_acc, "test/epoch": cur_epoch})

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
      np.save("out/"+model_id+"_test_entity_types_best.npy", t_all_et_preds)  #this will get overwritten everytime a better model is found
    else:
      np.save("out/"+model_id+"_ep"+str(cur_epoch)+"_test_entity_types.npy", t_all_et_preds)

  md = [true_ls, pred_ls, list(t_predicted_label_probs.flatten()),correct ]
  my_data = [ [md[0][i], md[1][i], md[2][i], md[3][i]] for i in range(len(true_ls)) ]
  df = pd.DataFrame(my_data, columns=["true label", "pred_label", "pred_prob", "correct?"])
  my_table = wandb.Table(data=df)

  #how to look at preds within WandB.. don't save text its too long!
  #wandb.log({"test/test_examples", my_table})  #gives wandb TypeError: "unhashable type: 'Table'"  so try with run as per https://docs.wandb.ai/guides/data-vis/log-tables#save-tables
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
  run_et.parser.add_argument('--num_labels', type=int, default=16, help='num labels in task')
  run_et.parser.add_argument("--save_model", action='store_const', default=False, const=True)
  run_et.parser.add_argument("--debug_mode", action='store_const', default=False, const=True)
  run_et.parser.add_argument("--by_epoch", action='store_const', default=False, const=True)
  run_et.parser.add_argument("--save_types", action='store_const', default=False, const=True)
  run_et.parser.add_argument('--starting_at', type=int, default=0, help='if by_epoch is set, what epoch to start showing at')
  run_et.parser.add_argument("--do_clipping", action='store_const', default=False, const=True)
  run_et.parser.add_argument("--do_grad_accum", action='store_const', default=False, const=True)
  run_et.parser.add_argument('--learning_rate_ffn', type=float, default=3e-5, help='learning_rate_for_ffn')
  run_et.parser.add_argument("--separate_opts", action='store_const', default=False, const=True)
  args = run_et.parser.parse_args()

  device = torch.device("cuda")

  label_map = {'Gene_or_gene_product': 0, 'Cell': 1, 'Cancer': 2, 'Simple_chemical': 3,  
               'Organism': 4, 'Multi-tissue_structure': 5, 'Tissue': 6, 'Cellular_component': 7, 
               'Organ': 8, 'Pathological_formation': 9, 'Organism_substance': 10, 'Amino_acid': 11, 
               'Immaterial_anatomical_entity': 12, 'Organism_subdivision': 13, 'Anatomical_system': 14, 
               'Developing_anatomical_structure': 15}

  #get elc data
  ret = load_elc_data()
  train_ids, train_data, ents_per_file_train, dev_ids, dev_data, ents_per_file_dev, test_ids, test_data, ents_per_file_test = ret

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

  if args.debug_mode:
    num_epoch = 1

  # randomly initialize weights to test importance of pretraining!
  set_init_weights = args.set_init_weights

  torch.cuda.empty_cache()
  
  run = wandb.init(project="itsirl_ELC", entity="xxxxx", name=model_id, config=vars(args))

  #Load OLD MODEL Architecture and get optimizers
  model = PriorTransformerModelwithFFN(args, transformer_constant.ANSWER_NUM_DICT[env])

  if args.separate_opts:
    optimizer_enc, optimizer_cls = load_optimizers(model, args)
    run_et.load_model(mod, transformer_constant.get(env,'EXP_ROOT'), model_id, model, optimizer_enc, optimizer_cls)
  else:
    run_et.load_model(mod, transformer_constant.get(env,'EXP_ROOT'), model_id, model)

  print("MODEL:",type(model))

  train_dataloader = get_train_data(model, train_ids, dev_ids, train_data, dev_data, args.debug_mode, label_map)
  test_prediction_dataloader, test_texts = get_test_data(model, test_ids, test_data, label_map)

  if modtype == "ffn_with_linear_head_to_tune":
    # add linear layer on top of model, freeze rest of model (IER and FFN) and update only linear layer
    print("Loading ", modtype, " which Freezes model and adds linear head.. Randomize FFN weights?", set_init_weights )
    pmodel = PTMFlinear(model, args.num_labels, True, set_init_weights)

  elif modtype == "ffn_with_softmax":
    # add linear layer, freeze IER but update FFN AND soft max on top
    print("Loading ", modtype, " which Freezes IER, but updates FFN.. Randomize FFN weights?", set_init_weights )
    pmodel = PTMFlinear(model, args.num_labels, False, set_init_weights)

  else:
    # tune end to end
    print("Loading End to End.. Randomize weights?", set_init_weights )
    pmodel = PTMFlinear(model, args.num_labels, "endtoend", set_init_weights)

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


  test_vars = [test_prediction_dataloader, model_id, label_map, test_texts, run]
  pmodel, optimizer = train_model(pmodel, optimizer, train_dataloader, device, num_epoch, args, test_vars, label_map)

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
    test_acc = do_test(pmodel, test_prediction_dataloader, model_id, device, label_map, test_texts, run, args.save_types)
