# 1a) load dependencies
# 1b) add softmax on end of model, fine tune just decoder and see how it does - ItsIRL decoder only
# 1c) add softmax on end of model, fine tune whol model and see how it does   - ItsIRL end2end
# 1d) howd does using random weights for ffn effects 1a and 1b - in notebooks/

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

from scipy.stats import pearsonr
from torchmetrics import PearsonCorrCoef

import wandb
WANDA_LOGIN_KEY='xxxxxx'


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

    #do loss directly in training
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
    """
    if labels is not None:
      #loss = self.loss_fn(label_logits.view(-1, self.num_labels), labels.view(-1))
      loss = self.loss_fn(label_logits, labels)
    else:
      loss = None
    """
    return _, ier_layer, label_logits, sig_logits



def load_biosses_data(combine_train_dev=False, use_mini=False):
  base_path = 'biosses_data/'
  # dev.tsv  test.tsv  traindev.tsv  train.tsv
  if combine_train_dev == False and use_mini == False:
    print("Load Train and Dev sets seperately for validation tuning ( selection of epochs to run)")
    train_df = pd.read_csv(base_path + 'train.tsv', sep='\t')
    dev_df = pd.read_csv(base_path + 'dev.tsv', sep='\t')
  elif combine_train_dev and not use_mini:
    print("Load Combined Train and Dev sets into TRaining")
    train_df = pd.read_csv(base_path + 'traindev.tsv', sep='\t')
    dev_df = None

  test_df = pd.read_csv(base_path + 'test.tsv', sep='\t')  #TODO: verify this order is same as train/dev
  print("TRAIN DF HEAD: ", train_df.head(5))
  print("TEST DF HEAD: ", test_df.head(5))
  print(train_df.shape, test_df.shape)

  return train_df, dev_df, test_df


def get_data_loader(model, df, batch_size=8):
  #df can be train_df or test_df
  st = time.time()
  max_len = 512
  train_input_ids, train_attention_masks = [], []
  train_labels = []
  train_info = []

  for id in list(df["index"].values):
    row = df[df["index"] == id]
    ex = [row.sentence1.values[0], row.sentence2.values[0], row.score.values[0]]
    torch.cuda.empty_cache()
    context_tokens = model.transformer_tokenizer.encode_plus(ex[0],ex[1])
    cur_len = len(context_tokens['input_ids'])
    encoded_dict = model.transformer_tokenizer.encode_plus(    
          ex[0],ex[1],
          add_special_tokens=True,        
          max_length=max_len,
          truncation_strategy='only_second',
          pad_to_max_length=True,
          return_tensors='pt'
        )
    train_input_ids.append(encoded_dict['input_ids'][0])
    train_attention_masks.append(encoded_dict['attention_mask'][0]) 
    train_labels.append(ex[2])
    train_info.append([id,ex[0]])
  print("Elapsed.", time.time() - st)     #119

  # Convert the lists into PyTorch tensors.
  train_pt_input_ids = torch.stack(train_input_ids, dim=0)
  train_pt_attention_masks = torch.stack(train_attention_masks, dim=0)
  train_pt_labels = torch.tensor(train_labels)  

  # Create the DataLoader for our training set.
  train_data_set = TensorDataset(train_pt_input_ids, train_pt_attention_masks, train_pt_labels)
  train_sampler = RandomSampler(train_data_set)
  train_dataloader = DataLoader(train_data_set, sampler=train_sampler, batch_size=batch_size)
  return train_dataloader

def train_model(model, optimizer, train_dataloader, dev_dataloader, run, device, epochs, args, test_vars, print_out=True):
  by_epoch = args.by_epoch
  losstype = args.losstype
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
  cur_best_mse, cur_best_pearson = 100, -100
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
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_labels = b_labels.to(torch.float32)
        
        torch.cuda.empty_cache()
        #model.zero_grad()     <-- this is taken care of by optimizer.zero_grad() since all model.params() are there!  TODO: DID THIS CAUSE ISSUES IN EITHER HOC OR ELC?
        inputs = {'input_ids': b_input_ids, 'token_type_ids': None, 'attention_mask': b_input_mask}  #token_type_ids was None
        outputs = model(inputs, labels=b_labels)    #_, ier_layer, label_logits, sig_logits

        _, cls_emb, label_logits, ier_logits = outputs
        if losstype != "pearson":
          loss_fct = nn.MSELoss()   #  incorrect for regression 
          loss = loss_fct(label_logits.squeeze(), b_labels.squeeze())
        else:
          #https://stackoverflow.com/questions/54165651/how-to-use-a-numpy-function-as-the-loss-function-in-pytorch-and-avoid-getting-er
          #loss = torch.tensor(np.corrcoef(label_logits.squeeze(), b_labels.squeeze())[1][1]).float()   #TODO maybe easier with nn.corrcoef ??
          #loss = torch.tensor(np.corrcoef(label_logits.squeeze().detach().cpu().numpy(), b_labels.squeeze().detach().cpu().numpy())[1][1]).float()   #no grads
          #loss = torch.corrcoef(label_logits.squeeze(), b_labels.squeeze())   #requires torch 1.11 and we have 1.5.1 
          pearson = PearsonCorrCoef().to(device)
          loss = -1 * pearson(label_logits.squeeze(), b_labels.squeeze()).to(device)   #without the minus 1 it minimizes this and really we want to maximize this
          
        
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
        

        if step % 100 == 0 and not step == 0:
            elapsed = int(round(time.time() - t0))
            loss_v = float(1.0 * loss.clone().item())
            print('  Batch {:>5,}  of  {:>5,}.   Loss: {:} Elapsed: {:}.'.format(step, len(train_dataloader), loss_v, elapsed))
            if run != None:
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
            optimizer.zero_grad()  #what does this do?

        #TODO 
        if dev_dataloader != None:
          TODO=1

    avg_train_loss = total_loss / len(train_dataloader)            
    loss_values.append(avg_train_loss)
    et_preds.append(cur_et_preds)

    t_all_predictions = np.concatenate(predictions, axis=0)
    t_all_true_labels = np.concatenate(labels, axis=0)
     

    #t_all_et_preds = np.concatenate(et_preds, axis=0)
    print(t_all_predictions.shape, t_all_true_labels.shape)  

    errs_sqd = []
    for i in range(t_all_predictions.shape[0]):
      v = t_all_predictions[i][0] - t_all_true_labels[i]
      if print_out:
        print(i,t_all_predictions[i][0], t_all_true_labels[i], v*v)
      errs_sqd.append(v*v)
  
    mse = sum(errs_sqd)/len(errs_sqd)
    corr, _ = pearsonr(t_all_predictions.squeeze(), t_all_true_labels)

    print("\n  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(int(round(time.time() - t0))))        
    print("Train mse:",mse ,"Train pearson corr:",corr )

    
    if args.save_types and epoch_i+1 == epochs:
      print("save ET PREDS", len(et_preds), len(et_preds[0]))
      
      # we are only storing final run of train as opposed to overtime
      st = time.time()
      test_prediction_dataloader, model_id, run = test_vars
      t_all_et_preds = np.concatenate(cur_et_preds, axis=0)
      np.save("out/"+model_id+"_ep"+str(epochs)+"_train_full_predictions.npy", t_all_predictions)
      np.save("out/"+model_id+"_ep"+str(epochs)+"_train_true_labels.npy", t_all_true_labels)
      np.save("out/"+model_id+"_ep"+str(epochs)+"_entity_types.npy", t_all_et_preds)

    wandb.log({"train/avg_train_loss": avg_train_loss, "train/epoch": epoch_i+1, "train/mse": mse, "train/pearsons": corr})

    if by_epoch and epoch_i + 1 >= args.starting_at:
      #write out perf on train AND do test on model at this point
      print("Do Test Eval")
      model.eval()
      test_prediction_dataloader, model_id, run = test_vars
   
      #TODO maybe re-introduce
      #outfile = "out/train_"+model_id+"_ep"+str(epoch_i+1)+"_res.pkl"
      #with open(outfile, 'wb') as fp:
      #  pickle.dump(my_data, fp)
 
      cur_mets = [cur_best_mse, cur_best_pearson]
      metrics = do_test(model, test_prediction_dataloader, model_id, device, run, args.save_types, epoch_i+1, epochs, cur_mets )
      #TODO do for mse and pearsonr
      test_mse, test_pearson = metrics
      if test_mse < cur_best_mse:
        cur_best_mse = test_mse
        save_fname = '{0:s}/{1:s}_best_mse.pt'.format(transformer_constant.get(env,'EXP_ROOT'), model_id)
        print("Found better mse model at epoch ", str(epoch_i+1), " now: ", test_mse, " and saving best mse model to ", save_fname)
        torch.save( { 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'args': args }, save_fname)
      if test_pearson > cur_best_pearson:  #TODO: do i need to do absolute value or something ( don't care about sign )
        cur_best_pearson = test_pearson
        save_fname = '{0:s}/{1:s}_best_pearson.pt'.format(transformer_constant.get(env,'EXP_ROOT'), model_id)
        print("Found better pearson model at epoch ", str(epoch_i+1), " now: ", test_pearson, " and saving model to ", save_fname)
        torch.save( { 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'args': args }, save_fname)

      model.train()
      
    t0 = time.time()

  print("")
  print("Training complete!")   #20 mins per epoch on gpu  so 80 minutes for 4 epochs | 1,822 batches of size 8

  return model, optimizer


def do_test(model, test_prediction_dataloader, model_id, device, run, save_types, cur_epoch=8, last_epoch=8, cur_mets = [100,-100], print_out=True):
  #Eval test 
  model.eval()
  cur_best_mse, cur_best_pearson = cur_mets
  print("Evaling ", model_id)
  test_predictions , test_true_labels, et_preds, multi_preds  = [], [], [], []
  
  for tbatch in test_prediction_dataloader:

    test_b_input_ids = tbatch[0].to(device)
    test_b_input_mask = tbatch[1].to(device)
    test_b_labels = tbatch[2].to(device)
    test_b_labels = test_b_labels.to(torch.float32)
  
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

      test_outs = test_logits >= 0.5
      multi_preds.append(test_outs)
  
      test_label_ids = test_b_labels.to('cpu').numpy()
      test_predictions.append(test_logits)
      test_true_labels.append(test_label_ids)


  t_all_predictions = np.concatenate(test_predictions, axis=0)
  t_all_true_labels = np.concatenate(test_true_labels, axis=0)

  print(t_all_predictions.shape, t_all_true_labels.shape, )  #(6955, 16) (6955,) (6955,), (8 x 63808 )
  errs_sqd = []
  for i in range(t_all_predictions.shape[0]):
    v = t_all_predictions[i][0] - t_all_true_labels[i]
    if print_out:
      print(i,t_all_predictions[i][0], t_all_true_labels[i], v*v)
    errs_sqd.append(v*v)

  test_mse = sum(errs_sqd)/len(errs_sqd)
  test_pearson, _ = pearsonr(t_all_predictions.squeeze(), t_all_true_labels)

  print("Test mse: ", test_mse, "pearson: ", test_pearson, "at epoch", cur_epoch)
  if run != None:
    wandb.log({"test/mse": test_mse, "test/epoch": cur_epoch, "test/pearson": test_pearson})

  #ADD TEXT OF EACH EXAMPLE,

  if save_types or ( cur_best_mse != 100 and test_mse < cur_best_mse) or ( cur_best_pearson != 100 and test_pearson < cur_best_pearson):
    print("TEST ET PREDS")
    st = time.time()
    t_all_et_preds = np.concatenate(et_preds, axis=0)
    cur_found = 0
    if cur_best_mse != 100 and test_mse < cur_best_mse:
      cur_found = 1
      np.save("out/hoc_"+model_id+"_test_entity_types_best_mse.npy", t_all_et_preds)  #this will get overwritten everytime a better model is found

    if cur_best_pearson != 100 and test_pearson < cur_best_pearson:
      cur_found = 1
      np.save("out/hoc_"+model_id+"_test_entity_types_best_pearson.npy", t_all_et_preds)  #this will get overwritten everytime a better model is found

    if not cur_found:
      np.save("out/hoc_"+model_id+"_ep"+str(cur_epoch)+"_test_entity_types.npy", t_all_et_preds)

  #wandb.log({"test/test_examples", my_table})  #gives wandb TypeError: "unhashable type: 'Table'"  so try with run as per https://docs.wandb.ai/guides/data-vis/log-tables#save-tables

  # save out results
  #outfile = "out/test_"+model_id+"_ep"+str(cur_epoch)+"_res.pkl"
  #with open(outfile, 'wb') as fp:
  #  pickle.dump(my_data, fp)

  # save this out only for final round
  if cur_epoch == last_epoch:
    np.save("out/"+model_id+"_ep"+str(cur_epoch)+"_test_full_predictions.npy", t_all_predictions)
    np.save("out/"+model_id+"_ep"+str(cur_epoch)+"_test_true_labels.npy", t_all_true_labels)

  return test_mse, test_pearson

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
  run_et.parser.add_argument("--losstype", default=None, type=str, choices=['mse', 'pearson'], help="loss method to minimize")
  run_et.parser.add_argument('--num_labels', type=int, default=1, help='num labels in task')
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
    modtype = "ffn_with_linear_head_to_tune"  #ffn_with_linear_head_to_tune,  ffn_with_softmax, endtoend
  
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

  wandb.login(key=WANDA_LOGIN_KEY)

  if args.debug_mode:
    # IN DEBUG MODE DON'T LOG TO WANDB ( to avoid clutter there )
    run = None
  else:
    run = wandb.init(project="itsirl_BIOSSES", entity="xxxxx", name=model_id, config=vars(args))

  #1. get hoc data
  train_df, dev_df, test_df = load_biosses_data(args.combine_train_dev, use_mini)  
  
  #2. Load OLD MODEL Architecture and get optimizers
  model = PriorTransformerModelwithFFN(args, transformer_constant.ANSWER_NUM_DICT[env])

  if args.separate_opts:
    optimizer_enc, optimizer_cls = load_optimizers(model, args)
    run_et.load_model(mod, transformer_constant.get(env,'EXP_ROOT'), model_id, model, optimizer_enc, optimizer_cls)
  else:
    run_et.load_model(mod, transformer_constant.get(env,'EXP_ROOT'), model_id, model)

  #print(model)
  print("MODEL:",type(model))

  #default to batch_size=8
  train_dataloader = get_data_loader(model, train_df)
  if not args.combine_train_dev:
    dev_dataloader = get_data_loader(model, dev_df)
  else:
    dev_dataloader = None
  test_dataloader = get_data_loader(model, test_df)
  
  if modtype == "ffn_with_linear_head_to_tune":
    # add linear layer on top of model, freeze rest of model (IER and FFN) and update only linear layer
    print("Loading ", modtype, " which Freezes model and adds linear head.. Randomize FFN weights?", set_init_weights,"pretrained has 8 layers?", args.pretrained_has_8layers )
    pmodel = PTMFlinear(model, args.num_labels, True, set_init_weights, args.pretrained_has_8layers)

  elif modtype == "ffn_with_softmax":
    # add linear layer, freeze IER but update decoder FFN AND soft max on top
    print("Loading ", modtype, " which Freezes IER, but updates FFN.. Randomize FFN weights?", set_init_weights,"pretrained has 8 layers?", args.pretrained_has_8layers )
    pmodel = PTMFlinear(model, args.num_labels, False, set_init_weights, args.pretrained_has_8layers)

  else:
    # tune whole model end to end
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

  test_vars = [test_dataloader, model_id, run]
  pmodel, optimizer = train_model(pmodel, optimizer, train_dataloader, dev_dataloader, run, device, num_epoch, args, test_vars)

  if args.save_model:
    save_fname = '{0:s}/{1:s}_{2:s}_{3:s}.pt'.format(transformer_constant.get(env,'EXP_ROOT'), model_id, modtype, mod)
    print("Saving model to ", save_fname)
    torch.save( { 'state_dict': pmodel.state_dict(), 'optimizer': optimizer.state_dict(), 'args': args }, save_fname)

  if not args.by_epoch:
    pmodel.eval()
    test_acc = do_test(pmodel, test_prediction_dataloader, model_id, device, run, args.save_types)
