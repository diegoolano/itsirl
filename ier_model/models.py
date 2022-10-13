import argparse
import numpy as np
import sys
import torch
import torch.nn as nn

from transformers import AutoConfig
from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer
from transformers import AutoTokenizer, AutoModel

import transformer_constant   #just for debug
import wandb

TRANSFORMER_MODELS = {
  'bert-base-uncased': (BertModel, BertTokenizer),
  'bert-large-uncased-whole-word-masking': (BertModel, BertTokenizer),
  'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext': (AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'), AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'))
}

class SimpleDecoder(nn.Module):
  def __init__(self, output_dim, answer_num):
    super(SimpleDecoder, self).__init__()
    self.answer_num = answer_num
    self.linear = nn.Linear(output_dim, answer_num, bias=False)

  def forward(self, inputs, output_type=None):
    output_embed = self.linear(inputs)
    return output_embed


class DownProject(nn.Module):
  def __init__(self, output_dim, answer_num):
    super(DownProject, self).__init__()
    self.answer_num = answer_num
    self.linear = nn.Linear(answer_num, output_dim, bias=False)

  def forward(self, inputs, output_type=None):
    output_embed = self.linear(inputs)
    return output_embed


class IERDecoder(nn.Module):
  def __init__(self, output_dim, dropout_rate=.1):
    super(IERDecoder, self).__init__()    
    self.linear = nn.Linear(output_dim, output_dim, bias=False)
    self.layer_norm = nn.LayerNorm(output_dim)
    self.dropout = nn.Dropout(dropout_rate)

  def forward(self, inputs, output_type=None):
    output_embed = self.linear(inputs)
    output_embed = self.layer_norm(output_embed)
    output_embed = self.dropout(output_embed)        
    return output_embed


class ModelBase(nn.Module):
  def __init__(self):
    super(ModelBase, self).__init__()
    self.type_loss = nn.BCEWithLogitsLoss()
    self.sigmoid_fn = nn.Sigmoid()
    self.mse_loss = nn.MSELoss()

  def define_loss(self, logits, targets, bert_cls, ier_layer):
    #this is reconstruction loss between output logits and the initial BERT cls rep
    mse_err = self.mse_loss(logits, bert_cls) 

    #this is the entity typing loss over the sparse layer and entity type labels
    type_err = self.type_loss(ier_layer, targets)
    return [mse_err, type_err]

  def forward(self, feed_dict):
    pass


class TransformerModel(ModelBase):
  def __init__(self, args, answer_num):
    super(TransformerModel, self).__init__()
    print('In TransformerModel Initializing <{}> model...'.format(args.model_type))

    _model_class, _tokenizer_class = TRANSFORMER_MODELS[args.model_type]
    self.transformer_tokenizer = _tokenizer_class.from_pretrained(args.model_type)
    self.transformer_config = AutoConfig.from_pretrained(args.model_type)

    self.encoder = _model_class.from_pretrained(args.model_type)
    self.classifier = SimpleDecoder(self.transformer_config.hidden_size, answer_num)
    self.dropout = nn.Dropout(args.hidden_dropout_prob)
    self.avg_pooling = args.avg_pooling

    # add a projection layer to get to embedding dimension expected 
    self.down_project =  DownProject(self.transformer_config.hidden_size, answer_num)

    if args.num_ff_layers == 1:
        self.decoder_l1 = IERDecoder(self.transformer_config.hidden_size)
    else:
        self.decoder_l1 = IERDecoder(self.transformer_config.hidden_size)
        self.decoder_l2 = IERDecoder(self.transformer_config.hidden_size)
        self.decoder_l3 = IERDecoder(self.transformer_config.hidden_size)

        if args.num_ff_layers >= 5:
          self.decoder_l4 = IERDecoder(self.transformer_config.hidden_size)
          self.decoder_l5 = IERDecoder(self.transformer_config.hidden_size)

        if args.num_ff_layers == 8:
          self.decoder_l6 = IERDecoder(self.transformer_config.hidden_size)
          self.decoder_l7 = IERDecoder(self.transformer_config.hidden_size)
          self.decoder_l8 = IERDecoder(self.transformer_config.hidden_size)


    self.id2ans = transformer_constant.ID2ANS_MEDWIKI_DICT["0720_600k_full_orig"]
    self.et_lambda = args.et_lambda
    self.num_ff_layers = args.num_ff_layers

  def forward(self, inputs, targets=None):

    #1. encode input sentence
    outputs = self.encoder(
      inputs["input_ids"],
      attention_mask=inputs["attention_mask"],
      token_type_ids=inputs["token_type_ids"] if "token_type_ids" in inputs else (None)
    )

    if self.avg_pooling:  # Averaging all hidden states
      orig_outputs = (outputs[0] * inputs["attention_mask"].unsqueeze(-1)).sum(1)\
                / inputs["attention_mask"].sum(1).unsqueeze(-1)
    else:  # Use [CLS]
      orig_outputs = outputs[0][:, 0, :]


    #2. do dropout on output of encoder (CLS by default)
    orig_outputs_drop = self.dropout(orig_outputs)

    #3. pass through linear layer to get interpretable sparse layer (ie, prior logits)  #this layer has not been transformed via sigmoid to keep logits unbounded!
    ier_layer = self.classifier(orig_outputs_drop)

    #4. new pass through linear layer returning to 768 embedding size expected by BERT
    down_project = self.down_project(ier_layer)

    #5. feed downprojected sparse layer into BERT encoder ( feed in embeddings)
    outputs_l1 = self.decoder_l1(down_project)
    if self.num_ff_layers == 3:
      outputs_l2 = self.decoder_l2(outputs_l1)
      outputs_l3 = self.decoder_l3(outputs_l2)
      logits = outputs_l3
    else:
      logits = outputs_l1

    if targets is not None:
      #this is now Entity Typing Loss + Reconstruction Loss!
      mse_err, type_err = self.define_loss(logits, targets, orig_outputs, ier_layer)
      #loss = mse_err + (self.et_lambda * type_err)   
      #loss = [mse_err, type_err]
    else:
      #loss = None
      mse_err, type_err = None, None

    #logging works in general and NOW get logging for multiple losses working
    #THEN DO HYPER PARAM SWEEP
    #return loss, logits, ier_layer
    return mse_err, type_err, logits, ier_layer

class PriorModelBase(nn.Module):
  def __init__(self):
    super(PriorModelBase, self).__init__()
    self.type_loss = nn.BCEWithLogitsLoss()
    self.sigmoid_fn = nn.Sigmoid()
    self.mse_loss = nn.MSELoss()

  def prior_define_loss(self, logits, targets):
    loss = self.type_loss(logits, targets)
    return loss

  def define_loss(self, logits, targets, bert_cls, ier_layer):
    #this is reconstruction loss between output logits and the initial BERT cls rep
    mse_err = self.mse_loss(logits, bert_cls) 

    #this is the entity typing loss over the sparse layer and entity type labels
    type_err = self.type_loss(ier_layer, targets)
    return [mse_err, type_err]

  def forward(self, feed_dict):
    pass

class PriorTransformerModel(PriorModelBase):
  def __init__(self, args, answer_num):
    super(PriorTransformerModel, self).__init__()
    print('In PriorTransformerModel Initializing <{}> model...'.format(args.model_type))
    _model_class, _tokenizer_class = TRANSFORMER_MODELS[args.model_type]
    self.transformer_tokenizer = _tokenizer_class.from_pretrained(args.model_type)
    self.transformer_config = AutoConfig.from_pretrained(args.model_type)
    self.encoder = _model_class.from_pretrained(args.model_type)
    self.classifier = SimpleDecoder(self.transformer_config.hidden_size, answer_num)
    self.dropout = nn.Dropout(args.hidden_dropout_prob)
    self.avg_pooling = args.avg_pooling

  def forward(self, inputs, targets=None):
    outputs = self.encoder(
      inputs["input_ids"],
      attention_mask=inputs["attention_mask"],
      token_type_ids=inputs["token_type_ids"] if "token_type_ids" in inputs else (None)
    )
    if self.avg_pooling:  # Averaging all hidden states
      outputs = (outputs[0] * inputs["attention_mask"].unsqueeze(-1)).sum(1)\
                / inputs["attention_mask"].sum(1).unsqueeze(-1)
    else:  # Use [CLS]
      outputs = outputs[0][:, 0, :]
    outputs = self.dropout(outputs)
    logits = self.classifier(outputs)
    if targets is not None:
      loss = self.prior_define_loss(logits, targets)
    else:
      loss = None
    return loss, logits

class PriorTransformerModelwithFFN(PriorModelBase):
  def __init__(self, args, answer_num):
    super(PriorTransformerModelwithFFN, self).__init__()
    print('In PriorTransformerModelwithFFN Initializing <{}> model...'.format(args.model_type))
    _model_class, _tokenizer_class = TRANSFORMER_MODELS[args.model_type]
    self.transformer_tokenizer = _tokenizer_class.from_pretrained(args.model_type)
    self.transformer_config = AutoConfig.from_pretrained(args.model_type)
    self.encoder = _model_class.from_pretrained(args.model_type)
    self.classifier = SimpleDecoder(self.transformer_config.hidden_size, answer_num)
    self.dropout = nn.Dropout(args.hidden_dropout_prob)
    self.avg_pooling = args.avg_pooling
    self.freeze_ier = args.freeze_ier
    self.et_lambda = args.et_lambda
    self.num_ff_layers = args.num_ff_layers

    hidden_size = self.transformer_config.hidden_size
    dropout_rate = args.hidden_dropout_prob

    self.down_project =  DownProject(hidden_size, answer_num)
    if self.num_ff_layers == 1:
        print("USE Decoder with only 1 layer ")
        self.decoder_l1 = IERDecoder(hidden_size, dropout_rate)
    else:
        print("USE Decoder with ",args.num_ff_layers," layers ")
        self.decoder_l1 = IERDecoder(hidden_size, dropout_rate)
        self.decoder_l2 = IERDecoder(hidden_size, dropout_rate)
        self.decoder_l3 = IERDecoder(hidden_size, dropout_rate)

        if args.num_ff_layers >= 5:
          print("USE Decoder with ",args.num_ff_layers," layers ")
          self.decoder_l4 = IERDecoder(hidden_size, dropout_rate)
          self.decoder_l5 = IERDecoder(hidden_size, dropout_rate)

        if args.num_ff_layers == 8:
          print("USE Decoder with ",args.num_ff_layers," layers ")
          self.decoder_l6 = IERDecoder(hidden_size, dropout_rate)
          self.decoder_l7 = IERDecoder(hidden_size, dropout_rate)
          self.decoder_l8 = IERDecoder(hidden_size, dropout_rate)
    
    if args.set_init_weights:
      # initialize down_project onward
      self.init_weights(self.down_project)
      self.init_weights(self.decoder_l1)
  
      if self.num_ff_layers >= 3:
        self.init_weights(self.decoder_l2)
        self.init_weights(self.decoder_l3)

        if self.num_ff_layers >= 5:
          self.init_weights(self.decoder_l4)
          self.init_weights(self.decoder_l5)

          if self.num_ff_layers >= 8:
            self.init_weights(self.decoder_l6)
            self.init_weights(self.decoder_l7)
            self.init_weights(self.decoder_l8)

    # if freeze is True, hold encoder/classifier as is and then only do updates to weights for ffn layers
    if self.freeze_ier:
      for param in self.encoder.parameters():
        param.requires_grad = False
      for param in self.classifier.parameters():
        param.requires_grad = False 

      self.et_lambda = 0 # if we are freezing prior model, we really only want to optimize for MSE ( since by this point ET has been optimized for already )

  def init_weights(self, module):
    """ Initialize the weights """
    #print("In init weights: ", type(module))
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.02) 
    elif isinstance(module, nn.LayerNorm):   #https://github.com/huggingface/transformers/issues/10892
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

  def forward(self, inputs, targets=None):
    outputs = self.encoder(
      inputs["input_ids"],
      attention_mask=inputs["attention_mask"],
      token_type_ids=inputs["token_type_ids"] if "token_type_ids" in inputs else (None)
    )
    if self.avg_pooling:  # Averaging all hidden states
      orig_outputs = (outputs[0] * inputs["attention_mask"].unsqueeze(-1)).sum(1)\
                / inputs["attention_mask"].sum(1).unsqueeze(-1)
    else:  # Use [CLS]
      orig_outputs = outputs[0][:, 0, :]

    #2. do dropout on output of encoder (CLS by default)
    orig_outputs_drop = self.dropout(orig_outputs)

    #3. pass through linear layer to get interpretable sparse layer (ie, prior logits)  
    #this layer has not been transformed via sigmoid to keep logits unbounded!
    ier_layer = self.classifier(orig_outputs_drop)

    #4. new pass through linear layer returning to 768 embedding size expected by BERT
    down_project = self.down_project(ier_layer)

    #5. feed downprojected sparse layer into BERT encoder ( feed in embeddings)
    outputs_l1 = self.decoder_l1(down_project)
    if self.num_ff_layers:
      logits = outputs_l1

    elif self.num_ff_layers == 3:
      outputs_l2 = self.decoder_l2(outputs_l1)
      outputs_l3 = self.decoder_l3(outputs_l2)
      logits = outputs_l3

    elif self.num_ff_layers == 5:
      outputs_l2 = self.decoder_l2(outputs_l1)
      outputs_l3 = self.decoder_l3(outputs_l2)
      outputs_l4 = self.decoder_l4(outputs_l3)
      outputs_l5 = self.decoder_l5(outputs_l4)
      logits = outputs_l5

    elif self.num_ff_layers == 8:
      outputs_l2 = self.decoder_l2(outputs_l1)
      outputs_l3 = self.decoder_l3(outputs_l2)
      outputs_l4 = self.decoder_l2(outputs_l3)
      outputs_l5 = self.decoder_l4(outputs_l4)
      outputs_l6 = self.decoder_l5(outputs_l5)
      outputs_l7 = self.decoder_l6(outputs_l6)
      outputs_l8 = self.decoder_l7(outputs_l7)
      logits = outputs_l8

    if targets is not None:
      mse_err, type_err = self.define_loss(logits, targets, orig_outputs, ier_layer)
    else:      
      mse_err, type_err = None, None

    #return mse_err, type_err, orig_outputs_drop, logits, ier_layer
    return mse_err, type_err, logits, ier_layer


if __name__ == '__main__':

  # TEST
  parser = argparse.ArgumentParser()
  parser.add_argument("-model_type", default='bert-large-uncased-whole-word-masking')
  parser.add_argument("-hidden_dropout_prob", help="Dropout rate", default=.1, type=float)
  args = parser.parse_args()
  args.avg_pooling = False
  model = TransformerModel(args, 60000)
  for n, p in model.named_parameters():
    print(n)
  print(model.transformer_config)
