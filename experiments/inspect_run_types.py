import argparse
import glob
import numpy as np
import sys
import time

# NOT USED 

sys.path.insert(0, '../ier_model')
import transformer_constant

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

def loc_sigmoid(x):
  #sig = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
  sig = 1/(1 + np.exp(-x))
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
    return probs
  else:
    sig_prob = loc_sigmoid(prob)
    sorted_prob = sorted( [(p, id2ans[i]) for i, p in enumerate(sig_prob) if i != 0], key=lambda x: x[0], reverse=True)  #instead of id2ans[i+1]
    if k == -1:
      return [(v1, v2) for v1, v2 in sorted_prob if v1 > threshhold]
    else:
      return sorted_prob[:k] + [(k, v) for k, v in sorted_prob if k <= 0. and v > threshhold]

def show_types(i,pmid, test_data, probs):
  tmp_pmid = "PMID" + pmid.split("PMID")[1]
  pmid = tmp_pmid.split(".")[0]
  exs = test_data[pmid]
  title = exs[0]
  abstract = exs[1]
  mentions = exs[2]
  
  print("Case: ", pmid)
  print("Title: ",title)
  print("Abstract: ", abstract)
  print("Mention: " ,  mentions[i] )
  for i in range(len(probs)):
    if probs[i][0] > .1 :
      print(i, probs[i])

def time_show_single(ets, test_ids, test_data, i=0):
  print("Getting Topk for single")
  t = ets[i]
  pmid = test_ids[i]
  print(type(t), t.shape, t.ndim, t )
  st = time.time()
  probs = get_topk_types(t, k=-1)
  print("Elapsed:", time.time() - st )
  print("Found", len(probs))   #this should have been a matrix and not a list
  print(test_ids[0])
  show_types(i, pmid, test_data, probs)
  print("Total Elapsed: ", time.time() - st)


def time_show_all(ets, test_ids, test_data):
  print("Getting Topk for 10")
  i = 0
  pmid = test_ids[i]
  st = time.time()
  probs = get_topk_types(ets[0:10], k=-1)
  print("Elapsed:", time.time() - st )
  print("Found", len(probs))   #this should have been a matrix and not a list
  print(test_ids[0])
  show_types(i, pmid, test_data, probs[0])
  print("Total Elapsed: ", time.time() - st)


def get_topk_types_v2(prob, k=-1, threshhold = .01):
  #pass in ets that has sig values and threshhold
  id2ans = transformer_constant.ID2ANS_MEDWIKI_DICT['0720_600k_full_orig']
  prob_filt = np.where(prob >= threshhold) 

  probs = [[(0,"") for i in range(150)] for i in range(prob.shape[0])]
  cur_x, cur_y = 0, 0
  n = len(prob_filt[0])
  for i in range(n):
    x,y = prob_filt[0][i], prob_filt[1][i]
    v = prob[x][y]
    if x != cur_x:
      cur_x = x
      cur_y = 0
    try:
      probs[x][cur_y] = (v, id2ans[y])
      cur_y += 1
    except Exception as e:
      print(x,y,v,cur_x,cur_y)  #2731 9891 0.017303403 2731 100

  #sort each row ( prob, entity_type )
  for i in range(prob.shape[0]):
    probs[i].sort(key=lambda y: y[0],reverse=True) 

  if k != -1:
    fin_probs = [[probs[r][i] for i in range(k)] for r in range(prob.shape[0])]
    return fin_probs 
  else:
    return probs

def load_entity_types(f):
  
  train_ids, train_data, ents_per_file_train, dev_ids, dev_data, ents_per_file_dev, test_ids, test_data, ents_per_file_test = load_elc_data()

  print("Loading EntityTypes file: ",f)
  st = time.time()
  ets = np.load(f)
  ets_sig = loc_sigmoid(ets)
  print(ets.shape)   #(6955, 68304)
  print(ets_sig.shape)

  ets0_sig = loc_sigmoid(ets[0])
  print(sum(ets0_sig))   #both give 11.938
  print(sum(ets_sig[0]))

  probs = get_topk_types_v2(ets_sig)
  print(len(probs), len(probs[0]))
  for i in range(40):
    print(i, probs[0][i])
  
  print("\n\n")
  time_show_single(ets, test_ids, test_data)
  print("\n\n")
  #time_show_all(ets, test_ids, test_data)
  print("Final Elapsed:", time.time() - st )

  # 1. NOW look at if we are saving types that we need via --save_types 
  # then run experiment with softmax ffn with save types and ten epochs
  # then run this and save out results ( along with right/wrong pred for each )
  
  # 2. run experiment with end to end and 10 epochs
  # then run this and save out resuling ets as well
  
  # compare two 
  # if in fact we don't have a bug and our work is working
  #   then see how "fixing does on cases" where it didn't work !!


  # look into HOC work
 
  

def load_true_preds():
  f = "out/elc_0516_ft_softmax_8eps_ep8_test_true_labels.npy"


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--types_file", help="types file to load", default="out/elc_0516_ft_softmax_8eps_ep8_test_entity_types.npy")
  parser.add_argument("--single_preds_file", help="types file to load", default="out/elc_0516_ft_softmax_8eps_ep8_test_single_predictions.npy")
  parser.add_argument("--single_preds_probs_file", help="types file to load", default="out/elc_0516_ft_softmax_8eps_ep8_test_single_pred_probs.npy")
  parser.add_argument("--full_preds_file", help="types file to load", default="out/elc_0516_ft_softmax_8eps_ep8_test_full_predictions.npy")
  parser.add_argument("--model_file", help="model to load from which we'll get entity types from test", default="")
  parser.add_argument("--percent_load", help="percent of cases to load ( between 0 and 1 )", default=-1)
  
  args = parser.parse_args()
  if args.model_file == "":
    # load entity types ( in out/ )  examples:
    #   1.8G May 16 14:15 elc_0516_ft_softmax_8eps_ep8_test_entity_types.npy
    #   3.8G May 16 14:12 elc_0516_ft_softmax_8eps_ep8_entity_types.npy
    load_entity_types(args.types_file)
  else:
    #load model and generate entity types for first 
    #1) load model

    #2) load test data
    if args.percent_load == -1:
      #load all tests cases to generate types for
      TODO=2
    else:
      #load percentage of cases to generate types for
      TODO=3
    
    #3) use model to generate entity types on loaded test data
