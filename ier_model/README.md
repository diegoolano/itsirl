# How to pre-train ItsIRL models
<pre>
project uses Weights And Biases ( wandb ) for logging experiments so you'll need to have a login key (free for academic purposes)

1. get data, ontology and model checkpoint from the BIERs repo
2. set locations for where data/code located in transformer_contstants.py and train.sh, and see wandb login/details in run_et.py
3. see train.sh for how to pre-train ItsIRL model
4. see experiments/ for how to use pre-trained ItsIRL model on downstream tasks
</pre>
