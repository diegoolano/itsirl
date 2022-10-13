# Downstream Experiments for evaluation of ItsIRL 


all files currently expect to log runs via Weights And Biases ( wandb ) so you'll need a login key to use. 

# 1. Entity Label Classification on Cancer Genetics Data
<pre>
 - See experiments/run_elc.sh to generate command to run
 - training data expected to be found in ./elc_data/

 - hyperparams for ItsIRL End2End fine-tuned run reported in Table 1 ( Acc. 95.73 )

   CUDA_DEVICES=1 python3 elc.py 
        --model_id=debug_elc_0519_endtoend_no_clipping_lr2e5 
        --model_type=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext 
        --mtype=endtoend 
        --reload_model_name=0504_prior_ier_frozen_with_ffn_400000 
        --goal=medwiki 
        --env=0720_600k_full_orig 
        --learning_rate_enc=2e-5 
        --learning_rate_cls=1e-3 
        --per_gpu_train_batch_size=8 
        --per_gpu_eval_batch_size=8 
        --gradient_accumulation_steps=4 
        --num_epoch=10 
        --save_model --by_epoch --do_clipping
 
 - hyperparams for ItsIRL Decoder fine-tuned run reported in Table 1 ( Acc. 91.95 )
   CUDA_DEVICES=1 python3  elc.py 
        --model_id=debug_elc_0523_v1_ft_softmax_200eps_bothon_load_optimizers_enc_cls_ffn2e5 
        --model_type=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext 
        --mtype=ffn_with_softmax 
        --reload_model_name=0504_prior_ier_frozen_with_ffn_400000 
        --goal=medwiki 
        --env=0720_600k_full_orig 
        --learning_rate_enc=2e-5 
        --learning_rate_cls=1e-3 
        --learning_rate_ffn=2e-5 
        --per_gpu_train_batch_size=8 
        --per_gpu_eval_batch_size=8 
        --gradient_accumulation_steps=4 
        --num_epoch=200 
        --save_model --by_epoch --do_clipping --do_grad_acc
</pre>
 Both the above runs are based on this ItsIRL base model checkpoint which is then fine tuned (endtoend, or decoderonly) on ELC task


BIER interpretable (87.5) and PubMedBERT dense (96.1) numbers found in [ [Colab url](https://colab.research.google.com/drive/1CDwTG71UkTKLxMhk7uDm4DHX2YABYbEf?usp=sharing) ] 




# 2. Sentence Similarity on BIOSSES data
<pre>
  - see experiments/run_biosses.sh to generate commands for run
  - training data expected in ...

  - hyperparams for ItsIRL End2End fine-tuned run reported in Table 2 ( 1.14 MSE )

    CUDA_DEVICES=1 python3  biosses.py 
        --model_id=bs_0602_e2e_redo_60eps_3e5_ffn_0518 
        --model_type=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext 
        --mtype=endtoend 
        --reload_model_name=0518_prior_ier_frozen_with_ffn_200eps_1300000 
        --goal=medwiki 
        --env=0720_600k_full_orig 
        --learning_rate_enc=2e-5 
        --learning_rate_cls=1e-3 
        --per_gpu_train_batch_size=8 
        --per_gpu_eval_batch_size=8 
        --gradient_accumulation_steps=4 
        --learning_rate_ffn=3e-5 
        --num_epoch=60 
        --by_epoch --do_clipping --do_grad_accum --save_model

 - hyperparams for ItsIRL Decoder fine-tuned run reported in Table 2 ( MSE 1.59 )
    CUDA_DEVICES=1 python3  biosses.py 
        --model_id=bs_0603_ft_60eps_1p5e5_ffn_0518 
        --model_type=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext 
        --mtype=ffn_with_softmax 
        --reload_model_name=0518_prior_ier_frozen_with_ffn_200eps_1300000 
        --goal=medwiki 
        --env=0720_600k_full_orig 
        --learning_rate_enc=2e-5 
        --learning_rate_cls=1e-3 
        --per_gpu_train_batch_size=8 
        --per_gpu_eval_batch_size=8 
        --gradient_accumulation_steps=4 
        --learning_rate_ffn=1.5e-5 
        --num_epoch=60 
        --by_epoch --do_clipping --do_grad_accum --save_model

    Baseline BIER (5.05) and PubMedBERT dense (1.14) numbers are in this repo notebooks/BIOSSES_using_BIERs_and_PubMedBERT_akbc.ipynb

    Entity Type Sparsity numbers found in notebooks/ItsIRL_BIOSSES_entity_type_sparsity_akbc.ipynb

</pre>
