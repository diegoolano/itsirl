#!/usr/bin/env bash
export PY_PATH="xxxx"
export PROJ_BASE="xxxx"
export SCRIPT_PATH="${PROJ_BASE}experiments/"
export MEDWIKI_ENV="0720_600k_full_orig"   
export MODELTYPE="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

export GPU=$1
export OUTHANDLE=$2
export RELOAD=$3   #"0504_prior_ier_frozen_with_ffn_best"
export MTYPE=$4    #"ffn_with_linear"  #ffn_with_linear_head_to_tune,  ffn_with_softmax 

#ADDITIONAL PARAMS
str=$5
str="${str#?}" # removes first character
str="${str%?}"  # removes last character


#example call
#sh run_elc.sh 3 debug_elc_0506_ft_linear 0504_prior_ier_frozen_with_ffn_best ffn_with_linear
#sh run_elc.sh 3 debug_elc_0509_ft_linear_one_epoch 0504_prior_ier_frozen_with_ffn_best ffn_with_linear_head_to_tune

#sh run_elc.sh 3 debug_elc_0511_ft_linear_8eps 0504_prior_ier_frozen_with_ffn_best ffn_with_linear_head_to_tune
#sh run_elc.sh 1 debug_elc_0511_ft_softmax_8eps 0504_prior_ier_frozen_with_ffn_best ffn_with_softmax

#test et
#sh run_elc.sh 1 debug_elc_0513_softmax_1eps 0504_prior_ier_frozen_with_ffn_best ffn_with_softmax " --debug_mode "


#0516
# sh run_elc.sh 1 debug_elc_0516_ft_softmax_1eps 0504_prior_ier_frozen_with_ffn_best ffn_with_softmax " --debug_mode --save_types "
# if the above works do it for 8 epochs and without debug_mode ( but yes with save_types)


#sh run_elc.sh 0 elc_0606_ft_softmax_300eps_ffn2e5_w_gradacc_sepopts 0504_prior_ier_frozen_with_ffn_400000 ffn_with_softmax " --learning_rate_ffn=2e-5 --num_epoch=200 --save_model --by_epoch --do_clipping --do_grad_accum --separate_opts "


launch_cmd="CUDA_VISIBLE_DEVICES=${GPU} CUDA_CACHE_PATH='/home/xxx/.cache/' ${PY_PATH}python -u ${SCRIPT_PATH}elc.py \
--model_id=${OUTHANDLE} \
--model_type=${MODELTYPE} \
--mtype=${MTYPE} \
--reload_model_name=${RELOAD} \
--goal=medwiki \
--env=${MEDWIKI_ENV} \
--learning_rate_enc=2e-5 \
--learning_rate_cls=1e-3 \
--per_gpu_train_batch_size=8 \
--per_gpu_eval_batch_size=8 \
--gradient_accumulation_steps=4 \
${str}
| tee log/${OUTHANDLE}.log"

echo $launch_cmd
