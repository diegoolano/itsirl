#!/usr/bin/env bash
export PY_PATH="/home/diego/new_biers22/nbiers_env/bin/"
export PROJ_BASE="/home/diego/new_biers22/"
export SCRIPT_PATH="${PROJ_BASE}nbiers/experiments/"
export MEDWIKI_ENV="0720_600k_full_orig"   
export MODELTYPE="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

export GPU=$1
export OUTHANDLE=$2
export RELOAD=$3   #"0504_prior_ier_frozen_with_ffn_best"
export MTYPE=$4    #ffn_with_linear_head_to_tune,  ffn_with_softmax , endtoend

#ADDITIONAL PARAMS
str=$5
str="${str#?}" # removes first character
str="${str%?}"  # removes last character


#example call
#sh run_hoc.sh 3 debug_hoc_0506_ft_linear 0504_prior_ier_frozen_with_ffn_best ffn_with_linear



launch_cmd="CUDA_VISIBLE_DEVICES=${GPU} CUDA_CACHE_PATH='/home/diego/.cache/' ${PY_PATH}python -u ${SCRIPT_PATH}hoc.py \
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
