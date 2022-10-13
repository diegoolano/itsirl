#!/usr/bin/env bash
export PY_PATH="xxx"
export PROJ_BASE="xxx"
export SCRIPT_PATH="${PROJ_BASE}/experiments/"
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
#sh run_biosses.sh 3 bs_0602_ft_10eps_2e5_ffn8_pretrainedhas8 0528_pretrain_8layers_400000 ffn_with_softmax "  --learning_rate_ffn=2e-5 --num_epoch=10 --num_ff_layers=8 --by_epoch --do_clipping --do_grad_accum --pretrained_has_8layers  --save_model "


launch_cmd="CUDA_VISIBLE_DEVICES=${GPU} CUDA_CACHE_PATH='/home/xxxx/.cache/' ${PY_PATH}python -u ${SCRIPT_PATH}biosses.py \
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
