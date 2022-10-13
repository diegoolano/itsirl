#!/usr/bin/env bash

# expected call:
# sh validation.sh 3 SAVED_MODEL_NAME "--dev_tiny --et_labmda=10"
#  1st param is GPU
#  2nd param is trained model to eval
#  3nd param is outputname_piece
#  3rd param are any additional flags ( --dev_tiny, --et_lambda 1 ,  --num_ff_layers 1, etc)

export PY_PATH="/home/diego/new_biers22/nbiers_env/bin/"
export DATA_PATH="/data/diego/new_biers22/data/"
export PROJ_BASE="/home/diego/new_biers22/"
export SCRIPT_PATH="${PROJ_BASE}nbiers/ier_model/"
export OUTPUT_PATH="${PROJ_BASE}data/model_out/"



#1. SELECT MODEL BASE
#export MODELTYPE="bert-large-uncased-whole-word-masking"
export MODELTYPE="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

#2.  SELECT DATASET / TYPE SYSTEM  
# "data sub for medwiki", choices=["0720_3k_full","0720_3k_full_orig","0720_3k_drugs","0720_600k_full","0720_600k_full_orig","0720_600k_drugs"])
export MEDWIKI_ENV="0720_600k_full_orig"   #should this be 0720_600k_full ?


#IMPORTANT the model ckpts path is set in transformer_constant.py 

#3.  SELECT MODEL CHECKPOINT TO LOAD AND USE FOR TESTING (with no .pt at end )
#export SAVEDMODEL="0923_e1_run_0720_600k_full_microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

export GPU=$1
export SAVEDMODEL=$2

#4.  SELECT OUTPUT NAME for metric result storage
#export OUTHANDLE="0923_e1_run_0720_600k_full_microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext.results"

export OUT_DEETS=$3
export OUTHANDLE="${SAVEDMODEL}.${OUT_DEETS}_results"


#5. DECIDE IF YOU ARE DOING val or test ( validation data or test data evaluation )
export MODE="val"

export LOG_PERIOD=100
export TRAIN_ASSESS=100
export EVAL_ASSESS=5000
export SAVE_MODEL_PER=100000

#6. ADDITIONAL PARAMS
str=$4
str="${str#?}" # removes first character
str="${str%?}"  # removes last character


launch_cmd="CUDA_VISIBLE_DEVICES=${GPU} CUDA_CACHE_PATH='/home/diego/.cache/' ${PY_PATH}python3 -u ${SCRIPT_PATH}run_et.py \
--model_id=${OUTHANDLE} \
--model_type=${MODELTYPE} \
--load \
--reload_model_name=${SAVEDMODEL} \
--mode=val \
--examples_limit=2000 \
--goal=medwiki \
--env=${MEDWIKI_ENV} \
--learning_rate_enc=2e-5 \
--learning_rate_cls=1e-3 \
--per_gpu_train_batch_size=8 \
--per_gpu_eval_batch_size=8 \
--gradient_accumulation_steps=4 \
${str}
| tee log/${OUTHANDLE}.log"

#-log_period ${LOG_PERIOD} \
#-eval_period ${TRAIN_ASSESS} \
#-eval_after ${EVAL_ASSESS} \
#-save_period ${SAVE_MODEL_PER} \

#echo "Launching testing job and saving results to $OUTPUT_PATH and outfiles $outfile_f"
echo $launch_cmd

#result=$($launch_cmd)  #doesn't work.. might have been lack of being in right env
#echo $result
