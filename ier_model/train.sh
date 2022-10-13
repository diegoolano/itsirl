#!/usr/bin/env bash

# expected call:
# sh train.sh 3 0419_e1_wanda_test "--dev_tiny --et_labmda=10"
#  1st param is GPU
#  2nd param is outputname of model
#  3rd param are any additional flags ( " --dev_tiny --et_lambda=100 --num_ff_layers=1, etc)

# --prior_ier with --freeze_ier loads prior trained IER model and adds 3 layer ffn after it

export PY_PATH="xxxx"
export DATA_PATH="xxxx"
export PROJ_BASE="xxxx"
export SCRIPT_PATH="${PROJ_BASE}ier_model/"
export OUTPUT_PATH="${PROJ_BASE}model_out/"

#1. SELECT MODEL BASE
export MODELTYPE="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

#2. SELECT DATA SET / TYPE SYSTEM
export MEDWIKI_ENV="0720_600k_full_orig"   

#3. SELECT OUTPUT NAME ( THIS IS NOW PASSED IN FROM COMMAND LINE )
export OUTHANDLE=$2

#4. SET LOG AND SAVE FREQUENCY
# SAVE IS  BASED ON 
#  1) if batch_num % args.eval_period == 0 and batch_num > args.eval_after: 
export LOG_PERIOD=100
export TRAIN_ASSESS=1000      #eval_period: do eval at every X batches once eval_after has been reached!
export EVAL_ASSESS=1000      #eval_after: start doing eval after this point step/batch wise  ( if Ma_F1 of model is above prior high, save this run as new best )
#export EVAL_ASSESS=1000000      #eval_after: start doing eval after this point step/batch wise  ( if Ma_F1 of model is above prior high, save this run as new best )
export SAVE_MODEL_PER=100000 #save out 

#5. SET GPU
export GPU=$1

#6. ADDITIONAL PARAMS
str=$3
str="${str#?}" # removes first character
str="${str%?}"  # removes last character

launch_cmd="CUDA_VISIBLE_DEVICES=${GPU} CUDA_CACHE_PATH='/home/xxxx/.cache/' ${PY_PATH}python -u ${SCRIPT_PATH}run_et.py \
--model_id=${OUTHANDLE} \
--model_type=${MODELTYPE} \
--mode=train \
--goal=medwiki \
--env=${MEDWIKI_ENV} \
--learning_rate_enc=2e-5 \
--learning_rate_cls=1e-3 \
--per_gpu_train_batch_size=8 \
--per_gpu_eval_batch_size=8 \
--gradient_accumulation_steps=4 \
--log_period=${LOG_PERIOD} \
--eval_period=${TRAIN_ASSESS} \
--eval_after=${EVAL_ASSESS} \
--save_period=${SAVE_MODEL_PER} \
${str}
| tee log/${OUTHANDLE}.log"

outfile_f="${OUTPUT_PATH}${OUTHANDLE}"

#echo "Launching job and saving results to $OUTPUT_PATH and outfiles $outfile_f"
echo $launch_cmd
