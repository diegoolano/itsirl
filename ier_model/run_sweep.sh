export GPU=$1
launch_cmd="CUDA_VISIBLE_DEVICES=${GPU} CUDA_CACHE_PATH='/home/diego/.cache/' wandb sweep sweep_bayes.yaml"
echo $launch_cmd
