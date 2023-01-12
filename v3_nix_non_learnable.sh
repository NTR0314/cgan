#!/bin/sh
PATH=$PATH:"/opt/bwhpc/common/devel/miniconda3/bin:/opt/bwhpc/common/devel/miniconda3/condabin"
source activate cGAN 
export LD_LIBRARY_PATH="/home/kit/stud/ufkpt/.conda/envs/cGAN/lib:"$LD_LIBRARY_PATH

model_name='v3_nix_non_learnable'
model_path="/home/kit/stud/ufkpt/NN_pr/models/${model_name}"
log_path="${model_path}/training.log"

echo "Creating path"
mkdir -p $model_path
echo "Starting training"
python3 "/home/kit/stud/ufkpt/NN_pr/train_${model_name}.py" -m $model_name -g -t &> $log_path
echo "Ending training"
