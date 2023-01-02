#!/bin/sh
PATH=$PATH:"/opt/bwhpc/common/devel/miniconda3/bin:/opt/bwhpc/common/devel/miniconda3/condabin"
source activate cGAN 
export LD_LIBRARY_PATH="/home/kit/stud/ufkpt/.conda/envs/cGAN/lib:"$LD_LIBRARY_PATH

model_name='v2_stride'
model_path="/home/kit/stud/ufkpt/NN_pr/models/${model_name}"
log_path="${model_path}/training.log"

mkdir -p $model_path
python3 "/home/kit/stud/ufkpt/NN_pr/train_${model_name}.py" -m $model_name -g -t &> $log_path
