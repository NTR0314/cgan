#!/bin/sh
PATH=$PATH:"/opt/bwhpc/common/devel/miniconda3/bin:/opt/bwhpc/common/devel/miniconda3/condabin"
source activate cGAN 
export LD_LIBRARY_PATH="/home/kit/stud/ufkpt/.conda/envs/cGAN/lib:"$LD_LIBRARY_PATH

model_name='v2_all_both_do'
model_path="/home/kit/stud/ufkpt/NN_pr/models/${model_name}"
log_path="${model_path}/training.log"

mkdir -p $model_path
python3 "/home/kit/stud/ufkpt/NN_pr/train_v2_sn_lrelu_tconv_ls_bn.py" -m $model_name -g -t --dog --dod &> $log_path
