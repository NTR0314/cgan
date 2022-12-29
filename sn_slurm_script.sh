#!/bin/sh
PATH=$PATH:"/opt/bwhpc/common/devel/miniconda3/bin:/opt/bwhpc/common/devel/miniconda3/condabin"
source activate cGAN 
export LD_LIBRARY_PATH="/home/kit/stud/ufkpt/.conda/envs/cGAN/lib:"$LD_LIBRARY_PATH


model_name='alpha_sn'


python3 /home/kit/stud/ufkpt/NN_pr/train_sn.py -m $model_name -g -t &> /home/kit/stud/ufkpt/NN_pr/$model_name.log
