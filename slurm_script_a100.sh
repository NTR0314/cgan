#!/bin/sh
#echo $PATH
PATH=$PATH:"/opt/bwhpc/common/devel/miniconda3/bin:/opt/bwhpc/common/devel/miniconda3/condabin"
#echo $PATH
source activate cGAN 
#DEBUG_PATH='/home/kit/stud/ufkpt/NN_pr/cGAN_env.debug'
export LD_LIBRARY_PATH="/home/kit/stud/ufkpt/.conda/envs/cGAN/lib:"$LD_LIBRARY_PATH
# conda list > $DEBUG_PATH
# pip list >> $DEBUG_PATH
# which pip >> $DEBUG_PATH
# which python3 >> $DEBUG_PATH
python3 /home/kit/stud/ufkpt/NN_pr/train.py -t -m 'alpha_a100' -g &> /home/kit/stud/ufkpt/NN_pr/trainig_alpha_a100.log
