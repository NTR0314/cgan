# Neural Nets Praktikum WS22/23

## CGAN
This repository contains the code used for the NN Praktikum WS22/23 of the **conditional Generative Advesarial Network** group.

## Dependencies
The repository depends on several libraries:
- scipy
- torchvision
- torchmetrics
- matplotlib

## How to run
The file `train_cgan.py` is the entrypoint for training a cGAN model.
It expects different arguments which can be shown by using `python3 train_cgan.py --help`

The arguments are:

| Argument | Description |
| --- | --- |
| `-h, --help` | show this help message and exit |
| `-g, --gen_images` | Don't generate images (default: True) |
| `-t, --training` | (Continue to) train the model (default: False) |
| `-m MODEL_NAME, --model_name MODEL_NAME` | Model name. If specified the model will be saved to that directory and if already existing the model will be loaded. (default: None) |
| `--no_last_inception` | If this arg is set then the last inception scores will not be calculated. This is mainly used for local computation. (default: False) |
| `-s, --sloppy` | If this arg is set then a sloppy IS of 50 images will be calculated instead of FID and IS of 1000 images. (default: False) |
| `--ngf NGF` | ngf dim (default: 64) |
| `--spectral` | Use Spectral normalization (default: False) |
| `--lrelu` | Use leaky relu (default: False) |
| `--tconv` | Use transposed convulation (default: False) |
| `--leastsquare` | Use least square loss (default: False) |
| `--batchnorm` | Use batch norm (default: False) |
| `--embedding` | Use embedding matrix (default: False) |
| `--noresidual` | don't use residual path (default: False) |
| `--thousand_fid` | use 1k images for each class instead of 100 (default: False) |

## Example script
This script is an example of how to train the cGAN model with all the improvements with a set up conda environment `cGAN`.
```shell
#!/bin/sh
PATH=$PATH:"/opt/bwhpc/common/devel/miniconda3/bin:/opt/bwhpc/common/devel/miniconda3/condabin"
source activate cGAN 
export LD_LIBRARY_PATH="/home/kit/stud/ufkpt/.conda/envs/cGAN/lib:"$LD_LIBRARY_PATH

model_name='v3_all_learnable'
# Used for loading a checkpoint to continue to train on.
model_path="/home/kit/stud/<user>/NN_pr/models/${model_name}"
log_path="${model_path}/training.log"

mkdir -p $model_path
python3 "/home/kit/stud/ufkpt/NN_pr/train_cgan.py" -m $model_name -g -t -s --batchnorm --leastsquare --tconv --lrelu --spectral &>> $log_path

```