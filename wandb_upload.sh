#!/bin/bash

git pull origin master

export CUDA_VISIBLE_DEVICES=0
torchrun --nnodes=1 --nproc_per_node=1 --master_port=29505 wandb_upload.py 