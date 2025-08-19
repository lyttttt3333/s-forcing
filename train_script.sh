#!/bin/bash

git pull origin dmd

export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nnodes=1 --nproc_per_node=4 --master_port=29505 train.py \
  --config_path configs/self_forcing_gan.yaml \
  --logdir logs/