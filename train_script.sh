#!/bin/bash


conda activate self_forcing
git pull origin dmd

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
torchrun --nnodes=1 --nproc_per_node=7 --master_port=29505 train.py \
  --config_path configs/self_forcing_gan.yaml \
  --logdir logs/