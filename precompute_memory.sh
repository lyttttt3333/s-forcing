#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
torchrun --nnodes=1 --nproc_per_node=6 --master_port=29505 precompute_memory.py