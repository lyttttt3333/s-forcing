#!/bin/bash

set -e
workdir='..'
model_name='StreamVGGT'
model_weights="/lustre/fsw/portfolios/av/users/shiyil/jfxiao/StreamVGGT/ckpt/checkpoints.pth"

export CUDA_VISIBLE_DEVICES=0,1,2,3
output_dir="/lustre/fsw/portfolios/av/users/shiyil/jfxiao/StreamVGGT/eval_results/mv_recon/${model_name}_${ckpt_name}"
echo "$output_dir"
accelerate launch --num_processes 1 --main_process_port 29572 minimal_inference.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --model_name "$model_name" \
     