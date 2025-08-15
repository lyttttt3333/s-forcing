#!/bin/bash
#SBATCH --job-name=video_generation               # Job name
#SBATCH --output=out_%j.txt             # Output file
#SBATCH --error=err_%j.txt              # Error file
#SBATCH --partition=a100                # Requested partition
#SBATCH --gpus=4                       # Requested number of GPUs
#SBATCH --time=20:00:00                 # Time limit (hh:mm:ss)
#SBATCH --mail-type=BEGIN               # Send email when job starts
#SBATCH --mail-use=liyitong.thu@gmail.com   # Destination Email

git pull origin master

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nnodes=1 --nproc_per_node=8 --master_port=29505 train.py \
  --config_path configs/self_forcing_gan.yaml \
  --logdir logs/