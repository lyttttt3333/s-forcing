#!/bin/bash
#SBATCH --job-name=video_generation               # Job name
#SBATCH --output=out_%j.txt             # Output file
#SBATCH --error=err_%j.txt              # Error file
#SBATCH --partition=a100                # Requested partition
#SBATCH --gpus=4                       # Requested number of GPUs
#SBATCH --time=20:00:00                 # Time limit (hh:mm:ss)
#SBATCH --mail-type=BEGIN               # Send email when job starts
#SBATCH --mail-use=liyitong.thu@gmail.com   # Destination Email

source ~/miniforge3/bin/activate
conda activate verl

torchrun --nnodes=1 --nproc_per_node=4 --master_port=29509 train.py \
  --config_path configs/self_forcing_dmd.yaml \
  --logdir logs/self_forcing_afetr_3000 \