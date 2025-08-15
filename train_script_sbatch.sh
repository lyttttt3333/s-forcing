#!/bin/bash
#SBATCH --job-name=self_forcing_gan
#SBATCH --partition interactive,polar,grizzly,polar3,polar4,backfill_singlenode,batch_singlenode,backfill_block1
#SBATCH --account=av_alpamayo_cosmos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=8
#SBATCH --mem=512G
#SBATCH --time=12:10:00
#SBATCH --output=logs/hello-%j.out
#SBATCH --error=logs/hello-%j.err

git pull origin master

source /lustre/fsw/portfolios/av/users/shiyil/anaconda3/etc/profile.d/conda.sh
conda activate self_forcing

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nnodes=1 --nproc_per_node=8 --master_port=29505 train.py \
  --config_path configs/self_forcing_gan.yaml \
  --logdir logs/