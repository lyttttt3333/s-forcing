export CUDA_VISIBLE_DEVICES=0,1
torchrun --nnodes=1 --nproc_per_node=2 --master_port=29505 train.py \
  --config_path configs/self_forcing_dmd.yaml \
  --logdir logs/ \
