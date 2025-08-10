export CUDA_VISIBLE_DEVICES=0,1,2,3 #,4,5,6,7
torchrun --nnodes=1 --nproc_per_node=4 --master_port=29505 train.py \
  --config_path configs/self_forcing_dmd.yaml \
  --logdir logs/ \