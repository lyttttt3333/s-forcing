export CUDA_VISIBLE_DEVICES=0,1
git pull origin dmd
torchrun --nnodes=1 --nproc_per_node=2 --master_port=29505 test_pretrained.py 