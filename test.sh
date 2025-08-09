export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nnodes=1 --nproc_per_node=4 --master_port=29505 test.py 