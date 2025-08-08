CUDA_VISIBLE_DEVICES=0 python inference.py \
    --config_path configs/self_forcing_dmd_infer.yaml \
    --output_folder videos/self_forcing_dmd \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --output_folder video/test \
    --num_output_frames 168 \
    --use_ema

#--data_path prompts/MovieGenVideoBench_extended.txt \