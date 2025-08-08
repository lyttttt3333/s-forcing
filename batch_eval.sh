#!/bin/bash
#SBATCH --job-name=video_generation               # Job name
#SBATCH --output=out_%j.txt             # Output file
#SBATCH --error=err_%j.txt              # Error file
#SBATCH --partition=a100                # Requested partition
#SBATCH --gpus=1                        # Requested number of GPUs
#SBATCH --time=5:00:00                 # Time limit (hh:mm:ss)
#SBATCH --mail-type=BEGIN               # Send email when job starts
#SBATCH --mail-use=liyitong.thu@gmail.com   # Destination Email

source ~/miniforge3/bin/activate
conda activate verl

frames_list=(168)

for frames in "${frames_list[@]}"; do
    echo "Running for frames=$frames at $(date)"

    python inference.py \
        --config_path configs/self_forcing_dmd.yaml \
        --checkpoint_path logs/self_forcing_afetr_3000/checkpoint_model_004000/model.pt \
        --data_path prompts/MovieGenVideoBench_extended.txt \
        --output_folder video/21train_7000steps \
        --num_output_frames ${frames} \
        --use_ema
done