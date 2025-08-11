import os
import pandas as pd
import torch
import torch.distributed as dist
from accelerate import Accelerator

from utils.wan_wrapper import WanTextEncoder

def get_text_token(text_encoder, prompt):
    embed = text_encoder(
        text_prompts=[prompt])["prompt_embeds"]
    return embed

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    
    input_csv = "/lustre/fsw/portfolios/av/users/shiyil/jfxiao/AirVuz-V2-08052025/metadata.csv"  
    output_dir = "/lustre/fsw/portfolios/av/users/shiyil/jfxiao/AirVuz-V2-08052025/text_tokens"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    total_rows = len(df)

    device = "cuda"

    text_encoder = WanTextEncoder().to(torch.float16).to(device)
    for i in range(rank, total_rows, world_size):
        row = df.iloc[i]
        prompt = str(row['prompt']) if pd.notna(row['prompt']) else ""
        video = row['video']
        tokens = get_text_token(text_encoder, prompt).to(torch.bfloat16)
        basename = video.split(".")[0]
        save_path = os.path.join(output_dir, f"{basename}.pth")
        torch.save(tokens, save_path)
        print(f"[GPU {rank}] done {basename}")

if __name__ == "__main__":
    main()