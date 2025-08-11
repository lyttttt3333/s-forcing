import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import time
import torch
import argparse
import numpy as np
# import open3d as o3d
import os.path as osp
from torch.utils.data import DataLoader
# from add_ckpt_path import add_path_to_dust3r
from accelerate import Accelerator
from torch.utils.data._utils.collate import default_collate
import tempfile
from tqdm import tqdm
import uuid
import json
from collections import defaultdict

def create_fake_frames(num_frames=7, img_channels=3, img_height=518, img_width=392):
    images = torch.zeros((num_frames, img_channels, img_height, img_width), dtype=torch.bfloat16).to("cuda")
    return images

def get_args_parser():
    parser = argparse.ArgumentParser("3D Reconstruction evaluation", add_help=False)
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="ckpt name",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument(
        "--conf_thresh", type=float, default=0.0, help="confidence threshold"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--revisit", type=int, default=1, help="revisit times")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--use_proj", action="store_true")
    return parser


def generate_tokens(model, frames):
    past_key_values = [None] * model.aggregator.depth
    output_tokens = []
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype = torch.bfloat16):
            with torch.no_grad():
                for i in range(frames.shape[0]):
                    frame = frames[i].unsqueeze(0)
                    aggregated_token, patch_start_idx, past_key_values = model.inference(frame, i, past_key_values=past_key_values)
                    output_tokens.append(aggregated_token)
    output_tokens = torch.cat(output_tokens, dim=0)
    return output_tokens

def save_tokens(output_tokens, save_path):
    torch.save(output_tokens, save_path)



if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device
    model_name = args.model_name
    if model_name == "StreamVGGT":
        from streamvggt.models.streamvggt import StreamVGGT
        model = StreamVGGT()
        ckpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        model.eval()
        model = model.to("cuda").to(torch.bfloat16)
    del ckpt

    frames = create_fake_frames()

    output_tokens = generate_tokens(model, frames)
    print(output_tokens.shape)
    save_tokens(output_tokens, "/lustre/fsw/portfolios/av/users/shiyil/jfxiao/AirVuz-V2-08052025/memory_tokens/test.pth")