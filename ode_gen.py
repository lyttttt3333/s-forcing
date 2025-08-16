import gc
import logging

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import TextDataset, MixedDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from model import CausVid, DMD, SiD, GAN, DMD_GEN
from utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.wan_wrapper import WanDiffusionWrapper

from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from torch.nn import ModuleList
from omegaconf import OmegaConf
import torch
import wandb
import time
import os
import imageio
from tqdm import tqdm
import torchvision
import numpy as np
import math

def generate_from_latent(real_score, sample_scheduler, frame_token, uncond_dict, cond_dict, device, select_index):
    z = frame_token
    sp_size = 1
    patch_size = (1, 2, 2)
    vae_stride = (4, 16, 16)
    ow = 1248 
    oh = 704
    F = 84
    real_guidance_scale = 5.0
    
    seq_len = ((F - 1) // vae_stride[0] + 1) * (
        oh // vae_stride[1]) * (ow // vae_stride[2]) // (
            patch_size[1] * patch_size[2])
    seq_len = int(math.ceil(seq_len / sp_size)) * sp_size

    noise = torch.randn(
        48, (F - 1) // vae_stride[0] + 1,
        oh // vae_stride[1],
        ow // vae_stride[2],
        dtype=torch.bfloat16,
        device=device)

    with (
            torch.amp.autocast('cuda'),
            torch.no_grad(),
    ):

        # sample videos
        latent = noise # shape [48, 21, 44, 78]
        mask = torch.ones_like(noise)
        mask[:, 0] = 0
        latent = (1. - mask) * z + mask * latent

        trajectory = []

        for _, t in enumerate(tqdm(sample_scheduler.timesteps)):

            latent_model_input = latent.to(device)
            timestep = [t]

            timestep = torch.stack(timestep).to(device)
            print(f"before {mask.shape}")
            temp_ts = (mask[0][:, ::2, ::2] * timestep).flatten()
            # print(f"first stage {temp_ts.shape}")
            # print(f"##################time_step", temp_ts.size(0), seq_len)
            temp_ts = torch.cat([
                temp_ts,
                temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
            ])
            # print(f"second stage {temp_ts.shape}")
            timestep = temp_ts.unsqueeze(0)
            # print("##################time_step",timestep.shape)
            # print("##################", timestep)


            pred_real_image_cond = real_score(
                noisy_image_or_video=latent_model_input.unsqueeze(0),
                conditional_dict=cond_dict,
                timestep=timestep,
                memory_condition=False,
                seq_len=seq_len,
            )

            pred_real_image_uncond = real_score(
                noisy_image_or_video=latent_model_input.unsqueeze(0),
                conditional_dict=uncond_dict,
                timestep=timestep,
                memory_condition=False,
                seq_len=seq_len,
            )

            pred_real_image = pred_real_image_cond + (
                pred_real_image_cond - pred_real_image_uncond
            ) * real_guidance_scale

            temp_x0 = sample_scheduler.step(
                pred_real_image.unsqueeze(0),
                t,
                latent_model_input.unsqueeze(0),
                return_dict=False)[0]
            latent = temp_x0.squeeze(0)
            latent = (1. - mask) * z + mask * latent

            trajectory.append(latent.unsqueeze(0))

    trajectory = torch.stack(trajectory, dim=1)
    trajectory = trajectory[:, select_index]

    return trajectory

def load_batch(batch, dtype, device):
    for key in batch.keys():
        if key != "base_name" and key != "clean_token":
            path = batch[key]
            tensor = torch.load(path, map_location="cpu").to(dtype).to(device)
            batch[key] = tensor
        if key == "clean_token":
            path = batch[key]
            tensor = torch.load(path, map_location="cpu").to(dtype).to(device)
            batch[key] = tensor.unsqueeze(0)  # Ensure it has batch dimension
    return batch

def init_model(device):
    model = WanDiffusionWrapper().to(device).to(torch.float32)
    model.set_module_grad(
        {
            "model": False
        }
    )

    scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=1000,
                    shift=1,
                    use_dynamic_shifting=False)

    global_dict_path = "/lustre/fsw/portfolios/av/users/shiyil/jfxiao/s-forcing/ref_lib/global_embed_dict.pt"
    global_text_token = torch.load(global_dict_path, map_location="cpu").to(device).to(torch.float32)
    unconditional_dict = {'prompt_embeds': global_text_token}

    meta_path = "/lustre/fsw/portfolios/av/users/shiyil/jfxiao/AirVuz-V2-08052025/meta.csv"
    root_path = "/lustre/fsw/portfolios/av/users/shiyil/jfxiao/AirVuz-V2-08052025"
    dataset = MixedDataset(meta_path, root_path)

    return model, scheduler, unconditional_dict, dataset
    

        


if __name__ == "__main__":

    output_folder = "xxx"

    launch_distributed_job()
    global_rank = dist.get_rank()

    device = torch.cuda.current_device()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, sample_scheduler, unconditional_dict, dataset = init_model(device=device)

    if global_rank == 0:
        os.makedirs(output_folder, exist_ok=True)

    for index in tqdm(range(int(math.ceil(len(dataset) / dist.get_world_size()))), disable=dist.get_rank() != 0):
        prompt_index = index * dist.get_world_size() + dist.get_rank()
        if prompt_index >= len(dataset):
            continue

        ### unwrap data
        data = dataset[prompt_index]
        data = load_batch(data, torch.float32, device)
        text_token = data["text_token"]
        frame_token = data["frame_token"]
        base_name = data["base_name"]
        conditional_dict = {'prompt_embeds': text_token}


        sample_scheduler.set_timesteps(5, device=device, shift=5)

        
        trajectory = generate_from_latent(real_score=model.generator,
                                sample_scheduler=sample_scheduler,
                                frame_token=frame_token,
                                uncond_dict=unconditional_dict,
                                cond_dict=conditional_dict,
                                device=device,
                                select_index=[0,1,3,4])
                                #[0, 36, 44, -1])
        
        torch.save(
            {base_name: trajectory.cpu().detach()},
            os.path.join(output_folder, f"{base_name}.pt")
        )

        print(f"GPU[{global_rank}]: {base_name} {trajectory.shape}")

    dist.barrier()
        

