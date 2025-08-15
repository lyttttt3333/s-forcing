import cv2
from PIL import Image
import torch
import torch.distributed as dist
import os

from utils.wan_wrapper import WanVAEWrapper
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy 

import torchvision.transforms.functional as TF
import imageio
from tqdm import tqdm

def save_video(video_tensor, save_path, fps=30, quality=9, ffmpeg_params=None):
    """
    保存一个形状为 [C, T, H, W] 的视频张量到文件
    video_tensor: torch.Tensor, float32, 值域 [-1, 1] 或 [0, 1]
    """
    assert video_tensor.dim() == 4, "video_tensor 必须是 4 维 [C, T, H, W]"
    
    # 转到 CPU，避免 GPU 张量直接参与 numpy 转换
    video_tensor = video_tensor.detach().cpu()

    # [-1, 1] → [0, 255]
    if video_tensor.min() < 0:
        video_tensor = (video_tensor + 1) / 2  # [-1,1] → [0,1]
    video_tensor = (video_tensor * 255).clamp(0, 255).byte()

    # [C, T, H, W] → [T, H, W, C]
    video_tensor = video_tensor.permute(1, 2, 3, 0).numpy()

    # 保存视频
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in tqdm(video_tensor, desc="Saving video"):
        writer.append_data(frame)
    writer.close()


def center_crop_resize(img: Image.Image, target_h: int, target_w: int) -> Image.Image:
    ih, iw = img.height, img.width
    target_ratio = target_w / target_h
    input_ratio = iw / ih

    if input_ratio > target_ratio:
        # 输入更宽 → 按高度对齐，裁掉左右多余部分
        new_width = round(ih * target_ratio)
        x1 = (iw - new_width) // 2
        y1 = 0
        cropped = img.crop((x1, y1, x1 + new_width, ih))
    else:
        # 输入更高 → 按宽度对齐，裁掉上下多余部分
        new_height = round(iw / target_ratio)
        x1 = 0
        y1 = (ih - new_height) // 2
        cropped = img.crop((x1, y1, iw, y1 + new_height))

    return cropped.resize((target_w, target_h), Image.LANCZOS)


def video_to_tensor(video_path: str, target_frames: int, target_h: int, target_w: int, device: str):
    """
    从视频读取帧，每隔一帧取一次，并裁剪成 (C, F, H, W) tensor。
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    for i in range(0, total_frames, 2):  # 每隔一帧取一次
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = cap.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = center_crop_resize(img, target_h, target_w)
        img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5)  # [-1, 1]
        frames.append(img_tensor)

        if len(frames) >= target_frames:
            break

    print(f"############ {len(frames)} and {target_frames} ###########")

    cap.release()
    if len(frames) < target_frames:
        # 补齐帧
        while len(frames) < target_frames:
            frames.append(frames[-1].clone())

    video_tensor = torch.stack(frames, dim=1).unsqueeze(0).to(device)  # (1, C, F, H, W)
    return video_tensor

def encode_video(vae, video_tensor):
    """
    对视频 tensor 进行 VAE 编码
    """
    vae = vae.to(torch.bfloat16)
    z = vae.encode_to_latent([video_tensor])[0]
    return z


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = "cuda"

    input_dir = "/lustre/fsw/portfolios/av/users/shiyil/jfxiao/AirVuz-V2-08052025/videos"
    output_dir = "/lustre/fsw/portfolios/av/users/shiyil/jfxiao/AirVuz-V2-08052025/video_token"
    os.makedirs(output_dir, exist_ok=True)

    video_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".mp4")
    ]
    video_files.sort()[100:]

    vae = WanVAEWrapper().to(torch.float16).to(device)

    # 设置目标视频形状
    target_frames = 84
    target_h = 480
    target_w = 640

    for i in range(rank, len(video_files), world_size):
        video_path = video_files[i]
        base_name = os.path.basename(video_path).split(".")[0]
        save_path = os.path.join(output_dir, f"{base_name}.pth")

        video_tensor = video_to_tensor(video_path, target_frames, target_h, target_w, device)
        video_tensor = video_tensor[0]
        save_path = "test.mp4" # For testing, change this to your desired path
        save_video(video_tensor, save_path)
        break
        # latent = encode_video(vae, video_tensor).to(torch.bfloat16)
        # torch.save(latent, save_path)
        # print(f"[GPU {rank}] done {base_name}")


if __name__ == "__main__":
    main()