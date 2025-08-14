import cv2
from PIL import Image
import torch
import torch.distributed as dist
import os

from utils.wan_wrapper import WanVAEWrapper
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy 

import torchvision.transforms.functional as TF
from PIL import Image


def best_output_size(w, h, dw, dh, expected_area):
    # float output size
    ratio = w / h
    ow = (expected_area * ratio)**0.5
    oh = expected_area / ow

    # process width first
    ow1 = int(ow // dw * dw)
    oh1 = int(expected_area / ow1 // dh * dh)
    assert ow1 % dw == 0 and oh1 % dh == 0 and ow1 * oh1 <= expected_area
    ratio1 = ow1 / oh1

    # process height first
    oh2 = int(oh // dh * dh)
    ow2 = int(expected_area / oh2 // dw * dw)
    assert oh2 % dh == 0 and ow2 % dw == 0 and ow2 * oh2 <= expected_area
    ratio2 = ow2 / oh2

    # compare ratios
    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2,
                                                 ratio2 / ratio):
        return ow1, oh1
    else:
        return ow2, oh2

def get_first_frame_as_pil(video_path: str) -> Image.Image:
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    if not success:
        raise ValueError(f"无法读取视频 {video_path} 的第一帧")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

def encode_images(vae, img, device):
    # ih, iw = img.height, img.width
    # patch_size = (1, 2, 2)
    # vae_stride = (4, 16, 16)
    # max_area=704 * 1280
    # dh, dw = patch_size[1] * vae_stride[1], patch_size[
    #     2] * vae_stride[2]
    # ow, oh = best_output_size(iw, ih, dw, dh, max_area)
    # print(ow, oh)

    # scale = max(ow / iw, oh / ih)
    # img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)

    # # center-crop
    # x1 = (img.width - ow) // 2
    # y1 = (img.height - oh) // 2
    # img = img.crop((x1, y1, x1 + ow, y1 + oh))
    # assert img.width == ow and img.height == oh
    target_w, target_h = 640, 480
    iw, ih = img.width, img.height
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

    cropped.resize((target_w, target_h), Image.LANCZOS)

    # to tensor
    img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device).unsqueeze(1)
    print(f"Image size after processing: {img.shape}")
    img.to(torch.bfloat16)
    z = vae.encode_to_latent([img])[0]
    return z 


def main():
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = "cuda"

    input_dir = "/lustre/fsw/portfolios/av/users/shiyil/jfxiao/AirVuz-V2-08052025/videos"
    output_dir = "/lustre/fsw/portfolios/av/users/shiyil/jfxiao/AirVuz-V2-08052025/first_frame"
    os.makedirs(output_dir, exist_ok=True)
    video_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".mp4")
    ]
    video_files.sort()

    vae = WanVAEWrapper().to(torch.float16).to(device)

    for i in range(rank, len(video_files), world_size):
        video_path = video_files[i]
        base_name = os.path.basename(video_path).split(".")[0]
        save_path = os.path.join(output_dir,f"{base_name}.pth")
        image = get_first_frame_as_pil(video_path)
        latent = encode_images(vae, image, device).to(torch.bfloat16)
        print(latent.shape)
        break
        # torch.save(latent, save_path)
        print(f"[GPU {rank}] done {base_name}")

if __name__ == "__main__":
    main()