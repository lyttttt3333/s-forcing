from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from wan.modules.causal_model import CausalWanModel
from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy 
import imageio
from tqdm import tqdm

import torch

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

vae = WanVAEWrapper()
vae.requires_grad_(False)
latent = torch.load("pred_image.pt", map_location="cpu").to("cuda")[0][:,:3]
latent = torch.zeros([48, 1, 30, 60], device="cuda", dtype=torch.bfloat16) 
with torch.no_grad():
    # latent = vae.encode_to_latent([video])[0]
    print(latent.shape)
    video = vae.decode_to_pixel([latent])
    print(video[0].shape)

# output_path = "pred_video.mp4"
# save_video(video[0], output_path, fps=16, quality=5)

# import wandb
# import os

# wandb.login(host="https://api.wandb.ai", key="5409d3b960b01b25cec0f6abb5361b4022f0cc41")
# wandb.init(
#     mode="online",
#     entity="liyitong-Tsinghua University",
#     project="self-forcing",
# )

# video_path = "pred_video.mp4"
# basename = os.path.basename(video_path)
# wandb.log({f"{basename}": wandb.Video(video_path, fps=16, format="mp4")})



# import os
# import torch
# import torch.distributed as dist

# def main():
#     dist.init_process_group(backend="nccl")  # 使用NCCL后端（适合GPU）
#     local_rank = int(os.environ["LOCAL_RANK"])  # 获取本地进程排名
    
#     # 设置当前进程使用的设备
#     torch.cuda.set_device(local_rank)
#     device = torch.device(f"cuda:{local_rank}")

#     model_name = "Wan2.2-TI2V-5B"
#     print(f"Loading model: {model_name}")
    
#     # 加载模型
#     model = CausalWanModel.from_pretrained(f"wan_models/{model_name}/")
    
#     local_rank = int(os.environ["LOCAL_RANK"])
#     torch.cuda.set_device(local_rank)  # 每个进程绑定到对应的 GPU（0~7）

#     # 3. FSDP 包装模型：自动分片并分配到对应 GPU
#     model = FSDP(
#         model,  # 传入 CPU 上的模型
#         device_id=torch.cuda.current_device(),  # 当前进程绑定的 GPU
#         sharding_strategy=ShardingStrategy.FULL_SHARD  # 全分片模式
#     )
    
#     # 查看当前进程的显存占用
#     allocated = torch.cuda.memory_allocated() / (1024 **3)  # 转换为GB
#     reserved = torch.cuda.memory_reserved() / (1024** 3)
    
#     print(f" using device {model.device}")
#     print(f"进程 {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0} 显存使用情况:")
#     print(f"  已分配: {allocated:.2f} GB")
#     print(f"  已保留: {reserved:.2f} GB")


# if __name__ == "__main__":
#     main()

# real_model_name = "Wan2.2-TI2V-5B"
# self.real_score = WanDiffusionWrapper(model_name=real_model_name, is_causal=False)