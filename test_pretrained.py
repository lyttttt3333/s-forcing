from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from wan.modules.causal_model import CausalWanModel
from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy 

import torch



vae = WanVAEWrapper()
video = torch.zeros([3, 12, 512, 512]).to("cuda")
latent = vae.encode_to_latent([video])[0]
print(latent.shape)
video = vae.decode_to_pixel([latent])
print(video[0].shape)


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