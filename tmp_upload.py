import wandb
import os
import glob

# 登录 W&B
wandb.login(host="https://api.wandb.ai", key="5409d3b960b01b25cec0f6abb5361b4022f0cc41")
wandb.init(
    mode="online",
    entity="liyitong-Tsinghua University",
    project="self-forcing",
)

# 搜索 /tmp 下所有 mp4 文件（递归搜索）
tmp_dir = "tmp"
mp4_files = glob.glob(os.path.join(tmp_dir, "**", "*.mp4"), recursive=True)

# 批量上传
for video_path in mp4_files:
    basename = os.path.basename(video_path)
    wandb.log({basename: wandb.Video(video_path, fps=16, format="mp4")})

wandb.finish()