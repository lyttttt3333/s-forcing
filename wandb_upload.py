import wandb
import os

wandb.login(host="https://api.wandb.ai", key="5409d3b960b01b25cec0f6abb5361b4022f0cc41")
wandb.init(
    mode="online",
    entity="liyitong-Tsinghua University",
    project="self-forcing",
)

video_path = "/lustre/fsw/portfolios/av/users/shiyil/jfxiao/Wan2.2/ti2v-5B_1280*704_4_Summer_beach_vacation_style,_a_white_cat_wearing_s_20250813_093505.mp4"
basename = os.path.basename(video_path)
wandb.log({f"{basename}": wandb.Video(video_path, fps=16, format="mp4")})