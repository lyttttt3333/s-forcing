# import pandas as pd
# import os

# # 1. 读取CSV
# input_path = '/lustre/fsw/portfolios/av/users/shiyil/jfxiao/AirVuz-V2-08052025/metadata.csv' 
# output_path = '/lustre/fsw/portfolios/av/users/shiyil/jfxiao/AirVuz-V2-08052025/meta.csv' 
# df = pd.read_csv(input_path)

# # 2. 获取video列
# video_list = df['video'].tolist()
# root_dir = '/lustre/fsw/portfolios/av/users/shiyil/jfxiao/AirVuz-V2-08052025'  # 替换为实际的视频根目录

# base_name_list = []

# print("initial video list length:", len(video_list))

# # 3. 验证存在性并生成basename列表
# for video_path in video_list:
#     base_name = video_path.split(".")[0]
#     text_token_dir = os.path.join(root_dir, "text_token")
#     text_token_path = os.path.join(text_token_dir, base_name + '.pth')
#     if not os.path.exists(text_token_path):
#         continue
#     memory_token_dir = os.path.join(root_dir, "memory_token")
#     memory_token_path = os.path.join(memory_token_dir, base_name + '.pth')
#     if not os.path.exists(memory_token_path):
#         continue
#     frame_token_dir = os.path.join(root_dir, "frame_token")
#     frame_token_path = os.path.join(frame_token_dir, base_name + '.pth')
#     if not os.path.exists(frame_token_path):
#         continue
#     base_name_list.append(base_name)

# print("final video list length:", len(base_name_list))

# output_df = pd.DataFrame({'basename': base_name_list})
# output_df.to_csv(output_path, index=False, encoding='utf-8-sig')

# print("处理完成，结果已保存到 output.csv")

import torch
import glob

text_token_dir = '/lustre/fsw/portfolios/av/users/shiyil/jfxiao/AirVuz-V2-08052025/text_token'
memory_token_dir = '/lustre/fsw/portfolios/av/users/shiyil/jfxiao/AirVuz-V2-08052025/memory_token'
frame_token_dir = '/lustre/fsw/portfolios/av/users/shiyil/jfxiao/AirVuz-V2-08052025/frame_token'

text_pth_path = glob.glob(text_token_dir + '/*.pth')[0]
print("Text token shape:", torch.load(text_pth_path).shape)
memory_pth_path = glob.glob(memory_token_dir + '/*.pth')[0]
print("Memory token shape:", torch.load(memory_pth_path).shape)
frame_pth_path = glob.glob(frame_token_dir + '/*.pth')[0]
print("Frame token shape:", torch.load(frame_pth_path).shape)
