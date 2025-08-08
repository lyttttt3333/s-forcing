import json
import numpy as np
import matplotlib.pyplot as plt

# 读取 JSON 文件
with open("clip_time_series.json", "r") as f:
    data = json.load(f)

# 取任意一个视频获取统一时间点（假设至少有一个视频）
sample_video = next(iter(data))
time_points = [p["time"] for p in data[sample_video]]

# 遍历所有时间点，收集所有视频对应分数
avg_scores = []
for i, t in enumerate(time_points):
    scores_at_t = []
    for video in data:
        scores_at_t.append(data[video][i]["score"])
    avg_scores.append(np.mean(scores_at_t))

# 画图
plt.figure(figsize=(10,6))
plt.plot(time_points, avg_scores, marker='o')
plt.xlabel("Time (seconds)")
plt.ylabel("Average CLIP Similarity Score")
plt.title("Average CLIP Score Across Videos Over Time")
plt.grid(True)
plt.tight_layout()
plt.savefig("clip_score_time_series.png", dpi=300)
