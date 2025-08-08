import os
import cv2
import torch
import clip
import json
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from tqdm import tqdm

# ===== 参数设置 =====
video_path = "video/dense_168"         # ← 修改为你的视频路径
frame_count = 8                             # 每个视频采样帧数
output_file = "clip_time_series.json"       # 输出文件名
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== 加载 CLIP 模型 =====
model, preprocess = clip.load("ViT-B/32", device=device)

# ===== 采样帧函数：返回 [(timestamp, frame_tensor), ...] =====
def sample_video_frames_with_timestamps(video_file, num_frames):
    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total_frames == 0 or fps == 0:
        return []

    frame_indices = [int(i) for i in torch.linspace(0, total_frames - 1, steps=num_frames)]
    print(frame_indices)
    results = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        timestamp = idx / fps
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        tensor = preprocess(pil_img)
        results.append((timestamp, tensor))

    cap.release()
    return results

# ===== 主流程 =====
results = {}

for filename in tqdm(os.listdir(video_path)):
    if not filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        continue

    full_path = os.path.join(video_path, filename)
    sampled = sample_video_frames_with_timestamps(full_path, frame_count)
    if not sampled:
        continue

    prompt = os.path.splitext(filename)[0]
    text = clip.tokenize([prompt]).to(device)

    timeline = []

    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        for timestamp, img_tensor in sampled:
            img_tensor = img_tensor.unsqueeze(0).to(device)
            image_features = model.encode_image(img_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).item()
            timeline.append({
                "time": round(timestamp, 3),
                "score": round(similarity, 5)
            })

    results[filename] = timeline

# ===== 保存为 JSON =====
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Saved time-aligned CLIP scores to: {output_file}")
