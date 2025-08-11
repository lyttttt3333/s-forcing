import os
import cv2
import torch
import torch.distributed as dist

def extract_memory_tokens(frames_tensor):
    return frames_tensor.mean(dim=0)

def extract_frames_from_video(video_path, interval=18, max_frames=7):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor_frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(tensor_frame)
            if len(frames) >= max_frames:
                break
        frame_count += 1

    cap.release()
    if frames:
        return torch.stack(frames)  # (N, C, H, W)
    else:
        return torch.empty(0)

def main():
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    input_dir = "/lustre/fsw/portfolios/av/users/shiyil/jfxiao/AirVuz-V2-08052025/videos"
    output_dir = "/lustre/fsw/portfolios/av/users/shiyil/jfxiao/AirVuz-V2-08052025/memory_tokens"
    os.makedirs(output_dir, exist_ok=True)
    video_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".mp4")
    ][:20]
    video_files.sort()

    for i in range(rank, len(video_files), world_size):
        video_path = video_files[i]
        base_name = os.path.basename(video_path)
        frames_tensor = extract_frames_from_video(video_path)
        print(frames_tensor.shape)
        break
        if frames_tensor.numel() > 0:
            frames_tensor = frames_tensor.to(rank, non_blocking=True)
            print(f"[GPU {rank}] begin {base_name}")
            result = extract_memory_tokens(frames_tensor)
            print(f"[GPU {rank}] done {base_name}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
