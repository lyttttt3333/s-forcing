import os
import cv2
import torch
import torch.distributed as dist
from accelerate import Accelerator

from uuid import uuid4
from PIL import Image
from torchvision import transforms as TF

from torchvision.transforms import ToPILImage
to_pil = ToPILImage()

def generate_tokens(model, frames):
    past_key_values = [None] * model.aggregator.depth
    output_tokens = []
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype = torch.bfloat16):
            with torch.no_grad():
                for i in range(frames.shape[0]):
                    frame = frames[i].unsqueeze(0)
                    aggregated_token, patch_start_idx, past_key_values = model.inference(frame, i, past_key_values=past_key_values)
                    output_tokens.append(aggregated_token)
    output_tokens = torch.cat(output_tokens, dim=0)
    return output_tokens


def load_and_preprocess_images(image_path_list, mode="crop"):
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:

        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y: start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images


def delete_saved_paths(paths):
    for p in paths:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception as e:
            print(f"删除 {p} 失败: {e}")

def extract_frames_from_video(video_path, preprocess_function, tmp_dir="tmp_frames", interval=18, max_frames=7):
    rank = dist.get_rank() if dist.is_initialized() else 0
    save_dir = os.path.join(tmp_dir, f"rank_{rank}")
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_paths = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            img_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_{len(saved_paths)+1}_{uuid4().hex[:8]}.jpg"
            img_path = os.path.join(save_dir, img_name)
            cv2.imwrite(img_path, frame)
            saved_paths.append(img_path)

            if len(saved_paths) >= max_frames:
                break
        frame_count += 1

    cap.release()

    processed_result = preprocess_function(saved_paths)
    delete_saved_paths(saved_paths)
    return processed_result


def main():
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    input_dir = "/lustre/fsw/portfolios/av/users/shiyil/jfxiao/AirVuz-V2-08052025/videos"
    output_dir = "/lustre/fsw/portfolios/av/users/shiyil/jfxiao/AirVuz-V2-08052025/memory_tokens"
    tmp_dir = "/lustre/fsw/portfolios/av/users/shiyil/jfxiao/AirVuz-V2-08052025/tmp_frames"
    weight_path = "/lustre/fsw/portfolios/av/users/shiyil/jfxiao/StreamVGGT/ckpt/checkpoints.pth"
    os.makedirs(output_dir, exist_ok=True)
    video_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".mp4")
    ][:20]
    video_files.sort()


    accelerator = Accelerator()
    device = accelerator.device
    from streamvggt.models.streamvggt import StreamVGGT
    model = StreamVGGT()
    ckpt = torch.load(weight_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    model = model.to("cuda").to(torch.bfloat16)
    del ckpt
    print(f"Load at {device}")

    for i in range(rank, len(video_files), world_size):
        video_path = video_files[i]
        base_name = os.path.basename(video_path).split([.])[0]
        frames_tensor = extract_frames_from_video(
                video_path=video_path,
                preprocess_function=load_and_preprocess_images,  
                tmp_dir=tmp_dir,
                interval=18,
                max_frames=7
            )
        frames_tensor = frames_tensor.to(rank, non_blocking=True)
        print(f"[GPU {rank}] begin {base_name}")
        save_path = os.path.join(output_dir,f"{base_name}.pth")
        output_tokens = generate_tokens(model, frames_tensor)
        print(output_tokens.shape, output_tokens.dtype)
        # torch.save(output_tokens, save_path)
        print(f"[GPU {rank}] done {base_name}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
