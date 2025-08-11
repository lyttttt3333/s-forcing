import cv2
from PIL import Image

from utils.wan_wrapper import WanVAEWrapper
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy 


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
    ih, iw = img.height, img.width
    patch_size = (1, 2, 2)
    vae_stride = (4, 16, 16)
    dh, dw = patch_size[1] * vae_stride[1], patch_size[
        2] * vae_stride[2]
    ow, oh = best_output_size(iw, ih, dw, dh, max_area)

    scale = max(ow / iw, oh / ih)
    img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)

    # center-crop
    x1 = (img.width - ow) // 2
    y1 = (img.height - oh) // 2
    img = img.crop((x1, y1, x1 + ow, y1 + oh))
    assert img.width == ow and img.height == oh

    # to tensor
    img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device).unsqueeze(1)


    print("before")
    print(img.shape)
    img.to(torch.bfloat16)
    z = vae.encode([img])
    return z 

video_path = "/lustre/fsw/portfolios/av/users/shiyil/jfxiao/AirVuz-V2-08052025/videos/fff82aff-7041-43f2-8e1c-96436a87797e.mp4"
vae = WanVAEWrapper().to(torch.float16)

image = get_first_frame_as_pil(video_path)
latent = encode_images(vae, image)
print(latent)