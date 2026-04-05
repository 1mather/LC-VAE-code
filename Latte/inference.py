# Please update the version of diffusers at leaset to 0.30.0
from diffusers import LattePipeline
from diffusers.models import AutoencoderKLTemporalDecoder
from torchvision.utils import save_image
import torch
import imageio

torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
video_length = 16 # 1 (text-to-image) or 16 (text-to-video)
pipe = LattePipeline.from_pretrained("maxin-cn/Latte-1", torch_dtype=torch.float16).to(device)

# Using temporal decoder of VAE
vae = AutoencoderKLTemporalDecoder.from_pretrained("maxin-cn/Latte-1", subfolder="vae_temporal_decoder", torch_dtype=torch.float16).to(device)
pipe.vae = vae

prompt = "old man reading a book on cloud"
videos = pipe(prompt, video_length=video_length, output_type='pt').frames.cpu()

# Save video
output_path = "output_video.mp4"
# videos shape: (batch, time, channels, height, width) -> convert to (time, height, width, channels)
video = videos[0].permute(0, 2, 3, 1)  # (T, H, W, C)
video = (video * 255).numpy().astype('uint8')  # Convert to uint8
imageio.mimwrite(output_path, video, fps=8, quality=8)
print(f"Video saved to {output_path}")