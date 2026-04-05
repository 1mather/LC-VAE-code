import argparse
from tqdm import tqdm
import torch
import sys
from torch.utils.data import DataLoader, Subset
import os
sys.path.append(".")
from causalvideovae.model import *
from causalvideovae.dataset.video_dataset import ValidVideoDataset, ValidParquetVideoDataset
from causalvideovae.utils.video_utils import custom_to_video
from causalvideovae.model.utils.wavelet_utils import HaarWaveletTransform3D
from accelerate import Accelerator
from scipy.stats import entropy

def calculate_energy(subband):
    return torch.sum(torch.square(subband))

def calculate_entropy(subband):
    p = torch.abs(subband) / torch.sum(torch.abs(subband))
    return torch.tensor(entropy(p.float().cpu().numpy().flatten()))

@torch.no_grad()
def main(args: argparse.Namespace):
    accelerator = Accelerator()
    #此处的accelerator是
    device = accelerator.device
    
    real_video_dir = args.real_video_dir
    generated_video_dir = args.generated_video_dir
    sample_rate = args.sample_rate
    resolution = args.resolution
    num_frames = args.num_frames
    sample_rate = args.sample_rate
    device = args.device
    sample_fps = args.sample_fps
    batch_size = args.batch_size
    num_workers = args.num_workers
    subset_size = args.subset_size
    crop_size_width = args.crop_size_width
    crop_size_height = args.crop_size_height
    
    if not os.path.exists(args.generated_video_dir):
        os.makedirs(args.generated_video_dir, exist_ok=True)
    
    data_type = torch.bfloat16
    
    # ---- Load Model ----
    device = args.device
    model_cls = ModelRegistry.get_model(args.model_name)

    if args.from_pretrained and os.path.isfile(args.from_pretrained) and args.from_pretrained.endswith(".ckpt"):
        assert args.model_config is not None and os.path.isfile(args.model_config), \
            "--model_config must be provided when --from_pretrained is a .ckpt file"
        vae = model_cls.from_config(args.model_config)
        checkpoint = torch.load(args.from_pretrained, map_location="cpu")
        state_dict = checkpoint.get("state_dict", {}).get("gen_model", None)
        if state_dict is None:
            raise RuntimeError("Invalid checkpoint: missing state_dict.gen_model")
        try:
            missing, unexpected = vae.load_state_dict(state_dict, strict=False)
        except Exception:
            # Fallback for models that might be wrapped during training
            missing, unexpected = vae.module.load_state_dict(state_dict, strict=False)
        if len(unexpected) > 0:
            print(f"Warning: unexpected keys when loading ckpt: {unexpected}")
        if len(missing) > 0:
            print(f"Warning: missing keys when loading ckpt: {missing}")
    else:
        vae = model_cls.from_pretrained(args.from_pretrained)
    vae = vae.to(device).to(data_type)
    vae.eval()
    if args.enable_tiling:
        vae.enable_tiling()

    # ---- Prepare Dataset ----
    # Use Parquet dataset if parquet_path is provided, otherwise use directory dataset
    if args.parquet_path:
        print(f"Using ValidParquetVideoDataset with parquet file: {args.parquet_path}")
        dataset = ValidParquetVideoDataset(
            parquet_path=args.parquet_path,
            num_frames=num_frames,
            sample_rate=sample_rate,
            crop_size_width=crop_size_width,
            crop_size_height=crop_size_height,
            resolution=resolution,
            video_column=args.video_column,
            is_main_process=accelerator.is_main_process,
        )
    else:
        print(f"Using ValidVideoDataset with directory: {real_video_dir}")
        dataset = ValidVideoDataset(
            real_video_dir=real_video_dir,
            num_frames=num_frames,
            sample_rate=sample_rate,
            crop_size_width=crop_size_width,
            crop_size_height=crop_size_height,
            resolution=resolution,
        )
    
    if subset_size:
        indices = range(subset_size)
        dataset = Subset(dataset, indices=indices)
        
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=False, num_workers=num_workers
    )


    energy_list = {}
    entropy_list = {}
    # ---- Create output directories ----
    # generated/ for reconstructed videos, real/ for sampled original videos
    generated_dir = os.path.join(generated_video_dir, "generated")
    real_dir = os.path.join(generated_video_dir, "real")
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)
    
    print(f"Saving reconstructed videos to: {generated_dir}")
    print(f"Saving sampled real videos to: {real_dir}")
    
    # ---- Inference ----
    count = 0
    precision = torch.float16
    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        count += 1
        if count % args.eval_interval == 0:
            print(f"Processed {count} videos")
            break

        x, file_names = batch['video'], batch['file_name']
        #x_normalized = x * 2 - 1  # Normalize to [-1, 1] for model input
        x_normalized = x

        with torch.amp.autocast("cuda", dtype=precision):
            video_recon = vae.forward(x_normalized)
        video_recon = video_recon.sample

        # Save each item in the batch
        recon_tensor = getattr(video_recon, "sample", video_recon)
        for i, video in enumerate(recon_tensor):
            # Save reconstructed video to generated/
            generated_output_path = os.path.join(generated_dir, file_names[i])
            custom_to_video(
                video, fps=sample_fps / sample_rate, output_file=generated_output_path
            )
            
            # Save sampled real video to real/
            real_output_path = os.path.join(real_dir, file_names[i])
            custom_to_video(
                x_normalized[i], fps=sample_fps / sample_rate, output_file=real_output_path
            )
            
        if accelerator.is_main_process and count % 100 == 0:
            print(f"Processed {count} videos...")
    

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data source - either directory or parquet file
    parser.add_argument("--real_video_dir", type=str, default="", 
                        help="Directory containing video files (ignored if --parquet_path is provided)")
    parser.add_argument("--parquet_path", type=str, default="",
                        help="Path to parquet file containing video paths (takes priority over --real_video_dir)")
    parser.add_argument("--video_column", type=str, default="video_path",
                        help="Column name in parquet file containing video paths")
    
    # Output settings
    parser.add_argument("--generated_video_dir", type=str, default="",
                        help="Base directory for output (will create 'generated/' and 'real/' subdirectories)")
    
    # Model settings
    parser.add_argument("--from_pretrained", type=str, default="")
    parser.add_argument("--model_name", type=str, default=None, help="Model class name")
    parser.add_argument("--model_config", type=str, default=None, help="Model config file path")
    parser.add_argument('--enable_tiling', action='store_true',
                        help="Enable tiling for processing large videos")
    parser.add_argument("--tile_overlap_factor", type=float, default=0.25)
    
    # Video parameters
    parser.add_argument("--sample_fps", type=int, default=30,
                        help="FPS for output videos")
    parser.add_argument("--resolution", type=int, default=336,
                        help="Resolution to resize videos to")
    parser.add_argument("--crop_size_width", type=int, default=None,
                        help="Width for center crop (optional)")
    parser.add_argument("--crop_size_height", type=int, default=None,
                        help="Height for center crop (optional)")
    parser.add_argument("--num_frames", type=int, default=17,
                        help="Number of frames to sample from each video")
    parser.add_argument("--sample_rate", type=int, default=1,
                        help="Frame sampling rate (stride)")
    
    # DataLoader settings
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--subset_size", type=int, default=None,
                        help="Limit dataset to first N samples (for testing)")
    
    # Computation settings
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_interval", type=int, default=50,
                        help="Process and stop after this many videos")

    args = parser.parse_args()
    
    # Validate input arguments
    if not args.parquet_path and not args.real_video_dir:
        parser.error("Either --parquet_path or --real_video_dir must be provided")
    
    main(args)
    
