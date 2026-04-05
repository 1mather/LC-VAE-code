"""
Debug script to check input/output ranges
"""
import argparse
import torch
import sys
sys.path.append(".")
from torch.utils.data import DataLoader, Subset
from causalvideovae.model import *
from causalvideovae.dataset.video_dataset import ValidVideoDataset
from accelerate import Accelerator
from einops import rearrange
import os

@torch.no_grad()
def main():
    accelerator = Accelerator()
    device = accelerator.device
    
    # Model
    model_name = "WVAE_Compressed_TopK_multi_wavelet"
    ckpt = "/scratch/cs/vidgen/guanjr/experiment/WFVAE_Experiment/WVAE_Channel_Keepratio_0.5_no_teacher_loss_32_channels_fix_topk_32frame_8gpu-lr2.00e-05-bs4-rs256-sr1-fr32-cons0.0-p1_1200000-p2_0-modetv_l1-20251028_140418/checkpoint-100000.ckpt"
    model_config = "/scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/examples/WVAE_compressed_topk_multi_wavelet/wfvae-temporal-compressed-example-fix-topk-handcraft-32chan.json"
    
    model_cls = ModelRegistry.get_model(model_name)
    vae = model_cls.from_config(model_config)
    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint["state_dict"]["gen_model"]
    vae.load_state_dict(state_dict, strict=False)
    vae = vae.to(device).to(torch.bfloat16)
    vae.eval()
    
    # Dataset
    dataset = ValidVideoDataset(
        real_video_dir="/scratch/cs/vidgen/data/kinetics-dataset/k400/test",
        num_frames=32,
        sample_rate=1,
        crop_size=256,  # Use crop_size like training
        resolution=256,
    )
    dataset = Subset(dataset, indices=range(3))
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    dataloader = accelerator.prepare(dataloader)
    
    print("Testing first 3 videos...")
    for i, batch in enumerate(dataloader):
        inputs = batch['video'].to(device)
        
        print(f"\nVideo {i+1}:")
        print(f"  Input shape: {inputs.shape}")
        print(f"  Input range: [{inputs.min().item():.4f}, {inputs.max().item():.4f}]")
        print(f"  Input mean: {inputs.mean().item():.4f}")
        
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = vae(inputs)
            video_recon = output.sample
        
        print(f"  Output shape: {video_recon.shape}")
        print(f"  Output range: [{video_recon.min().item():.4f}, {video_recon.max().item():.4f}]")
        print(f"  Output mean: {video_recon.mean().item():.4f}")
        
        # Calculate PSNR
        inputs_reshaped = rearrange(inputs, "b c t h w -> (b t) c h w").contiguous()
        video_recon_reshaped = rearrange(video_recon, "b c t h w -> (b t) c h w").contiguous()
        mse = torch.mean(torch.square(inputs_reshaped - video_recon_reshaped), dim=(1, 2, 3))
        psnr = 20 * torch.log10(1 / torch.sqrt(mse))
        psnr = psnr.mean().detach().cpu().item()
        print(f"  PSNR: {psnr:.4f}")

if __name__ == "__main__":
    main()

