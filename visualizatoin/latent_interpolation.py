#!/usr/bin/env python3
"""
Create interpolation videos between different latent representations
"""

import torch
import numpy as np
import cv2
import os
import argparse
import json
from causalvideovae.model.vae.modeling_latent_wf_AE import WFVAEModelV2

def interpolate_latents(latent1, latent2, num_steps=50):
    """
    Interpolate between two latent representations
    """
    alphas = torch.linspace(0, 1, num_steps).view(-1, 1, 1, 1, 1)
    
    interpolated = []
    for alpha in alphas:
        interp = (1 - alpha) * latent1 + alpha * latent2
        interpolated.append(interp)
    
    return torch.cat(interpolated, dim=0)  # (num_steps, C, T, H, W)

def create_interpolation_video(model, latent1, latent2, output_path, fps=10, num_steps=50):
    """
    Create video showing interpolation between two latents through decoder
    """
    device = latent1.device
    
    # Interpolate latents
    interpolated_latents = interpolate_latents(latent1, latent2, num_steps)
    
    with torch.no_grad():
        # Decode all interpolated latents
        reconstructed_videos = []
        for i in range(num_steps):
            decoded = model.decode(interpolated_latents[i:i+1])  # Add batch dim
            reconstructed_videos.append(decoded.sample[0])  # Remove batch dim
        
        # Stack all reconstructions
        all_reconstructions = torch.stack(reconstructed_videos, dim=0)  # (num_steps, C, T, H, W)
        
        # Create video from middle time frame of each reconstruction
        middle_t = all_reconstructions.shape[2] // 2
        frames = all_reconstructions[:, :, middle_t, :, :]  # (num_steps, C, H, W)
        
        # Convert to video format
        frames = (frames + 1) / 2  # Denormalize from [-1,1] to [0,1]
        frames = torch.clamp(frames, 0, 1)
        frames = frames.permute(0, 2, 3, 1)  # (num_steps, H, W, C)
        frames_np = (frames.cpu().numpy() * 255).astype(np.uint8)
        
        # Write video
        H, W = frames_np.shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
        
        for frame in frames_np:
            if frame.shape[2] == 3:  # RGB
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:  # Grayscale
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Saved interpolation video: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create latent interpolation videos")
    parser.add_argument("--model_config", type=str, help="Path to model config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, default="./latent_interpolation", help="Output directory")
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of interpolation steps")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.model_config:
        with open(args.model_config, 'r') as f:
            config = json.load(f)
        model = WFVAEModelV2(**config)
    else:
        model = WFVAEModelV2()
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']['gen_model'] if 'gen_model' in checkpoint['state_dict'] else checkpoint['state_dict']
        elif 'ema_state_dict' in checkpoint:
            state_dict = checkpoint['ema_state_dict']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    
    # Create two different video inputs
    video1 = torch.randn(1, 3, 25, 256, 256).to(device)
    video2 = torch.randn(1, 3, 25, 256, 256).to(device)
    
    with torch.no_grad():
        # Encode both videos
        latent1 = model.encode(video1)
        latent2 = model.encode(video2)
        
        # Create interpolation video
        interp_path = os.path.join(args.output_dir, 'latent_interpolation.mp4')
        create_interpolation_video(model, latent1, latent2, interp_path, 
                                 fps=args.fps, num_steps=args.num_steps)
    
    print("Interpolation video created!")

if __name__ == "__main__":
    main()