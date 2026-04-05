#!/usr/bin/env python3
"""
Visualize latent representations and their properties
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from causalvideovae.model.vae.modeling_latent_wf_AE import WFVAEModelV2
import json
import argparse
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import cv2

def load_model(model_config, checkpoint_path, device):
    """Load model from config and checkpoint"""
    if model_config:
        with open(model_config, 'r') as f:
            config = json.load(f)
        model = WFVAEModelV2(**config)
    else:
        model = WFVAEModelV2()
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
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
    return model

def visualize_latent_distribution(latents, save_path):
    """Visualize latent distribution statistics"""
    # latents: (B, C, T, H, W)
    B, C, T, H, W = latents.shape
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Channel-wise mean
    channel_means = latents.mean(dim=(0, 2, 3, 4)).cpu().numpy()
    axes[0, 0].bar(range(C), channel_means)
    axes[0, 0].set_title('Channel-wise Mean')
    axes[0, 0].set_xlabel('Channel')
    axes[0, 0].set_ylabel('Mean Value')
    
    # 2. Channel-wise std
    channel_stds = latents.std(dim=(0, 2, 3, 4)).cpu().numpy()
    axes[0, 1].bar(range(C), channel_stds)
    axes[0, 1].set_title('Channel-wise Std')
    axes[0, 1].set_xlabel('Channel')
    axes[0, 1].set_ylabel('Std Value')
    
    # 3. Temporal variance
    temporal_vars = latents.var(dim=2).mean(dim=(0, 3, 4)).cpu().numpy()
    axes[0, 2].bar(range(C), temporal_vars)
    axes[0, 2].set_title('Temporal Variance by Channel')
    axes[0, 2].set_xlabel('Channel')
    axes[0, 2].set_ylabel('Temporal Variance')
    
    # 4. Spatial variance
    spatial_vars = latents.var(dim=(3, 4)).mean(dim=(0, 2)).cpu().numpy()
    axes[1, 0].bar(range(C), spatial_vars)
    axes[1, 0].set_title('Spatial Variance by Channel')
    axes[1, 0].set_xlabel('Channel')
    axes[1, 0].set_ylabel('Spatial Variance')
    
    # 5. Overall distribution
    latent_flat = latents.flatten().cpu().numpy()
    axes[1, 1].hist(latent_flat, bins=50, alpha=0.7)
    axes[1, 1].set_title('Overall Latent Distribution')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Frequency')
    
    # 6. Channel correlation heatmap
    latent_reshaped = latents.permute(1, 0, 2, 3, 4).contiguous().view(C, -1)  # (C, B*T*H*W)
    correlation = torch.corrcoef(latent_reshaped).cpu().numpy()
    im = axes[1, 2].imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 2].set_title('Channel Correlation')
    axes[1, 2].set_xlabel('Channel')
    axes[1, 2].set_ylabel('Channel')
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()



def visualize_latent_spatial_patterns(latents, save_path):
    """Visualize spatial patterns in latents"""
    # latents: (B, C, T, H, W)
    B, C, T, H, W = latents.shape
    
    # Average over time and batch to get (C, H, W)
    spatial_latents = latents.mean(dim=(0, 2))  # (C, H, W)
    
    # Plot first 16 channels as heatmaps
    num_channels_to_plot = min(16, C)
    rows = 4
    cols = 4
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()
    
    for c in range(num_channels_to_plot):
        im = axes[c].imshow(spatial_latents[c].cpu().numpy(), cmap='viridis')
        axes[c].set_title(f'Channel {c}')
        axes[c].axis('off')
        plt.colorbar(im, ax=axes[c], fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for c in range(num_channels_to_plot, len(axes)):
        axes[c].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_wavelet_components(model, video_tensor, save_path):
    """Visualize wavelet decomposition components"""
    with torch.no_grad():
        # Get latent
        z = model.encode(video_tensor)
        
        if model.use_latent_wavelet:
            z_tilde, coeffs, l1_coeffs = model.latent_wav(z)
            
            # coeffs: (B, 8*C, T/2, H/2, W/2)
            # l1_coeffs: (B, C, T/2, H/2, W/2) - LLL components
            
            B, full_C, T_w, H_w, W_w = coeffs.shape
            C = z.shape[1]
            
            # Reshape to separate wavelet components
            coeffs_reshaped = coeffs.view(B, C, 8, T_w, H_w, W_w)
            
            # Visualize different wavelet components
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            component_names = ['LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH']
            
            for i in range(8):
                # Average over batch, channels, and time for visualization
                component = coeffs_reshaped[0, :, i, :, :, :].mean(dim=(0, 1))  # (H_w, W_w)
                
                row = i // 4
                col = i % 4
                im = axes[row, col].imshow(component.cpu().numpy(), cmap='RdBu_r')
                axes[row, col].set_title(f'{component_names[i]} Component')
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return l1_coeffs
    return None

def visualize_consistency_loss_components(l1_coeffs, save_path, mode='tv_l1_v2'):
    """Visualize components that contribute to consistency loss"""
    if l1_coeffs is None:
        return
    
    # l1_coeffs: (B, C, T, H, W)
    B, C, T, H, W = l1_coeffs.shape
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Temporal differences (what TV-L1 measures)
    if T > 1:
        temporal_diffs = torch.abs(l1_coeffs[:, :, 1:] - l1_coeffs[:, :, :-1])  # (B, C, T-1, H, W)
        
        # Average temporal difference per channel
        avg_temp_diff = temporal_diffs.mean(dim=(0, 2, 3, 4)).cpu().numpy()
        axes[0, 0].bar(range(C), avg_temp_diff)
        axes[0, 0].set_title('Average Temporal Difference by Channel')
        axes[0, 0].set_xlabel('Channel')
        axes[0, 0].set_ylabel('Avg Temporal Diff')
        
        # Temporal difference over time
        temp_diff_over_time = temporal_diffs.mean(dim=(0, 1, 3, 4)).cpu().numpy()  # (T-1,)
        axes[0, 1].plot(temp_diff_over_time)
        axes[0, 1].set_title('Temporal Difference Over Time')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Avg Temporal Diff')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Spatial map of temporal differences
        spatial_temp_diff = temporal_diffs[0].mean(dim=(0, 1))  # (H, W)
        im = axes[0, 2].imshow(spatial_temp_diff.cpu().numpy(), cmap='hot')
        axes[0, 2].set_title('Spatial Distribution of Temporal Differences')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2])
    
    # 2. Channel-wise variance
    channel_vars = l1_coeffs.var(dim=(0, 2, 3, 4)).cpu().numpy()
    axes[1, 0].bar(range(C), channel_vars)
    axes[1, 0].set_title('Channel-wise Variance')
    axes[1, 0].set_xlabel('Channel')
    axes[1, 0].set_ylabel('Variance')
    
    # 3. Temporal variance (what temporal_variance mode measures)
    temporal_vars = l1_coeffs.var(dim=2).mean(dim=(0, 3, 4)).cpu().numpy()
    axes[1, 1].bar(range(C), temporal_vars)
    axes[1, 1].set_title('Temporal Variance by Channel')
    axes[1, 1].set_xlabel('Channel')
    axes[1, 1].set_ylabel('Temporal Variance')
    
    # 4. Overall distribution of LLL coefficients
    l1_flat = l1_coeffs.flatten().cpu().numpy()
    axes[1, 2].hist(l1_flat, bins=50, alpha=0.7)
    axes[1, 2].set_title('LLL Coefficients Distribution')
    axes[1, 2].set_xlabel('Value')
    axes[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_latent_video(model, video_tensor, output_dir):
    """Complete analysis of latent representation for a video"""
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        # Encode video
        print("Encoding video...")
        z = model.encode(video_tensor)
        print(f"Latent shape: {z.shape}")
        
        # Basic latent visualization
        print("Visualizing latent distribution...")
        visualize_latent_distribution(z, os.path.join(output_dir, 'latent_distribution.png'))
        
        print("Visualizing spatial patterns...")
        visualize_latent_spatial_patterns(z, os.path.join(output_dir, 'spatial_patterns.png'))
        
        # Wavelet analysis if available
        if model.use_latent_wavelet:
            print("Analyzing wavelet components...")
            l1_coeffs = visualize_wavelet_components(model, video_tensor, 
                                                   os.path.join(output_dir, 'wavelet_components.png'))
            
            if l1_coeffs is not None:
                print("Analyzing consistency loss components...")
                visualize_consistency_loss_components(l1_coeffs, 
                                                    os.path.join(output_dir, 'consistency_components.png'))
                
                # Calculate actual consistency loss
                from causalvideovae.model.vae.modeling_latent_wf_AE import lowfreq_consistency_loss
                
                modes = ['temporal_variance', 'tv_l1', 'tv_l1_v2']
                consistency_values = {}
                
                for mode in modes:
                    try:
                        loss_val = lowfreq_consistency_loss(l1_coeffs, mode=mode).item()
                        consistency_values[mode] = loss_val
                        print(f"Consistency loss ({mode}): {loss_val:.6f}")
                    except Exception as e:
                        print(f"Error calculating {mode}: {e}")
                
                # Save consistency values
                with open(os.path.join(output_dir, 'consistency_values.json'), 'w') as f:
                    json.dump(consistency_values, f, indent=2)
        
        # Save latent statistics
        stats = {
            'shape': list(z.shape),
            'mean': z.mean().item(),
            'std': z.std().item(),
            'min': z.min().item(),
            'max': z.max().item(),
            'channel_means': z.mean(dim=(0, 2, 3, 4)).cpu().numpy().tolist(),
            'channel_stds': z.std(dim=(0, 2, 3, 4)).cpu().numpy().tolist(),
        }
        
        with open(os.path.join(output_dir, 'latent_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Analysis complete! Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Visualize latent representations")
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file")
    parser.add_argument("--model_config", type=str, help="Path to model config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, default="./latent_analysis", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = load_model(args.model_config, args.checkpoint, args.device)
    
    # Load video (simplified - you might need to adapt this)
    # For now, create dummy video tensor
    print("Creating dummy video tensor...")
    video_tensor = torch.randn(1, 3, 25, 256, 256).to(args.device)  # (B, C, T, H, W)
    
    # Analyze
    analyze_latent_video(model, video_tensor, args.output_dir)

if __name__ == "__main__":
    main()