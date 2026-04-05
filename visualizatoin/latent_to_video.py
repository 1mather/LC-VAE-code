#!/usr/bin/env python3
"""
Visualize WFVAE latent representations after Haar wavelet transform
Specifically visualize LLL (low-frequency) and HHH (high-frequency) components
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import argparse
import json
import sys
from pathlib import Path
import glob

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from causalvideovae.model.vae.modeling_latent_wf_AE import LatentWFAEModelV1
from causalvideovae.model.utils.module_utils import resolve_str_to_obj
from causalvideovae.model.registry import ModelRegistry

def load_video_from_file(video_path, target_size=(256, 256), target_frames=25, device='cpu'):
    """
    Load video from file and preprocess it
    Args:
        video_path: Path to video file
        target_size: (height, width) target size
        target_frames: Number of frames to extract
        device: Device to place tensor on
    Returns:
        video_tensor: (1, 3, T, H, W) tensor
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {original_width}x{original_height}")
    
    # Calculate frame indices to sample
    if total_frames <= target_frames:
        frame_indices = list(range(total_frames))
        # Pad with last frame if needed
        while len(frame_indices) < target_frames:
            frame_indices.append(total_frames - 1)
    else:
        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
    
    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize
            frame = cv2.resize(frame, (target_size[1], target_size[0]))
            frames.append(frame)
        else:
            print(f"Warning: Cannot read frame {frame_idx}")
            # Use last successful frame
            if frames:
                frames.append(frames[-1])
    
    cap.release()
    
    if not frames:
        raise ValueError("No frames could be read from video")
    
    # Convert to tensor
    video_array = np.array(frames)  # (T, H, W, 3)
    video_tensor = torch.from_numpy(video_array).permute(3, 0, 1, 2).float()  # (3, T, H, W)
    video_tensor = video_tensor / 255.0  # Normalize to [0, 1]
    video_tensor = video_tensor * 2.0 - 1.0  # Normalize to [-1, 1] (same as training)
    video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension (1, 3, T, H, W)
    video_tensor = video_tensor.to(device)  # Move to specified device
    
    print(f"Loaded video tensor shape: {video_tensor.shape} on {device}")
    print(f"Video tensor range: [{video_tensor.min():.3f}, {video_tensor.max():.3f}]")
    return video_tensor

def load_video_from_directory(video_dir, target_size=(256, 256), target_frames=25, max_videos=1, device='cpu'):
    """
    Load videos from directory
    Args:
        video_dir: Directory containing video files
        target_size: (height, width) target size
        target_frames: Number of frames to extract
        max_videos: Maximum number of videos to process
        device: Device to place tensors on
    Returns:
        List of video tensors
    """
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))
        video_files.extend(glob.glob(os.path.join(video_dir, '**', ext), recursive=True))
    
    if not video_files:
        raise ValueError(f"No video files found in directory: {video_dir}")
    
    print(f"Found {len(video_files)} video files")
    
    video_tensors = []
    for i, video_path in enumerate(video_files[:max_videos]):
        print(f"Loading video {i+1}/{min(max_videos, len(video_files))}: {os.path.basename(video_path)}")
        try:
            video_tensor = load_video_from_file(video_path, target_size, target_frames, device)
            video_tensors.append(video_tensor)
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            continue
    
    return video_tensors

def normalize_for_visualization(tensor, method='robust'):
    """Normalize tensor for better visualization"""
    if method == 'minmax':
        # Min-max normalization to [0, 1]
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val > min_val:
            normalized = (tensor - min_val) / (max_val - min_val)
        else:
            normalized = torch.zeros_like(tensor)
    
    elif method == 'robust':
        # Robust normalization using percentiles
        # Use 1st and 99th percentiles to avoid outliers
        flat_tensor = tensor.flatten()
        p1 = torch.quantile(flat_tensor, 0.01)
        p99 = torch.quantile(flat_tensor, 0.99)
        
        if p99 > p1:
            normalized = (tensor - p1) / (p99 - p1)
            normalized = torch.clamp(normalized, 0, 1)
        else:
            normalized = torch.zeros_like(tensor)
    
    elif method == 'zscore_robust':
        # Robust z-score normalization
        mean_val = tensor.mean()
        std_val = tensor.std()
        if std_val > 0:
            normalized = (tensor - mean_val) / std_val
            # Use 2-sigma clipping instead of 3-sigma for better contrast
            normalized = torch.clamp(normalized, -2, 2)
            normalized = (normalized + 2) / 4  # Map [-2, 2] to [0, 1]
        else:
            normalized = torch.zeros_like(tensor)
    
    elif method == 'adaptive':
        # Adaptive normalization based on data distribution
        mean_val = tensor.mean()
        std_val = tensor.std()
        
        if std_val > 0:
            # Use sigmoid to compress extreme values
            normalized = torch.sigmoid((tensor - mean_val) / (std_val + 1e-8))
        else:
            normalized = torch.zeros_like(tensor)
    
    elif method == 'tanh':
        # Tanh normalization
        normalized = torch.tanh(tensor)
        normalized = (normalized + 1) / 2  # Map [-1, 1] to [0, 1]
    
    return normalized

def print_tensor_stats(tensor, name="tensor"):
    """Print detailed statistics of a tensor for debugging"""
    print(f"\n=== {name} Statistics ===")
    print(f"Shape: {tensor.shape}")
    print(f"Min: {tensor.min():.6f}")
    print(f"Max: {tensor.max():.6f}")
    print(f"Mean: {tensor.mean():.6f}")
    print(f"Std: {tensor.std():.6f}")
    print(f"Median: {tensor.median():.6f}")
    
    # Percentiles
    flat_tensor = tensor.flatten()
    p1 = torch.quantile(flat_tensor, 0.01)
    p5 = torch.quantile(flat_tensor, 0.05)
    p95 = torch.quantile(flat_tensor, 0.95)
    p99 = torch.quantile(flat_tensor, 0.99)
    
    print(f"P1: {p1:.6f}, P5: {p5:.6f}, P95: {p95:.6f}, P99: {p99:.6f}")
    
    # Check for extreme values
    extreme_ratio = (torch.abs(tensor) > 3 * tensor.std()).float().mean()
    print(f"Extreme values (>3σ) ratio: {extreme_ratio:.4f}")
    print("=" * 30)

def tensor_to_video(tensor, output_path, fps=10, colormap='viridis', title=None, norm_method='robust'):
    """
    Convert tensor to video
    tensor: (T, H, W) or (C, T, H, W)
    if colormap is None and tensor has 3 channels, treat as RGB video
    norm_method: normalization method ('robust', 'minmax', 'adaptive', etc.)
    """
    if tensor.dim() == 4:  # Multi-channel (C, T, H, W)
        C, T, H, W = tensor.shape
        
        # Special case: RGB video (3 channels, no colormap)
        if C == 3 and colormap is None:
            print("Processing RGB video...")
            # 确保在 [0, 1] 范围内（用于可视化的tensor应该已经是[0,1]范围）
            tensor = torch.clamp(tensor, 0, 1)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
            
            for t in range(T):
                # Get RGB frame: (3, H, W) -> (H, W, 3)
                frame = tensor[:, t, :, :].permute(1, 2, 0).cpu().numpy()
                frame = np.clip(frame, 0, 1)
                frame_uint8 = (frame * 255).astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
        
        else:
            # Create grid layout for multi-channel visualization
            cols = int(np.ceil(np.sqrt(C)))
            rows = int(np.ceil(C / cols))
            
            grid_height = rows * H
            grid_width = cols * W
            
            # Normalize each channel independently with specified method
            normalized_channels = []
            for c in range(C):
                normalized = normalize_for_visualization(tensor[c], method=norm_method)
                normalized_channels.append(normalized)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (grid_width, grid_height))
            
            for t in range(T):
                # Create grid frame
                grid_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
                
                for c in range(C):
                    row = c // cols
                    col = c % cols
                    
                    if row < rows and col < cols:
                        # Get normalized frame
                        frame = normalized_channels[c][t].cpu().numpy()  # (H, W)
                        
                        # Apply colormap
                        cmap = plt.colormaps[colormap]
                        colored_frame = cmap(frame)[:, :, :3]  # RGB
                        
                        # Convert to uint8
                        frame_uint8 = (colored_frame * 255).astype(np.uint8)
                        
                        # Place in grid
                        start_row = row * H
                        end_row = start_row + H
                        start_col = col * W
                        end_col = start_col + W
                        
                        grid_frame[start_row:end_row, start_col:end_col] = frame_uint8
                
                # Convert RGB to BGR for OpenCV
                grid_frame_bgr = cv2.cvtColor(grid_frame, cv2.COLOR_RGB2BGR)
                out.write(grid_frame_bgr)
    
    elif tensor.dim() == 3:  # Single tensor (T, H, W)
        T, H, W = tensor.shape
        
        # Normalize for visualization with specified method
        normalized = normalize_for_visualization(tensor, method=norm_method)
        
        # Convert to numpy
        frames_np = normalized.cpu().numpy()
        
        # Apply colormap
        cmap = plt.colormaps[colormap]
        norm = Normalize(vmin=0, vmax=1)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
        
        for t in range(T):
            frame = frames_np[t]  # (H, W)
            
            # Apply colormap
            colored_frame = cmap(norm(frame))  # Returns RGBA
            colored_frame_rgb = colored_frame[:, :, :3]  # Take RGB only
            
            # Convert to uint8 and BGR for OpenCV
            frame_uint8 = (colored_frame_rgb * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            
            out.write(frame_bgr)
    
    out.release()
    print(f"Saved video: {output_path}")

def swap_lll_and_decode(model, video1_tensor, video2_tensor, output_dir, fps=10):
    """
    Swap LLL components between two videos and decode the results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        print("Encoding first video...")
        z1 = model.encode(video1_tensor)  # (B, C, T, H, W)
        z2 = model.encode(video2_tensor)  # (B, C, T, H, W)
        if hasattr(z1, 'latent_dist'):
            z_sample1 = z1.latent_dist.sample()
            z_sample2 = z2.latent_dist.sample()
            print(f"Video 1 latent shape: {z_sample1.shape}")
            print(f"Video 2 latent shape: {z_sample2.shape}")
        else:
            z_sample1 = z1
            z_sample2 = z2
            print(f"Video 1 latent shape: {z_sample1.shape}")
            print(f"Video 2 latent shape: {z_sample2.shape}")
        
        if hasattr(model, 'use_latent_wavelet') and model.use_latent_wavelet:
            print("Applying Haar wavelet transform to both videos...")
            
            # Get wavelet coefficients for both videos
            z1_tilde, coeffs1, l1_coeffs1 = model.latent_wav(z_sample1)
            z2_tilde, coeffs2, l1_coeffs2 = model.latent_wav(z_sample2)
            
            print(f"Video 1 LLL shape: {l1_coeffs1.shape}")
            print(f"Video 2 LLL shape: {l1_coeffs2.shape}")
            
            # Swap each temporal low-frequency subband individually (LLL, LLH, LHL, LHH)
            print("Swapping each temporal low-frequency subband individually...")
            
            B, full_C, T_w, H_w, W_w = coeffs1.shape
            C = z_sample1.shape[1]  # Number of channels
            
            # 8个小波子带：[LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH]
            # 前4个是时间低频子带：LLL, LLH, LHL, LHH
            # 每个子带占用C个通道
            temporal_low_freq_names = ['LLL', 'LLH', 'LHL', 'LHH']
            
            # 为每个子带创建交换实验
            for subband_idx, subband_name in enumerate(temporal_low_freq_names):
                print(f"Processing {subband_name} subband swap...")
                
                # 创建新的系数副本
                coeffs1_swapped = coeffs1.clone()
                coeffs2_swapped = coeffs2.clone()
                
                # 计算当前子带的通道范围
                subband_start = subband_idx * C
                subband_end = (subband_idx + 1) * C
                
                # 交换当前子带的系数
                coeffs1_swapped[:, subband_start:subband_end, :, :, :] = coeffs2[:, subband_start:subband_end, :, :, :]
                coeffs2_swapped[:, subband_start:subband_end, :, :, :] = coeffs1[:, subband_start:subband_end, :, :, :]
                
                # 解码交换后的视频
                try:
                    z1_mixed = model.latent_wav.idwt3d(coeffs1_swapped)
                    z2_mixed = model.latent_wav.idwt3d(coeffs2_swapped)
                    
                    if hasattr(model, 'decode'):
                        try:
                            decoded_output1 = model.decode(z1_mixed)
                            decoded_output2 = model.decode(z2_mixed)
                            
                            decoded1 = decoded_output1.sample
                            decoded2 = decoded_output2.sample
                            
                            # 保存解码视频，使用子带名称命名
                            decoded1_path = os.path.join(output_dir, f'decoded_video1_{subband_name}_swapped.mp4')
                            decoded2_path = os.path.join(output_dir, f'decoded_video2_{subband_name}_swapped.mp4')
                            
                            # 转换到 [0, 1] 范围用于可视化
                            decoded1_vis = (decoded1[0] + 1.0) / 2.0
                            decoded2_vis = (decoded2[0] + 1.0) / 2.0
                            
                            # 确保在 [0, 1] 范围内
                            decoded1_vis = torch.clamp(decoded1_vis, 0, 1)
                            decoded2_vis = torch.clamp(decoded2_vis, 0, 1)
                            
                            tensor_to_video(decoded1_vis, decoded1_path, fps=fps, colormap=None)
                            tensor_to_video(decoded2_vis, decoded2_path, fps=fps, colormap=None)
                            
                            print(f"Successfully decoded {subband_name} swapped videos!")
                            
                        except Exception as e:
                            print(f"Decode failed for {subband_name}: {e}")
                    else:
                        print(f"Model does not have a decode method for {subband_name}")
                        
                except Exception as e:
                    print(f"Subband {subband_name} swapping failed: {e}")
            
            # 所有子带交换实验已完成
            print("All subband swap experiments completed!")
            
            # 现在进行全部时间低频子带同时交换的实验
            print("Performing all temporal low-frequency subbands swap experiment...")
            
            # 创建新的系数副本用于全部交换
            coeffs1_all_swapped = coeffs1.clone()
            coeffs2_all_swapped = coeffs2.clone()
            
            # 交换所有时间低频子带 (LLL, LLH, LHL, LHH)
            temporal_low_freq_start = 0
            temporal_low_freq_end = 4 * C  # 前4个子带，每个C个通道
            
            coeffs1_all_swapped[:, temporal_low_freq_start:temporal_low_freq_end, :, :, :] = coeffs2[:, temporal_low_freq_start:temporal_low_freq_end, :, :, :]
            coeffs2_all_swapped[:, temporal_low_freq_start:temporal_low_freq_end, :, :, :] = coeffs1[:, temporal_low_freq_start:temporal_low_freq_end, :, :, :]
            
            # 解码全部交换后的视频
            try:
                z1_all_mixed = model.latent_wav.idwt3d(coeffs1_all_swapped)
                z2_all_mixed = model.latent_wav.idwt3d(coeffs2_all_swapped)
                
                if hasattr(model, 'decode'):
                    try:
                        decoded_output1_all = model.decode(z1_all_mixed)
                        decoded_output2_all = model.decode(z2_all_mixed)
                        
                        decoded1_all = decoded_output1_all.sample
                        decoded2_all = decoded_output2_all.sample
                        
                        # 保存全部交换后的解码视频
                        decoded1_all_path = os.path.join(output_dir, 'decoded_video1_all_temporal_lowfreq_swapped.mp4')
                        decoded2_all_path = os.path.join(output_dir, 'decoded_video2_all_temporal_lowfreq_swapped.mp4')
                        
                        # 转换到 [0, 1] 范围用于可视化
                        decoded1_all_vis = (decoded1_all[0] + 1.0) / 2.0
                        decoded2_all_vis = (decoded2_all[0] + 1.0) / 2.0
                        
                        # 确保在 [0, 1] 范围内
                        decoded1_all_vis = torch.clamp(decoded1_all_vis, 0, 1)
                        decoded2_all_vis = torch.clamp(decoded2_all_vis, 0, 1)
                        
                        tensor_to_video(decoded1_all_vis, decoded1_all_path, fps=fps, colormap=None)
                        tensor_to_video(decoded2_all_vis, decoded2_all_path, fps=fps, colormap=None)
                        
                        print("Successfully decoded videos with all temporal low-frequency subbands swapped!")
                        
                    except Exception as e:
                        print(f"Decode failed for all subbands swap: {e}")
                else:
                    print("Model does not have a decode method for all subbands swap")
                    
            except Exception as e:
                print(f"All temporal low-frequency subbands swapping failed: {e}")
            
            # 进行两两组合交换实验
            print("Performing pairwise subband swap experiments...")
            
            # 生成所有两两组合
            from itertools import combinations
            subband_combinations = list(combinations(range(4), 2))  # (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
            subband_names = ['LLL', 'LLH', 'LHL', 'LHH']
            
            for combo in subband_combinations:
                subband1_idx, subband2_idx = combo
                combo_name = f"{subband_names[subband1_idx]}_{subband_names[subband2_idx]}"
                print(f"Processing {combo_name} combination swap...")
                
                # 创建新的系数副本
                coeffs1_combo_swapped = coeffs1.clone()
                coeffs2_combo_swapped = coeffs2.clone()
                
                # 计算两个子带的通道范围
                subband1_start = subband1_idx * C
                subband1_end = (subband1_idx + 1) * C
                subband2_start = subband2_idx * C
                subband2_end = (subband2_idx + 1) * C
                
                # 交换两个子带的系数
                coeffs1_combo_swapped[:, subband1_start:subband1_end, :, :, :] = coeffs2[:, subband1_start:subband1_end, :, :, :]
                coeffs1_combo_swapped[:, subband2_start:subband2_end, :, :, :] = coeffs2[:, subband2_start:subband2_end, :, :, :]
                coeffs2_combo_swapped[:, subband1_start:subband1_end, :, :, :] = coeffs1[:, subband1_start:subband1_end, :, :, :]
                coeffs2_combo_swapped[:, subband2_start:subband2_end, :, :, :] = coeffs1[:, subband2_start:subband2_end, :, :, :]
                
                # 解码组合交换后的视频
                try:
                    z1_combo_mixed = model.latent_wav.idwt3d(coeffs1_combo_swapped)
                    z2_combo_mixed = model.latent_wav.idwt3d(coeffs2_combo_swapped)
                    
                    if hasattr(model, 'decode'):
                        try:
                            decoded_output1_combo = model.decode(z1_combo_mixed)
                            decoded_output2_combo = model.decode(z2_combo_mixed)
                            
                            decoded1_combo = decoded_output1_combo.sample
                            decoded2_combo = decoded_output2_combo.sample
                            
                            # 保存组合交换后的解码视频
                            decoded1_combo_path = os.path.join(output_dir, f'decoded_video1_{combo_name}_swapped.mp4')
                            decoded2_combo_path = os.path.join(output_dir, f'decoded_video2_{combo_name}_swapped.mp4')
                            
                            # 转换到 [0, 1] 范围用于可视化
                            decoded1_combo_vis = (decoded1_combo[0] + 1.0) / 2.0
                            decoded2_combo_vis = (decoded2_combo[0] + 1.0) / 2.0
                            
                            # 确保在 [0, 1] 范围内
                            decoded1_combo_vis = torch.clamp(decoded1_combo_vis, 0, 1)
                            decoded2_combo_vis = torch.clamp(decoded2_combo_vis, 0, 1)
                            
                            tensor_to_video(decoded1_combo_vis, decoded1_combo_path, fps=fps, colormap=None)
                            tensor_to_video(decoded2_combo_vis, decoded2_combo_path, fps=fps, colormap=None)
                            
                            print(f"Successfully decoded {combo_name} combination swapped videos!")
                            
                        except Exception as e:
                            print(f"Decode failed for {combo_name} combination: {e}")
                    else:
                        print(f"Model does not have a decode method for {combo_name} combination")
                        
                except Exception as e:
                    print(f"Subband combination {combo_name} swapping failed: {e}")
            
            print("All pairwise combination swap experiments completed!")
            
            # 新增实验1: 只保留时间低频子带，时间高频子带设为0
            print("\n=== Experiment 1: Temporal Low-Frequency Only (Zero out Temporal High-Freq) ===")
            try:
                # 创建新的系数副本
                coeffs1_temporal_low_only = coeffs1.clone()
                coeffs2_temporal_low_only = coeffs2.clone()
                
                # 将时间高频子带（HLL, HLH, HHL, HHH）设为0
                # 这些是第4-7个子带（索引4-7），每个占C个通道
                temporal_high_start = 4 * C
                temporal_high_end = 8 * C
                
                coeffs1_temporal_low_only[:, temporal_high_start:temporal_high_end, :, :, :] = 0
                coeffs2_temporal_low_only[:, temporal_high_start:temporal_high_end, :, :, :] = 0
                
                print(f"Zeroed out temporal high-frequency subbands (HLL, HLH, HHL, HHH)")
                print(f"Channel range: {temporal_high_start} to {temporal_high_end}")
                
                # 逆小波变换
                z1_temporal_low_only = model.latent_wav.idwt3d(coeffs1_temporal_low_only)
                z2_temporal_low_only = model.latent_wav.idwt3d(coeffs2_temporal_low_only)
                
                # 解码
                if hasattr(model, 'decode'):
                    try:
                        decoded_output1_low = model.decode(z1_temporal_low_only)
                        decoded_output2_low = model.decode(z2_temporal_low_only)
                        
                        decoded1_low = decoded_output1_low.sample
                        decoded2_low = decoded_output2_low.sample
                        
                        # 保存只有时间低频的重建视频
                        decoded1_low_path = os.path.join(output_dir, 'decoded_video1_temporal_lowfreq_only.mp4')
                        decoded2_low_path = os.path.join(output_dir, 'decoded_video2_temporal_lowfreq_only.mp4')
                        
                        # 转换到 [0, 1] 范围用于可视化
                        decoded1_low_vis = (decoded1_low[0] + 1.0) / 2.0
                        decoded2_low_vis = (decoded2_low[0] + 1.0) / 2.0
                        
                        # 确保在 [0, 1] 范围内
                        decoded1_low_vis = torch.clamp(decoded1_low_vis, 0, 1)
                        decoded2_low_vis = torch.clamp(decoded2_low_vis, 0, 1)
                        
                        tensor_to_video(decoded1_low_vis, decoded1_low_path, fps=fps, colormap=None)
                        tensor_to_video(decoded2_low_vis, decoded2_low_path, fps=fps, colormap=None)
                        
                        print("✓ Successfully decoded temporal low-frequency only videos!")
                        
                    except Exception as e:
                        print(f"✗ Decode failed for temporal low-freq only: {e}")
                else:
                    print("✗ Model does not have a decode method")
                    
            except Exception as e:
                print(f"✗ Temporal low-frequency only experiment failed: {e}")
            
            # 新增实验2: 时间低频子带做temporal pooling，与时间高频子带重建
            print("\n=== Experiment 2: Temporal Pooling of Low-Freq + High-Freq Subbands ===")
            try:
                # 创建新的系数副本
                coeffs1_pooled = coeffs1.clone()
                coeffs2_pooled = coeffs2.clone()
                
                # 对时间低频子带（LLL, LLH, LHL, LHH）做temporal pooling
                B, full_C, T_w, H_w, W_w = coeffs1.shape
                
                for subband_idx in range(4):  # 前4个子带：LLL, LLH, LHL, LHH
                    subband_start = subband_idx * C
                    subband_end = (subband_idx + 1) * C
                    
                    # 提取子带
                    subband1 = coeffs1[:, subband_start:subband_end, :, :, :]  # (B, C, T, H, W)
                    subband2 = coeffs2[:, subband_start:subband_end, :, :, :]
                    
                    # 对时间维度做平均池化
                    subband1_pooled = subband1.mean(dim=2, keepdim=True)  # (B, C, 1, H, W)
                    subband2_pooled = subband2.mean(dim=2, keepdim=True)
                    
                    # 扩展到原来的时间维度
                    subband1_expanded = subband1_pooled.expand(-1, -1, T_w, -1, -1)  # (B, C, T, H, W)
                    subband2_expanded = subband2_pooled.expand(-1, -1, T_w, -1, -1)
                    
                    # 替换到系数中
                    coeffs1_pooled[:, subband_start:subband_end, :, :, :] = subband1_expanded
                    coeffs2_pooled[:, subband_start:subband_end, :, :, :] = subband2_expanded
                    
                    print(f"Pooled {['LLL', 'LLH', 'LHL', 'LHH'][subband_idx]} from shape {subband1.shape} to single frame")
                
                print(f"Temporal low-frequency subbands pooled to single frame and replicated")
                print(f"Temporal high-frequency subbands (HLL, HLH, HHL, HHH) kept as is")
                
                # 逆小波变换
                z1_pooled = model.latent_wav.idwt3d(coeffs1_pooled)
                z2_pooled = model.latent_wav.idwt3d(coeffs2_pooled)
                
                # 解码
                if hasattr(model, 'decode'):
                    try:
                        decoded_output1_pooled = model.decode(z1_pooled)
                        decoded_output2_pooled = model.decode(z2_pooled)
                        
                        decoded1_pooled = decoded_output1_pooled.sample
                        decoded2_pooled = decoded_output2_pooled.sample
                        
                        # 保存pooling后的重建视频
                        decoded1_pooled_path = os.path.join(output_dir, 'decoded_video1_temporal_lowfreq_pooled.mp4')
                        decoded2_pooled_path = os.path.join(output_dir, 'decoded_video2_temporal_lowfreq_pooled.mp4')
                        
                        # 转换到 [0, 1] 范围用于可视化
                        decoded1_pooled_vis = (decoded1_pooled[0] + 1.0) / 2.0
                        decoded2_pooled_vis = (decoded2_pooled[0] + 1.0) / 2.0
                        
                        # 确保在 [0, 1] 范围内
                        decoded1_pooled_vis = torch.clamp(decoded1_pooled_vis, 0, 1)
                        decoded2_pooled_vis = torch.clamp(decoded2_pooled_vis, 0, 1)
                        
                        tensor_to_video(decoded1_pooled_vis, decoded1_pooled_path, fps=fps, colormap=None)
                        tensor_to_video(decoded2_pooled_vis, decoded2_pooled_path, fps=fps, colormap=None)
                        
                        print("✓ Successfully decoded temporal pooled low-freq + high-freq videos!")
                        
                    except Exception as e:
                        print(f"✗ Decode failed for temporal pooled: {e}")
                else:
                    print("✗ Model does not have a decode method")
                    
            except Exception as e:
                print(f"✗ Temporal pooling experiment failed: {e}")
            
            # Save original videos for comparison
            print("\n=== Saving Original Videos ===")
            original1_path = os.path.join(output_dir, 'original_video1.mp4')
            original2_path = os.path.join(output_dir, 'original_video2.mp4')
            
            # 转换从 [-1, 1] 到 [0, 1] 用于可视化
            original1_tensor = (video1_tensor[0] + 1.0) / 2.0  # 从 [-1, 1] 到 [0, 1]
            original2_tensor = (video2_tensor[0] + 1.0) / 2.0  # 从 [-1, 1] 到 [0, 1]
            
            tensor_to_video(original1_tensor, original1_path, fps=fps, colormap=None)
            tensor_to_video(original2_tensor, original2_path, fps=fps, colormap=None)
            
            # Save temporal low-frequency components for visualization
            print("Saving temporal low-frequency components...")
            
            # 提取所有时间低频子带 (LLL, LLH, LHL, LHH)
            coeffs1_reshaped = coeffs1.view(B, C, 8, T_w, H_w, W_w)  # (B, C, 8, T/2, H/2, W/2)
            coeffs2_reshaped = coeffs2.view(B, C, 8, T_w, H_w, W_w)
            
            # 保存每个时间低频子带
            temporal_low_freq_names = ['LLL', 'LLH', 'LHL', 'LHH']
            for i, name in enumerate(temporal_low_freq_names):
                # 提取第i个子带的所有通道并合并
                subband1 = coeffs1_reshaped[0, :, i, :, :, :].sum(dim=0)  # (T/2, H/2, W/2)
                subband2 = coeffs2_reshaped[0, :, i, :, :, :].sum(dim=0)  # (T/2, H/2, W/2)
                
                subband1_path = os.path.join(output_dir, f'{name}_video1.mp4')
                subband2_path = os.path.join(output_dir, f'{name}_video2.mp4')
                
                tensor_to_video(subband1, subband1_path, fps=fps, colormap='viridis')
                tensor_to_video(subband2, subband2_path, fps=fps, colormap='viridis')
            
            # 保存所有时间低频子带的合并版本
            lll1_combined = l1_coeffs1[0].sum(dim=0)  # (T/2, H/2, W/2) - 这只是LLL
            lll2_combined = l1_coeffs2[0].sum(dim=0)  # (T/2, H/2, W/2) - 这只是LLL
            
            lll1_path = os.path.join(output_dir, 'LLL_video1.mp4')
            lll2_path = os.path.join(output_dir, 'LLL_video2.mp4')
            
            tensor_to_video(lll1_combined, lll1_path, fps=fps, colormap='viridis')
            tensor_to_video(lll2_combined, lll2_path, fps=fps, colormap='viridis')
            
            # 计算并保存LLL子带的方差和能量统计图
            print("Computing LLL subband variance and energy statistics...")
            create_lll_statistics_plots(l1_coeffs1, l1_coeffs2, output_dir)
            
            # 诊断潜在空间问题
            print("Diagnosing latent space issues...")
            diagnose_latent_space_issues(z_sample1, z_sample2, l1_coeffs1, l1_coeffs2, output_dir)
            
            # 像素域对照分析
            print("Performing pixel domain baseline analysis...")
            analyze_pixel_domain_baseline(video1_tensor, video2_tensor, output_dir)
            
            # 子带层面能量分析
            print("Analyzing subband energy distribution...")
            analyze_subband_energy_distribution(coeffs1, coeffs2, output_dir)
            
            
            return True
        else:
            print("Model does not use latent wavelet transform!")
            return False

def visualize_haar_components(model, video_tensor, output_dir, fps=10):
    """
    Visualize LLL and HHH components after Haar wavelet transform
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        print("Encoding video to latent space...")
        # Get latent representation
        z = model.encode(video_tensor)  # (B, C, T, H, W)
        
        print(f"Latent shape: {z.shape}")
        print(f"Latent range: [{z.min():.3f}, {z.max():.3f}]")
        
        # Print detailed statistics for debugging
        print_tensor_stats(z[0], "Latent z")

        # 可视化z (所有通道合并)
        print("Visualizing latent z (all channels combined)...")
        z_combined = z[0].sum(dim=0)  # Sum over channels: (C, T, H, W) -> (T, H, W)
        latent_z_path = os.path.join(output_dir, 'latent_z_combined.mp4')
        tensor_to_video(z_combined, latent_z_path, fps=fps, colormap='viridis', norm_method='adaptive')
        
        # 可视化z的每个通道分别保存
        print("Visualizing individual z channels...")
        z_channels_dir = os.path.join(output_dir, 'z_channels')
        os.makedirs(z_channels_dir, exist_ok=True)
        
        C = z.shape[1]  # Number of channels
        for c in range(C):
            print(f"Processing channel {c}/{C-1}")
            z_channel = z[0, c, :, :, :]  # (T, H, W)
            channel_path = os.path.join(z_channels_dir, f'z_channel_{c:02d}.mp4')
            
            # Use different colormaps for different channels
            colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm', 
                        'seismic', 'hot', 'jet', 'turbo', 'rainbow', 'spring', 'summer', 
                        'autumn', 'winter', 'bone']
            colormap = colormaps[c % len(colormaps)]
            
            tensor_to_video(z_channel, channel_path, fps=fps, colormap=colormap, norm_method='adaptive')
        
        # 保存原始视频作为对比
        print("Saving original video for comparison...")
        original_video_path = os.path.join(output_dir, 'original_video.mp4')
        # 转换从 [-1, 1] 到 [0, 1] 用于可视化
        original_tensor = (video_tensor[0] + 1.0) / 2.0  # (3, T, H, W) 从 [-1, 1] 到 [0, 1]
        tensor_to_video(original_tensor, original_video_path, fps=fps, colormap=None)
        
        # 尝试重建视频来检查模型质量
        print("Attempting to reconstruct video to check model quality...")
        try:
            reconstructed_output = model.decode(z)
            reconstructed_video = reconstructed_output.sample[0]  # (3, T, H, W)
            
            # Save reconstructed video
            reconstructed_path = os.path.join(output_dir, 'reconstructed_video.mp4')
            
            # 检查模型输出范围并适当处理
            print(f"Reconstructed video range: [{reconstructed_video.min():.3f}, {reconstructed_video.max():.3f}]")
            
            # 模型输出应该在 [-1, 1] 范围，转换到 [0, 1] 用于可视化
            reconstructed_vis = (reconstructed_video + 1.0) / 2.0
                
            # 确保在 [0, 1] 范围内
            reconstructed_vis = torch.clamp(reconstructed_vis, 0, 1)
            
            tensor_to_video(reconstructed_vis, reconstructed_path, fps=fps, colormap=None)
            
            # Calculate reconstruction error
            mse_error = torch.mean((video_tensor[0] - reconstructed_video) ** 2).item()
            mae_error = torch.mean(torch.abs(video_tensor[0] - reconstructed_video)).item()
            
            print(f"Reconstruction MSE: {mse_error:.6f}")
            print(f"Reconstruction MAE: {mae_error:.6f}")
            
        except Exception as e:
            print(f"Reconstruction failed: {e}")
            print("Model might not have proper decode functionality")

        if hasattr(model, 'use_latent_wavelet') and model.use_latent_wavelet:
            print("Applying Haar wavelet transform to latent...")
            # Apply Haar wavelet transform
            z_tilde, coeffs, l1_coeffs = model.latent_wav(z)
            # coeffs: (B, 8*C, T/2, H/2, W/2) - all 8 wavelet components
            # l1_coeffs: (B, C, T/2, H/2, W/2) - LLL (low-frequency) components
            
            B, full_C, T_w, H_w, W_w = coeffs.shape
            C = z.shape[1]
            
            print(f"Wavelet coefficients shape: {coeffs.shape}")
            print(f"LLL coefficients shape: {l1_coeffs.shape}")
            
            # Print statistics for wavelet coefficients
            print_tensor_stats(l1_coeffs[0], "LLL coefficients")
            
            # Additional debugging for LLL components
            print("\n=== LLL Component Analysis ===")
            lll_sample = l1_coeffs[0, 0, 0, :, :]  # First channel, first frame
            print(f"LLL sample shape: {lll_sample.shape}")
            print(f"LLL sample min: {lll_sample.min():.6f}")
            print(f"LLL sample max: {lll_sample.max():.6f}")
            print(f"LLL sample mean: {lll_sample.mean():.6f}")
            print(f"LLL sample std: {lll_sample.std():.6f}")
            
            # Check if LLL is mostly zeros or constant
            lll_flat = l1_coeffs[0].flatten()
            zero_ratio = (torch.abs(lll_flat) < 1e-6).float().mean()
            constant_ratio = (torch.abs(lll_flat - lll_flat.mean()) < 1e-6).float().mean()
            print(f"Zero ratio: {zero_ratio:.4f}")
            print(f"Constant ratio: {constant_ratio:.4f}")
            print("=" * 40)
            
            # Reshape coeffs to separate components
            coeffs_reshaped = coeffs.view(B, C, 8, T_w, H_w, W_w)
            
            # Component names: [LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH]
            component_names = ['LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH']
            
            # Visualize LLL (low-frequency) components - all channels combined
            print("Creating LLL (low-frequency) visualization (all channels combined)...")
            lll_tensor = l1_coeffs[0]  # Remove batch dimension (C, T/2, H/2, W/2)
            lll_combined = lll_tensor.sum(dim=0)  # Sum over channels: (C, T/2, H/2, W/2) -> (T/2, H/2, W/2)
            lll_path = os.path.join(output_dir, 'LLL_low_frequency_combined.mp4')
            tensor_to_video(lll_combined, lll_path, fps=fps, colormap='viridis')
            
            # Visualize LLL的每个通道分别保存
            print("Visualizing individual LLL channels...")
            lll_channels_dir = os.path.join(output_dir, 'LLL_channels')
            os.makedirs(lll_channels_dir, exist_ok=True)
            
            C_lll = lll_tensor.shape[0]  # Number of LLL channels
            for c in range(C_lll):
                print(f"Processing LLL channel {c}/{C_lll-1}")
                lll_channel = lll_tensor[c, :, :, :]  # (T/2, H/2, W/2)
                lll_channel_path = os.path.join(lll_channels_dir, f'LLL_channel_{c:02d}.mp4')
                
                # Use different colormaps for different channels
                colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm', 
                            'seismic', 'hot', 'jet', 'turbo', 'rainbow', 'spring', 'summer', 
                            'autumn', 'winter', 'bone']
                colormap = colormaps[c % len(colormaps)]
                
                tensor_to_video(lll_channel, lll_channel_path, fps=fps, colormap=colormap, norm_method='adaptive')
            
            # Visualize HHH (high-frequency) components - all channels combined
            print("Creating HHH (high-frequency) visualization (all channels combined)...")
            hhh_tensor = coeffs_reshaped[0, :, 7, :, :, :]  # HHH is the 8th component (index 7)
            hhh_combined = hhh_tensor.sum(dim=0)  # Sum over channels: (C, T/2, H/2, W/2) -> (T/2, H/2, W/2)
            hhh_path = os.path.join(output_dir, 'HHH_high_frequency_combined.mp4')
            tensor_to_video(hhh_combined, hhh_path, fps=fps, colormap='plasma')
            
            # Create comparison video showing all components
            print("Creating all components comparison...")
            all_components_dir = os.path.join(output_dir, 'all_components')
            os.makedirs(all_components_dir, exist_ok=True)
            
            for comp_idx, comp_name in enumerate(component_names):
                comp_tensor = coeffs_reshaped[0, :, comp_idx, :, :, :]  # (C, T_w, H_w, W_w)
                comp_path = os.path.join(all_components_dir, f'{comp_name}.mp4')
                
                # Use different colormaps for different components
                colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm', 'seismic', 'hot']
                colormap = colormaps[comp_idx % len(colormaps)]
                
                tensor_to_video(comp_tensor, comp_path, fps=fps, colormap=colormap)
            
            # Create statistics comparison (using combined channels)
            print("Creating statistics comparison...")
            create_statistics_comparison(lll_combined, hhh_combined, output_dir, fps=fps)
            
            # Analyze energy distribution across all 8 subbands
            print("Analyzing energy distribution...")
            energies, energy_percentages = analyze_wavelet_energy_distribution(coeffs_reshaped, output_dir, fps=fps)
            
            return True
        else:
            print("Model does not use latent wavelet transform!")
            return False

def analyze_wavelet_energy_distribution(coeffs_reshaped, output_dir, fps=10):
    """
    Analyze energy distribution across all 8 wavelet subbands
    coeffs_reshaped: (B, C, 8, T_w, H_w, W_w) - all wavelet components
    """
    print("Analyzing wavelet energy distribution...")
    
    B, C, num_components, T_w, H_w, W_w = coeffs_reshaped.shape
    component_names = ['LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH']
    
    # Calculate energy for each component (sum over channels and spatial dimensions)
    energies = []
    energy_percentages = []
    
    for comp_idx in range(num_components):
        comp_tensor = coeffs_reshaped[0, :, comp_idx, :, :, :]  # (C, T_w, H_w, W_w)
        
        # Calculate total energy (sum of squares)
        energy = torch.sum(comp_tensor ** 2).item()
        energies.append(energy)
    
    # Calculate percentages
    total_energy = sum(energies)
    energy_percentages = [(e / total_energy * 100) if total_energy > 0 else 0 for e in energies]
    
    # Print energy distribution
    print("\n=== Wavelet Energy Distribution ===")
    for i, (comp_name, energy, percentage) in enumerate(zip(component_names, energies, energy_percentages)):
        print(f"{comp_name}: {energy:.6f} ({percentage:.2f}%)")
    print("=" * 40)
    
    # Create energy distribution visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Bar chart of energy distribution
    axes[0, 0].bar(component_names, energies, color=['blue', 'green', 'red', 'orange', 
                                                    'purple', 'brown', 'pink', 'gray'])
    axes[0, 0].set_title('Energy Distribution Across Subbands')
    axes[0, 0].set_ylabel('Energy (Sum of Squares)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Pie chart of energy percentages
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
              '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    axes[0, 1].pie(energy_percentages, labels=component_names, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
    axes[0, 1].set_title('Energy Percentage Distribution')
    
    # 3. Energy evolution over time for each component
    time_energies = []
    for comp_idx in range(num_components):
        comp_tensor = coeffs_reshaped[0, :, comp_idx, :, :, :]  # (C, T_w, H_w, W_w)
        comp_energy_over_time = []
        for t in range(T_w):
            frame_energy = torch.sum(comp_tensor[:, t, :, :] ** 2).item()
            comp_energy_over_time.append(frame_energy)
        time_energies.append(comp_energy_over_time)
    
    for i, (comp_name, time_energy) in enumerate(zip(component_names, time_energies)):
        axes[1, 0].plot(time_energy, label=comp_name, linewidth=2, color=colors[i])
    
    axes[1, 0].set_title('Energy Evolution Over Time')
    axes[1, 0].set_xlabel('Time Frame')
    axes[1, 0].set_ylabel('Energy')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Low-frequency vs High-frequency energy comparison
    low_freq_components = ['LLL', 'LLH', 'LHL', 'LHH']  # Low spatial frequency
    high_freq_components = ['HLL', 'HLH', 'HHL', 'HHH']  # High spatial frequency
    
    low_freq_indices = [i for i, name in enumerate(component_names) if name in low_freq_components]
    high_freq_indices = [i for i, name in enumerate(component_names) if name in high_freq_components]
    
    low_freq_energy = sum(energies[i] for i in low_freq_indices)
    high_freq_energy = sum(energies[i] for i in high_freq_indices)
    
    freq_labels = ['Low Spatial\nFrequency', 'High Spatial\nFrequency']
    freq_energies = [low_freq_energy, high_freq_energy]
    freq_colors = ['#3498db', '#e74c3c']
    
    bars = axes[1, 1].bar(freq_labels, freq_energies, color=freq_colors, alpha=0.8)
    axes[1, 1].set_title('Low vs High Spatial Frequency Energy')
    axes[1, 1].set_ylabel('Total Energy')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, energy in zip(bars, freq_energies):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{energy:.2f}', ha='center', va='bottom')
    
    plt.suptitle('Wavelet Subband Energy Analysis', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    energy_plot_path = os.path.join(output_dir, 'wavelet_energy_analysis.png')
    plt.savefig(energy_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved energy analysis plot: {energy_plot_path}")
    
    # Create energy distribution video
    create_energy_distribution_video(coeffs_reshaped, component_names, output_dir, fps)
    
    return energies, energy_percentages

def create_energy_distribution_video(coeffs_reshaped, component_names, output_dir, fps=10):
    """
    Create a video showing energy distribution over time
    """
    print("Creating energy distribution video...")
    
    B, C, num_components, T_w, H_w, W_w = coeffs_reshaped.shape
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
              '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    
    # Video dimensions
    video_width, video_height = 1600, 900
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    energy_video_path = os.path.join(output_dir, 'energy_distribution_over_time.mp4')
    out = cv2.VideoWriter(energy_video_path, fourcc, fps, (video_width, video_height))
    
    for t in range(T_w):
        # Calculate energy for each component at time t
        energies_at_t = []
        for comp_idx in range(num_components):
            comp_tensor = coeffs_reshaped[0, :, comp_idx, t, :, :]  # (C, H_w, W_w)
            energy = torch.sum(comp_tensor ** 2).item()
            energies_at_t.append(energy)
        
        # Normalize energies for visualization
        total_energy = sum(energies_at_t)
        normalized_energies = [(e / total_energy) if total_energy > 0 else 0 for e in energies_at_t]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 9))
        
        # 1. Bar chart
        bars = axes[0, 0].bar(component_names, normalized_energies, color=colors)
        axes[0, 0].set_title(f'Energy Distribution at Time {t}')
        axes[0, 0].set_ylabel('Normalized Energy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylim(0, max(normalized_energies) * 1.1 if max(normalized_energies) > 0 else 1)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Pie chart
        axes[0, 1].pie(normalized_energies, labels=component_names, autopct='%1.1f%%', 
                       colors=colors, startangle=90)
        axes[0, 1].set_title(f'Energy Percentage at Time {t}')
        
        # 3. Line plot showing evolution
        if t == 0:
            energy_history = {name: [energy] for name, energy in zip(component_names, normalized_energies)}
        else:
            for i, name in enumerate(component_names):
                energy_history[name].append(normalized_energies[i])
        
        for i, (name, history) in enumerate(energy_history.items()):
            axes[1, 0].plot(history, label=name, color=colors[i], linewidth=2)
        
        axes[1, 0].set_title('Energy Evolution Over Time')
        axes[1, 0].set_xlabel('Time Frame')
        axes[1, 0].set_ylabel('Normalized Energy')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(0, T_w-1)
        
        # 4. Text summary
        axes[1, 1].axis('off')
        summary_text = f"Time Frame: {t}/{T_w-1}\n\nEnergy Summary:\n"
        for name, energy, norm_energy in zip(component_names, energies_at_t, normalized_energies):
            summary_text += f"{name}: {energy:.4f} ({norm_energy*100:.1f}%)\n"
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                       fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'Wavelet Energy Distribution Analysis - Frame {t}', fontsize=16)
        plt.tight_layout()
        
        # Convert to video frame
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, 1:]  # Remove alpha channel
        
        # Resize to video dimensions
        img_resized = cv2.resize(img, (video_width, video_height))
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        
        out.write(img_bgr)
        plt.close()
    
    out.release()
    plt.close()
    print(f"Saved energy distribution video: {energy_video_path}")

def create_lll_statistics_plots(l1_coeffs1, l1_coeffs2, output_dir):
    """
    创建LLL子带的统计图，分别分析两个视频
    l1_coeffs1, l1_coeffs2: (B, C, T/2, H/2, W/2) LLL系数
    """
    print("Creating LLL subband statistics plots...")
    
    # 移除batch维度
    lll1 = l1_coeffs1[0]  # (C, T/2, H/2, W/2)
    lll2 = l1_coeffs2[0]  # (C, T/2, H/2, W/2)
    
    C, T, H, W = lll1.shape
    
    # 计算每个通道的L2能量占比和均值
    def calculate_channel_stats(channel_data):
        """计算单个通道的统计信息"""
        # L2能量（平方和）
        l2_energy = torch.sum(channel_data ** 2).item()
        # 均值
        mean_val = torch.mean(channel_data).item()
        # 标准差
        std_val = torch.std(channel_data).item()
        return l2_energy, mean_val, std_val
    
    # 计算所有通道的统计信息
    stats1 = [calculate_channel_stats(lll1[c]) for c in range(C)]
    stats2 = [calculate_channel_stats(lll2[c]) for c in range(C)]
    
    # 提取数据
    energies1 = [s[0] for s in stats1]
    means1 = [s[1] for s in stats1]
    stds1 = [s[2] for s in stats1]
    
    energies2 = [s[0] for s in stats2]
    means2 = [s[1] for s in stats2]
    stds2 = [s[2] for s in stats2]
    
    # 计算能量占比
    total_energy1 = sum(energies1)
    total_energy2 = sum(energies2)
    energy_ratios1 = [e / total_energy1 if total_energy1 > 0 else 0 for e in energies1]
    energy_ratios2 = [e / total_energy2 if total_energy2 > 0 else 0 for e in energies2]
    
    # 创建分别的统计图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    channels = list(range(C))
    x_pos = np.arange(len(channels))
    
    # 1. Video 1 - L2能量占比
    axes[0, 0].bar(x_pos, energy_ratios1, alpha=0.8, color='blue')
    axes[0, 0].set_xlabel('Channel Index')
    axes[0, 0].set_ylabel('L2 Energy Ratio')
    axes[0, 0].set_title('Video 1 - LLL Subband L2 Energy Ratio')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels([f'Ch{i}' for i in channels])
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Video 2 - L2能量占比
    axes[0, 1].bar(x_pos, energy_ratios2, alpha=0.8, color='red')
    axes[0, 1].set_xlabel('Channel Index')
    axes[0, 1].set_ylabel('L2 Energy Ratio')
    axes[0, 1].set_title('Video 2 - LLL Subband L2 Energy Ratio')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels([f'Ch{i}' for i in channels])
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Video 1 - 子带均值
    axes[1, 0].bar(x_pos, means1, alpha=0.8, color='blue')
    axes[1, 0].set_xlabel('Channel Index')
    axes[1, 0].set_ylabel('Mean Value')
    axes[1, 0].set_title('Video 1 - LLL Subband Mean Values')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([f'Ch{i}' for i in channels])
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Video 2 - 子带均值
    axes[1, 1].bar(x_pos, means2, alpha=0.8, color='red')
    axes[1, 1].set_xlabel('Channel Index')
    axes[1, 1].set_ylabel('Mean Value')
    axes[1, 1].set_title('Video 2 - LLL Subband Mean Values')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([f'Ch{i}' for i in channels])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('LLL Subband Statistics Analysis (Separate Videos)', fontsize=16)
    plt.tight_layout()
    
    # 保存统计图
    stats_plot_path = os.path.join(output_dir, 'LLL_subband_statistics.png')
    plt.savefig(stats_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved LLL subband statistics plot: {stats_plot_path}")
    
    # 打印统计信息
    print("\n=== LLL Subband Statistics Summary ===")
    print(f"Number of channels: {C}")
    print(f"Tensor shape: {lll1.shape}")
    
    print(f"\nVideo 1 Statistics:")
    print(f"  Total L2 Energy: {total_energy1:.6f}")
    print(f"  Mean Energy Ratio: {np.mean(energy_ratios1):.6f}")
    print(f"  Mean Value: {np.mean(means1):.6f}")
    print(f"  Std Value: {np.mean(stds1):.6f}")
    
    print(f"\nVideo 2 Statistics:")
    print(f"  Total L2 Energy: {total_energy2:.6f}")
    print(f"  Mean Energy Ratio: {np.mean(energy_ratios2):.6f}")
    print(f"  Mean Value: {np.mean(means2):.6f}")
    print(f"  Std Value: {np.mean(stds2):.6f}")
    
    # 详细打印每个通道的统计信息
    print("\n=== Detailed Channel Statistics ===")
    print("Video 1:")
    for c in range(C):
        print(f"  Channel {c:2d}: EnergyRatio={energy_ratios1[c]:.6f}, Mean={means1[c]:.6f}, Std={stds1[c]:.6f}")
    
    print("Video 2:")
    for c in range(C):
        print(f"  Channel {c:2d}: EnergyRatio={energy_ratios2[c]:.6f}, Mean={means2[c]:.6f}, Std={stds2[c]:.6f}")
    
    print("=" * 50)
    
    # 保存详细的统计信息到文件
    stats_file_path = os.path.join(output_dir, 'LLL_subband_statistics.txt')
    with open(stats_file_path, 'w') as f:
        f.write("LLL Subband Statistics Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of channels: {C}\n")
        f.write(f"Tensor shape: {lll1.shape}\n\n")
        
        f.write("Video 1 Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total L2 Energy: {total_energy1:.6f}\n")
        f.write(f"Mean Energy Ratio: {np.mean(energy_ratios1):.6f}\n")
        f.write(f"Mean Value: {np.mean(means1):.6f}\n")
        f.write(f"Std Value: {np.mean(stds1):.6f}\n\n")
        
        f.write("Video 2 Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total L2 Energy: {total_energy2:.6f}\n")
        f.write(f"Mean Energy Ratio: {np.mean(energy_ratios2):.6f}\n")
        f.write(f"Mean Value: {np.mean(means2):.6f}\n")
        f.write(f"Std Value: {np.mean(stds2):.6f}\n\n")
        
        f.write("Channel-wise Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write("Video 1:\n")
        for c in range(C):
            f.write(f"  Channel {c:2d}: EnergyRatio={energy_ratios1[c]:.6f}, Mean={means1[c]:.6f}, Std={stds1[c]:.6f}\n")
        
        f.write("Video 2:\n")
        for c in range(C):
            f.write(f"  Channel {c:2d}: EnergyRatio={energy_ratios2[c]:.6f}, Mean={means2[c]:.6f}, Std={stds2[c]:.6f}\n")
        
        f.write(f"\nAnalysis Method:\n")
        f.write("-" * 30 + "\n")
        f.write("L2 Energy Ratio = Channel Energy / Total Energy\n")
        f.write("Mean Value = Average of all coefficients in the channel\n")
        f.write("Std Value = Standard deviation of coefficients in the channel\n")
        f.write("This provides a normalized view of energy distribution and coefficient characteristics\n")
    
    print(f"Saved detailed statistics to: {stats_file_path}")

def diagnose_latent_space_issues(z1, z2, l1_coeffs1, l1_coeffs2, output_dir):
    """
    诊断潜在空间问题，特别是为什么只有ch0有能量
    """
    print("=== Latent Space Diagnosis ===")
    
    # 分析原始潜在空间
    print("\n1. Original Latent Space Analysis:")
    print(f"Z1 shape: {z1.shape}, Z2 shape: {z2.shape}")
    
    # 检查原始潜在空间的通道分布
    z1_analysis = analyze_tensor_channels(z1[0], "Z1")
    z2_analysis = analyze_tensor_channels(z2[0], "Z2")
    
    # 分析LLL系数
    print("\n2. LLL Coefficients Analysis:")
    lll1_analysis = analyze_tensor_channels(l1_coeffs1[0], "LLL1")
    lll2_analysis = analyze_tensor_channels(l1_coeffs2[0], "LLL2")
    
    # 检查小波变换的通道映射
    print("\n3. Wavelet Transform Channel Mapping:")
    analyze_wavelet_channel_mapping(z1, l1_coeffs1, "Video1")
    analyze_wavelet_channel_mapping(z2, l1_coeffs2, "Video2")
    
    # 保存诊断结果
    diagnosis_file = os.path.join(output_dir, 'latent_space_diagnosis.txt')
    with open(diagnosis_file, 'w') as f:
        f.write("Latent Space Diagnosis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. Original Latent Space Analysis:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Z1 shape: {z1.shape}\n")
        f.write(f"Z2 shape: {z2.shape}\n")
        f.write(f"Z1 analysis: {z1_analysis}\n")
        f.write(f"Z2 analysis: {z2_analysis}\n\n")
        
        f.write("2. LLL Coefficients Analysis:\n")
        f.write("-" * 30 + "\n")
        f.write(f"LLL1 analysis: {lll1_analysis}\n")
        f.write(f"LLL2 analysis: {lll2_analysis}\n\n")
        
        f.write("3. Potential Issues:\n")
        f.write("-" * 30 + "\n")
        f.write("If only ch0 has energy in LLL coefficients:\n")
        f.write("- The model might be using only the first latent channel\n")
        f.write("- Other channels might be underutilized or zero-initialized\n")
        f.write("- The wavelet transform might be working correctly but latent space is sparse\n")
        f.write("- This could indicate over-parameterization of the latent space\n")
        f.write("- The model might need regularization to encourage channel utilization\n")
        
        f.write("\n4. Recommendations:\n")
        f.write("-" * 30 + "\n")
        f.write("- Check if the model is properly trained\n")
        f.write("- Consider adding channel-wise regularization\n")
        f.write("- Verify if this is expected behavior for the specific model architecture\n")
        f.write("- Check if other subbands (LLH, LHL, LHH) have similar issues\n")
    
    print(f"Saved diagnosis report to: {diagnosis_file}")

def analyze_pixel_domain_baseline(video1_tensor, video2_tensor, output_dir):
    """
    在像素域进行小波变换分析，建立正常范围的对照
    """
    print("=== Pixel Domain Baseline Analysis ===")
    
    # 确保视频张量在正确的范围内
    video1 = video1_tensor[0]  # (C, T, H, W)
    video2 = video2_tensor[0]  # (C, T, H, W)
    
    # 转换到 [0, 1] 范围进行像素域分析
    video1_pixel = (video1 + 1.0) / 2.0
    video2_pixel = (video2 + 1.0) / 2.0
    
    print(f"Video 1 pixel shape: {video1_pixel.shape}")
    print(f"Video 2 pixel shape: {video2_pixel.shape}")
    
    # 对每个视频进行3D小波变换
    from causalvideovae.model.modules.wavelet import HaarWaveletTransform3D
    dwt3d = HaarWaveletTransform3D()
    
    # 将小波变换模块移到与输入相同的设备
    device = video1_pixel.device
    dwt3d = dwt3d.to(device)
    
    # 添加batch维度
    video1_batch = video1_pixel.unsqueeze(0)  # (1, C, T, H, W)
    video2_batch = video2_pixel.unsqueeze(0)  # (1, C, T, H, W)
    
    # 进行小波变换
    coeffs1_pixel = dwt3d(video1_batch)  # (1, 8*C, T/2, H/2, W/2)
    coeffs2_pixel = dwt3d(video2_batch)  # (1, 8*C, T/2, H/2, W/2)
    
    print(f"Pixel domain coeffs1 shape: {coeffs1_pixel.shape}")
    print(f"Pixel domain coeffs2 shape: {coeffs2_pixel.shape}")
    
    # 重新排列系数
    C = video1_pixel.shape[0]
    coeffs1_reshaped = coeffs1_pixel.view(1, C, 8, coeffs1_pixel.shape[2], coeffs1_pixel.shape[3], coeffs1_pixel.shape[4])
    coeffs2_reshaped = coeffs2_pixel.view(1, C, 8, coeffs2_pixel.shape[2], coeffs2_pixel.shape[3], coeffs2_pixel.shape[4])
    
    # 提取LLL系数
    lll1_pixel = coeffs1_reshaped[0, :, 0, :, :, :]  # (C, T/2, H/2, W/2)
    lll2_pixel = coeffs2_reshaped[0, :, 0, :, :, :]  # (C, T/2, H/2, W/2)
    
    print(f"Pixel domain LLL1 shape: {lll1_pixel.shape}")
    print(f"Pixel domain LLL2 shape: {lll2_pixel.shape}")
    
    # 计算像素域LLL的统计信息
    def calculate_pixel_stats(channel_data):
        """计算像素域通道的统计信息"""
        l2_energy = torch.sum(channel_data ** 2).item()
        mean_val = torch.mean(channel_data).item()
        std_val = torch.std(channel_data).item()
        return l2_energy, mean_val, std_val
    
    # 计算所有通道的统计信息
    stats1_pixel = [calculate_pixel_stats(lll1_pixel[c]) for c in range(C)]
    stats2_pixel = [calculate_pixel_stats(lll2_pixel[c]) for c in range(C)]
    
    # 提取数据
    energies1_pixel = [s[0] for s in stats1_pixel]
    means1_pixel = [s[1] for s in stats1_pixel]
    stds1_pixel = [s[2] for s in stats1_pixel]
    
    energies2_pixel = [s[0] for s in stats2_pixel]
    means2_pixel = [s[1] for s in stats2_pixel]
    stds2_pixel = [s[2] for s in stats2_pixel]
    
    # 计算能量占比
    total_energy1_pixel = sum(energies1_pixel)
    total_energy2_pixel = sum(energies2_pixel)
    energy_ratios1_pixel = [e / total_energy1_pixel if total_energy1_pixel > 0 else 0 for e in energies1_pixel]
    energy_ratios2_pixel = [e / total_energy2_pixel if total_energy2_pixel > 0 else 0 for e in energies2_pixel]
    
    # 打印像素域统计信息
    print("\n=== Pixel Domain LLL Statistics ===")
    print(f"Video 1 Pixel Domain:")
    print(f"  Total L2 Energy: {total_energy1_pixel:.6f}")
    print(f"  Mean Energy Ratio: {np.mean(energy_ratios1_pixel):.6f}")
    print(f"  Mean Value: {np.mean(means1_pixel):.6f}")
    print(f"  Std Value: {np.mean(stds1_pixel):.6f}")
    
    print(f"Video 2 Pixel Domain:")
    print(f"  Total L2 Energy: {total_energy2_pixel:.6f}")
    print(f"  Mean Energy Ratio: {np.mean(energy_ratios2_pixel):.6f}")
    print(f"  Mean Value: {np.mean(means2_pixel):.6f}")
    print(f"  Std Value: {np.mean(stds2_pixel):.6f}")
    
    # 保存像素域分析结果
    pixel_analysis_file = os.path.join(output_dir, 'pixel_domain_baseline.txt')
    with open(pixel_analysis_file, 'w') as f:
        f.write("Pixel Domain Baseline Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write("This analysis provides a baseline for normal wavelet behavior in pixel domain.\n")
        f.write("Compare these results with latent domain analysis to identify potential issues.\n\n")
        
        f.write("Video 1 Pixel Domain Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total L2 Energy: {total_energy1_pixel:.6f}\n")
        f.write(f"Mean Energy Ratio: {np.mean(energy_ratios1_pixel):.6f}\n")
        f.write(f"Mean Value: {np.mean(means1_pixel):.6f}\n")
        f.write(f"Std Value: {np.mean(stds1_pixel):.6f}\n\n")
        
        f.write("Video 2 Pixel Domain Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total L2 Energy: {total_energy2_pixel:.6f}\n")
        f.write(f"Mean Energy Ratio: {np.mean(energy_ratios2_pixel):.6f}\n")
        f.write(f"Mean Value: {np.mean(means2_pixel):.6f}\n")
        f.write(f"Std Value: {np.mean(stds2_pixel):.6f}\n\n")
        
        f.write("Channel-wise Pixel Domain Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write("Video 1:\n")
        for c in range(C):
            f.write(f"  Channel {c:2d}: EnergyRatio={energy_ratios1_pixel[c]:.6f}, Mean={means1_pixel[c]:.6f}, Std={stds1_pixel[c]:.6f}\n")
        
        f.write("Video 2:\n")
        for c in range(C):
            f.write(f"  Channel {c:2d}: EnergyRatio={energy_ratios2_pixel[c]:.6f}, Mean={means2_pixel[c]:.6f}, Std={stds2_pixel[c]:.6f}\n")
        
        f.write(f"\nBaseline Interpretation:\n")
        f.write("-" * 30 + "\n")
        f.write("In pixel domain, we expect:\n")
        f.write("1. More even energy distribution across channels\n")
        f.write("2. Reasonable mean values (not too close to zero)\n")
        f.write("3. Consistent statistics between videos\n")
        f.write("4. LLL subband should contain most of the energy\n")
        f.write("\nIf latent domain shows significantly different patterns,\n")
        f.write("it may indicate issues with the model's latent representation.\n")
    
    print(f"Saved pixel domain baseline analysis to: {pixel_analysis_file}")
    
    # 像素域子带能量分析
    print("Analyzing pixel domain subband energy distribution...")
    analyze_pixel_domain_subband_energy(coeffs1_pixel, coeffs2_pixel, output_dir)

def analyze_pixel_domain_subband_energy(coeffs1_pixel, coeffs2_pixel, output_dir):
    """
    分析像素域所有子带的能量分布
    coeffs1_pixel, coeffs2_pixel: (1, 8*C, T/2, H/2, W/2) 像素域小波系数
    """
    print("=== Pixel Domain Subband Energy Distribution Analysis ===")
    
    # 移除batch维度
    coeffs1 = coeffs1_pixel[0]  # (8*C, T/2, H/2, W/2)
    coeffs2 = coeffs2_pixel[0]  # (8*C, T/2, H/2, W/2)
    
    # 子带名称
    subband_names = ['LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH']
    
    # 重新排列系数: (8*C, T, H, W) -> (C, 8, T, H, W)
    C = coeffs1.shape[0] // 8
    coeffs1_reshaped = coeffs1.view(C, 8, coeffs1.shape[1], coeffs1.shape[2], coeffs1.shape[3])
    coeffs2_reshaped = coeffs2.view(C, 8, coeffs2.shape[1], coeffs2.shape[2], coeffs2.shape[3])
    
    print(f"Pixel domain reshaped coeffs1: {coeffs1_reshaped.shape}")
    print(f"Pixel domain reshaped coeffs2: {coeffs2_reshaped.shape}")
    
    # 计算每个子带的能量
    def calculate_pixel_subband_energy(coeffs_reshaped):
        """计算像素域每个子带的能量"""
        subband_energies = []
        for sb in range(8):  # 8个子带
            # 提取该子带的所有通道
            subband_data = coeffs_reshaped[:, sb, :, :, :]  # (C, T, H, W)
            # 计算该子带的总能量
            energy = torch.sum(subband_data ** 2).item()
            subband_energies.append(energy)
        return subband_energies
    
    # 计算两个视频的像素域子带能量
    energies1_pixel = calculate_pixel_subband_energy(coeffs1_reshaped)
    energies2_pixel = calculate_pixel_subband_energy(coeffs2_reshaped)
    
    # 计算能量占比
    total_energy1_pixel = sum(energies1_pixel)
    total_energy2_pixel = sum(energies2_pixel)
    energy_ratios1_pixel = [e / total_energy1_pixel if total_energy1_pixel > 0 else 0 for e in energies1_pixel]
    energy_ratios2_pixel = [e / total_energy2_pixel if total_energy2_pixel > 0 else 0 for e in energies2_pixel]
    
    # 打印像素域子带能量统计
    print("\n=== Pixel Domain Subband Energy Statistics ===")
    print("Video 1 Pixel Domain:")
    for i, (name, energy, ratio) in enumerate(zip(subband_names, energies1_pixel, energy_ratios1_pixel)):
        print(f"  {name}: Energy={energy:.6f}, Ratio={ratio:.4f} ({ratio*100:.2f}%)")
    
    print("Video 2 Pixel Domain:")
    for i, (name, energy, ratio) in enumerate(zip(subband_names, energies2_pixel, energy_ratios2_pixel)):
        print(f"  {name}: Energy={energy:.6f}, Ratio={ratio:.4f} ({ratio*100:.2f}%)")
    
    # 创建像素域子带能量分布图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    x_pos = np.arange(len(subband_names))
    
    # 1. Video 1 - 像素域子带能量占比
    axes[0, 0].bar(x_pos, energy_ratios1_pixel, alpha=0.8, color='lightblue')
    axes[0, 0].set_xlabel('Subband')
    axes[0, 0].set_ylabel('Energy Ratio')
    axes[0, 0].set_title('Video 1 - Pixel Domain Subband Energy Distribution')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(subband_names)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Video 2 - 像素域子带能量占比
    axes[0, 1].bar(x_pos, energy_ratios2_pixel, alpha=0.8, color='lightcoral')
    axes[0, 1].set_xlabel('Subband')
    axes[0, 1].set_ylabel('Energy Ratio')
    axes[0, 1].set_title('Video 2 - Pixel Domain Subband Energy Distribution')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(subband_names)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 像素域子带能量对比
    width = 0.35
    axes[1, 0].bar(x_pos - width/2, energy_ratios1_pixel, width, label='Video 1', alpha=0.8, color='lightblue')
    axes[1, 0].bar(x_pos + width/2, energy_ratios2_pixel, width, label='Video 2', alpha=0.8, color='lightcoral')
    axes[1, 0].set_xlabel('Subband')
    axes[1, 0].set_ylabel('Energy Ratio')
    axes[1, 0].set_title('Pixel Domain Subband Energy Comparison')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(subband_names)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 像素域子带能量相关性
    axes[1, 1].scatter(energy_ratios1_pixel, energy_ratios2_pixel, alpha=0.7, s=100)
    axes[1, 1].plot([min(energy_ratios1_pixel + energy_ratios2_pixel), max(energy_ratios1_pixel + energy_ratios2_pixel)], 
                    [min(energy_ratios1_pixel + energy_ratios2_pixel), max(energy_ratios1_pixel + energy_ratios2_pixel)], 
                    'r--', alpha=0.5, label='y=x')
    axes[1, 1].set_xlabel('Video 1 Energy Ratio')
    axes[1, 1].set_ylabel('Video 2 Energy Ratio')
    axes[1, 1].set_title('Pixel Domain Subband Energy Correlation')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 添加子带名称标注
    for i, name in enumerate(subband_names):
        axes[1, 1].annotate(name, (energy_ratios1_pixel[i], energy_ratios2_pixel[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.suptitle('Pixel Domain Subband Energy Distribution Analysis', fontsize=16)
    plt.tight_layout()
    
    # 保存像素域子带能量分布图
    pixel_subband_plot_path = os.path.join(output_dir, 'pixel_domain_subband_energy_distribution.png')
    plt.savefig(pixel_subband_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved pixel domain subband energy distribution plot: {pixel_subband_plot_path}")
    
    # 保存详细的像素域子带能量分析
    pixel_subband_analysis_file = os.path.join(output_dir, 'pixel_domain_subband_energy_analysis.txt')
    with open(pixel_subband_analysis_file, 'w') as f:
        f.write("Pixel Domain Subband Energy Distribution Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write("This analysis shows the energy distribution across all 8 wavelet subbands in pixel domain.\n")
        f.write("This serves as a baseline for comparison with latent domain analysis.\n")
        f.write("Subbands: LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH\n")
        f.write("Where L=Low frequency, H=High frequency\n\n")
        
        f.write("Video 1 Pixel Domain Subband Energy Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Energy: {total_energy1_pixel:.6f}\n")
        for i, (name, energy, ratio) in enumerate(zip(subband_names, energies1_pixel, energy_ratios1_pixel)):
            f.write(f"{name}: Energy={energy:.6f}, Ratio={ratio:.4f} ({ratio*100:.2f}%)\n")
        
        f.write(f"\nVideo 2 Pixel Domain Subband Energy Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Energy: {total_energy2_pixel:.6f}\n")
        for i, (name, energy, ratio) in enumerate(zip(subband_names, energies2_pixel, energy_ratios2_pixel)):
            f.write(f"{name}: Energy={energy:.6f}, Ratio={ratio:.4f} ({ratio*100:.2f}%)\n")
        
        f.write(f"\nPixel Domain Subband Energy Comparison:\n")
        f.write("-" * 40 + "\n")
        f.write("Subband | Video 1 Ratio | Video 2 Ratio | Difference\n")
        f.write("-" * 60 + "\n")
        for i, name in enumerate(subband_names):
            diff = abs(energy_ratios1_pixel[i] - energy_ratios2_pixel[i])
            f.write(f"{name:7s} | {energy_ratios1_pixel[i]:12.4f} | {energy_ratios2_pixel[i]:12.4f} | {diff:10.4f}\n")
        
        f.write(f"\nPixel Domain Energy Distribution Analysis:\n")
        f.write("-" * 40 + "\n")
        
        # 分析低频子带
        low_freq_energy1_pixel = sum(energy_ratios1_pixel[:4])  # LLL, LLH, LHL, LHH
        low_freq_energy2_pixel = sum(energy_ratios2_pixel[:4])
        f.write(f"Low-frequency subbands (LLL+LLH+LHL+LHH):\n")
        f.write(f"  Video 1: {low_freq_energy1_pixel:.4f} ({low_freq_energy1_pixel*100:.2f}%)\n")
        f.write(f"  Video 2: {low_freq_energy2_pixel:.4f} ({low_freq_energy2_pixel*100:.2f}%)\n")
        
        # 分析高频子带
        high_freq_energy1_pixel = sum(energy_ratios1_pixel[4:])  # HLL, HLH, HHL, HHH
        high_freq_energy2_pixel = sum(energy_ratios2_pixel[4:])
        f.write(f"High-frequency subbands (HLL+HLH+HHL+HHH):\n")
        f.write(f"  Video 1: {high_freq_energy1_pixel:.4f} ({high_freq_energy1_pixel*100:.2f}%)\n")
        f.write(f"  Video 2: {high_freq_energy2_pixel:.4f} ({high_freq_energy2_pixel*100:.2f}%)\n")
        
        # 分析时间维度
        temporal_low_energy1_pixel = sum(energy_ratios1_pixel[:4])  # LLL, LLH, LHL, LHH
        temporal_high_energy1_pixel = sum(energy_ratios1_pixel[4:])  # HLL, HLH, HHL, HHH
        temporal_low_energy2_pixel = sum(energy_ratios2_pixel[:4])
        temporal_high_energy2_pixel = sum(energy_ratios2_pixel[4:])
        
        f.write(f"\nPixel Domain Temporal Frequency Analysis:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Temporal Low (LLL+LLH+LHL+LHH):\n")
        f.write(f"  Video 1: {temporal_low_energy1_pixel:.4f} ({temporal_low_energy1_pixel*100:.2f}%)\n")
        f.write(f"  Video 2: {temporal_low_energy2_pixel:.4f} ({temporal_low_energy2_pixel*100:.2f}%)\n")
        f.write(f"Temporal High (HLL+HLH+HHL+HHH):\n")
        f.write(f"  Video 1: {temporal_high_energy1_pixel:.4f} ({temporal_high_energy1_pixel*100:.2f}%)\n")
        f.write(f"  Video 2: {temporal_high_energy2_pixel:.4f} ({temporal_high_energy2_pixel*100:.2f}%)\n")
        
        f.write(f"\nPixel Domain Baseline Interpretation:\n")
        f.write("-" * 40 + "\n")
        f.write("In pixel domain, we expect:\n")
        f.write("1. LLL (Low-Low-Low) should contain most energy in typical video content\n")
        f.write("2. High-frequency subbands should have lower but non-zero energy\n")
        f.write("3. More even energy distribution compared to latent domain\n")
        f.write("4. Consistent patterns between different videos\n")
        f.write("5. This serves as a reference for normal wavelet behavior\n")
        f.write("\nCompare these results with latent domain analysis to identify:\n")
        f.write("- Model training issues\n")
        f.write("- Latent space over-parameterization\n")
        f.write("- Wavelet transform implementation problems\n")
        f.write("- Channel utilization problems\n")
    
    print(f"Saved pixel domain subband energy analysis to: {pixel_subband_analysis_file}")
    
    # 计算像素域子带能量相关性
    correlation_pixel = np.corrcoef(energy_ratios1_pixel, energy_ratios2_pixel)[0, 1]
    print(f"\nPixel domain subband energy correlation between videos: {correlation_pixel:.4f}")
    
    # 分析像素域能量分布特征
    print(f"\n=== Pixel Domain Energy Distribution Features ===")
    print(f"Video 1 - LLL dominance: {energy_ratios1_pixel[0]:.4f} ({energy_ratios1_pixel[0]*100:.2f}%)")
    print(f"Video 2 - LLL dominance: {energy_ratios2_pixel[0]:.4f} ({energy_ratios2_pixel[0]*100:.2f}%)")
    print(f"Video 1 - Low-freq total: {low_freq_energy1_pixel:.4f} ({low_freq_energy1_pixel*100:.2f}%)")
    print(f"Video 2 - Low-freq total: {low_freq_energy2_pixel:.4f} ({low_freq_energy2_pixel*100:.2f}%)")
    
    if energy_ratios1_pixel[0] > 0.8 or energy_ratios2_pixel[0] > 0.8:
        print("⚠️  WARNING: LLL subband dominates in pixel domain (>80% energy)")
    elif energy_ratios1_pixel[0] > 0.5 or energy_ratios2_pixel[0] > 0.5:
        print("✓ NORMAL: LLL subband has majority energy in pixel domain (>50%)")
    else:
        print("⚠️  UNUSUAL: LLL subband has less than 50% energy in pixel domain")
    
    print("=" * 60)

def analyze_subband_energy_distribution(coeffs1, coeffs2, output_dir):
    """
    分析所有子带的能量分布
    coeffs1, coeffs2: (B, 8*C, T/2, H/2, W/2) 小波系数
    """
    print("=== Subband Energy Distribution Analysis ===")
    
    # 移除batch维度
    coeffs1 = coeffs1[0]  # (8*C, T/2, H/2, W/2)
    coeffs2 = coeffs2[0]  # (8*C, T/2, H/2, W/2)
    
    # 子带名称
    subband_names = ['LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH']
    
    # 重新排列系数: (8*C, T, H, W) -> (C, 8, T, H, W)
    C = coeffs1.shape[0] // 8
    coeffs1_reshaped = coeffs1.view(C, 8, coeffs1.shape[1], coeffs1.shape[2], coeffs1.shape[3])
    coeffs2_reshaped = coeffs2.view(C, 8, coeffs2.shape[1], coeffs2.shape[2], coeffs2.shape[3])
    
    print(f"Reshaped coeffs1: {coeffs1_reshaped.shape}")
    print(f"Reshaped coeffs2: {coeffs2_reshaped.shape}")
    
    # 计算每个子带的能量
    def calculate_subband_energy(coeffs_reshaped):
        """计算每个子带的能量"""
        subband_energies = []
        for sb in range(8):  # 8个子带
            # 提取该子带的所有通道
            subband_data = coeffs_reshaped[:, sb, :, :, :]  # (C, T, H, W)
            # 计算该子带的总能量
            energy = torch.sum(subband_data ** 2).item()
            subband_energies.append(energy)
        return subband_energies
    
    # 计算两个视频的子带能量
    energies1 = calculate_subband_energy(coeffs1_reshaped)
    energies2 = calculate_subband_energy(coeffs2_reshaped)
    
    # 计算能量占比
    total_energy1 = sum(energies1)
    total_energy2 = sum(energies2)
    energy_ratios1 = [e / total_energy1 if total_energy1 > 0 else 0 for e in energies1]
    energy_ratios2 = [e / total_energy2 if total_energy2 > 0 else 0 for e in energies2]
    
    # 打印子带能量统计
    print("\n=== Subband Energy Statistics ===")
    print("Video 1:")
    for i, (name, energy, ratio) in enumerate(zip(subband_names, energies1, energy_ratios1)):
        print(f"  {name}: Energy={energy:.6f}, Ratio={ratio:.4f} ({ratio*100:.2f}%)")
    
    print("Video 2:")
    for i, (name, energy, ratio) in enumerate(zip(subband_names, energies2, energy_ratios2)):
        print(f"  {name}: Energy={energy:.6f}, Ratio={ratio:.4f} ({ratio*100:.2f}%)")
    
    # 创建子带能量分布图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    x_pos = np.arange(len(subband_names))
    
    # 1. Video 1 - 子带能量占比
    axes[0, 0].bar(x_pos, energy_ratios1, alpha=0.8, color='blue')
    axes[0, 0].set_xlabel('Subband')
    axes[0, 0].set_ylabel('Energy Ratio')
    axes[0, 0].set_title('Video 1 - Subband Energy Distribution')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(subband_names)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Video 2 - 子带能量占比
    axes[0, 1].bar(x_pos, energy_ratios2, alpha=0.8, color='red')
    axes[0, 1].set_xlabel('Subband')
    axes[0, 1].set_ylabel('Energy Ratio')
    axes[0, 1].set_title('Video 2 - Subband Energy Distribution')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(subband_names)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 子带能量对比
    width = 0.35
    axes[1, 0].bar(x_pos - width/2, energy_ratios1, width, label='Video 1', alpha=0.8, color='blue')
    axes[1, 0].bar(x_pos + width/2, energy_ratios2, width, label='Video 2', alpha=0.8, color='red')
    axes[1, 0].set_xlabel('Subband')
    axes[1, 0].set_ylabel('Energy Ratio')
    axes[1, 0].set_title('Subband Energy Comparison')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(subband_names)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 子带能量相关性
    axes[1, 1].scatter(energy_ratios1, energy_ratios2, alpha=0.7, s=100)
    axes[1, 1].plot([min(energy_ratios1 + energy_ratios2), max(energy_ratios1 + energy_ratios2)], 
                    [min(energy_ratios1 + energy_ratios2), max(energy_ratios1 + energy_ratios2)], 
                    'r--', alpha=0.5, label='y=x')
    axes[1, 1].set_xlabel('Video 1 Energy Ratio')
    axes[1, 1].set_ylabel('Video 2 Energy Ratio')
    axes[1, 1].set_title('Subband Energy Correlation')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 添加子带名称标注
    for i, name in enumerate(subband_names):
        axes[1, 1].annotate(name, (energy_ratios1[i], energy_ratios2[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.suptitle('Subband Energy Distribution Analysis', fontsize=16)
    plt.tight_layout()
    
    # 保存子带能量分布图
    subband_plot_path = os.path.join(output_dir, 'subband_energy_distribution.png')
    plt.savefig(subband_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved subband energy distribution plot: {subband_plot_path}")
    
    # 保存详细的子带能量分析
    subband_analysis_file = os.path.join(output_dir, 'subband_energy_analysis.txt')
    with open(subband_analysis_file, 'w') as f:
        f.write("Subband Energy Distribution Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write("This analysis shows the energy distribution across all 8 wavelet subbands.\n")
        f.write("Subbands: LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH\n")
        f.write("Where L=Low frequency, H=High frequency\n\n")
        
        f.write("Video 1 Subband Energy Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Energy: {total_energy1:.6f}\n")
        for i, (name, energy, ratio) in enumerate(zip(subband_names, energies1, energy_ratios1)):
            f.write(f"{name}: Energy={energy:.6f}, Ratio={ratio:.4f} ({ratio*100:.2f}%)\n")
        
        f.write(f"\nVideo 2 Subband Energy Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Energy: {total_energy2:.6f}\n")
        for i, (name, energy, ratio) in enumerate(zip(subband_names, energies2, energy_ratios2)):
            f.write(f"{name}: Energy={energy:.6f}, Ratio={ratio:.4f} ({ratio*100:.2f}%)\n")
        
        f.write(f"\nSubband Energy Comparison:\n")
        f.write("-" * 30 + "\n")
        f.write("Subband | Video 1 Ratio | Video 2 Ratio | Difference\n")
        f.write("-" * 50 + "\n")
        for i, name in enumerate(subband_names):
            diff = abs(energy_ratios1[i] - energy_ratios2[i])
            f.write(f"{name:7s} | {energy_ratios1[i]:12.4f} | {energy_ratios2[i]:12.4f} | {diff:10.4f}\n")
        
        f.write(f"\nEnergy Distribution Analysis:\n")
        f.write("-" * 30 + "\n")
        
        # 分析低频子带
        low_freq_energy1 = sum(energy_ratios1[:4])  # LLL, LLH, LHL, LHH
        low_freq_energy2 = sum(energy_ratios2[:4])
        f.write(f"Low-frequency subbands (LLL+LLH+LHL+LHH):\n")
        f.write(f"  Video 1: {low_freq_energy1:.4f} ({low_freq_energy1*100:.2f}%)\n")
        f.write(f"  Video 2: {low_freq_energy2:.4f} ({low_freq_energy2*100:.2f}%)\n")
        
        # 分析高频子带
        high_freq_energy1 = sum(energy_ratios1[4:])  # HLL, HLH, HHL, HHH
        high_freq_energy2 = sum(energy_ratios2[4:])
        f.write(f"High-frequency subbands (HLL+HLH+HHL+HHH):\n")
        f.write(f"  Video 1: {high_freq_energy1:.4f} ({high_freq_energy1*100:.2f}%)\n")
        f.write(f"  Video 2: {high_freq_energy2:.4f} ({high_freq_energy2*100:.2f}%)\n")
        
        # 分析时间维度
        temporal_low_energy1 = sum(energy_ratios1[:4])  # LLL, LLH, LHL, LHH
        temporal_high_energy1 = sum(energy_ratios1[4:])  # HLL, HLH, HHL, HHH
        temporal_low_energy2 = sum(energy_ratios2[:4])
        temporal_high_energy2 = sum(energy_ratios2[4:])
        
        f.write(f"\nTemporal Frequency Analysis:\n")
        f.write(f"Temporal Low (LLL+LLH+LHL+LHH):\n")
        f.write(f"  Video 1: {temporal_low_energy1:.4f} ({temporal_low_energy1*100:.2f}%)\n")
        f.write(f"  Video 2: {temporal_low_energy2:.4f} ({temporal_low_energy2*100:.2f}%)\n")
        f.write(f"Temporal High (HLL+HLH+HHL+HHH):\n")
        f.write(f"  Video 1: {temporal_high_energy1:.4f} ({temporal_high_energy1*100:.2f}%)\n")
        f.write(f"  Video 2: {temporal_high_energy2:.4f} ({temporal_high_energy2*100:.2f}%)\n")
        
        f.write(f"\nInterpretation:\n")
        f.write("-" * 30 + "\n")
        f.write("1. LLL (Low-Low-Low) should contain most energy in typical video content\n")
        f.write("2. High-frequency subbands (HLL, HLH, HHL, HHH) should have lower energy\n")
        f.write("3. Temporal low-frequency subbands (LLL, LLH, LHL, LHH) contain motion information\n")
        f.write("4. Temporal high-frequency subbands (HLL, HLH, HHL, HHH) contain frame-to-frame changes\n")
        f.write("5. If energy distribution is very uneven, it may indicate:\n")
        f.write("   - Model training issues\n")
        f.write("   - Latent space over-parameterization\n")
        f.write("   - Wavelet transform implementation problems\n")
    
    print(f"Saved subband energy analysis to: {subband_analysis_file}")
    
    # 计算子带能量相关性
    correlation = np.corrcoef(energy_ratios1, energy_ratios2)[0, 1]
    print(f"\nSubband energy correlation between videos: {correlation:.4f}")
    
    # 分析能量分布特征
    print(f"\n=== Energy Distribution Features ===")
    print(f"Video 1 - LLL dominance: {energy_ratios1[0]:.4f} ({energy_ratios1[0]*100:.2f}%)")
    print(f"Video 2 - LLL dominance: {energy_ratios2[0]:.4f} ({energy_ratios2[0]*100:.2f}%)")
    print(f"Video 1 - Low-freq total: {low_freq_energy1:.4f} ({low_freq_energy1*100:.2f}%)")
    print(f"Video 2 - Low-freq total: {low_freq_energy2:.4f} ({low_freq_energy2*100:.2f}%)")
    
    if energy_ratios1[0] > 0.8 or energy_ratios2[0] > 0.8:
        print("⚠️  WARNING: LLL subband dominates (>80% energy)")
    elif energy_ratios1[0] > 0.5 or energy_ratios2[0] > 0.5:
        print("✓ NORMAL: LLL subband has majority energy (>50%)")
    else:
        print("⚠️  UNUSUAL: LLL subband has less than 50% energy")
    
    print("=" * 50)

def analyze_tensor_channels(tensor, name):
    """分析张量的通道分布"""
    C = tensor.shape[0]
    analysis = {}
    
    for c in range(C):
        channel = tensor[c]
        energy = torch.sum(channel ** 2).item()
        mean_val = channel.mean().item()
        std_val = channel.std().item()
        zero_ratio = (torch.abs(channel) < 1e-6).float().mean().item()
        
        analysis[f'ch{c}'] = {
            'energy': energy,
            'mean': mean_val,
            'std': std_val,
            'zero_ratio': zero_ratio
        }
    
    # 打印摘要
    energies = [analysis[f'ch{c}']['energy'] for c in range(C)]
    zero_ratios = [analysis[f'ch{c}']['zero_ratio'] for c in range(C)]
    
    print(f"{name} - Total channels: {C}")
    print(f"  Energy range: [{min(energies):.6f}, {max(energies):.6f}]")
    print(f"  Zero ratio range: [{min(zero_ratios):.2%}, {max(zero_ratios):.2%}]")
    
    # 检查异常通道
    max_energy = max(energies)
    for c in range(C):
        if energies[c] > 0.1 * max_energy:  # 能量超过最大值的10%
            print(f"  Channel {c}: Energy={energies[c]:.6f}, ZeroRatio={zero_ratios[c]:.2%}")
    
    return analysis

def analyze_wavelet_channel_mapping(z, lll_coeffs, name):
    """分析小波变换的通道映射关系"""
    print(f"\n{name} Wavelet Channel Mapping:")
    
    # 检查原始潜在空间和小波系数的关系
    z_channels = z[0]  # (C, T, H, W)
    lll_channels = lll_coeffs[0]  # (C, T/2, H/2, W/2)
    
    print(f"  Original latent: {z_channels.shape}")
    print(f"  LLL coefficients: {lll_channels.shape}")
    
    # 计算每个通道的能量比例
    z_energies = [torch.sum(z_channels[c] ** 2).item() for c in range(z_channels.shape[0])]
    lll_energies = [torch.sum(lll_channels[c] ** 2).item() for c in range(lll_channels.shape[0])]
    
    print(f"  Original latent energy distribution:")
    for c, energy in enumerate(z_energies):
        if energy > 0.01 * max(z_energies):  # 只显示有意义的通道
            print(f"    Channel {c}: {energy:.6f}")
    
    print(f"  LLL coefficients energy distribution:")
    for c, energy in enumerate(lll_energies):
        if energy > 0.01 * max(lll_energies):  # 只显示有意义的通道
            print(f"    Channel {c}: {energy:.6f}")
    
    # 检查能量保持
    total_z_energy = sum(z_energies)
    total_lll_energy = sum(lll_energies)
    energy_ratio = total_lll_energy / total_z_energy if total_z_energy > 0 else 0
    
    print(f"  Energy preservation ratio: {energy_ratio:.4f}")
    print(f"  (LLL energy / Original energy)")

def create_statistics_comparison(lll_tensor, hhh_tensor, output_dir, fps=10):
    """
    Create statistical comparison between LLL and HHH components
    Now expects 2D tensors (T, H, W) - combined channels
    """
    T, H, W = lll_tensor.shape
    
    # Calculate statistics over time
    lll_means = []
    lll_stds = []
    hhh_means = []
    hhh_stds = []
    
    for t in range(T):
        lll_frame = lll_tensor[t, :, :]  # (H, W)
        hhh_frame = hhh_tensor[t, :, :]  # (H, W)
        
        lll_means.append(lll_frame.mean().cpu().numpy())  # scalar
        lll_stds.append(lll_frame.std().cpu().numpy())
        hhh_means.append(hhh_frame.mean().cpu().numpy())
        hhh_stds.append(hhh_frame.std().cpu().numpy())
    
    lll_means = np.array(lll_means)  # (T,)
    lll_stds = np.array(lll_stds)
    hhh_means = np.array(hhh_means)
    hhh_stds = np.array(hhh_stds)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Video dimensions
    video_width, video_height = 1500, 1000
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    stats_path = os.path.join(output_dir, 'LLL_vs_HHH_statistics.mp4')
    out = cv2.VideoWriter(stats_path, fourcc, fps, (video_width, video_height))
    
    for t in range(T):
        # Clear axes
        for ax in axes.flat:
            ax.clear()
        
        # Plot LLL vs HHH means (single values now)
        axes[0, 0].bar(['LLL (Low-freq)', 'HHH (High-freq)'], [lll_means[t], hhh_means[t]], 
                      color=['blue', 'red'], alpha=0.8)
        axes[0, 0].set_title(f'Mean Values Comparison at Time {t}')
        axes[0, 0].set_ylabel('Mean Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot LLL vs HHH stds (single values now)
        axes[0, 1].bar(['LLL (Low-freq)', 'HHH (High-freq)'], [lll_stds[t], hhh_stds[t]], 
                      color=['blue', 'red'], alpha=0.8)
        axes[0, 1].set_title(f'Std Values Comparison at Time {t}')
        axes[0, 1].set_ylabel('Std Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot LLL evolution over time
        axes[1, 0].clear()
        axes[1, 0].plot(lll_means[:t+1], label='LLL', color='blue', linewidth=2)
        axes[1, 0].set_title('LLL Mean Evolution Over Time')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Mean Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot HHH evolution over time
        axes[1, 1].clear()
        axes[1, 1].plot(hhh_means[:t+1], label='HHH', color='red', linewidth=2)
        axes[1, 1].set_title('HHH Mean Evolution Over Time')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Mean Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'LLL vs HHH Components Analysis - Time {t}', fontsize=16)
        plt.tight_layout()
        
        # Convert matplotlib figure to OpenCV format
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, 1:]  # Remove alpha channel, keep RGB
        
        # Resize to video dimensions
        img_resized = cv2.resize(img, (video_width, video_height))
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        
        out.write(img_bgr)
    
    out.release()
    plt.close()
    print(f"Saved statistics comparison: {stats_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize WFVAE latent Haar wavelet components")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--model_config", type=str, required=True, help="Path to model config JSON file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--video_path", type=str, help="Path to input video file or directory")
    parser.add_argument("--output_dir", type=str, default="./haar_visualization", help="Output directory")
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--video_size", type=str, default="256x256", help="Video size (e.g., 256x256)")
    parser.add_argument("--num_frames", type=int, default=25, help="Number of frames to extract from video")
    parser.add_argument("--max_videos", type=int, default=1, help="Maximum number of videos to process")
    parser.add_argument("--use_random", action="store_true", help="Use random noise instead of real video")
    parser.add_argument("--swap_lll", action="store_true", help="Swap LLL components between two videos and decode")
    
    args = parser.parse_args()
    
    # Parse video size
    width, height = map(int, args.video_size.split('x'))
    
    # Load model
    print("Loading model...")
    with open(args.model_config, 'r') as f:
        config = json.load(f)
    
    model_cls = ModelRegistry.get_model(args.model_name)
    model = model_cls.from_config(args.model_config)
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        if 'gen_model' in state_dict:
            state_dict = state_dict['gen_model']
    elif 'ema_state_dict' in checkpoint:
        state_dict = checkpoint['ema_state_dict']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.to(args.device)
    model.eval()
    
    # Load video(s)
    video_tensors = []
    
    if args.swap_lll:
        # For LLL swap experiment, we need exactly 2 videos
        print("LLL swap experiment mode - loading 2 videos...")
        if not args.video_path:
            print("Using random noise for both videos...")
            video1_tensor = torch.randn(1, 3, args.num_frames, height, width).to(args.device)
            video2_tensor = torch.randn(1, 3, args.num_frames, height, width).to(args.device)
            video_tensors = [video1_tensor, video2_tensor]
        else:
            video_path = args.video_path
            if os.path.isfile(video_path):
                print("Error: For LLL swap, provide a directory with at least 2 videos")
                print("Using random noise for both videos...")
                video1_tensor = torch.randn(1, 3, args.num_frames, height, width).to(args.device)
                video2_tensor = torch.randn(1, 3, args.num_frames, height, width).to(args.device)
                video_tensors = [video1_tensor, video2_tensor]
            elif os.path.isdir(video_path):
                print(f"Loading 2 videos from directory: {video_path}")
                try:
                    video_tensors = load_video_from_directory(video_path, (height, width), args.num_frames, 2, args.device)
                    if len(video_tensors) < 2:
                        print(f"Only found {len(video_tensors)} videos, using random for second video...")
                        video2_tensor = torch.randn(1, 3, args.num_frames, height, width).to(args.device)
                        video_tensors.append(video2_tensor)
                except Exception as e:
                    print(f"Error loading videos: {e}")
                    print("Using random noise for both videos...")
                    video1_tensor = torch.randn(1, 3, args.num_frames, height, width).to(args.device)
                    video2_tensor = torch.randn(1, 3, args.num_frames, height, width).to(args.device)
                    video_tensors = [video1_tensor, video2_tensor]
            else:
                print(f"Video path does not exist: {video_path}")
                print("Using random noise for both videos...")
                video1_tensor = torch.randn(1, 3, args.num_frames, height, width).to(args.device)
                video2_tensor = torch.randn(1, 3, args.num_frames, height, width).to(args.device)
                video_tensors = [video1_tensor, video2_tensor]
    
    elif args.use_random or not args.video_path:
        print(f"Using random noise tensor: ({args.num_frames}, 3, {height}, {width})")
        video_tensor = torch.randn(1, 3, args.num_frames, height, width).to(args.device)
        video_tensors.append(video_tensor)
    else:
        video_path = args.video_path
        
        if os.path.isfile(video_path):
            print(f"Loading single video: {video_path}")
            try:
                video_tensor = load_video_from_file(video_path, (height, width), args.num_frames, args.device)
                video_tensors.append(video_tensor)
            except Exception as e:
                print(f"Error loading video: {e}")
                print("Falling back to random noise...")
                video_tensor = torch.randn(1, 3, args.num_frames, height, width).to(args.device)
                video_tensors.append(video_tensor)
                
        elif os.path.isdir(video_path):
            print(f"Loading videos from directory: {video_path}")
            try:
                video_tensors = load_video_from_directory(video_path, (height, width), args.num_frames, args.max_videos, args.device)
                if not video_tensors:
                    print("No videos loaded, using random noise...")
                    video_tensor = torch.randn(1, 3, args.num_frames, height, width).to(args.device)
                    video_tensors.append(video_tensor)
            except Exception as e:
                print(f"Error loading videos from directory: {e}")
                print("Falling back to random noise...")
                video_tensor = torch.randn(1, 3, args.num_frames, height, width).to(args.device)
                video_tensors.append(video_tensor)
        else:
            print(f"Video path does not exist: {video_path}")
            print("Using random noise...")
            video_tensor = torch.randn(1, 3, args.num_frames, height, width).to(args.device)
            video_tensors.append(video_tensor)
    
    # Process videos
    if args.swap_lll and len(video_tensors) >= 2:
        print(f"\nStarting LLL swap experiment...")
        print(f"Video 1 shape: {video_tensors[0].shape}")
        print(f"Video 2 shape: {video_tensors[1].shape}")
        
        # Create output directory for LLL swap experiment
        swap_output_dir = os.path.join(args.output_dir, "lll_swap_experiment")
        
        # Perform LLL swap experiment
        success = swap_lll_and_decode(model, video_tensors[0], video_tensors[1], swap_output_dir, fps=args.fps)
        
        if success:
            print(f"LLL swap experiment complete! Results saved to: {swap_output_dir}")
        else:
            print(f"LLL swap experiment failed!")
    else:
        # Process each video normally
        for i, video_tensor in enumerate(video_tensors):
            print(f"\nProcessing video {i+1}/{len(video_tensors)}")
            
            # Create output directory for this video
            if len(video_tensors) > 1:
                video_output_dir = os.path.join(args.output_dir, f"video_{i+1:03d}")
            else:
                video_output_dir = args.output_dir
            
            # Visualize Haar components
            print("Starting Haar wavelet visualization...")
            success = visualize_haar_components(model, video_tensor, video_output_dir, fps=args.fps)
            
            if success:
                print(f"Visualization complete for video {i+1}! Results saved to: {video_output_dir}")
            else:
                print(f"Visualization failed for video {i+1}!")
    
    print(f"\nAll visualizations complete! Results saved to: {args.output_dir}")
    
    if args.swap_lll:
        print("\nTemporal Low-Frequency Subband Swap Experiment Generated files:")
        print("- original_video1.mp4: Original first video")
        print("- original_video2.mp4: Original second video")
        print("- LLL_video1.mp4: LLL components of first video")
        print("- LLL_video2.mp4: LLL components of second video")
        print("- LLH_video1.mp4: LLH components of first video")
        print("- LLH_video2.mp4: LLH components of second video")
        print("- LHL_video1.mp4: LHL components of first video")
        print("- LHL_video2.mp4: LHL components of second video")
        print("- LHH_video1.mp4: LHH components of first video")
        print("- LHH_video2.mp4: LHH components of second video")
        print("- decoded_video1_LLL_swapped.mp4: First video with LLL swapped")
        print("- decoded_video2_LLL_swapped.mp4: Second video with LLL swapped")
        print("- decoded_video1_LLH_swapped.mp4: First video with LLH swapped")
        print("- decoded_video2_LLH_swapped.mp4: Second video with LLH swapped")
        print("- decoded_video1_LHL_swapped.mp4: First video with LHL swapped")
        print("- decoded_video2_LHL_swapped.mp4: Second video with LHL swapped")
        print("- decoded_video1_LHH_swapped.mp4: First video with LHH swapped")
        print("- decoded_video2_LHH_swapped.mp4: Second video with LHH swapped")
        print("- decoded_video1_all_temporal_lowfreq_swapped.mp4: First video with all temporal low-freq subbands swapped")
        print("- decoded_video2_all_temporal_lowfreq_swapped.mp4: Second video with all temporal low-freq subbands swapped")
        print("- decoded_video1_LLL_LLH_swapped.mp4: First video with LLL+LLH swapped")
        print("- decoded_video2_LLL_LLH_swapped.mp4: Second video with LLL+LLH swapped")
        print("- decoded_video1_LLL_LHL_swapped.mp4: First video with LLL+LHL swapped")
        print("- decoded_video2_LLL_LHL_swapped.mp4: Second video with LLL+LHL swapped")
        print("- decoded_video1_LLL_LHH_swapped.mp4: First video with LLL+LHH swapped")
        print("- decoded_video2_LLL_LHH_swapped.mp4: Second video with LLL+LHH swapped")
        print("- decoded_video1_LLH_LHL_swapped.mp4: First video with LLH+LHL swapped")
        print("- decoded_video2_LLH_LHL_swapped.mp4: Second video with LLH+LHL swapped")
        print("- decoded_video1_LLH_LHH_swapped.mp4: First video with LLH+LHH swapped")
        print("- decoded_video2_LLH_LHH_swapped.mp4: Second video with LLH+LHH swapped")
        print("- decoded_video1_LHL_LHH_swapped.mp4: First video with LHL+LHH swapped")
        print("- decoded_video2_LHL_LHH_swapped.mp4: Second video with LHL+LHH swapped")
        print("- decoded_video1_temporal_lowfreq_only.mp4: First video reconstructed with only temporal low-freq (zero out temporal high-freq)")
        print("- decoded_video2_temporal_lowfreq_only.mp4: Second video reconstructed with only temporal low-freq (zero out temporal high-freq)")
        print("- decoded_video1_temporal_lowfreq_pooled.mp4: First video with temporal low-freq pooled to single frame + temporal high-freq")
        print("- decoded_video2_temporal_lowfreq_pooled.mp4: Second video with temporal low-freq pooled to single frame + temporal high-freq")
        print("- LLL_subband_statistics.png: LLL subband variance and energy statistics plot")
        print("- LLL_subband_statistics.txt: Detailed LLL subband statistics")
        print("- latent_space_diagnosis.txt: Latent space diagnosis report")
        print("- pixel_domain_baseline.txt: Pixel domain baseline analysis for comparison")
        print("- pixel_domain_subband_energy_distribution.png: Pixel domain subband energy distribution plot")
        print("- pixel_domain_subband_energy_analysis.txt: Pixel domain subband energy analysis")
        print("- subband_energy_distribution.png: All 8 subbands energy distribution plot")
        print("- subband_energy_analysis.txt: Detailed subband energy analysis")
    else:
        print("\nGenerated files:")
        print("- original_video.mp4: Original input video")
        print("- reconstructed_video.mp4: Model reconstruction (quality check)")
        print("- latent_z_combined.mp4: Combined latent representation")
        print("- z_channels/: Individual latent channels (16 channels)")
        print("- LLL_low_frequency_combined.mp4: Low-frequency components (combined)")
        print("- LLL_channels/: Individual LLL channels (16 channels)")
        print("- HHH_high_frequency_combined.mp4: High-frequency components (combined)")
        print("- all_components/: All 8 wavelet components")
        print("- LLL_vs_HHH_statistics.mp4: Statistical comparison")
        print("- wavelet_energy_analysis.png: Energy distribution analysis plot")
        print("- energy_distribution_over_time.mp4: Energy evolution video")

if __name__ == "__main__":
    main()

"""
使用示例:

# 使用真实视频文件
python ./visualizatoin/latent_to_video.py \
    --model_config /scratch/work/guanj2/latent_wf_vae_new/examples/AE/wfvae-latent-large-16chn.json \
    --checkpoint /scratch/cs/aaltoml/users/guanjr/experiment/vae/AE_large_16chn/Latent_WFAE-lr1.00e-05-bs2-rs256-sr1-fr25/checkpoint-100000.ckpt \
    --video_path /path/to/your/video.mp4 \
    --output_dir haar_visualization \
    --fps 10 \
    --video_size 256x256 \
    --num_frames 25

# 使用视频目录（处理多个视频）
python ./visualizatoin/latent_to_video.py \
    --model_config /scratch/work/guanj2/latent_wf_vae_new/examples/AE/wfvae-latent-large-16chn.json \
    --checkpoint /scratch/cs/aaltoml/users/guanjr/experiment/vae/AE_large_16chn/Latent_WFAE-lr1.00e-05-bs2-rs256-sr1-fr25/checkpoint-100000.ckpt \
    --video_path /scratch/cs/aaltoml/users/guanjr/data/filtered_dataset/videos \
    --output_dir haar_visualization \
    --max_videos 5 \
    --fps 10 \
    --video_size 256x256 \
    --num_frames 25

# 使用随机噪声（测试用）
python ./visualizatoin/latent_to_video.py \
    --model_config /scratch/work/guanj2/latent_wf_vae_new/examples/AE/wfvae-latent-large-16chn.json \
    --checkpoint /scratch/cs/aaltoml/users/guanjr/experiment/vae/AE_large_16chn/Latent_WFAE-lr1.00e-05-bs2-rs256-sr1-fr25/checkpoint-100000.ckpt \
    --use_random \
    --output_dir haar_visualization \
    --fps 5 \
    --video_size 256x256 \
    --num_frames 25

# LLL交换实验（使用随机噪声）
python ./visualizatoin/latent_to_video.py \
    --model_config /scratch/work/guanj2/latent_wf_vae_new/examples/AE/wfvae-latent-large-16chn.json \
    --checkpoint /scratch/cs/aaltoml/users/guanjr/experiment/vae/AE_large_16chn/Latent_WFAE-lr1.00e-05-bs2-rs256-sr1-fr25/checkpoint-100000.ckpt \
    --swap_lll \
    --output_dir lll_swap_experiment \
    --fps 5 \
    --video_size 256x256 \
    --num_frames 25

# LLL交换实验（使用真实视频）
python ./visualizatoin/latent_to_video.py \
    --model_config /scratch/work/guanj2/latent_wf_vae_new/examples/AE/wfvae-latent-large-16chn.json \
    --checkpoint /scratch/cs/aaltoml/users/guanjr/experiment/vae/AE_large_16chn/Latent_WFAE-lr1.00e-05-bs2-rs256-sr1-fr25/checkpoint-100000.ckpt \
    --swap_lll \
    --video_path /path/to/video/directory \
    --output_dir lll_swap_experiment \
    --fps 10 \
    --video_size 256x256 \
    --num_frames 25

"""