"""
Pixel domain analysis utilities for wavelet subband analysis
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt


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
