"""
Statistics analysis utilities for wavelet subband analysis
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt


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
        f.write("-" * 30 + "\n")
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
