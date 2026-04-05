"""
Diagnosis utilities for latent space and wavelet analysis
"""

import os
import torch
import numpy as np


def diagnose_latent_space_issues(z1, z2, l1_coeffs1, l1_coeffs2, output_dir):
    """
    诊断潜在空间问题
    """
    print("=== Latent Space Diagnosis ===")
    
    # 分析原始潜在空间
    print("\n1. Original Latent Space Analysis:")
    z1_analysis = analyze_tensor_channels(z1[0], "Video 1 Original")
    z2_analysis = analyze_tensor_channels(z2[0], "Video 2 Original")
    
    # 分析LLL系数
    print("\n2. LLL Coefficients Analysis:")
    lll1_analysis = analyze_tensor_channels(l1_coeffs1[0], "Video 1 LLL")
    lll2_analysis = analyze_tensor_channels(l1_coeffs2[0], "Video 2 LLL")
    
    # 检查能量保持
    print("\n3. Energy Preservation Analysis:")
    z1_total_energy = sum(z1_analysis['energies'])
    z2_total_energy = sum(z2_analysis['energies'])
    lll1_total_energy = sum(lll1_analysis['energies'])
    lll2_total_energy = sum(lll2_analysis['energies'])
    
    preservation1 = lll1_total_energy / z1_total_energy if z1_total_energy > 0 else 0
    preservation2 = lll2_total_energy / z2_total_energy if z2_total_energy > 0 else 0
    
    print(f"Video 1 energy preservation: {preservation1:.4f} ({preservation1*100:.2f}%)")
    print(f"Video 2 energy preservation: {preservation2:.4f} ({preservation2*100:.2f}%)")
    
    # 检查通道利用率
    print("\n4. Channel Utilization Analysis:")
    active_channels_z1 = sum(1 for e in z1_analysis['energies'] if e > 0.01 * max(z1_analysis['energies']))
    active_channels_z2 = sum(1 for e in z2_analysis['energies'] if e > 0.01 * max(z2_analysis['energies']))
    active_channels_lll1 = sum(1 for e in lll1_analysis['energies'] if e > 0.01 * max(lll1_analysis['energies']))
    active_channels_lll2 = sum(1 for e in lll2_analysis['energies'] if e > 0.01 * max(lll2_analysis['energies']))
    
    print(f"Video 1 - Original: {active_channels_z1}/{len(z1_analysis['energies'])} channels active")
    print(f"Video 1 - LLL: {active_channels_lll1}/{len(lll1_analysis['energies'])} channels active")
    print(f"Video 2 - Original: {active_channels_z2}/{len(z2_analysis['energies'])} channels active")
    print(f"Video 2 - LLL: {active_channels_lll2}/{len(lll2_analysis['energies'])} channels active")
    
    # 保存诊断报告
    diagnosis_file = os.path.join(output_dir, 'latent_space_diagnosis.txt')
    with open(diagnosis_file, 'w') as f:
        f.write("Latent Space Diagnosis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. Original Latent Space Analysis:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Video 1: {z1_analysis}\n")
        f.write(f"Video 2: {z2_analysis}\n\n")
        
        f.write("2. LLL Coefficients Analysis:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Video 1: {lll1_analysis}\n")
        f.write(f"Video 2: {lll2_analysis}\n\n")
        
        f.write("3. Energy Preservation:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Video 1: {preservation1:.4f} ({preservation1*100:.2f}%)\n")
        f.write(f"Video 2: {preservation2:.4f} ({preservation2*100:.2f}%)\n\n")
        
        f.write("4. Channel Utilization:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Video 1 - Original: {active_channels_z1}/{len(z1_analysis['energies'])} channels\n")
        f.write(f"Video 1 - LLL: {active_channels_lll1}/{len(lll1_analysis['energies'])} channels\n")
        f.write(f"Video 2 - Original: {active_channels_z2}/{len(z2_analysis['energies'])} channels\n")
        f.write(f"Video 2 - LLL: {active_channels_lll2}/{len(lll2_analysis['energies'])} channels\n\n")
        
        f.write("5. Potential Issues:\n")
        f.write("-" * 30 + "\n")
        if preservation1 < 0.1 or preservation2 < 0.1:
            f.write("⚠️  Low energy preservation in LLL coefficients\n")
        if active_channels_lll1 < 2 or active_channels_lll2 < 2:
            f.write("⚠️  Very few active channels in LLL coefficients\n")
        if abs(preservation1 - preservation2) > 0.2:
            f.write("⚠️  Inconsistent energy preservation between videos\n")
        
        f.write("\n6. Recommendations:\n")
        f.write("-" * 30 + "\n")
        f.write("- Check if the model is properly trained\n")
        f.write("- Consider adding channel-wise regularization\n")
        f.write("- Verify if this is expected behavior for the specific model architecture\n")
        f.write("- Check if other subbands (LLH, LHL, LHH) have similar issues\n")
    
    print(f"Saved diagnosis report to: {diagnosis_file}")


def analyze_tensor_channels(tensor, name):
    """分析张量的通道分布"""
    C = tensor.shape[0]
    analysis = {}
    
    for c in range(C):
        channel = tensor[c]
        energy = torch.sum(channel ** 2).item()
        mean_val = torch.mean(channel).item()
        std_val = torch.std(channel).item()
        zero_ratio = (torch.abs(channel) < 1e-6).float().mean().item()
        
        analysis[f'channel_{c}'] = {
            'energy': energy,
            'mean': mean_val,
            'std': std_val,
            'zero_ratio': zero_ratio
        }
    
    # 计算总体统计
    energies = [analysis[f'channel_{c}']['energy'] for c in range(C)]
    means = [analysis[f'channel_{c}']['mean'] for c in range(C)]
    stds = [analysis[f'channel_{c}']['std'] for c in range(C)]
    zero_ratios = [analysis[f'channel_{c}']['zero_ratio'] for c in range(C)]
    
    analysis['energies'] = energies
    analysis['means'] = means
    analysis['stds'] = stds
    analysis['zero_ratios'] = zero_ratios
    analysis['total_energy'] = sum(energies)
    analysis['mean_energy'] = np.mean(energies)
    analysis['std_energy'] = np.std(energies)
    
    print(f"{name}:")
    print(f"  Total energy: {analysis['total_energy']:.6f}")
    print(f"  Mean energy: {analysis['mean_energy']:.6f}")
    print(f"  Std energy: {analysis['std_energy']:.6f}")
    print(f"  Active channels: {sum(1 for e in energies if e > 0.01 * max(energies))}/{C}")
    
    return analysis
