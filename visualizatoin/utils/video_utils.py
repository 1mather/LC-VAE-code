"""
Video processing utilities for visualization
"""

import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


def create_statistics_comparison(lll_tensor, hhh_tensor, output_dir, fps=10):
    """
    Create statistical comparison between LLL and HHH components
    Now expects 2D tensors (T, H, W) - combined channels
    """
    T, H, W = lll_tensor.shape
    
    # Calculate statistics for each frame
    lll_stats = []
    hhh_stats = []
    
    for t in range(T):
        lll_frame = lll_tensor[t]
        hhh_frame = hhh_tensor[t]
        
        # Calculate statistics for this frame
        lll_mean = torch.mean(lll_frame).item()
        lll_std = torch.std(lll_frame).item()
        lll_energy = torch.sum(lll_frame ** 2).item()
        
        hhh_mean = torch.mean(hhh_frame).item()
        hhh_std = torch.std(hhh_frame).item()
        hhh_energy = torch.sum(hhh_frame ** 2).item()
        
        lll_stats.append({
            'mean': lll_mean,
            'std': lll_std,
            'energy': lll_energy
        })
        
        hhh_stats.append({
            'mean': hhh_mean,
            'std': hhh_std,
            'energy': hhh_energy
        })
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    time_frames = list(range(T))
    
    # Mean comparison
    lll_means = [s['mean'] for s in lll_stats]
    hhh_means = [s['mean'] for s in hhh_stats]
    
    axes[0, 0].plot(time_frames, lll_means, 'b-', label='LLL', linewidth=2)
    axes[0, 0].plot(time_frames, hhh_means, 'r-', label='HHH', linewidth=2)
    axes[0, 0].set_xlabel('Time Frame')
    axes[0, 0].set_ylabel('Mean Value')
    axes[0, 0].set_title('Mean Value Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Standard deviation comparison
    lll_stds = [s['std'] for s in lll_stats]
    hhh_stds = [s['std'] for s in hhh_stats]
    
    axes[0, 1].plot(time_frames, lll_stds, 'b-', label='LLL', linewidth=2)
    axes[0, 1].plot(time_frames, hhh_stds, 'r-', label='HHH', linewidth=2)
    axes[0, 1].set_xlabel('Time Frame')
    axes[0, 1].set_ylabel('Standard Deviation')
    axes[0, 1].set_title('Standard Deviation Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Energy comparison
    lll_energies = [s['energy'] for s in lll_stats]
    hhh_energies = [s['energy'] for s in hhh_stats]
    
    axes[1, 0].plot(time_frames, lll_energies, 'b-', label='LLL', linewidth=2)
    axes[1, 0].plot(time_frames, hhh_energies, 'r-', label='HHH', linewidth=2)
    axes[1, 0].set_xlabel('Time Frame')
    axes[1, 0].set_ylabel('Energy')
    axes[1, 0].set_title('Energy Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Energy ratio
    energy_ratios = [lll_energies[t] / hhh_energies[t] if hhh_energies[t] > 0 else float('inf') 
                    for t in range(T)]
    
    axes[1, 1].plot(time_frames, energy_ratios, 'g-', linewidth=2)
    axes[1, 1].set_xlabel('Time Frame')
    axes[1, 1].set_ylabel('LLL/HHH Energy Ratio')
    axes[1, 1].set_title('Energy Ratio (LLL/HHH)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('LLL vs HHH Statistical Comparison', fontsize=16)
    plt.tight_layout()
    
    # Save the comparison plot
    comparison_path = os.path.join(output_dir, 'lll_hhh_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved LLL vs HHH comparison plot: {comparison_path}")
    
    # Save detailed statistics
    stats_path = os.path.join(output_dir, 'lll_hhh_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("LLL vs HHH Statistical Comparison\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Frame-wise Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write("Frame | LLL Mean | LLL Std | LLL Energy | HHH Mean | HHH Std | HHH Energy | Ratio\n")
        f.write("-" * 80 + "\n")
        
        for t in range(T):
            f.write(f"{t:5d} | {lll_means[t]:8.4f} | {lll_stds[t]:7.4f} | {lll_energies[t]:10.2f} | "
                   f"{hhh_means[t]:8.4f} | {hhh_stds[t]:7.4f} | {hhh_energies[t]:10.2f} | {energy_ratios[t]:5.2f}\n")
        
        f.write(f"\nSummary Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"LLL - Mean: {np.mean(lll_means):.4f}, Std: {np.mean(lll_stds):.4f}, Energy: {np.mean(lll_energies):.2f}\n")
        f.write(f"HHH - Mean: {np.mean(hhh_means):.4f}, Std: {np.mean(hhh_stds):.4f}, Energy: {np.mean(hhh_energies):.2f}\n")
        f.write(f"Average Energy Ratio (LLL/HHH): {np.mean(energy_ratios):.2f}\n")
    
    print(f"Saved detailed statistics: {stats_path}")


def create_energy_distribution_video(tensor, output_path, fps=10, title="Energy Distribution"):
    """
    Create a video showing the energy distribution of a tensor over time
    tensor: (T, H, W) tensor
    """
    T, H, W = tensor.shape
    
    # Calculate energy for each frame
    energies = []
    for t in range(T):
        frame = tensor[t]
        energy = torch.sum(frame ** 2).item()
        energies.append(energy)
    
    # Normalize energies for visualization
    max_energy = max(energies)
    min_energy = min(energies)
    energy_range = max_energy - min_energy
    
    if energy_range > 0:
        normalized_energies = [(e - min_energy) / energy_range for e in energies]
    else:
        normalized_energies = [0.5] * T
    
    # Create video frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    for t in range(T):
        # Create a frame with the energy value as text
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Add energy information
        energy_text = f"Frame {t}: Energy = {energies[t]:.2f}"
        cv2.putText(frame, energy_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add normalized energy bar
        bar_width = int(normalized_energies[t] * W * 0.8)
        cv2.rectangle(frame, (10, 50), (10 + bar_width, 80), (0, 255, 0), -1)
        
        out.write(frame)
    
    out.release()
    plt.close()
    print(f"Saved energy distribution video: {output_path}")
