#!/usr/bin/env python3
"""
从通道统计数据生成固定mask
用于统一压缩策略：所有样本使用相同的通道mask
"""

import json
import argparse
import numpy as np
import torch
import os
from pathlib import Path

def generate_fixed_mask_from_stats(mask_stats_file, handcraft, keep_ratio=0.2, output_file=None):
    """
    从统计数据生成固定的通道mask
    
    Args:
        mask_stats_file: 统计数据JSON文件路径
        keep_ratio: 保留通道的比例（0.2表示保留20%）
        output_file: 输出文件路径（.pth格式）
    
    Returns:
        fixed_mask: torch.Tensor of shape (num_channels,) with 0s and 1s
    """
    # 加载统计数据
    if mask_stats_file is not None and not handcraft:
        with open(mask_stats_file, 'r') as f:
            data = json.load(f)
        
        freqs = np.array(data['avg_selection_frequency'])
        num_channels = len(freqs)
        num_keep = int(num_channels * keep_ratio)
        
        print(f"\n{'='*80}")
        print(f"生成固定通道Mask")
        print(f"{'='*80}")
        print(f"总通道数: {num_channels}")
        print(f"保留比例: {keep_ratio:.1%}")
        print(f"保留通道数: {num_keep}")
        print(f"丢弃通道数: {num_channels - num_keep}")
        
        # 根据选择频率排序，选择top-k
        top_indices = np.argsort(freqs)[-num_keep:]  # 选择频率最高的
        top_indices = np.sort(top_indices)  # 排序以便查看
        
        # 创建固定mask（binary）
        fixed_mask = np.zeros(num_channels, dtype=np.float32)
        fixed_mask[top_indices] = 1.0
            # 分析保留的通道
        print(f"\n保留通道的统计:")
        selected_freqs = freqs[top_indices]
        print(f"  - 平均选择频率: {selected_freqs.mean():.4f}")
        print(f"  - 最小选择频率: {selected_freqs.min():.4f}")
        print(f"  - 最大选择频率: {selected_freqs.max():.4f}")
        
        # 丢弃通道的统计
        discarded_indices = np.setdiff1d(np.arange(num_channels), top_indices)
        discarded_freqs = freqs[discarded_indices]
        print(f"\n丢弃通道的统计:")
        print(f"  - 平均选择频率: {discarded_freqs.mean():.4f}")
        print(f"  - 最小选择频率: {discarded_freqs.min():.4f}")
        print(f"  - 最大选择频率: {discarded_freqs.max():.4f}")
        
        # 按64通道块分析保留情况
        print(f"\n按64通道块的保留情况:")
        for i in range(num_channels // 64):
            block_start = i * 64
            block_end = (i + 1) * 64
            block_mask = fixed_mask[block_start:block_end]
            kept = int(block_mask.sum())
            print(f"  Block {i} (ch {block_start:3d}-{block_end-1:3d}): "
                f"保留 {kept:2d}/{64} ({kept/64*100:5.1f}%)")
        
        # 显示保留的通道索引（前50个）
        print(f"\n保留的通道索引 (前50个):")
        print(f"  {top_indices[:50].tolist()}")
        if len(top_indices) > 50:
            print(f"  ... 还有 {len(top_indices)-50} 个")
    else:
        if keep_ratio == 0.2:
            block_0=np.zeros(64, dtype=np.float32)
            for i in range(40):
                block_0[i] = 1.0
                
            block_1=np.zeros(64, dtype=np.float32)
            for i in range(2):
                block_1[i] = 1.0

            block_2=np.zeros(64, dtype=np.float32)
            for i in range(20):
                block_2[i] = 1.0

            block_3=np.zeros(64, dtype=np.float32)
            for i in range(2):
                block_3[i] = 1.0

            block_4=np.zeros(64, dtype=np.float32)
            for i in range(20):
                block_4[i] = 1.0

            block_5=np.zeros(64, dtype=np.float32)
            for i in range(2):
                block_5[i] = 1.0

            block_6=np.zeros(64, dtype=np.float32)
            for i in range(20):
                block_6[i] = 1.0

            block_7=np.zeros(64, dtype=np.float32)
            for i in range(2):
                block_7[i] = 1.0


        # elif keep_ratio == 1.0:
        #     block_0=np.zeros(128, dtype=np.float32)
        #     for i in range(128):
        #         block_0[i] = 1.0
                
        #     block_1=np.zeros(128, dtype=np.float32)
        #     for i in range(30):
        #         block_1[i] = 1.0

        #     block_2=np.zeros(128, dtype=np.float32)
        #     for i in range(80):
        #         block_2[i] = 1.0

        #     block_3=np.zeros(128, dtype=np.float32)
        #     for i in range(30):
        #         block_3[i] = 1.0

        #     block_4=np.zeros(128, dtype=np.float32)
        #     for i in range(80):
        #         block_4[i] = 1.0

        #     block_5=np.zeros(128, dtype=np.float32)
        #     for i in range(30):
        #         block_5[i] = 1.0

        #     block_6=np.zeros(128, dtype=np.float32)
        #     for i in range(80):
        #         block_6[i] = 1.0

        #     block_7=np.zeros(128, dtype=np.float32)
        #     for i in range(30):
        #         block_7[i] = 1.0

        # elif keep_ratio == 0.5:
        #     block_0=np.zeros(64, dtype=np.float32)
        #     for i in range(64):
        #         block_0[i] = 1.0
                
        #     block_1=np.zeros(64, dtype=np.float32)
        #     for i in range(15):
        #         block_1[i] = 1.0

        #     block_2=np.zeros(64, dtype=np.float32)
        #     for i in range(40):
        #         block_2[i] = 1.0

        #     block_3=np.zeros(64, dtype=np.float32)
        #     for i in range(15):
        #         block_3[i] = 1.0

        #     block_4=np.zeros(64, dtype=np.float32)
        #     for i in range(40):
        #         block_4[i] = 1.0

        #     block_5=np.zeros(64, dtype=np.float32)
        #     for i in range(15):
        #         block_5[i] = 1.0

        #     block_6=np.zeros(64, dtype=np.float32)
        #     for i in range(40):
        #         block_6[i] = 1.0

        #     block_7=np.zeros(64, dtype=np.float32)
        #     for i in range(15):
        #         block_7[i] = 1.0

        # elif keep_ratio == 0.5:
        #     block_0=np.zeros(64, dtype=np.float32) #LLL
        #     for i in range(64):
        #         block_0[i] = 1.0
                
        #     block_1=np.zeros(64, dtype=np.float32)#LLH
        #     for i in range(60):
        #         block_1[i] = 1.0

        #     block_2=np.zeros(64, dtype=np.float32)#LHL
        #     for i in range(60):
        #         block_2[i] = 1.0

        #     block_3=np.zeros(64, dtype=np.float32)#LHH
        #     for i in range(4):
        #         block_3[i] = 1.0

        #     block_4=np.zeros(64, dtype=np.float32)#HLL
        #     for i in range(60):
        #         block_4[i] = 1.0

        #     block_5=np.zeros(64, dtype=np.float32)#HLH
        #     for i in range(4):
        #         block_5[i] = 1.0

        #     block_6=np.zeros(64, dtype=np.float32)#HHL
        #     for i in range(4):
        #         block_6[i] = 1.0

        #     block_7=np.zeros(64, dtype=np.float32)#HHH
        #     for i in range(0):
        #         block_7[i] = 1.0


        elif keep_ratio == 0.5:
            block_0=np.zeros(32, dtype=np.float32)
            for i in range(32):
                block_0[i] = 1.0
                
            block_1=np.zeros(32, dtype=np.float32)
            for i in range(30):
                block_1[i] = 1.0

            block_2=np.zeros(32, dtype=np.float32)
            for i in range(30):
                block_2[i] = 1.0

            block_3=np.zeros(32, dtype=np.float32)
            for i in range(2):
                block_3[i] = 1.0

                
            block_4=np.zeros(32, dtype=np.float32)
            for i in range(30):
                block_4[i] = 1.0

            block_5=np.zeros(32, dtype=np.float32)
            for i in range(2):
                block_5[i] = 1.0

            block_6=np.zeros(32, dtype=np.float32)
            for i in range(2):
                block_6[i] = 1.0

            block_7=np.zeros(32, dtype=np.float32)
            for i in range(0):
                block_7[i] = 1.0

        elif keep_ratio == 1.0:
            block_0=np.zeros(128, dtype=np.float32) #LLL
            for i in range(124):
                block_0[i] = 1.0
                
            block_1=np.zeros(128, dtype=np.float32)#LLH
            for i in range(124):
                block_1[i] = 1.0

            block_2=np.zeros(128, dtype=np.float32)#LHL
            for i in range(124):
                block_2[i] = 1.0

            block_3=np.zeros(128, dtype=np.float32)#LHH
            for i in range(4):
                block_3[i] = 1.0

            block_4=np.zeros(128, dtype=np.float32)#HLL
            for i in range(124):
                block_4[i] = 1.0

            block_5=np.zeros(128, dtype=np.float32)#HLH
            for i in range(4):
                block_5[i] = 1.0

            block_6=np.zeros(128, dtype=np.float32)#HHL
            for i in range(4):
                block_6[i] = 1.0

            block_7=np.zeros(128, dtype=np.float32)#HHH
            for i in range(4):
                block_7[i] = 1.0

        fixed_mask = np.concatenate([block_0, block_1, block_2, block_3, block_4, block_5, block_6, block_7], axis=0)
        
        # 手工crafted mask的元数据
        num_channels = len(fixed_mask)
        num_keep = int(fixed_mask.sum())
        top_indices = np.where(fixed_mask > 0)[0]

    # 转换为torch tensor并保存
    fixed_mask_tensor = torch.from_numpy(fixed_mask)
    if output_file:
        # 构造输出路径
        if mask_stats_file is not None and not handcraft:
            output_path = os.path.join(output_file, 'fixed_mask_from_stats.pth')
        else:
            output_path = os.path.join(output_file, 'fixed_mask_from_handcraft.pth')
        
        # 确保目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:  # 如果不是空字符串（当前目录）
            os.makedirs(output_dir, exist_ok=True)
        
        # 保存mask和元数据
        save_data = {
            'fixed_mask': fixed_mask_tensor,
            'keep_ratio': keep_ratio,
            'num_channels': num_channels,
            'num_keep': num_keep,
            'selected_indices': torch.from_numpy(top_indices),
            'source_stats_file': str(mask_stats_file) if mask_stats_file else 'handcrafted',
        }
        
        torch.save(save_data, output_path)
        print(f"\n✓ 固定mask已保存到: {output_path}")
        
        # 同时保存一个JSON版本便于查看
        # 构造JSON路径（替换.pth为.json）
        json_path = output_path.replace('.pth', '.json')
        json_data = {
            'fixed_mask': fixed_mask.tolist(),
            'keep_ratio': keep_ratio,
            'num_channels': num_channels,
            'num_keep': num_keep,
            'selected_indices': top_indices.tolist(),
            'source_stats_file': str(mask_stats_file) if mask_stats_file else 'handcrafted',
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"✓ 可读版本已保存到: {json_path}")
    
    return fixed_mask_tensor, top_indices

def main():
    parser = argparse.ArgumentParser(
        description='从通道统计数据生成固定的通道mask'
    )
    parser.add_argument(
        '--handcraft', 
        action='store_true',
        default=True,
        help='是否手工crafted mask'
    )
    parser.add_argument(
        '--stats-file', 
        type=str, 
        default='/scratch/work/guanj2/latent_wf_vae_new/causalvideovae/model/channel_name/${EXP_NAME}-lr1.00e-05-bs1-rs256-sr1-fr25-cons1.0-modetv_l1-20251020_172139/channel_masks/step_80001_hard_masks.json',
        help='通道统计JSON文件路径'
    )
    parser.add_argument(
        '--keep-ratio', 
        type=float, 
        default=1.0,
        help='保留通道的比例（默认0.2，即20%%）'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='/scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/causalvideovae/model/channel_name/keep_ratio_0.5_32_channel_50%/channel_masks',
        help='输出文件路径（.pth格式）'
    )
    
    args = parser.parse_args()
    
    # 如果没有指定输出路径，自动生成
    if args.output is None:
        stats_path = Path(args.stats_file)
        output_dir = stats_path.parent
        args.output = output_dir / f'fixed_mask_keep{args.keep_ratio:.2f}.pth'
    
    # 生成固定mask
    fixed_mask, selected_indices = generate_fixed_mask_from_stats(
        args.stats_file,
        args.handcraft,
        keep_ratio=args.keep_ratio,
        output_file=args.output
    )
    
    print(f"\n{'='*80}")
    print("完成！")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

