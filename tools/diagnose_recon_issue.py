"""
诊断脚本：检查训练和推理时的数据范围差异

运行方法：
python tools/diagnose_recon_issue.py
"""

import torch
import sys
import os
sys.path.append(".")

from causalvideovae.dataset.video_dataset import ValidVideoDataset
from causalvideovae.model import ModelRegistry

# 测试1: 检查 ValidVideoDataset 输出的数据范围
print("="*70)
print("测试1: 检查 ValidVideoDataset 的数据范围")
print("="*70)

# 假设你有一些视频文件用于测试
TEST_VIDEO_DIR = "/scratch/cs/vidgen/data/kinetics-dataset/k400/test"  # 修改为你的测试视频目录

if os.path.exists(TEST_VIDEO_DIR):
    dataset = ValidVideoDataset(
        real_video_dir=TEST_VIDEO_DIR,
        num_frames=16,
        sample_rate=1,
        resolution=256,
    )
    
    if len(dataset) > 0:
        sample = dataset[0]
        x = sample['video']  # shape: (C, T, H, W)
        
        print(f"Video shape: {x.shape}")
        print(f"Video dtype: {x.dtype}")
        print(f"Video min: {x.min().item():.4f}")
        print(f"Video max: {x.max().item():.4f}")
        print(f"Video mean: {x.mean().item():.4f}")
        
        # 检查是否在 [0,1] 范围
        if x.min() >= 0 and x.max() <= 1:
            print("✓ 数据在 [0, 1] 范围")
        elif x.min() >= -1 and x.max() <= 1:
            print("✗ 数据在 [-1, 1] 范围！这是问题所在！")
        else:
            print(f"✗ 数据范围异常: [{x.min():.4f}, {x.max():.4f}]")
        
        # 测试2: 检查模型前向传播的输入要求
        print("\n" + "="*70)
        print("测试2: 检查模型对输入数据范围的期望")
        print("="*70)
        
        print("\n情况A: 输入 [0,1] 范围的数据")
        x_01 = x.clone()
        print(f"  输入范围: [{x_01.min():.4f}, {x_01.max():.4f}]")
        
        print("\n情况B: 输入 *2-1 归一化到 [-1,1] 的数据")
        x_normalized = x * 2 - 1
        print(f"  输入范围: [{x_normalized.min():.4f}, {x_normalized.max():.4f}]")
        
        print("\n" + "="*70)
        print("结论")
        print("="*70)
        print("""
如果 ValidVideoDataset 输出 [0,1]:
  - recon_video.py 使用 x*2-1 转到 [-1,1] ✓ 正确
  - 但 eval.py 直接使用 x ✗ 错误！应该也要 x*2-1

如果 ValidVideoDataset 输出 [-1,1]:
  - recon_video.py 使用 x*2-1 ✗ 错误！变成 [-3,1]
  - eval.py 直接使用 x ✓ 正确

关键是要确认 ValidVideoDataset 的 ToTensorVideo() 到底输出什么范围！
        """)
        
    else:
        print(f"✗ 测试视频目录为空: {TEST_VIDEO_DIR}")
else:
    print(f"✗ 测试视频目录不存在: {TEST_VIDEO_DIR}")
    print("请修改脚本中的 TEST_VIDEO_DIR 变量")

# 测试3: 直接检查 ToTensorVideo 的行为
print("\n" + "="*70)
print("测试3: 直接测试 ToTensorVideo 的转换")
print("="*70)

from causalvideovae.dataset.transform import ToTensorVideo
import numpy as np

# 创建一个假的视频数据 (T, H, W, C) in [0, 255]
fake_video = np.random.randint(0, 256, (16, 64, 64, 3), dtype=np.uint8)
print(f"原始 numpy 数据范围: [{fake_video.min()}, {fake_video.max()}]")

transform = ToTensorVideo()
tensor_video = transform(fake_video)
print(f"ToTensorVideo 后的范围: [{tensor_video.min():.4f}, {tensor_video.max():.4f}]")

if tensor_video.max() <= 1.0:
    print("✓ ToTensorVideo 将 [0,255] 转换到 [0,1]")
else:
    print("✗ ToTensorVideo 保持 [0,255] 范围")

print("\n" + "="*70)
print("最终诊断")
print("="*70)
print("""
1. 检查 ToTensorVideo 的输出范围
2. 确保 recon_video.py 和 eval.py 使用相同的归一化
3. 确保模型训练和推理时使用相同的输入范围

修复建议：
- 如果模型期望 [-1,1] 输入，确保 recon_video.py 和 eval.py 都做 x*2-1
- 如果模型期望 [0,1] 输入，确保都不做额外归一化
""")

