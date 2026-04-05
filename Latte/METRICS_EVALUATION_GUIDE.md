# Latte 视频生成指标评估指南

## 支持的指标

Latte 提供了完整的视频质量评估指标，包括：

### 视频指标
- **FVD (Fréchet Video Distance)**
  - `fvd2048_16f`: 16 帧 FVD，使用 2048 个真实和生成样本
  - `fvd2048_128f`: 128 帧 FVD
  - `fvd2048_128f_subsample8f`: 128 帧采样每 8 帧计算 FVD

- **ISv (Inception Score for Videos)**
  - `isv2048_ucf`: UCF-101 上训练的 C3D 模型计算的 IS

### 图像指标
- **FID (Fréchet Inception Distance)**
  - `fid50k_full`: 使用 50k 样本的 FID
  
- **IS (Inception Score)**
  - `is50k`: 使用 50k 样本的 IS

- **KID (Kernel Inception Distance)**
  - `kid50k_full`: 使用 50k 样本的 KID

## 快速开始

### 1. 准备数据

#### 真实数据 (Real Data)
将真实视频转换为帧：
```bash
# 数据结构
/path/to/real_data/
├── video1/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
├── video2/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
└── ...
```

#### 生成数据 (Fake Data)
将生成的视频也转换为帧，结构同上：
```bash
/path/to/fake_data/
├── sample0000/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
├── sample0001/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
└── ...
```

**注意：** 所有图像需要做 **center-crop-resize** 到指定分辨率（如 256x256）

### 2. 计算 FVD16 指标

```bash
cd Latte

export CUDA_VISIBLE_DEVICES=0
python tools/calc_metrics_for_dataset.py \
    --real_data_path /path/to/real_data/images \
    --fake_data_path /path/to/fake_data/images \
    --mirror 1 \
    --gpus 1 \
    --resolution 256 \
    --metrics fvd2048_16f \
    --verbose 1 \
    --use_cache 0
```

### 3. 计算 IS (UCF-101)

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/calc_metrics_for_dataset.py \
    --real_data_path /path/to/ucf101_test/images \
    --fake_data_path /path/to/generated_videos/images \
    --mirror 0 \
    --gpus 1 \
    --resolution 256 \
    --metrics isv2048_ucf \
    --verbose 1 \
    --use_cache 0
```

### 4. 同时计算多个指标

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/calc_metrics_for_dataset.py \
    --real_data_path /path/to/real_data/images \
    --fake_data_path /path/to/fake_data/images \
    --mirror 1 \
    --gpus 1 \
    --resolution 256 \
    --metrics fvd2048_16f,isv2048_ucf,fid50k_full \
    --verbose 1 \
    --use_cache 1
```

## 参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--real_data_path` | 真实数据路径 | `/data/ucf101/test/frames` |
| `--fake_data_path` | 生成数据路径 | `./results/generated_videos` |
| `--resolution` | 图像分辨率 | `256` |

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--metrics` | `fvd2048_16f,fid50k_full` | 要计算的指标列表 |
| `--mirror` | - | 是否对真实数据镜像增强 |
| `--gpus` | `1` | 使用的 GPU 数量 |
| `--verbose` | `False` | 是否打印详细信息 |
| `--use_cache` | `True` | 是否使用统计缓存 |
| `--num_runs` | `1` | 运行次数 |

## 可用指标列表

### 视频指标
- `fvd2048_16f` - FVD (16 frames)
- `fvd2048_128f` - FVD (128 frames)
- `fvd2048_128f_subsample8f` - FVD (128 frames, subsample 8)
- `isv2048_ucf` - Inception Score for Videos (C3D-UCF101)

### 图像指标
- `fid50k_full` - Fréchet Inception Distance
- `is50k` - Inception Score (mean + std)
- `kid50k_full` - Kernel Inception Distance

## 数据预处理脚本

### 视频转帧脚本

```python
#!/usr/bin/env python3
"""
将视频转换为帧并保存
"""
import os
import cv2
from pathlib import Path
from tqdm import tqdm

def video_to_frames(video_path, output_dir, fps=None):
    """
    将视频转换为帧
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        fps: 目标帧率（None 表示保持原帧率）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps is None:
        frame_interval = 1
    else:
        frame_interval = int(original_fps / fps)
    
    frame_idx = 0
    saved_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            # Center crop and resize to 256x256
            h, w = frame.shape[:2]
            min_dim = min(h, w)
            
            # Center crop to square
            top = (h - min_dim) // 2
            left = (w - min_dim) // 2
            frame_cropped = frame[top:top+min_dim, left:left+min_dim]
            
            # Resize to 256x256
            frame_resized = cv2.resize(frame_cropped, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            
            # Save
            output_path = os.path.join(output_dir, f'{saved_idx:06d}.jpg')
            cv2.imwrite(output_path, frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_idx += 1
        
        frame_idx += 1
    
    cap.release()
    return saved_idx

def process_video_dataset(video_dir, output_dir, fps=None):
    """
    批量处理视频数据集
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    
    # 支持的视频格式
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(list(video_dir.glob(f'**/*{ext}')))
    
    print(f"Found {len(video_files)} videos")
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        # 创建对应的输出目录
        relative_path = video_path.relative_to(video_dir)
        video_name = relative_path.stem
        video_output_dir = output_dir / video_name
        
        # 转换视频为帧
        num_frames = video_to_frames(video_path, video_output_dir, fps)
        
        if num_frames == 0:
            print(f"Warning: No frames extracted from {video_path}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True, help='输入视频目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出帧目录')
    parser.add_argument('--fps', type=int, default=None, help='目标帧率（None 保持原帧率）')
    
    args = parser.parse_args()
    
    process_video_dataset(args.video_dir, args.output_dir, args.fps)
    print("Done!")
```

### 使用预处理脚本

```bash
# 处理真实数据
python video_to_frames.py \
    --video_dir /path/to/real_videos \
    --output_dir /path/to/real_data/frames \
    --fps 8

# 处理生成数据
python video_to_frames.py \
    --video_dir /path/to/generated_videos \
    --output_dir /path/to/fake_data/frames \
    --fps 8
```

## 常见问题

### Q1: 如何从 validation 结果计算指标？

如果你已经在训练中保存了验证视频（如 `validation_samples/step_0005000_sample_0000.mp4`），需要先转换为帧：

```bash
# 转换验证视频为帧
python video_to_frames.py \
    --video_dir results/experiment_dir/validation_samples \
    --output_dir results/experiment_dir/validation_frames \
    --fps 8

# 计算指标
python tools/calc_metrics_for_dataset.py \
    --real_data_path /path/to/real_test_data/frames \
    --fake_data_path results/experiment_dir/validation_frames \
    --resolution 256 \
    --metrics fvd2048_16f,isv2048_ucf \
    --verbose 1
```

### Q2: 显存不足怎么办？

减少批处理大小（在代码中硬编码）或减少评估样本数：

```python
# 修改 tools/metrics/frechet_video_distance.py
NUM_FRAMES_IN_BATCH = {128: 64, 256: 64, 512: 32, 1024: 16}  # 减小 batch size
```

### Q3: 计算速度慢怎么办？

1. **使用缓存**：`--use_cache 1`
2. **多 GPU**：`--gpus 4`
3. **减少样本数**：修改 metric 定义中的 `num_gen` 和 `max_real`

### Q4: 需要多少视频样本？

根据论文标准：
- **FVD**: 通常使用 2048 个真实样本和 2048 个生成样本
- **IS**: 通常使用 2048 或更多生成样本
- 更多样本会得到更稳定的结果，但计算时间更长

### Q5: 如何解释结果？

- **FVD**: 越低越好（Lower is better）
  - FVD < 100: 优秀
  - FVD 100-500: 良好
  - FVD > 500: 需要改进

- **IS**: 越高越好（Higher is better）
  - IS > 10: 优秀（UCF-101）
  - IS 5-10: 良好
  - IS < 5: 需要改进

## 示例输出

```bash
Calculating fvd2048_16f...
Loading real data features...
Loading generated data features...
Computing statistics...
fvd2048_16f: 245.32

Calculating isv2048_ucf...
Loading generated data features...
Computing inception score...
isv2048_ucf_mean: 12.45
isv2048_ucf_std: 0.34
```

## 参考

- 原始实现基于 [StyleGAN-V](https://github.com/universome/stylegan-v)
- FVD 实现参考 [Google Research](https://github.com/google-research/google-research/tree/master/frechet_video_distance)
- 使用 I3D 模型计算视频特征
- 使用 C3D (UCF-101) 模型计算 IS

## 相关文件

- `tools/calc_metrics_for_dataset.py` - 主评估脚本
- `tools/metrics/frechet_video_distance.py` - FVD 实现
- `tools/metrics/video_inception_score.py` - IS 实现
- `tools/metrics/metric_main.py` - 指标注册和管理
- `tools/eval_metrics.sh` - 示例脚本

