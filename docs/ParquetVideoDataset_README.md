# ParquetVideoDataset 使用指南

## 概述

`ParquetVideoDataset` 是一个从 Parquet 文件读取视频路径并进行随机帧采样的 PyTorch Dataset 类。

## 特性

- ✅ 从 Parquet 文件读取视频路径
- ✅ 随机帧采样（支持动态采样率）
- ✅ 自动处理视频加载错误（递归重试）
- ✅ 支持视频转换和标准化
- ✅ 支持额外的元数据列

## Parquet 文件格式

Parquet 文件至少需要包含一列视频路径，例如：

| video_path | label | duration | fps |
|------------|-------|----------|-----|
| /path/to/video1.mp4 | cat | 10.5 | 30 |
| /path/to/video2.mp4 | dog | 15.2 | 25 |
| /path/to/video3.mp4 | bird | 8.3 | 30 |

## 使用方法

### 基本用法

```python
from causalvideovae.dataset.video_dataset import ParquetVideoDataset
from torch.utils.data import DataLoader

# 创建数据集
dataset = ParquetVideoDataset(
    parquet_path='/path/to/videos.parquet',
    sequence_length=16,          # 每个视频采样16帧
    resolution=256,              # 分辨率调整为256x256
    sample_rate=4,               # 最大采样步长为4
    dynamic_sample=True,         # 启用动态采样（随机步长）
    video_column='video_path',   # Parquet中包含视频路径的列名
    train=True,
)

# 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
)

# 训练循环
for batch in dataloader:
    video = batch['video']      # Shape: (B, C, T, H, W)
    metadata = batch['label']   # Dict containing metadata from parquet
    
    # 你的训练代码...
```

### 参数说明

- **parquet_path** (str): Parquet 文件路径
- **sequence_length** (int): 从每个视频采样的帧数
- **resolution** (int): 目标分辨率（正方形）
- **sample_rate** (int): 帧采样步长（最大值）
  - 如果 `sample_rate=4`, `sequence_length=16`，则需要覆盖 16×4=64 帧
- **dynamic_sample** (bool): 是否启用动态采样
  - True: 每次随机选择 1 到 `sample_rate` 之间的步长
  - False: 固定使用 `sample_rate` 作为步长
- **video_column** (str): Parquet 中包含视频路径的列名（默认: 'video_path'）
- **train** (bool): 训练模式标志

### 创建 Parquet 文件

```python
import pandas as pd
from glob import glob

# 方法1: 从视频文件夹创建
video_paths = glob('/path/to/videos/**/*.mp4', recursive=True)
df = pd.DataFrame({'video_path': video_paths})
df.to_parquet('videos.parquet')

# 方法2: 包含额外元数据
data = {
    'video_path': ['/path/to/video1.mp4', '/path/to/video2.mp4'],
    'label': ['cat', 'dog'],
    'duration': [10.5, 15.2],
    'fps': [30, 25],
}
df = pd.DataFrame(data)
df.to_parquet('videos_with_metadata.parquet')
```

## 数据处理流程

1. **读取视频**: 使用 Decord 高效读取视频
2. **随机采样步长**: 如果启用 `dynamic_sample`，随机选择采样步长
3. **时间裁剪**: 随机选择视频的一个时间段
4. **帧采样**: 在选定的时间段内均匀采样 `sequence_length` 帧
5. **空间变换**:
   - 转换为 Tensor
   - Resize 到目标分辨率
   - Center Crop
   - 标准化到 [-1, 1]

## 错误处理

如果加载视频时出错（文件损坏、路径错误等），Dataset 会：
1. 打印错误信息
2. 递归调用 `__getitem__` 随机选择另一个样本
3. 确保训练不会因单个样本错误而中断

## 输出格式

```python
batch = {
    'video': torch.Tensor,  # Shape: (B, C, T, H, W), Range: [-1, 1]
    'label': dict,          # 包含 Parquet 中的所有额外列
}
```

## 示例脚本

运行示例脚本测试数据集：

```bash
python examples/parquet_dataset_example.py \
    --parquet_path /path/to/videos.parquet \
    --sequence_length 16 \
    --resolution 256 \
    --sample_rate 4 \
    --batch_size 2 \
    --num_workers 4 \
    --dynamic_sample
```

## 注意事项

1. **视频长度**: 如果视频太短（帧数少于 `sequence_length * sample_rate`），会自动调整采样步长
2. **性能**: 建议使用多个 workers (`num_workers > 0`) 提高数据加载速度
3. **内存**: 大分辨率和长序列会占用较多内存，请根据 GPU 显存调整参数
4. **Parquet 缓存**: Pandas 会将整个 Parquet 文件加载到内存，适合中等规模数据集

## 与现有数据集的对比

| Dataset | 数据源 | 采样方式 | 错误处理 |
|---------|--------|---------|---------|
| `TrainVideoDataset` | 文件夹遍历 | 随机动态采样 | ✅ 递归重试 |
| `ValidVideoDataset` | 文件夹遍历 | 固定起始点 | ⚠️ 返回索引0 |
| `ParquetVideoDataset` | Parquet文件 | 随机动态采样 | ✅ 递归重试 |

## 性能优化建议

1. **预处理**: 如果数据集很大，考虑预先验证视频路径的有效性
2. **分片**: 对于超大数据集，可以将 Parquet 文件分片
3. **SSD**: 将视频文件存储在 SSD 上以提高 I/O 性能
4. **Workers**: 根据 CPU 核心数调整 `num_workers` 参数

