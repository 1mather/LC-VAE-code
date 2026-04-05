# ParquetVideoDataset 快速开始

## 5分钟快速上手

### 步骤 1: 创建 Parquet 文件

假设你的视频文件夹结构如下：

```
/data/videos/
├── cat/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── video3.mp4
├── dog/
│   ├── video4.mp4
│   └── video5.mp4
└── bird/
    └── video6.mp4
```

运行以下命令创建 Parquet 文件：

```bash
python tools/create_parquet_from_videos.py \
    --video_dir /data/videos \
    --output_parquet /data/videos.parquet \
    --recursive
```

这会创建一个包含所有视频路径的 parquet 文件，并自动从文件夹名称提取标签。

### 步骤 2: 使用 Dataset

```python
from causalvideovae.dataset.video_dataset import ParquetVideoDataset
from torch.utils.data import DataLoader

# 创建数据集
dataset = ParquetVideoDataset(
    parquet_path='/data/videos.parquet',
    sequence_length=16,      # 采样16帧
    resolution=256,          # 256x256分辨率
    sample_rate=4,           # 最大步长为4
    dynamic_sample=True,     # 随机步长
)

# 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=8,
)

# 使用数据
for batch in dataloader:
    video = batch['video']      # Shape: (4, 3, 16, 256, 256)
    metadata = batch['label']   # Dict with 'label', 'filename', etc.
    
    print(f"Video shape: {video.shape}")
    print(f"Label: {metadata['label']}")
    
    # 你的训练代码...
    break
```

### 步骤 3: 验证数据（可选）

使用示例脚本测试数据集：

```bash
python examples/parquet_dataset_example.py \
    --parquet_path /data/videos.parquet \
    --sequence_length 16 \
    --resolution 256 \
    --batch_size 2 \
    --num_workers 4
```

## 高级用法

### 添加视频验证

如果担心某些视频文件可能损坏，可以在创建 Parquet 时添加验证：

```bash
python tools/create_parquet_from_videos.py \
    --video_dir /data/videos \
    --output_parquet /data/videos.parquet \
    --recursive \
    --verify  # 验证每个视频是否可以打开
```

这会添加额外的列：`num_frames`、`fps`、`valid`

### 自定义列名

如果你的 Parquet 文件使用不同的列名：

```python
dataset = ParquetVideoDataset(
    parquet_path='/data/custom.parquet',
    video_column='path_to_video',  # 自定义列名
    sequence_length=16,
    resolution=256,
)
```

### 固定采样率

如果不想使用动态采样：

```python
dataset = ParquetVideoDataset(
    parquet_path='/data/videos.parquet',
    sequence_length=16,
    sample_rate=2,           # 固定步长为2
    dynamic_sample=False,    # 禁用动态采样
    resolution=256,
)
```

### 从已有的 DataFrame 创建

如果已经有包含视频路径的 DataFrame：

```python
import pandas as pd

# 从CSV或其他来源加载
df = pd.read_csv('video_list.csv')

# 确保有视频路径列
# df should have columns like: ['video_path', 'label', 'duration', ...]

# 保存为 parquet
df.to_parquet('videos.parquet', index=False)

# 使用 Dataset
dataset = ParquetVideoDataset(
    parquet_path='videos.parquet',
    video_column='video_path',
    sequence_length=16,
    resolution=256,
)
```

## 常见问题

### Q1: 视频太短怎么办？

A: Dataset 会自动调整采样步长。如果视频帧数少于 `sequence_length * sample_rate`，会使用更小的步长。

### Q2: 如果某些视频加载失败？

A: Dataset 会自动跳过失败的视频并随机选择另一个样本，训练不会中断。

### Q3: 如何处理超大数据集？

A: 考虑：
- 将 Parquet 文件分片
- 使用更多的 DataLoader workers
- 预先过滤无效视频
- 使用 SSD 存储视频

### Q4: 支持哪些视频格式？

A: 支持所有 Decord 支持的格式，包括：mp4, avi, mov, mkv, webm, flv, wmv 等。

### Q5: 输出视频的值范围是什么？

A: 输出的视频 tensor 已经标准化到 [-1, 1] 范围。

## 完整训练示例

```python
import torch
from torch.utils.data import DataLoader
from causalvideovae.dataset.video_dataset import ParquetVideoDataset

def train():
    # 创建数据集
    train_dataset = ParquetVideoDataset(
        parquet_path='/data/train_videos.parquet',
        sequence_length=16,
        resolution=256,
        sample_rate=4,
        dynamic_sample=True,
        train=True,
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    
    # 训练循环
    model = YourModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            video = batch['video'].cuda()  # (B, C, T, H, W)
            
            # 前向传播
            output = model(video)
            loss = compute_loss(output)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

if __name__ == '__main__':
    train()
```

## 性能建议

1. **num_workers**: 根据CPU核心数设置，通常 4-8 个 workers 效果较好
2. **pin_memory**: 在使用 GPU 时设置为 True
3. **prefetch_factor**: 可以增加 DataLoader 的 prefetch_factor 参数以提高吞吐量
4. **分辨率**: 较小的分辨率可以显著提高加载速度

```python
dataloader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=2,  # 每个 worker 预取2个batch
    persistent_workers=True,  # 保持 workers 存活
)
```

## 下一步

- 查看完整文档: `docs/ParquetVideoDataset_README.md`
- 查看代码实现: `causalvideovae/dataset/video_dataset.py`
- 运行示例: `examples/parquet_dataset_example.py`

