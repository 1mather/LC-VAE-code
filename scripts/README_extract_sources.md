# 提取Validation源视频工具

## 功能说明

这个脚本可以快速提取在validation时会被记录到wandb的源视频，**不需要加载模型**，运行速度非常快。

它会：
1. 使用与 `train_ddp.py` 完全相同的dataloader配置
2. 使用相同的seed和采样顺序
3. 保存会被记录到wandb的源视频及其帧
4. 生成详细的追踪信息

## 使用方法

### 单进程模式（推荐用于快速测试）

```bash
python scripts/extract_validation_sources.py \
    --eval_video_path /path/to/validation/videos \
    --seed 1234 \
    --eval_num_frames 17 \
    --eval_resolution 256 \
    --eval_sample_rate 1 \
    --eval_batch_size 8 \
    --eval_subset_size 100 \
    --eval_num_video_log 2 \
    --output_dir ./validation_sources
```

### DDP模式（与训练时完全一致）

如果你想确保100%复现训练时的采样顺序，使用DDP模式：

```bash
# 使用2个GPU
torchrun --nproc_per_node=2 scripts/extract_validation_sources.py \
    --ddp \
    --eval_video_path /path/to/validation/videos \
    --seed 1234 \
    --eval_num_frames 17 \
    --eval_resolution 256 \
    --eval_sample_rate 1 \
    --eval_batch_size 8 \
    --eval_subset_size 100 \
    --eval_num_video_log 2 \
    --output_dir ./validation_sources
```

## 参数说明

**重要：所有参数应该与你的训练脚本中使用的参数完全一致！**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--seed` | 随机种子（必须与训练一致） | 1234 |
| `--eval_video_path` | 验证视频目录 | 必填 |
| `--eval_num_frames` | 每个视频的帧数 | 17 |
| `--eval_resolution` | 分辨率 | 256 |
| `--eval_sample_rate` | 采样率 | 1 |
| `--eval_batch_size` | batch大小 | 8 |
| `--eval_subset_size` | 使用的视频子集大小 | 100 |
| `--eval_num_video_log` | 记录的视频数量 | 2 |
| `--output_dir` | 输出目录 | ./validation_sources |
| `--ddp` | 使用DDP模式 | False |

## 输出结果

脚本会在输出目录创建以下文件：

```
validation_sources/
├── video_source_tracking.json    # 详细的追踪信息
├── frames/                        # 提取的视频帧
│   ├── video_000_xxx_frame_000.png
│   ├── video_000_xxx_frame_001.png
│   └── ...
└── source_videos/                 # 源视频文件的副本
    ├── video_000_original_name.mp4
    └── video_001_original_name.mp4
```

### video_source_tracking.json 示例

```json
{
  "args": {
    "seed": 1234,
    "eval_batch_size": 8,
    ...
  },
  "sources": [
    {
      "video_log_idx": 0,
      "batch_idx": 0,
      "index_in_batch": 0,
      "file_name": "video_123.mp4",
      "shape": [3, 17, 256, 256],
      "dtype": "torch.float32",
      "value_range": [-1.0, 1.0]
    },
    {
      "video_log_idx": 1,
      "batch_idx": 0,
      "index_in_batch": 1,
      "file_name": "video_456.mp4",
      ...
    }
  ],
  "total_saved": 2
}
```

## 使用场景

### 场景1：找到重建视频对应的源视频

如果你有一个重建视频，想知道它是从哪个源视频生成的：

```bash
# 1. 运行提取脚本
python scripts/extract_validation_sources.py \
    --eval_video_path /your/val/path \
    --seed 1234 \
    --output_dir ./sources

# 2. 查看 sources/video_source_tracking.json
# 找到 video_log_idx 对应的 file_name

# 3. 对比源视频和重建视频
# 源视频在: sources/source_videos/video_000_xxx.mp4
# 源视频帧在: sources/frames/video_000_xxx_frame_*.png
```

### 场景2：检查视频混叠问题

```bash
# 1. 提取源视频帧
python scripts/extract_validation_sources.py ...

# 2. 对比每一帧
# 如果重建视频的某些帧和源视频完全不匹配，可能存在混叠问题

# 3. 检查帧连续性
cd validation_sources/frames
# 查看 video_000_xxx_frame_*.png 是否连续
```

### 场景3：复现训练时的validation行为

```bash
# 使用与训练完全相同的参数
python scripts/extract_validation_sources.py \
    --seed 1234 \
    --eval_video_path /data/val_videos \
    --eval_batch_size 8 \
    --eval_subset_size 100 \
    --eval_num_video_log 2
```

## 常见问题

### Q: 如何确保提取的视频与训练时一致？

A: 确保以下参数与训练脚本完全一致：
- `--seed`
- `--eval_batch_size`
- `--eval_subset_size`
- 所有其他 `--eval_*` 参数

### Q: 为什么提取的视频和wandb上的不一样？

A: 可能原因：
1. seed不一致
2. dataloader的shuffle顺序不同（使用DDP模式）
3. 训练脚本修改过，但没有更新这个脚本

### Q: 可以只提取信息不保存视频吗？

A: 可以，注释掉脚本中的 `save_video_as_images()` 和 `shutil.copy2()` 调用即可。

## 与 find_source_video.py 的区别

| 工具 | 用途 | 速度 | 准确性 |
|------|------|------|--------|
| `extract_validation_sources.py` | 直接提取会被记录的源视频 | 快 | 100%准确 |
| `find_source_video.py` | 通过相似度搜索找源视频 | 慢 | 近似匹配 |

推荐先用 `extract_validation_sources.py` 快速定位，如果需要分析相似度再用 `find_source_video.py`。



