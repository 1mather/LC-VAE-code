# 验证过程中的 FVD 计算指南

## 概述

训练脚本现在支持在验证阶段自动计算 FVD (Fréchet Video Distance)，用于实时监控生成质量。

## 功能特点

- ✅ 在验证过程中自动计算 FVD
- ✅ 使用 I3D 模型提取视频特征
- ✅ 自动记录到 WandB
- ✅ 支持多 GPU 训练（FVD 只在 rank 0 计算）
- ✅ 轻量级实现，不影响训练速度

## 配置方法

### 1. 准备真实验证视频

将测试集视频放在一个目录中：

```bash
/path/to/validation_videos/
├── video_0000.mp4
├── video_0001.mp4
├── video_0002.mp4
└── ...
```

### 2. 配置训练参数

在训练配置文件（如 `ucf101_train.yaml`）中添加：

```yaml
# validation config:
val_every: 5000                          # 每 5000 步运行一次验证
val_num_samples: 8                       # 生成 8 个验证样本
val_batch_size: 2                        # 验证批次大小
val_real_video_path: /path/to/validation_videos  # 真实验证视频路径
```

**重要参数说明：**
- `val_real_video_path`: 真实验证视频的路径
  - 如果不设置此参数，验证仍然会运行，但不会计算 FVD
  - 视频格式：支持 `.mp4`, `.avi`, `.mov` 等常见格式

## 工作原理

### FVD 计算流程

```
验证开始
    ↓
1. 加载真实验证视频（rank 0）
    ↓
2. 生成视频样本（所有 ranks）
    ↓
3. 收集生成视频（rank 0，最多8个）
    ↓
4. 使用 I3D 模型提取特征
    ↓
5. 计算 FVD = FID(real_features, fake_features)
    ↓
6. 记录到 WandB
    ↓
验证结束
```

### FVD 计算细节

```python
def compute_simple_fvd(real_videos, fake_videos, device):
    """
    简化的 FVD 计算
    
    1. 加载 I3D 模型 (Kinetics-400 预训练)
    2. 提取真实视频特征: real_features (N, D)
    3. 提取生成视频特征: fake_features (N, D)
    4. 计算均值和协方差
    5. 计算 FVD 距离
    
    Args:
        real_videos: (N, T, C, H, W) 真实视频 [0, 255]
        fake_videos: (N, T, C, H, W) 生成视频 [0, 255]
    
    Returns:
        fvd: FVD 值（越小越好）
    """
```

## 使用示例

### 完整训练命令

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 \
    Latte/train.py \
    --config Latte/configs/ucf101/ucf101_train.yaml \
    --val_every 5000 \
    --val_num_samples 8 \
    --val_real_video_path /scratch/cs/vidgen/guanjr/UCF-101-test
```

### 验证输出示例

```bash
Step 5000: Running validation...
Generating validation samples at results/experiment_name/validation_samples
Loaded 8 real videos for FVD calculation
Generating samples: 100%|████████████| 4/4 [00:10<00:00, 2.50s/it]
Validation samples saved to results/experiment_name/validation_samples
Computing FVD with 8 samples...
FVD at step 5000: 245.32

Step 10000: Running validation...
...
FVD at step 10000: 198.47
```

### WandB 可视化

验证指标会自动记录到 WandB：

- `validation/sample_0` - 生成的视频样本
- `validation/sample_1` - ...
- `validation/fvd` - FVD 值（折线图）
- `validation/num_fvd_samples` - 用于计算的样本数

## 性能优化

### 样本数量建议

| 样本数 | FVD 稳定性 | 计算时间 | 推荐场景 |
|--------|-----------|---------|---------|
| 4 | 低 | 快 (~10s) | 快速调试 |
| 8 | 中 | 中 (~20s) | 日常训练 |
| 16 | 高 | 慢 (~40s) | 最终评估 |

### 计算开销

- **显存占用**: 约 2-3 GB (I3D 模型 + 特征提取)
- **时间开销**: 
  - 8 个样本: ~20 秒
  - 包括视频生成、特征提取、FVD 计算
- **推荐设置**: 
  - 训练时: 4-8 个样本
  - 最终评估: 使用完整 FVD2048 计算

## 与完整 FVD 的对比

### 验证中的 FVD (简化版)

**优点：**
- ✅ 实时反馈
- ✅ 低计算开销
- ✅ 集成到训练流程
- ✅ 自动记录到 WandB

**限制：**
- ⚠️ 样本数较少 (4-8 个)
- ⚠️ FVD 值可能不够稳定
- ⚠️ 适合趋势监控，不适合论文报告

### 完整 FVD2048 (tools/calc_metrics_for_dataset.py)

**优点：**
- ✅ 使用 2048 个样本
- ✅ 结果更稳定
- ✅ 适合论文报告
- ✅ 支持多种指标

**限制：**
- ⚠️ 需要单独运行
- ⚠️ 计算时间长 (~30 分钟)
- ⚠️ 需要预先生成所有样本

### 使用建议

```python
# 训练过程中：使用简化 FVD
val_every: 5000
val_num_samples: 8
val_real_video_path: /path/to/validation_videos

# 最终评估：使用完整 FVD2048
bash tools/eval_metrics.sh
```

## 常见问题

### Q1: 如果没有设置 `val_real_video_path` 会怎样？

验证仍然会运行并生成视频样本，但不会计算 FVD。会看到警告：

```
Warning: No real validation videos loaded. Set 'val_real_video_path' in config to enable FVD calculation.
```

### Q2: FVD 计算失败怎么办？

可能原因：
1. **I3D 模型下载失败**: 检查网络连接
2. **显存不足**: 减少 `val_num_samples` 或 `val_batch_size`
3. **视频加载失败**: 检查 `val_real_video_path` 是否正确

解决方法：
```bash
# 手动下载 I3D 模型
wget https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1 -O i3d_torchscript.pt
```

### Q3: FVD 值很高怎么办？

- 初始阶段 FVD 很高是正常的 (>1000)
- 随着训练进行应该逐渐降低
- 参考值：
  - FVD < 200: 优秀
  - FVD 200-500: 良好
  - FVD > 500: 需要更多训练

### Q4: 能否使用自定义的真实视频？

可以！只需确保：
1. 视频格式正确（常见格式即可）
2. 分辨率匹配训练设置
3. 帧数足够（至少 16 帧）

```yaml
val_real_video_path: /path/to/your/custom/videos
```

### Q5: 如何只计算 FVD 而不生成新样本？

可以使用已保存的验证样本：

```python
# 使用 tools/calc_metrics_for_dataset.py
python tools/calc_metrics_for_dataset.py \
    --real_data_path /path/to/real_videos/frames \
    --fake_data_path results/experiment_dir/validation_samples/frames \
    --metrics fvd2048_16f
```

## 技术细节

### I3D 模型

- **预训练数据集**: Kinetics-400
- **输入格式**: (N, C, T, H, W), 范围 [-1, 1]
- **输出**: 400 维特征向量
- **模型大小**: ~50 MB

### FVD 计算公式

```
FVD = ||μ_real - μ_fake||² + Tr(Σ_real + Σ_fake - 2√(Σ_real · Σ_fake))
```

其中：
- `μ_real`, `μ_fake`: 真实和生成视频的特征均值
- `Σ_real`, `Σ_fake`: 真实和生成视频的特征协方差矩阵

### 数据流

```
真实视频 (mp4) -> 加载到内存 (T,C,H,W) -> 归一化 [0,255]
                                                    ↓
生成潜在向量 (z) -> Diffusion 去噪 -> VAE 解码 -> 归一化 [0,255]
                                                    ↓
                        I3D 特征提取
                                ↓
                    计算 FVD (均值+协方差)
                                ↓
                        记录到 WandB
```

## 相关文件

- `Latte/train.py` - 主训练脚本（包含验证和 FVD 计算）
- `Latte/tools/metrics/frechet_video_distance.py` - 完整 FVD 实现
- `Latte/tools/calc_metrics_for_dataset.py` - 完整指标计算脚本
- `Latte/METRICS_EVALUATION_GUIDE.md` - 完整指标评估指南

## 参考文献

1. **FVD**: [Towards Accurate Generative Models of Video: A New Metric & Challenges](https://arxiv.org/abs/1812.01717)
2. **I3D**: [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750)
3. **StyleGAN-V**: [StyleGAN-V: A Continuous Video Generator with the Price, Image Quality and Perks of StyleGAN2](https://arxiv.org/abs/2112.14683)

