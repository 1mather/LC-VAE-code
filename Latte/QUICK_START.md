# Latte + 自定义 VAE 快速开始指南

## 📋 修改完成清单

✅ **已修改的文件：**
- `train.py` - 主训练脚本
- `sample/sample.py` - 采样脚本
- `sample/sample_ddp.py` - 分布式采样脚本

✅ **新增的文件：**
- `VAE_MODIFICATION_README.md` - 详细修改说明
- `test_vae_loading.py` - VAE 测试脚本
- `QUICK_START.md` - 本文件

## 🚀 使用步骤

### 1. 安装依赖（如果还没安装）

```bash
# 进入 Latte 目录
cd /scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/Latte

# 安装依赖
pip install timm diffusers[torch]==0.24.0 accelerate tensorboard einops transformers av scikit-image decord pandas imageio-ffmpeg sentencepiece beautifulsoup4 ftfy omegaconf
```

### 2. 测试 VAE 加载

```bash
# 编辑 test_vae_loading.py，修改 VAE 路径（第14行）
# vae_path = "/your/vae/checkpoint/path"

# 运行测试
python test_vae_loading.py
```

### 3. 修改 VAE 路径

在以下文件中更新 VAE checkpoint 路径：

**train.py (第101行):**
```python
vae = WVAE_Compressed_TopK_multi_wavelet.from_pretrained(
    "你的VAE路径",  # 例如: "/scratch/cs/.../results/checkpoint-xxx"
    subfolder="vae"  # 如果不需要 subfolder，删除这一行
).to(device)
```

**sample/sample.py (第73行):**
```python
vae = WVAE_Compressed_TopK_multi_wavelet.from_pretrained(
    "你的VAE路径",
    subfolder="vae"
).to(device)
```

### 4. 选择编码方式

在 `train.py` 第223行，根据你的 VAE 类型选择：

**选项A：逐帧编码（默认，推荐先尝试这个）**
```python
# 已启用，不需要修改
x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
x = vae.encode(x).latent_dist.sample().mul_(0.18215)
x = rearrange(x, '(b f) c h w -> b f c h w', b=b).contiguous()
```

**选项B：视频级编码（如果你的 VAE 是 3D 的）**
```python
# 注释掉选项A的代码，取消注释这些行：
output = vae.encode(x, return_dict=True)
x = output.latent_dist.sample()
x = x.mul_(0.18215)  # 可能需要调整缩放因子
```

### 5. 运行训练

```bash
# 单 GPU 训练
python train.py --config configs/ucf101/ucf101_train.yaml

# 多 GPU 训练 (DDP)
torchrun --nproc_per_node=4 train.py --config configs/ucf101/ucf101_train.yaml
```

### 6. 运行采样

```bash
# 单 GPU 采样
python sample/sample.py \
    --config configs/ucf101/ucf101_sample.yaml \
    --ckpt /path/to/latte/checkpoint.pt \
    --pretrained_model_path /path/to/your/vae

# 多 GPU 采样
python sample/sample_ddp.py \
    --config configs/ucf101/ucf101_sample.yaml \
    --ckpt /path/to/latte/checkpoint.pt \
    --pretrained_model_path /path/to/your/vae
```

## ⚠️ 重要注意事项

### 1. 缩放因子
- Latte 默认使用 `0.18215` 作为 latent 缩放因子
- 你的 VAE 可能有不同的 scale 参数
- 检查你的 VAE config 并相应调整

### 2. 内存使用
- 3D VAE 通常比 2D VAE 占用更多内存
- 如果遇到 OOM (Out of Memory)：
  - 减小 batch size
  - 减小视频长度
  - 使用 gradient checkpointing
  - 使用 mixed precision (fp16)

### 3. VAE 模式
- 确保 VAE 处于评估模式：`vae.eval()`
- 编码时使用 `torch.no_grad()` 禁用梯度计算

## 🔍 调试技巧

### 检查 VAE 输出形状
```python
import torch
x = torch.randn(2, 3, 16, 256, 256).cuda()  # B, C, T, H, W
output = vae.encode(x)
latent = output.latent_dist.sample()
print(f"Input shape: {x.shape}")
print(f"Latent shape: {latent.shape}")
```

### 检查缩放因子
```python
# 查看你的 VAE config
print(vae.config)
# 查找 'scale' 参数
```

### 监控训练
```bash
# 启动 tensorboard
tensorboard --logdir results/your_experiment

# 查看日志
tail -f results/your_experiment/log.txt
```

## 📝 常见问题

### Q: ImportError: No module named 'causalvideovae'
**A:** 确保路径正确：
```python
sys.path.insert(0, '/scratch/cs/vidgen/guanjr/project/latent_wf_vae_new')
```

### Q: 训练时 loss 是 NaN
**A:** 可能原因：
1. 学习率太大
2. 缩放因子不正确
3. VAE 输出不稳定

### Q: 生成的视频质量很差
**A:** 检查：
1. VAE 是否训练良好
2. Latent 的缩放因子是否正确
3. 是否使用了正确的解码方式

### Q: 需要使用 VAE 的 extra_output
**A:** 修改训练循环以使用额外输出：
```python
output = vae.encode(x, return_dict=True)
latent = output.latent_dist.sample()
coeffs_low, coeffs_high, loss, teacher = output.extra_output
# 使用这些额外信息...
```

## 📚 参考文档

- **详细修改说明**: `VAE_MODIFICATION_README.md`
- **VAE 测试脚本**: `test_vae_loading.py`
- **Latte 原始 README**: `README.md`

## 🆘 需要帮助？

如果遇到问题：
1. 查看 `VAE_MODIFICATION_README.md` 获取更多细节
2. 运行 `test_vae_loading.py` 测试 VAE
3. 检查上述常见问题
4. 查看训练日志中的错误信息

---

**最后更新**: 2025-10-27
**修改人**: AI Assistant


