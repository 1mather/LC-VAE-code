# VAE 评估脚本使用指南

## 📊 支持的指标

- **PSNR** (↑): Peak Signal-to-Noise Ratio - 越高越好
- **SSIM** (↑): Structural Similarity Index - 越高越好
- **LPIPS** (↓): Learned Perceptual Image Patch Similarity - 越低越好
- **rFVD** (↓): Reconstruction Fréchet Video Distance - 越低越好

## 🚀 快速开始

### 基础评估（只计算 PSNR）
```bash
python scripts/eval_vae_psnr_fixed.py \
    --model_name WVAE_Compressed_TopK_multi_wavelet \
    --from_pretrained /path/to/checkpoint.ckpt \
    --model_config /path/to/config.json \
    --real_video_dir /path/to/videos \
    --num_frames 32 \
    --resolution 256
```

### 完整评估（所有四个指标）
```bash
python scripts/eval_vae_psnr_fixed.py \
    --model_name WVAE_Compressed_TopK_multi_wavelet \
    --from_pretrained /path/to/checkpoint.ckpt \
    --model_config /path/to/config.json \
    --real_video_dir /path/to/videos \
    --num_frames 32 \
    --resolution 256 \
    --compute_ssim \
    --eval_lpips \
    --eval_rfvd \
    --fvd_method styleganv
```

### 保存视频用于论文展示
```bash
python scripts/eval_vae_psnr_fixed.py \
    --model_name WVAE_Compressed_TopK_multi_wavelet \
    --from_pretrained /path/to/checkpoint.ckpt \
    --model_config /path/to/config.json \
    --real_video_dir /path/to/videos \
    --num_frames 32 \
    --resolution 256 \
    --save_videos \
    --paper_video_folder paper_videos_32ch \
    --max_videos_to_save 50 \
    --save_fps 8
```

## 📁 视频保存参数

### 方式 1: 简单文件夹名称（推荐）
```bash
--paper_video_folder paper_videos_8ch
--paper_video_folder paper_videos_16ch
--paper_video_folder paper_videos_32ch
```
文件夹会在当前目录下创建。

### 方式 2: 完整路径
```bash
--save_video_dir ./results/experiment_1/videos
--save_video_dir /absolute/path/to/videos
```

### 方式 3: 自动生成（不指定任何参数）
如果不指定 `--paper_video_folder` 或 `--save_video_dir`，会自动生成：
```
./saved_videos/WVAE_Compressed_TopK_multi_wavelet_checkpoint-290000_20251111_143052/
```

## 🔢 视频数量控制

### 限制评估的视频总数（快速测试）
```bash
--subset_size 100    # 只评估前100个视频
```

### 限制保存的视频数量（节省空间）
```bash
--max_videos_to_save 50    # 只保存前50个视频
```

### 组合使用示例
```bash
# 评估500个视频，但只保存前30个用于论文
python scripts/eval_vae_psnr_fixed.py \
    --subset_size 500 \
    --save_videos \
    --max_videos_to_save 30 \
    --paper_video_folder paper_videos_best \
    ...
```

## 🎥 视频/分辨率参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--resolution` | 目标分辨率 | 256 | 256, 512 |
| `--num_frames` | 每个视频的帧数 | 17 | 17, 32, 65 |
| `--sample_rate` | 帧采样率 | 1 | 1, 2 |
| `--batch_size` | 批次大小 | 1 | 1, 4, 10 |
| `--crop_size` | 裁剪大小 | 256 | 256, 512 |

## 📦 完整参数列表

### 必需参数
```bash
--model_name MODEL_NAME              # 模型类名
--from_pretrained PATH               # 模型检查点路径
--real_video_dir PATH                # 视频目录
```

### 模型参数
```bash
--model_config PATH                  # 模型配置文件（.ckpt需要）
--enable_tiling                      # 启用tiling（大视频）
--autocast_dtype {bf16,fp16,none}   # 混合精度类型（默认：bf16）
```

### 指标参数
```bash
--compute_ssim                       # 计算SSIM
--eval_lpips                         # 计算LPIPS
--eval_rfvd                          # 计算rFVD
--fvd_method {styleganv,videogpt}   # FVD计算方法（默认：styleganv）
```

### 数据加载参数
```bash
--num_workers N                      # DataLoader工作线程数（默认：8）
--batch_size N                       # 批次大小（默认：1）
```

### 输出参数
```bash
--output_file PATH                   # 保存结果JSON的路径
```

## 💡 实用示例

### 示例 1: 不同通道配置的对比评估
```bash
# 8通道配置
python scripts/eval_vae_psnr_fixed.py \
    --from_pretrained checkpoints/8ch.ckpt \
    --paper_video_folder paper_videos_8ch \
    --compute_ssim --eval_lpips --eval_rfvd \
    ...

# 16通道配置  
python scripts/eval_vae_psnr_fixed.py \
    --from_pretrained checkpoints/16ch.ckpt \
    --paper_video_folder paper_videos_16ch \
    --compute_ssim --eval_lpips --eval_rfvd \
    ...

# 32通道配置
python scripts/eval_vae_psnr_fixed.py \
    --from_pretrained checkpoints/32ch.ckpt \
    --paper_video_folder paper_videos_32ch \
    --compute_ssim --eval_lpips --eval_rfvd \
    ...
```

### 示例 2: 快速测试（50个视频，保存10个）
```bash
python scripts/eval_vae_psnr_fixed.py \
    --model_name WVAE_Compressed_TopK_multi_wavelet \
    --from_pretrained checkpoint.ckpt \
    --model_config config.json \
    --real_video_dir /path/to/videos \
    --subset_size 50 \
    --save_videos \
    --max_videos_to_save 10 \
    --paper_video_folder quick_test \
    --num_frames 32 \
    --resolution 256
```

### 示例 3: 完整评估不保存视频（最快）
```bash
python scripts/eval_vae_psnr_fixed.py \
    --model_name WVAE_Compressed_TopK_multi_wavelet \
    --from_pretrained checkpoint.ckpt \
    --model_config config.json \
    --real_video_dir /path/to/videos \
    --compute_ssim \
    --eval_lpips \
    --eval_rfvd \
    --fvd_method styleganv \
    --output_file results.json \
    --num_frames 32 \
    --resolution 256 \
    --batch_size 10
```

## 📊 输出说明

### 终端输出
```
======================================================================
EVALUATION RESULTS
======================================================================
Total videos evaluated: 1902
Total frames evaluated: 60864
Videos saved: 50 videos to paper_videos_32ch

PSNR (↑):        35.6386 ± 4.1832
SSIM (↑):        0.9234 ± 0.0156
LPIPS (↓):       0.0823 ± 0.0234
rFVD (↓):        45.2341
Flickering (↓):  0.009832 ± 0.002233
======================================================================
```

### JSON 输出文件
```json
{
  "psnr_mean": 35.6386,
  "psnr_std": 4.1832,
  "ssim_mean": 0.9234,
  "ssim_std": 0.0156,
  "lpips_mean": 0.0823,
  "lpips_std": 0.0234,
  "rfvd": 45.2341,
  "flickering_mean": 0.009832,
  "flickering_std": 0.002233,
  "num_videos": 1902,
  "num_frames": 60864,
  "config": {
    "model_name": "WVAE_Compressed_TopK_multi_wavelet",
    "from_pretrained": "/path/to/checkpoint.ckpt",
    "num_frames": 32,
    "resolution": 256,
    "sample_rate": 1,
    "fvd_method": "styleganv"
  }
}
```

### 保存的视频文件
```
paper_videos_32ch/
├── config.json                    # 评估配置
├── video_0000_real.mp4           # 原始视频
├── video_0000_recon.mp4          # 重建视频
├── video_0000_comparison.mp4     # 对比视频（左：真实，右：重建）
├── video_0001_real.mp4
├── video_0001_recon.mp4
├── video_0001_comparison.mp4
...
```

## ⚠️ 常见问题

### Q: 如何选择合适的 batch_size？
A: 
- GPU 内存充足：使用更大的 batch_size (如 10) 可以加速
- GPU 内存有限：使用 batch_size=1
- 计算 rFVD 时：batch_size 会影响结果，建议与训练时一致

### Q: 评估需要多长时间？
A:
- 只计算 PSNR：最快，约 0.5-1s/视频
- PSNR + SSIM + LPIPS：约 1-2s/视频
- 包含 rFVD：约 10-15s/视频（最慢，因为需要 I3D 特征提取）

### Q: 如何选择评估的视频数量？
A:
- 快速测试：50-100个视频
- 论文结果：建议至少 500-1000个视频
- 完整评估：使用全部测试集（如 1902个）

### Q: 保存的视频为什么看不出差异？
A: 这说明你的模型重建质量非常好！PSNR > 30 dB 时，人眼很难察觉差异。可以：
1. 放大查看细节
2. 使用视频播放器逐帧对比
3. 关注高频区域（纹理、边缘）

## 📚 相关文件

- 评估脚本：`scripts/eval_vae_psnr_fixed.py`
- 示例脚本：`examples/eval_all_metrics.sh`
- SSIM 计算：`causalvideovae/eval/cal_ssim.py`
- LPIPS 计算：使用 `lpips` 库
- FVD 计算：`causalvideovae/eval/cal_fvd.py`

