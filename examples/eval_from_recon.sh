#!/bin/bash

# 示例脚本：从 recon_video.py 生成的 real/ 和 generated/ 目录进行评估
# 这样可以确保完全对齐（使用相同采样的帧）

# 设置路径
BASE_OUTPUT_DIR="/scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/reconstruction_video_webvid"
REAL_DIR="${BASE_OUTPUT_DIR}/real"
GENERATED_DIR="${BASE_OUTPUT_DIR}/generated"

# 检查目录是否存在
if [ ! -d "$REAL_DIR" ]; then
    echo "Error: Real video directory does not exist: $REAL_DIR"
    echo "Please run recon_video.py first to generate the videos"
    exit 1
fi

if [ ! -d "$GENERATED_DIR" ]; then
    echo "Error: Generated video directory does not exist: $GENERATED_DIR"
    echo "Please run recon_video.py first to generate the videos"
    exit 1
fi

echo "Evaluating videos..."
echo "Real videos from: $REAL_DIR"
echo "Generated videos from: $GENERATED_DIR"
echo "========================================"

# 运行评估
accelerate launch \
    --config_file examples/accelerate_configs/default_config.yaml \
    scripts/eval.py \
    --batch_size 1 \
    --real_video_dir ${REAL_DIR} \
    --generated_video_dir ${GENERATED_DIR} \
    --device cuda \
    --sample_fps 24 \
    --sample_rate 1 \
    --num_frames 32 \
    --resolution 256 \
    --crop_size 256 \
    --subset_size 0 \
    --metric psnr

# 其他可用的 metrics: fvd, ssim, lpips
# 
# 例如评估 FVD:
# --metric fvd
# 
# 例如评估 SSIM:
# --metric ssim
# 
# 例如评估 LPIPS:
# --metric lpips

