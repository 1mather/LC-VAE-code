#!/bin/bash
# Example script for comprehensive VAE evaluation
# Evaluates PSNR (↑), SSIM (↑), LPIPS (↓), and rFVD (↓)

# Configuration
MODEL_NAME="Wan2_1_VAE_Trainable"
FROM_PRETRAINED="/scratch/cs/vidgen/guanjr/experiment/wan2_1_vae/WVAE_Channel_Keepratio_0.5_throw_highfreq-lr1.00e-05-bs2-rs256-sr1-fr33-cons0.0-p1_20000000-p2_0/checkpoint-100000.ckpt"
MODEL_CONFIG="/scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/examples/wan2_1_vae/wan2_1_vae_config.json"
REAL_VIDEO_DIR="/scratch/cs/vidgen/guanjr/Webvid"
NUM_FRAMES=33
RESOLUTION=256
SAMPLE_RATE=1
BATCH_SIZE=10
NUM_WORKERS=8

# # Run evaluation with all metrics
# python scripts/eval_vae_psnr_fixed.py \
#     --model_name ${MODEL_NAME} \
#     --from_pretrained ${FROM_PRETRAINED} \
#     --model_config ${MODEL_CONFIG} \
#     --real_video_dir ${REAL_VIDEO_DIR} \
#     --num_frames ${NUM_FRAMES} \
#     --resolution ${RESOLUTION} \
#     --sample_rate ${SAMPLE_RATE} \
#     --batch_size ${BATCH_SIZE} \
#     --num_workers ${NUM_WORKERS} \
#     --compute_ssim \
#     --eval_lpips \
#     --eval_rfvd \
#     --fvd_method styleganv \
#     --output_file results_all_metrics.json

# ==============================================================================
# Save videos for paper (optional)
# ==============================================================================
# Uncomment the following to save videos for visualization in your paper

python scripts/eval_vae_psnr_fixed.py \
    --model_name ${MODEL_NAME} \
    --from_pretrained ${FROM_PRETRAINED} \
    --model_config ${MODEL_CONFIG} \
    --real_video_dir ${REAL_VIDEO_DIR} \
    --num_frames ${NUM_FRAMES} \
    --resolution ${RESOLUTION} \
    --sample_rate ${SAMPLE_RATE} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --compute_ssim \
    --eval_lpips \
    --eval_rfvd \
    --fvd_method styleganv \
    --save_videos \
    --max_videos_to_save 400 \
    --subset_size 2000 \
    --save_fps 8 \
    --paper_video_folder wan2_1_results_all_metrics_8_webvid \
    --output_file wan2_1_results_all_metrics_8_webvid.json

# This will create:
# - video_XXXX_real.mp4: Original videos
# - video_XXXX_recon.mp4: Reconstructed videos  
# - video_XXXX_comparison.mp4: Side-by-side comparison (left: real, right: recon)

# ==============================================================================
# Parameter explanations:
# ==============================================================================
# Video folder options (pick one):
#   --paper_video_folder NAME     : Simple folder name (e.g. 'paper_videos_32ch')
#   --save_video_dir PATH         : Full path (e.g. './results/videos/experiment_1')
#   (auto-generated if neither specified)
#
# Video quantity control:
#   --subset_size N               : Evaluate only first N videos (for quick testing)
#   --max_videos_to_save N        : Save at most N videos (for paper figures)
#
# Resolution/size control:
#   --resolution R                : Target resolution (e.g. 256, 512)
#   --num_frames F                : Number of frames per video (e.g. 17, 32)
#   --batch_size B                : Batch size for evaluation
#
# ==============================================================================
# Usage examples:
# ==============================================================================

# Example 1: Custom folder name for different channel configurations
# python scripts/eval_vae_psnr_fixed.py ... --paper_video_folder paper_videos_8ch
# python scripts/eval_vae_psnr_fixed.py ... --paper_video_folder paper_videos_16ch
# python scripts/eval_vae_psnr_fixed.py ... --paper_video_folder paper_videos_32ch

# Example 2: Quick test on 50 videos
# python scripts/eval_vae_psnr_fixed.py ... --subset_size 50 --max_videos_to_save 10

# Example 3: Full evaluation without saving videos (faster)
# python scripts/eval_vae_psnr_fixed.py ... (without --save_videos flag)

# To evaluate only specific metrics:
# For PSNR only (default):
# python scripts/eval_vae_psnr_fixed.py ... (without --compute_ssim, --eval_lpips, --eval_rfvd)

# For PSNR + SSIM:
# python scripts/eval_vae_psnr_fixed.py ... --compute_ssim

# For PSNR + LPIPS:
# python scripts/eval_vae_psnr_fixed.py ... --eval_lpips

# For PSNR + rFVD:
# python scripts/eval_vae_psnr_fixed.py ... --eval_rfvd

# For all metrics (recommended):
# python scripts/eval_vae_psnr_fixed.py ... --compute_ssim --eval_lpips --eval_rfvd

