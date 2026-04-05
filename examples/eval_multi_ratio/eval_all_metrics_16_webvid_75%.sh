#!/bin/bash
# Example script for comprehensive VAE evaluation
# Evaluates PSNR (↑), SSIM (↑), LPIPS (↓), and rFVD (↓)

# Configuration
MODEL_NAME="WVAE_Compressed_TopK_multi_wavelet"
FROM_PRETRAINED="/scratch/cs/vidgen/guanjr/experiment/WFVAE_Experiment/WVAE_Channel_Keepratio_0.5_no_teacher_loss_16_channels_fix_topk_32frame_8gpu-lr2.00e-05-bs4-rs256-sr1-fr32-cons0.0-p1_1200000-p2_0-modetv_l1-20251107_033821/checkpoint-290000.ckpt"
MODEL_CONFIG="/scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/examples/WVAE_compressed_topk_multi_wavelet/wfvae-temporal-compressed-example-fix-topk-handcraft-16chan-keep-75.json"
REAL_VIDEO_DIR="/scratch/cs/vidgen/data/panda70m_mp4s"
NUM_FRAMES=32
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
    --max_videos_to_save 100 \
    --subset_size 2000 \
    --save_fps 8 \
    --paper_video_folder erase_75_16_panda70m  \
    --output_file results_all_metrics_16_panda70m_mp4s_75%.json

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

