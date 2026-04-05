#!/bin/bash

# Example script for reconstructing videos from parquet file
# This demonstrates how to use recon_video.py with parquet data source

# Set your paths here
#REAL_DATASET_DIR="/scratch/cs/vidgen/guanjr/Webvid"
REAL_DATASET_DIR="/scratch/cs/vidgen/data/kinetics-dataset/k400/val"
OUTPUT_DIR="/scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/reconstruction_video_k400_val"
CKPT="/scratch/cs/vidgen/guanjr/experiment/WFVAE_Experiment/WVAE_Channel_Keepratio_0.5_no_teacher_loss_32_channels_fix_topk_32frame_8gpu-lr2.00e-05-bs4-rs256-sr1-fr32-cons0.0-p1_1200000-p2_0-modetv_l1-20251028_140418/checkpoint-100000.ckpt"
MODEL_NAME="WVAE_Compressed_TopK_multi_wavelet"  # Change to your model name
MODEL_CONFIG="/scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/examples/WVAE_compressed_topk_multi_wavelet/wfvae-temporal-compressed-example-fix-topk-handcraft-32chan.json"
SAMPLE_RATE=1
NUM_FRAMES=32
RESOLUTION=256
SUBSET_SIZE=0
# Create output directory
mkdir -p ${OUTPUT_DIR}


# Optional: also save original videos for comparison
# Add --output_origin flag to the command above

# Optional: enable tiling for large videos
# Add --enable_tiling flag to the command above

# Optional: process only a subset (for testing)
# Add --subset_size 10 to process only first 10 videos

accelerate launch \
    --config_file examples/accelerate_configs/default_config.yaml \
    scripts/recon_video.py \
    --batch_size 1 \
    --real_video_dir ${REAL_DATASET_DIR} \
    --generated_video_dir ${OUTPUT_DIR} \
    --device cuda \
    --sample_fps 24 \
    --sample_rate ${SAMPLE_RATE} \
    --num_frames ${NUM_FRAMES} \
    --resolution ${RESOLUTION} \
    --subset_size ${SUBSET_SIZE} \
    --num_workers 16 \
    --model_config ${MODEL_CONFIG} \
    --from_pretrained ${CKPT} \
    --model_name ${MODEL_NAME} \
    --crop_size_width 256 \
    --crop_size_height 256 \
    --eval_interval 500