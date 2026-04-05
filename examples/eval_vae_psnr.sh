#!/bin/bash

# 直接计算 VAE PSNR 的脚本 (不保存视频)
# 完全对齐训练时的validation逻辑

# 设置路径
REAL_DATASET_DIR="/scratch/cs/vidgen/data/kinetics-dataset/k400/test"
CKPT="/scratch/cs/vidgen/guanjr/experiment/WFVAE_Experiment/WVAE_Channel_Keepratio_0.5_no_teacher_loss_32_channels_fix_topk_32frame_8gpu-lr2.00e-05-bs4-rs256-sr1-fr32-cons0.0-p1_1200000-p2_0-modetv_l1-20251028_140418/checkpoint-100000.ckpt"
MODEL_NAME="WVAE_Compressed_TopK_multi_wavelet"
MODEL_CONFIG="/scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/examples/WVAE_compressed_topk_multi_wavelet/wfvae-temporal-compressed-example-fix-topk-handcraft-32chan.json"

SAMPLE_RATE=1
NUM_FRAMES=32
RESOLUTION=256
SUBSET_SIZE=10  # 评估前1000个视频

echo "=========================================="
echo "VAE PSNR Evaluation (Training-Aligned)"
echo "=========================================="
echo ""
echo "使用与训练validation完全相同的逻辑："
echo "  - ValidVideoDataset (输入范围 [0,1])"
echo "  - 相同的模型前向传播"
echo "  - 相同的PSNR计算方式"
echo ""

accelerate launch \
    --config_file examples/accelerate_configs/default_config.yaml \
    scripts/eval_vae_psnr_fixed.py \
    --real_video_dir ${REAL_DATASET_DIR} \
    --from_pretrained ${CKPT} \
    --model_config ${MODEL_CONFIG} \
    --model_name ${MODEL_NAME} \
    --num_frames ${NUM_FRAMES} \
    --sample_rate ${SAMPLE_RATE} \
    --resolution ${RESOLUTION} \
    --crop_size 256 \
    --batch_size 2 \
    --num_workers 8 \
    --subset_size ${SUBSET_SIZE} \
    --output_file results_psnr.json

echo ""
echo "=========================================="
echo "评估完成！"
echo "=========================================="
echo ""
echo "PSNR结果:"
cat results_psnr.json | grep psnr_mean
echo ""
echo "期望值: ~35 (与训练validation一致)"
echo "=========================================="
