#!/bin/bash

# 快速测试脚本：用少量视频快速确定正确的输入范围

REAL_DATASET_DIR="/scratch/cs/vidgen/data/kinetics-dataset/k400/test"
CKPT="/scratch/cs/vidgen/guanjr/experiment/WFVAE_Experiment/WVAE_Channel_Keepratio_0.5_no_teacher_loss_32_channels_fix_topk_32frame_8gpu-lr2.00e-05-bs4-rs256-sr1-fr32-cons0.0-p1_1200000-p2_0-modetv_l1-20251028_140418/checkpoint-100000.ckpt"
MODEL_NAME="WVAE_Compressed_TopK_multi_wavelet"
MODEL_CONFIG="/scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/examples/WVAE_compressed_topk_multi_wavelet/wfvae-temporal-compressed-example-fix-topk-handcraft-32chan.json"

echo "=========================================="
echo "快速 PSNR 测试 (仅50个视频)"
echo "=========================================="

# 测试1: [0,1] 输入
echo ""
echo "测试 [0,1] 输入范围..."
python scripts/eval_vae_psnr.py \
    --real_video_dir ${REAL_DATASET_DIR} \
    --from_pretrained ${CKPT} \
    --model_config ${MODEL_CONFIG} \
    --model_name ${MODEL_NAME} \
    --num_frames 32 \
    --sample_rate 1 \
    --resolution 256 \
    --crop_size_width 256 \
    --crop_size_height 256 \
    --batch_size 4 \
    --num_workers 8 \
    --subset_size 50 \
    --output_file test_no_norm.json

# 测试2: [-1,1] 输入
echo ""
echo "测试 [-1,1] 输入范围..."
python scripts/eval_vae_psnr.py \
    --real_video_dir ${REAL_DATASET_DIR} \
    --from_pretrained ${CKPT} \
    --model_config ${MODEL_CONFIG} \
    --model_name ${MODEL_NAME} \
    --num_frames 32 \
    --sample_rate 1 \
    --resolution 256 \
    --crop_size_width 256 \
    --crop_size_height 256 \
    --batch_size 4 \
    --num_workers 8 \
    --subset_size 50 \
    --normalize_input \
    --output_file test_with_norm.json

echo ""
echo "=========================================="
echo "快速测试完成！"
echo "=========================================="
echo ""
echo "[0,1] 输入 PSNR:"
python -c "import json; print(f\"  {json.load(open('test_no_norm.json'))['psnr_mean']:.4f}\")"
echo ""
echo "[-1,1] 输入 PSNR:"
python -c "import json; print(f\"  {json.load(open('test_with_norm.json'))['psnr_mean']:.4f}\")"
echo ""
echo "期望值: ~35"
echo "=========================================="

