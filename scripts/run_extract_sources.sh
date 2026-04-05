#!/bin/bash

# 提取validation源视频的示例脚本
# 根据你的实际训练参数修改以下变量

# ============ 配置参数 ============
# 这些参数应该与你的训练脚本完全一致！

SEED=1234                        # 随机种子
EVAL_VIDEO_PATH=" /scratch/cs/aaltoml/users/guanjr/data/kinetics-dataset/k400/val"  # 验证视频目录（必填）
EVAL_NUM_FRAMES=17               # 帧数
EVAL_RESOLUTION=256              # 分辨率
EVAL_SAMPLE_RATE=1               # 采样率
EVAL_BATCH_SIZE=8                # batch大小
EVAL_SUBSET_SIZE=100             # 子集大小
EVAL_NUM_VIDEO_LOG=2             # 记录视频数量
OUTPUT_DIR="./validation_sources"  # 输出目录

# ============ 运行模式选择 ============

# 模式1：单进程模式（推荐，快速）
echo "Running in single process mode..."
python scripts/extract_validation_sources.py \
    --seed ${SEED} \
    --eval_video_path ${EVAL_VIDEO_PATH} \
    --eval_num_frames ${EVAL_NUM_FRAMES} \
    --eval_resolution ${EVAL_RESOLUTION} \
    --eval_sample_rate ${EVAL_SAMPLE_RATE} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --eval_subset_size ${EVAL_SUBSET_SIZE} \
    --eval_num_video_log ${EVAL_NUM_VIDEO_LOG} \
    --output_dir ${OUTPUT_DIR}

# 模式2：DDP模式（与训练完全一致，取消注释使用）
# NUM_GPUS=2  # 使用的GPU数量
# echo "Running in DDP mode with ${NUM_GPUS} GPUs..."
# torchrun --nproc_per_node=${NUM_GPUS} scripts/extract_validation_sources.py \
#     --ddp \
#     --seed ${SEED} \
#     --eval_video_path ${EVAL_VIDEO_PATH} \
#     --eval_num_frames ${EVAL_NUM_FRAMES} \
#     --eval_resolution ${EVAL_RESOLUTION} \
#     --eval_sample_rate ${EVAL_SAMPLE_RATE} \
#     --eval_batch_size ${EVAL_BATCH_SIZE} \
#     --eval_subset_size ${EVAL_SUBSET_SIZE} \
#     --eval_num_video_log ${EVAL_NUM_VIDEO_LOG} \
#     --output_dir ${OUTPUT_DIR}

echo ""
echo "============================================"
echo "Extraction completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Check ${OUTPUT_DIR}/video_source_tracking.json for source file names"
echo "2. View extracted frames in ${OUTPUT_DIR}/frames/"
echo "3. Compare with your reconstructed videos"

