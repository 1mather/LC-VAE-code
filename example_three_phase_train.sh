#!/bin/bash

# 三阶段时间压缩训练示例脚本
# 使用 Latent_WFVAE_TemporalCompressed_V2 模型

# ==================== 配置 ====================

# 实验名称
EXP_NAME="WFVAE_ThreePhase_Demo"

# 数据路径
VIDEO_PATH="/scratch/cs/vidgen/data/kinetics-dataset/k400/train"
EVAL_VIDEO_PATH="/scratch/cs/vidgen/data/kinetics-dataset/k400/val"

# 模型配置
MODEL_NAME="Latent_WFVAE_TemporalCompressed_V2"
MODEL_CONFIG="configs/wfvae_temporal_compressed_v2.json"  # 需要创建此配置文件

# 训练参数
BATCH_SIZE=1
LR=1e-5
NUM_FRAMES=17
RESOLUTION=256
SAMPLE_RATE=1

# 三阶段参数
PHASE1_STEPS=50000  # Phase 1: 0 -> 50k steps
PHASE2_STEPS=50000  # Phase 2: 50k -> 100k steps
# Phase 3: 100k -> end

# Consistency Loss
CONSISTENCE_WEIGHT=1.0
VARIANCE_FLOOR=0.02

# Phase 2 配置
PHASE2_MODE="gate"  # "gate" 或 "progressive_frames"
ALPHA_SCHEDULE="cosine"  # "cosine" 或 "linear"
MAX_ALPHA=0.9

# 动态权重调整
PHASE1_WEIGHT_MULT=1.0   # Phase 1: 1.0 * base_weight
PHASE2_WEIGHT_MULT=0.5   # Phase 2: 0.5 * base_weight
PHASE3_WEIGHT_MULT=0.3   # Phase 3: 0.3 * base_weight

# Discriminator 参数
DISC_START=5000
DISC_WEIGHT=0.5
KL_WEIGHT=1e-6
PERCEPTUAL_WEIGHT=1.0

# Checkpoint
CKPT_DIR="./results/three_phase"
SAVE_CKPT_STEP=5000

# Evaluation
EVAL_STEPS=5000
EVAL_BATCH_SIZE=4
EVAL_SUBSET_SIZE=100

# ==================== 运行训练 ====================

# 设置WANDB项目名
export WANDB_PROJECT="WFVAE-ThreePhase"

# 使用 torchrun 启动分布式训练
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    train_ddp_multi_phrase.py \
    --exp_name ${EXP_NAME} \
    --seed 1234 \
    \
    --epochs 1000 \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --weight_decay 1e-4 \
    --log_steps 50 \
    --save_ckpt_step ${SAVE_CKPT_STEP} \
    --ckpt_dir ${CKPT_DIR} \
    --mix_precision bf16 \
    \
    --video_path ${VIDEO_PATH} \
    --num_frames ${NUM_FRAMES} \
    --resolution ${RESOLUTION} \
    --sample_rate ${SAMPLE_RATE} \
    --dataset_num_worker 4 \
    --dynamic_sample \
    \
    --model_name ${MODEL_NAME} \
    --model_config ${MODEL_CONFIG} \
    \
    --enable_three_phase \
    --phase1_steps ${PHASE1_STEPS} \
    --phase2_steps ${PHASE2_STEPS} \
    --phase2_mode ${PHASE2_MODE} \
    --alpha_schedule ${ALPHA_SCHEDULE} \
    --max_alpha ${MAX_ALPHA} \
    --variance_floor ${VARIANCE_FLOOR} \
    --enable_dynamic_consistency_weight \
    --phase1_weight_mult ${PHASE1_WEIGHT_MULT} \
    --phase2_weight_mult ${PHASE2_WEIGHT_MULT} \
    --phase3_weight_mult ${PHASE3_WEIGHT_MULT} \
    \
    --consistence_weight ${CONSISTENCE_WEIGHT} \
    --disc_cls "causalvideovae.model.losses.LPIPSWithDiscriminator3D" \
    --disc_start ${DISC_START} \
    --disc_weight ${DISC_WEIGHT} \
    --kl_weight ${KL_WEIGHT} \
    --perceptual_weight ${PERCEPTUAL_WEIGHT} \
    --loss_type l1 \
    --disc_factor 1.0 \
    \
    --eval_steps ${EVAL_STEPS} \
    --eval_video_path ${EVAL_VIDEO_PATH} \
    --eval_num_frames ${NUM_FRAMES} \
    --eval_resolution ${RESOLUTION} \
    --eval_sample_rate 1 \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --eval_subset_size ${EVAL_SUBSET_SIZE} \
    --eval_num_video_log 2 \
    --eval_lpips \
    \
    --ema \
    --ema_decay 0.999 \
    --wandb

# ==================== 说明 ====================
# 
# 三阶段训练流程：
# 
# Phase 1 (0 -> 50k steps): 软一致性
#   - 仅添加 lowfreq_consistency_loss
#   - 不压缩时间维度，保持原始T帧
#   - 监控 PSNR/LPIPS 保持稳定，LLL variance 下降
# 
# Phase 2 (50k -> 100k steps): 软压缩
#   - 门控模式：LLL_soft = (1-α)*LLL + α*blur_t(LLL)
#   - α 从 0 → 0.9 按 cosine schedule 增长
#   - consistency_weight 降低到 0.5 * base_weight
# 
# Phase 3 (100k -> end): 硬压缩到1帧
#   - 时间低频压缩到 1 帧
#   - 解码时用 interpolate + blur_t，避免 expand 复制
#   - consistency_weight 进一步降低到 0.3 * base_weight
# 
# 监控指标（wandb）：
#   - train/consistence_loss: 一致性损失
#   - train/LLL_variance: 低频方差（应逐步下降但不坍缩）
#   - training/phase: 当前阶段 (phase1/phase2/phase3)
#   - training/alpha: Phase 2的门控系数
#   - training/consistency_weight: 动态调整的权重
# 
# 从checkpoint恢复：
#   --resume_from_checkpoint path/to/checkpoint.ckpt
# 
# 不恢复discriminator（仅加载generator）：
#   --not_resume_discriminator
# 
# 使用渐进减帧模式（Phase 2）：
#   --phase2_mode progressive_frames --phase2_target_frames 2

