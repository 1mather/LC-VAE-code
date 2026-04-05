export WANDB_PROJECT=WFVAE-Experiment
export WANDB_API_KEY=075cc222275dd179acb20fb979892510999e7864  # 请替换为你的wandb API密钥
export WANDB_RUN_ID=pnu0q0wl
export WANDB_RESUME=must 
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1,mlx5_14:1,mlx5_15:1,mlx5_16:1,mlx5_17:1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=162
export NCCL_IB_TIMEOUT=22
export NCCL_PXN_DISABLE=0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_ALGO=Ring
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export PYTHONPATH=$(pwd):$PYTHONPATH
EXP_NAME=WVAE_Channel_Keepratio_0.5_no_teacher_loss_8_channels_fix_topk_32frame_8gpu

# 注意：使用8卡训练，总 batch size = batch_size * 8
# 当前每卡 batch_size=4，总计32
# 如需保持与4卡相同的总 batch size (16)，请将 --batch_size 改为 2

# 使用随机端口避免冲突
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")


PHASE1_STEPS=1200000
PHASE2_STEPS=0

PHASE1_WEIGHT_MULT=0
PHASE2_WEIGHT_MULT=1.0
PHASE3_WEIGHT_MULT=0.3


torchrun \
    --nnodes=1 --nproc_per_node=8 \
    --master_addr=localhost \
    --master_port=${MASTER_PORT} \
    train_ddp_multi_phrase.py \
    --exp_name ${EXP_NAME} \
    --video_path /scratch/cs/vidgen/data/kinetics-dataset/k400/train \
    --eval_video_path /scratch/cs/vidgen/data/kinetics-dataset/k400/val \
    --model_name WVAE_Compressed_TopK_multi_wavelet \
    --model_config /scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/examples/WVAE_compressed_topk_multi_wavelet/wfvae-temporal-compressed-example-fix-topk-handcraft-8chan.json \
    --resolution 256 \
    --num_frames 32 \
    --batch_size 4 \
    --lr 0.00002 \
    --epochs 40 \
    --disc_start 0 \
    --save_ckpt_step 10000 \
    --eval_steps 1000 \
    --eval_batch_size 1 \
    --eval_num_frames 32 \
    --eval_sample_rate 1 \
    --eval_subset_size 1000 \
    --eval_lpips \
    --ema \
    --ema_decay 0.999 \
    --perceptual_weight 1.0 \
    --consistence_weight 0 \
    --loss_type l1 \
    --sample_rate 1 \
    --disc_cls causalvideovae.model.losses.LPIPSWithDiscriminator3D \
    --dataset_num_worker 8 \
    --ckpt_dir /scratch/cs/vidgen/guanjr/experiment/WFVAE_Experiment \
    --resume_from_checkpoint /scratch/cs/vidgen/guanjr/experiment/WFVAE_Experiment/WVAE_Channel_Keepratio_0.5_no_teacher_loss_8_channels_fix_topk_32frame_8gpu-lr2.00e-05-bs4-rs256-sr1-fr32-cons0.0-p1_1200000-p2_0-modetv_l1-20251103_152118/checkpoint-220000.ckpt \
    --enable_three_phase \
    --phase1_steps ${PHASE1_STEPS} \
    --phase2_steps ${PHASE2_STEPS} \
    --enable_dynamic_consistency_weight \
    --phase1_weight_mult ${PHASE1_WEIGHT_MULT} \
    --phase2_weight_mult ${PHASE2_WEIGHT_MULT} \
    --phase3_weight_mult ${PHASE3_WEIGHT_MULT} \