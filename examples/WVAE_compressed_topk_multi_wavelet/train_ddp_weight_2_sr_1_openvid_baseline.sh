

export WANDB_PROJECT=WFVAE-topk
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=162
export NCCL_IB_TIMEOUT=22
export NCCL_PXN_DISABLE=0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_ALGO=Ring
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

EXP_NAME=WVAE_Compressed_TopK_multi_wavelet

# 使用随机端口避免冲突
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

PHASE1_STEPS=0
PHASE2_STEPS=0

PHASE1_WEIGHT_MULT=0
PHASE2_WEIGHT_MULT=1.0
PHASE3_WEIGHT_MULT=0.3


torchrun \
    --nnodes=1 --nproc_per_node=4 \
    --master_addr=localhost \
    --master_port=${MASTER_PORT} \
    train_ddp_multi_phrase.py \
    --exp_name ${EXP_NAME} \
    --video_path /scratch/cs/vidgen/guanjr/OpenVid-1M/data/train \
    --eval_video_path /scratch/cs/vidgen/guanjr/OpenVid-1M/data/val \
    --model_name Latent_WFVAE_TemporalCompressed_TopK \
    --model_config /scratch/work/guanj2/latent_wf_vae_new/examples/Latent_WFVAE_compressed_topk/wfvae-temporal-compressed-example.json \
    --resolution 256 \
    --num_frames 25 \
    --batch_size 1 \
    --lr 0.00001 \
    --epochs 40 \
    --disc_start 0 \
    --save_ckpt_step 20000 \
    --eval_steps 1000 \
    --eval_batch_size 1 \
    --eval_num_frames 33 \
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
    --ckpt_dir /scratch/cs/aaltoml/users/guanjr/experiment/vae-fix/Latent_WFVAE_v1_large_16chn \
    --enable_three_phase \
    --phase1_steps ${PHASE1_STEPS} \
    --phase2_steps ${PHASE2_STEPS} \
    --enable_dynamic_consistency_weight \
    --phase1_weight_mult ${PHASE1_WEIGHT_MULT} \
    --phase2_weight_mult ${PHASE2_WEIGHT_MULT} \
    --phase3_weight_mult ${PHASE3_WEIGHT_MULT} \