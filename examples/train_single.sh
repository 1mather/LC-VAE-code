unset https_proxy
export WANDB_PROJECT=WFVAE
export CUDA_VISIBLE_DEVICES=0

# For single-GPU probing, avoid NIC/IB-specific overrides to prevent NCCL bootstrap errors
# export GLOO_SOCKET_IFNAME=bond0
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_TC=162
# export NCCL_IB_TIMEOUT=22
# export NCCL_PXN_DISABLE=0
# export NCCL_IB_QPS_PER_CONNECTION=4
# export NCCL_ALGO=Ring

# Explicitly disable InfiniBand to use TCP/SHM for local single process
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

EXP_NAME=WFVAE-probe

torchrun \
    --standalone --nnodes=1 --nproc_per_node=1 \
    /scratch/cs/imagedb/jiarui/latent_wf_vae/train_ddp.py \
    --exp_name ${EXP_NAME} \
    --video_path /scratch/cs/imagedb/jiarui/latent_wf_vae/dataset/kinetics-dataset/k400/train\
    --eval_video_path /scratch/cs/imagedb/jiarui/latent_wf_vae/dataset/kinetics-dataset/k400/val \
    --model_name WFVAE_V2 \
    --model_config examples/wfvae-large-16chn.json \
    --resolution 256 \
    --num_frames 25 \
    --batch_size 2 \
    --lr 0.00001 \
    --epochs 4 \
    --disc_start 0 \
    --save_ckpt_step 5000 \
    --eval_steps 1000 \
    --eval_batch_size 1 \
    --eval_num_frames 33 \
    --eval_sample_rate 1 \
    --eval_subset_size 10 \
    --eval_lpips \
    --ema \
    --ema_decay 0.999 \
    --perceptual_weight 1.0 \
    --loss_type l1 \
    --sample_rate 1 \
    --disc_cls causalvideovae.model.losses.LPIPSWithDiscriminator3D \
    --wavelet_loss \
    --wavelet_weight 0.1 \
    --wandb True