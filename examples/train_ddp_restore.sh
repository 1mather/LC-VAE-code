
export WANDB_PROJECT=WFVAE
# Respect scheduler-provided CUDA_VISIBLE_DEVICES; do not override
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



export WANDB_RUN_ID=g1gaxrcp
export WANDB_RESUME=must    # 必须接到同一个 run，不允许新建
EXP_NAME=latent-WFVAE-V1-exp1

# Determine number of visible GPUs
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -r -a _cuda_array <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#_cuda_array[@]}
else
    if command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    else
        NUM_GPUS=0
    fi
fi

if [ "$NUM_GPUS" -lt 1 ]; then
    echo "No GPUs detected. Aborting."
    exit 1
fi

# 使用随机端口避免冲突
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

torchrun \
    --nnodes=1 --nproc_per_node=${NUM_GPUS} \
    --master_addr=localhost \
    --master_port=${MASTER_PORT} \
    train_ddp.py \
    --exp_name ${EXP_NAME} \
    --video_path /scratch/cs/imagedb/jiarui/latent_wf_vae/dataset/kinetics-dataset/k400/train\
    --eval_video_path /scratch/cs/imagedb/jiarui/latent_wf_vae/dataset/kinetics-dataset/k400/val \
    --model_name WFVAE_V3_2 \
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
    --eval_subset_size 1000 \
    --eval_lpips \
    --ema \
    --ema_decay 0.999 \
    --perceptual_weight 1.0 \
    --consistence_weight 100 \
    --loss_type l1 \
    --sample_rate 1 \
    --disc_cls causalvideovae.model.losses.LPIPSWithDiscriminator3D \
    --wavelet_loss \
    --wavelet_weight 0.1 \
    --dataset_num_worker 6\
    --wandb True\
    --resume_from_checkpoint /scratch/cs/imagedb/jiarui/latent_wf_vae/results/lr1.00e-05-bs8-rs256-sr1-fr25/WFVAE-V1/checkpoint-90000.ckpt


