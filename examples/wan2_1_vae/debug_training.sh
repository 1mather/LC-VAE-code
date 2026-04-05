#!/bin/bash
# Quick debugging script to test if training setup works

export WANDB_PROJECT=wan2_1_vae_multi_wavelet_debug
export WANDB_API_KEY=075cc222275dd179acb20fb979892510999e7864
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd):$PYTHONPATH

# Use a random port
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# Quick test run with minimal settings
torchrun \
    --nnodes=1 --nproc_per_node=1 \
    --master_addr=localhost \
    --master_port=${MASTER_PORT} \
    train_ddp_multi_phrase.py \
    --exp_name DEBUG_TEST \
    --video_path /scratch/cs/vidgen/data/kinetics-dataset/k400/train \
    --eval_video_path /scratch/cs/vidgen/data/kinetics-dataset/k400/val \
    --model_name Wan2_1_VAE_MultiWavelet \
    --model_config /scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/examples/wan2_1_vae/wan2_1_vae_multi_wavelet_config.json \
    --resolution 256 \
    --num_frames 17 \
    --batch_size 1 \
    --lr 0.00001 \
    --epochs 1 \
    --disc_start 0 \
    --save_ckpt_step 100 \
    --eval_steps 50 \
    --eval_batch_size 1 \
    --eval_num_frames 17 \
    --eval_sample_rate 1 \
    --eval_subset_size 10 \
    --ema \
    --ema_decay 0.999 \
    --perceptual_weight 1.0 \
    --consistence_weight 0 \
    --loss_type l1 \
    --sample_rate 1 \
    --disc_cls causalvideovae.model.losses.LPIPSWithDiscriminator3D \
    --dataset_num_worker 2 \
    --ckpt_dir /scratch/cs/vidgen/guanjr/experiment/WFVAE_DEBUG \
    --enable_three_phase \
    --phase1_steps 100 \
    --phase2_steps 0 \
    --enable_dynamic_consistency_weight \
    --phase1_weight_mult 0 \
    --phase2_weight_mult 0 \
    --phase3_weight_mult 0

echo "Debug test completed. Check logs for errors."


