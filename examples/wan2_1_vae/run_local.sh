#!/bin/bash
export WANDB_PROJECT=wan2_1_vae_multi_wavelet
export WANDB_API_KEY=075cc222275dd179acb20fb979892510999e7864
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd):$PYTHONPATH

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

torchrun \
    --nnodes=1 --nproc_per_node=1 \
    --master_addr=localhost \
    --master_port=${MASTER_PORT} \
    train_ddp_multi_phrase.py \
    --exp_name WVAE_Channel_Keepratio_0.5_local \
    --video_path /scratch/cs/vidgen/data/kinetics-dataset/k400/train \
    --eval_video_path /scratch/cs/vidgen/data/kinetics-dataset/k400/val \
    --model_name Wan2_1_VAE_MultiWavelet \
    --model_config /scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/examples/wan2_1_vae/wan2_1_vae_multi_wavelet_config.json \
    --resolution 256 \
    --num_frames 33 \
    --batch_size 2 \
    --lr 0.00001 \
    --epochs 1000 \
    --disc_start 0 \
    --save_ckpt_step 20000 \
    --eval_steps 1000 \
    --eval_batch_size 1 \
    --eval_num_frames 33 \
    --eval_sample_rate 1 \
    --eval_subset_size 10 \
    --eval_lpips \
    --ema \
    --ema_decay 0.999 \
    --perceptual_weight 1.0 \
    --consistence_weight 0 \
    --loss_type l1 \
    --sample_rate 1 \
    --disc_cls causalvideovae.model.losses.LPIPSWithDiscriminator3D \
    --dataset_num_worker 6 \
    --ckpt_dir /scratch/cs/vidgen/guanjr/experiment/WFVAE_Experiment \
    --enable_three_phase \
    --phase1_steps 10000000 \
    --phase2_steps 0 \
    --enable_dynamic_consistency_weight \
    --phase1_weight_mult 5 \
    --phase2_weight_mult 2 \
    --phase3_weight_mult 1
