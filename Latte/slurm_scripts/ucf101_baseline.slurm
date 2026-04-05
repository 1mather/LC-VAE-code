#!/bin/bash
#SBATCH --job-name=latte_baseline_2gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --account=ellis_users
#SBATCH --output=./tiron_output/logs/%j.out
#SBATCH --error=./tiron_output/logs/%j.err
#SBATCH --constraint=hopper
cd /scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/Latte
source ~/.bashrc

conda activate wfvae
# Set environment variables for distributed training
export WANDB_PROJECT=Latte-Training
export WANDB_NAME=ucf101_wfvae_32chan_new_code

# Use srun for multi-GPU training (SLURM native way)
srun python train.py --config ./configs/ucf101/ucf101_train_baseline.yaml