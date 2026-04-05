#!/bin/bash
#SBATCH --job-name=latte_base
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --partition=gpu-a100-80g
#SBATCH --output=./tiron_output/logs/%j.out
#SBATCH --error=./tiron_output/logs/%j.err
#SBATCH --constraint=ampere
cd /scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/Latte
source ~/.bashrc

conda activate wfvae

# Set environment variables for distributed training
export WANDB_PROJECT=Latte-Training


# Use srun for multi-GPU training (SLURM native way)
srun python train_wfvae_new.py --config ./configs/ucf101/ucf101_train_32chan_a100.yaml