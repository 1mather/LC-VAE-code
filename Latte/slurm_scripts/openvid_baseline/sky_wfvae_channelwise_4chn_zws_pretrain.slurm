#!/bin/bash
#SBATCH --job-name=openvid_sky_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=60:00:00
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --account=ellis_users
#SBATCH --output=./tiron_output/logs/%j.out
#SBATCH --error=./tiron_output/logs/%j.err
#SBATCH --constraint=hopper
#SBATCH --tmp=1024G
cd /scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/Latte
export NCCL_TIMEOUT=1800 
source ~/.bashrc

module load mamba
source activate /scratch/cs/aaltoml/users/guanjr/conda/envs/wfvae
export PYTHONPATH=$(pwd):$PYTHONPATH
# Set environment variables for distributed training
export WANDB_PROJECT=Latte-Training-split-chn
export WANDB_API_KEY=075cc222275dd179acb20fb979892510999e7864  # 请替换为你的wandb API密钥
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=$((12000 + SLURM_JOB_ID % 20000)) 

# Use srun for multi-GPU training (SLURM native way)
srun python train_wfvae_new_4chan.py --config /scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/Latte/configs/open_vid/sky_train_4chan_channelwise_pretrain.yaml