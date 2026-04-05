#!/bin/bash
#SBATCH --job-name=wan_multi_nocache
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=50:00:00
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --account=ellis_users
#SBATCH --output=./tiron_output/logs/%j.out
#SBATCH --error=./tiron_output/logs/%j.err
#SBATCH --constraint=hopper


eval "$(conda shell.bash hook)"
source ~/.bashrc

module load mamba
source activate /scratch/cs/aaltoml/users/guanjr/conda/envs/wfvae

cd /scratch/cs/vidgen/guanjr/project/latent_wf_vae_new
bash /scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/examples/wan2_1_vae/train_wan2_1_multi_wavelet.sh

