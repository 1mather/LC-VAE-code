# LC-VAE: Latent Channel-wise Compression for Video VAE




## Overview

**LC-VAE** introduces channel-wise latent compression into wavelet-based video VAEs. Building on [WF-VAE](https://arxiv.org/abs/2411.17459) (CVPR 2025), LC-VAE selectively compresses latent channels using a fixed TopK mask, enabling flexible trade-offs between reconstruction quality and latent dimensionality.

Key features:
- **Channel-wise TopK compression** with configurable keep ratio (25% / 50% / 75%)
- **Fixed handcrafted masks** for stable training and deterministic channel selection
- **4 / 8 / 16 latent channel** variants
- **Wan2.1 VAE integration** for multi-wavelet latent compression
- Downstream video generation training via **Latte**

## Installation

```bash
git clone https://github.com/1mather/LC-VAE-code.git
cd LC-VAE-code
conda create -n lcvae python=3.10 -y
conda activate lcvae
pip install -r requirements.txt
pip install -e .
```

## Pretrained Models

| Model | Channels | Keep Ratio | Download |
|-------|----------|------------|----------|
| LC-VAE-4ch-75% | 4 | 75% | coming soon |
| LC-VAE-8ch-75% | 8 | 75% | coming soon |
| LC-VAE-16ch-75% | 16 | 75% | coming soon |
| LC-VAE-8ch-50% | 8 | 50% | coming soon |
| LC-VAE-16ch-50% | 16 | 50% | coming soon |

## Model Configurations

Model configs are in `examples/WVAE_compressed_topk_multi_wavelet/`. The key parameters:

| Parameter | Description |
|-----------|-------------|
| `latent_dim` | Number of latent channels (4 / 8 / 16) |
| `keep_ratio` | Fraction of channels retained after compression (0.25 / 0.5 / 0.75) |
| `use_fixed_mask` | Use handcrafted channel mask for deterministic compression |
| `fixed_mask_path` | Path to the `.json` mask file in `causalvideovae/model/channel_name/` |
| `not_compress` | Set `true` to disable compression (full-channel baseline) |

Example config (`8-channel, 75% keep ratio`):
```json
{
  "latent_dim": 8,
  "keep_ratio": 0.5,
  "use_fixed_mask": true,
  "fixed_mask_path": "causalvideovae/model/channel_name/multi_keep_ratio/fixed_mask_8ch_ratio75_handcraft.json",
  "not_compress": false
}
```

Available mask files are under `causalvideovae/model/channel_name/`:
- `multi_keep_ratio/fixed_mask_{8,16}ch_ratio{25,50,75}_handcraft.json`
- `t_t_t/` — 8 / 16 / 32 channel masks
- `wan_vae_32/` — 32-channel masks for Wan2.1

## Video Reconstruction

```bash
python scripts/recon_single_video.py \
    --model_name WVAE_Compressed_TopK_multi_wavelet \
    --from_pretrained /path/to/checkpoint \
    --model_config examples/WVAE_compressed_topk_multi_wavelet/wfvae-temporal-compressed-example-fix-topk-handcraft-8chan-keep-75.json \
    --video_path /path/to/video.mp4 \
    --rec_path rec.mp4 \
    --device cuda \
    --num_frames 32 \
    --resolution 256
```

## Evaluation

Compute PSNR / SSIM / LPIPS / rFVD on a video folder:

```bash
python scripts/eval_vae_psnr_fixed.py \
    --model_name WVAE_Compressed_TopK_multi_wavelet \
    --from_pretrained /path/to/checkpoint \
    --model_config /path/to/config.json \
    --real_video_dir /path/to/videos \
    --num_frames 32 \
    --resolution 256 \
    --compute_ssim \
    --eval_lpips \
    --eval_rfvd
```

Batch evaluation scripts for different channel counts and datasets are provided in `examples/`:

```bash
# 8-channel on UCF-101
bash examples/eval_all_metrics_8_ucf.sh

# 16-channel on Sky
bash examples/eval_all_metrics_16_sky.sh

# Wan2.1 VAE evaluation
bash examples/wan_eval_all_metrics_8_openvid.sh
```

## Training

### VAE Training

The main training script is `train_ddp_multi_phrase.py`. Update the config paths before running:

```bash
torchrun --nproc_per_node=4 train_ddp_multi_phrase.py \
    --config /path/to/config.json \
    --data_path /path/to/videos \
    --results_dir /path/to/output
```

### Latte Video Generation Training

After training the VAE, train a Latte video diffusion model with the compressed latents:

```bash
# 8-channel on Sky dataset
python Latte/train_wfvae_new_8chan.py \
    --config Latte/configs/sky/sky_train_8chan_channelwise_pretrain.yaml

# 16-channel on OpenVid
python Latte/train_wfvae_new_16chan.py \
    --config Latte/configs/open_vid/sky_train_16chan_channelwise_pretrain.yaml
```

SLURM scripts for cluster training are in `Latte/slurm_scripts/`. Update the paths (data, checkpoint, output directories) before submitting.

### Evaluation of Generation Quality

```bash
# Compute FVD/FID on generated videos
python Latte/cal_fvd.py

# Full generation metric evaluation
bash Latte/tools/eval_metrics.sh
```

## Project Structure

```
LC-VAE-code/
├── causalvideovae/
│   ├── model/
│   │   ├── vae/                  # VAE model implementations
│   │   │   ├── WVAE_Compressed_TopK_multi_wavelet.py        # Main LC-VAE model
│   │   │   ├── WVAE_Compressed_TopK_multi_wavelet_multi_keep_ratio.py
│   │   │   ├── Wan_2_1.py                                   # Wan2.1 integration
│   │   │   ├── Wan_2_1_multi_wavelet.py
│   │   │   └── modeling_wfvae_temporal_compressed_topk.py   # Base TopK model
│   │   ├── channel_name/         # Fixed channel mask configs
│   │   └── modules/              # Shared modules (wavelet, attention, etc.)
│   └── dataset/                  # Dataset loaders
├── Latte/
│   ├── train_wfvae_new_{4,8,16}chan.py  # Latte training scripts
│   ├── eval_wfvae_new_{4,8,16}chan.py   # Latte evaluation scripts
│   ├── configs/                  # Training configs (sky, ucf101, open_vid)
│   ├── slurm_scripts/            # Cluster job scripts
│   └── tools/                    # FVD/FID evaluation tools
├── scripts/
│   ├── eval_vae_psnr_fixed.py    # Main evaluation script
│   └── recon_single_video.py     # Video reconstruction demo
├── examples/
│   ├── WVAE_compressed_topk_multi_wavelet/  # Model configs
│   └── wan2_1_vae/               # Wan2.1 configs and scripts
└── train_ddp_multi_phrase.py     # Main VAE training script
```

## Acknowledgement

- [WF-VAE](https://github.com/PKU-YuanGroup/WF-VAE) — base wavelet video VAE (CVPR 2025)
- [Latte](https://github.com/Vchitect/Latte) — video generation training framework
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) — video generation model

## Citation

If you find this work useful, please cite:
```bibtex
@inproceedings{guan2026lcvae,
  title     = {Latent-Compressed Variational Autoencoder for Video Diffusion Models},
  author    = {Guan, Jiarui and Zhao, Wenshuai and Zou, Zhengtao and Kannala, Juho and Solin, Arno},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
}
```

## License

This project is released under the [Apache 2.0 License](LICENSE).
