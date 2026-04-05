# Wan 2.1 VAE Training Integration

This directory contains the integration of Wan 2.1 VAE into the multi-phrase training pipeline.

## Overview

The `Wan2_1_VAE_Trainable` model is a trainable version of the Wan 2.1 VAE that is compatible with the DDP training infrastructure in `train_ddp_multi_phrase.py`.

### Key Features

- **VideoBaseAE Compatible**: Implements the same interface as other VAE models
- **DDP Training Support**: Works with PyTorch DistributedDataParallel
- **Temporal Caching**: Maintains the original efficient temporal processing
- **Learned Normalization**: Uses per-channel mean/std statistics for latent normalization
- **Architecture**: 
  - 16-channel latents
  - 3D causal convolutions with temporal downsampling
  - Encoder with 3 downsample stages (spatial only, then 2x temporal+spatial)
  - Decoder with corresponding upsample stages

## Files

- `wan2_1_vae_config.json`: Model configuration file
- `train_wan2_1_vae.sh`: Training script template
- `convert_wan2_1_checkpoint.py`: Utility to convert pretrained checkpoints
- `README.md`: This file

## Quick Start

### 1. Training from Scratch

Edit `train_wan2_1_vae.sh` to set your data paths:

```bash
VIDEO_PATH="/path/to/your/training/videos"
EVAL_VIDEO_PATH="/path/to/your/eval/videos"
```

Then run:

```bash
cd /scratch/cs/vidgen/guanjr/project/latent_wf_vae_new
bash examples/wan2_1_vae/train_wan2_1_vae.sh
```

### 2. Fine-tuning from Pretrained Checkpoint

If you have a pretrained Wan 2.1 checkpoint (e.g., `vae_step_411000.pth`), first convert it:

```bash
python examples/wan2_1_vae/convert_wan2_1_checkpoint.py \
    --input_checkpoint /path/to/vae_step_411000.pth \
    --output_checkpoint /path/to/wan2_1_trainable.ckpt \
    --config examples/wan2_1_vae/wan2_1_vae_config.json
```

Then uncomment and set the resume flag in `train_wan2_1_vae.sh`:

```bash
RESUME_FROM_CHECKPOINT="/path/to/wan2_1_trainable.ckpt"
```

### 3. Using in Your Own Script

```python
from causalvideovae.model import ModelRegistry

# Create model from config
model_cls = ModelRegistry.get_model("Wan2_1_VAE_Trainable")
model = model_cls.from_config("examples/wan2_1_vae/wan2_1_vae_config.json")

# Or load pretrained
model = model_cls.from_pretrained("/path/to/checkpoint")

# Use in training
outputs = model(video_batch)  # Returns ForwardOutput
reconstruction = outputs.sample
latent_dist = outputs.latent_dist
```

## Training Configuration

### Recommended Settings

- **Batch Size**: 4-8 per GPU (depending on memory)
- **Learning Rate**: 2e-5 with AdamW
- **Frames**: 17 frames with sample_rate=2
- **Resolution**: 256x256
- **Mix Precision**: bf16 (better than fp16 for stability)
- **EMA**: Enabled with decay=0.999

### Loss Weights

- **Reconstruction**: L1 loss (default)
- **KL Weight**: 1e-6
- **Discriminator Weight**: 0.5 (starts after 5000 steps)
- **Perceptual Weight**: 1.0
- **Consistency Weight**: 0.0 (no wavelet loss for this model)

### Discriminator

The model uses `LPIPSWithDiscriminator3D` which combines:
- LPIPS perceptual loss
- 3D PatchGAN discriminator
- KL divergence regularization

## Model Architecture

```
Input Video: [B, 3, T, H, W]
    ↓
Encoder (Encoder3d):
    - CausalConv3d (3 → 96)
    - Downsample Block 1 (96 → 192, spatial only)
    - Downsample Block 2 (192 → 384, temporal+spatial, T/2)
    - Downsample Block 3 (384 → 384, temporal+spatial, T/4)
    - Middle Block (384 → 384)
    - Head (384 → 32)  # 32 = 16*2 for mu and log_var
    ↓
Conv1: 32 → 32
    ↓
Split to mu and log_var: [B, 16, T/4, H/8, W/8]
    ↓
Normalize: (mu - mean) * (1/std)
    ↓
Sample z ~ N(mu, exp(0.5*log_var))
    ↓
Denormalize: z / (1/std) + mean
    ↓
Conv2: 16 → 16
    ↓
Decoder (Decoder3d):
    - CausalConv3d (16 → 384)
    - Middle Block (384 → 384)
    - Upsample Block 1 (384 → 384, spatial only)
    - Upsample Block 2 (384 → 192, temporal+spatial, T*2)
    - Upsample Block 3 (192 → 96, temporal+spatial, T*4)
    - Head (96 → 3)
    ↓
Output Video: [B, 3, T, H, W]
```

### Temporal Compression

- **Encoder**: Downsamples time by 4x (T → T/4)
- **Decoder**: Upsamples time back to T
- **Spatial Compression**: 8x8 (256x256 → 32x32 latents)
- **Total Latent Shape**: [B, 16, T/4, H/8, W/8]

### Temporal Caching

The model uses temporal caching for efficient processing:
- Encoder processes first frame, then 4-frame chunks
- Decoder processes frame-by-frame with caching
- This reduces memory usage and enables long video generation

## Differences from Original Wan2_1_VAE

The `Wan2_1_VAE_Trainable` is designed for training, while the original `Wan2_1_VAE` is for inference:

1. **Interface**: Implements VideoBaseAE interface for training compatibility
2. **Modules**: Exposes encoder/decoder as separate modules for DDP
3. **Output Format**: Returns `ForwardOutput` with all necessary fields
4. **Normalization**: Uses registered buffers for mean/std (device-agnostic)
5. **Training Features**: Supports gradient computation, EMA, mixed precision

## Monitoring Training

The training script logs to WandB with the following metrics:

- **Generator Losses**:
  - `train/generator_loss`: Total generator loss
  - `train/rec_loss`: Reconstruction loss (L1 or L2)
  - `train/latents_std`: Standard deviation of latent samples
  
- **Discriminator Losses**:
  - `train/discriminator_loss`: Total discriminator loss
  
- **Validation Metrics** (every `eval_steps`):
  - `val_hard/psnr`: Peak Signal-to-Noise Ratio
  - `val_hard/lpips`: Learned Perceptual Image Patch Similarity
  - `val_hard/flickering`: Temporal consistency metric
  - `val_hard/recon`: Reconstructed video samples
  
- **EMA Validation** (if enabled):
  - `val_ema_hard/psnr`
  - `val_ema_hard/lpips`
  - `val_ema_hard/flickering`
  - `val_ema_hard/recon`

## Troubleshooting

### Out of Memory

- Reduce `batch_size`
- Reduce `num_frames`
- Use `--enable_tiling` flag
- Use gradient checkpointing (requires model modification)

### Training Unstable

- Lower learning rate
- Increase warmup steps
- Check discriminator start step
- Adjust loss weights
- Enable EMA (helps stability)

### Poor Reconstruction Quality

- Check if pretrained weights loaded correctly
- Increase perceptual weight
- Adjust discriminator weight
- Train longer (this model may need 300K+ steps)

## Citation

If you use this model, please cite:

```
[Original Wan 2.1 paper citation - to be added]
```

