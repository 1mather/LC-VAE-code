# Changes Summary - Wan2_1_VAE_Trainable

## Overview
Fixed issues with latent normalization and made it optional for training.

## Changes Made

### 1. **Made Latent Normalization Optional** 
   - **File**: `causalvideovae/model/vae/Wan_2_1.py`
   - **Why**: During VAE training, you don't need latent normalization. It's only needed for inference with diffusion models.
   - **Change**: Added `use_latent_normalization` parameter (default: `False`)
   
   ```python
   # New parameter in __init__
   use_latent_normalization: bool = False
   ```
   
   - When disabled, no normalization is applied during encode/decode
   - When enabled, applies: `z_normalized = (z - mean) / std`

### 2. **Fixed DiagonalGaussianDistribution Usage**
   - **File**: `causalvideovae/model/vae/Wan_2_1.py`, `Wan_2_1_multi_wavelet.py`
   - **Issue**: Incorrectly passing `mu` and `log_var` as separate arguments
   - **Fix**: Concatenate them before passing to `DiagonalGaussianDistribution`
   
   ```python
   # Before (WRONG):
   posterior = DiagonalGaussianDistribution(mu, log_var)
   
   # After (CORRECT):
   parameters = torch.cat([mu, log_var], dim=1)
   posterior = DiagonalGaussianDistribution(parameters)
   ```

### 3. **Added Generator Support to sample() Method**
   - **File**: `causalvideovae/model/utils/distrib_utils.py`
   - **Why**: For reproducible sampling during training/testing
   - **Change**: Added optional `generator` parameter to `sample()` method
   
   ```python
   def sample(self, generator=None):
       x = self.mean + self.std * torch.randn(
           self.mean.shape, 
           device=self.parameters.device, 
           dtype=self.parameters.dtype,
           generator=generator
       )
       return x
   ```

### 4. **Updated Config Files**

   **Training Config** (`wan2_1_vae_config.json`):
   ```json
   {
     "model_name": "Wan2_1_VAE_Trainable",
     "z_channels": 16,
     "dim": 96,
     "dim_mult": [1, 2, 4, 4],
     "num_res_blocks": 2,
     "attn_scales": [],
     "temporal_downsample": [false, true, true],
     "dropout": 0.0,
     "use_latent_normalization": false  // No normalization for training
   }
   ```
   
   **Inference Config** (`wan2_1_vae_config_inference.json`):
   ```json
   {
     "model_name": "Wan2_1_VAE_Trainable",
     ...
     "use_latent_normalization": true,  // Enable normalization for inference
     "latent_mean": [...],
     "latent_std": [...]
   }
   ```

## When to Use Each Config

- **For Training**: Use `wan2_1_vae_config.json` (normalization disabled)
  - Train the VAE from scratch or fine-tune
  - No need for pre-computed statistics
  
- **For Inference with Diffusion Models**: Use `wan2_1_vae_config_inference.json` (normalization enabled)
  - When using trained VAE with a diffusion model
  - Ensures latents are properly normalized for the diffusion model

## Background: What is Latent Normalization?

The `latent_mean` and `latent_std` are per-channel statistics computed by:
1. Taking a pretrained VAE
2. Encoding a large dataset (e.g., UCF-101)
3. Computing mean and std for each of the 16 latent channels
4. Using these for standardization: `z_norm = (z - mean) / std`

This helps stabilize diffusion model training by ensuring the latent space has unit variance.

## Testing

Run the integration test:
```bash
python examples/wan2_1_vae/test_integration.py
```

The test should now pass with the updated configuration.

