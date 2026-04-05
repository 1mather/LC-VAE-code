# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluation script for Latte using PyTorch DDP.

This script can evaluate multiple checkpoints automatically:
- Set `eval_checkpoint_dir` in config to scan a directory for checkpoints
- Set `eval_interval` (default: 20000) to evaluate checkpoints at regular step intervals
- The script will find all checkpoints matching the interval (e.g., 0010000.pt, 0020000.pt, ...)
- Each checkpoint will be loaded and evaluated sequentially

Alternatively, use `ckpt` argument for single checkpoint evaluation (backward compatibility).
"""
import sys
sys.path.insert(0, '/scratch/cs/vidgen/guanjr/project/latent_wf_vae_new')
import os
import torch
# Maybe use fp16 percision training need to set to False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader, DistributedSampler, Subset
import argparse
import logging
import tqdm
from itertools import chain
import wandb
import random
import numpy as np
from pathlib import Path
from einops import rearrange
from causalvideovae.model import *
from causalvideovae.model.ema_model import EMA
from causalvideovae.dataset.ddp_sampler import CustomDistributedSampler
from causalvideovae.dataset.video_dataset import TrainVideoDataset, ValidVideoDataset
from causalvideovae.model.utils.module_utils import resolve_str_to_obj
from causalvideovae.utils.video_utils import tensor_to_video
 

import io
import os
import math
import argparse
from glob import glob
from time import time
from copy import deepcopy
from models import get_models
from causalvideovae.model.registry import ModelRegistry
from datasets import get_dataset
from models.clip import TextEmbedder
from diffusion import create_diffusion
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
# Modified: Use custom VAE instead of AutoencoderKL
from diffusers.models import AutoencoderKL 

from causalvideovae.model.vae.WVAE_Compressed_TopK_multi_wavelet import WVAE_Compressed_TopK_multi_wavelet
from diffusers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils import (clip_grad_norm_, create_logger, update_ema, 
                   requires_grad, cleanup, create_tensorboard, 
                   write_tensorboard, setup_distributed,
                   get_experiment_dir, text_preprocessing)
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer
from datetime import datetime, timedelta
import random
import imageio
import json
import logging


import os
import sys
try:
    import utils

    from diffusion import create_diffusion
    from utils import find_model
except:
    sys.path.append(os.path.split(sys.path[0])[0])

    import utils

    from diffusion import create_diffusion
    from utils import find_model

import argparse
import torchvision

from einops import rearrange
from models import get_models
from torchvision.utils import save_image
# Modified: Use custom VAE instead of AutoencoderKL
from diffusers.models import AutoencoderKL 
import sys
sys.path.insert(0, '/scratch/cs/vidgen/guanjr/project/latent_wf_vae_new')
from causalvideovae.model.vae.WVAE_Compressed_TopK_multi_wavelet import WVAE_Compressed_TopK_multi_wavelet
from models.clip import TextEmbedder
import imageio
from omegaconf import OmegaConf

from Latte.tools.metrics.frechet_video_distance import compute_fvd

from tools.metrics.metric_utils import MetricOptions
from tools.metrics.frechet_video_distance import compute_fvd
from tools import dnnlib
from omegaconf import OmegaConf
import os

# ------------------------------------------------------------------------------------
# Utilities: fixed-mask compression / decompression
# ------------------------------------------------------------------------------------
def _compress_with_indices(z: torch.Tensor, keep_indices: torch.Tensor) -> torch.Tensor:
    """Select channel subset by indices. z: (B, C, T, H, W)"""
    return z.index_select(dim=1, index=keep_indices)

def _decompress_with_indices(coeffs: torch.Tensor, keep_indices: torch.Tensor, original_channels: int, device) -> torch.Tensor:
    """Scatter selected channels back to original channel dim; fill zero for removed channels.
       coeffs: (B, K, T, H, W) -> out: (B, C, T, H, W)"""
    B, K, T, H, W = coeffs.shape
    out = torch.zeros(B, original_channels, T, H, W, device=device, dtype=coeffs.dtype)
    out[:, keep_indices, :, :, :] = coeffs
    return out

def load_fixed_mask_indices_json(mask_path: str, device: torch.device) -> torch.LongTensor:
    """Load fixed channel indices from JSON with key 'selected_indices'."""
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Fixed mask file not found: {mask_path}")
    with open(mask_path, "r") as f:
        data = json.load(f)
    indices = data.get("selected_indices", None)
    if indices is None:
        raise ValueError(f"Invalid mask file: missing 'selected_indices' in {mask_path}")
    idx = torch.tensor(indices, dtype=torch.long, device=device)
    return idx


# ------------------------------------------------------------------------------------
# Evaluation: use your causalvideovae.eval pipeline over saved video folders
# ------------------------------------------------------------------------------------
def evaluate_generated_videos(real_dir, gen_dir, device="cuda", metrics=["fvd"], fvd_method="styleganv"):
    """
    Evaluate saved mp4/avi videos in two folders using direct video loading.
    Returns: dict like {"FVD": v1, ...}
    """
    from glob import glob
    from torch.utils.data import DataLoader
    from causalvideovae.dataset.video_dataset import (
        ValidVideoDataset, DecordInit, Compose, Lambda, resize, CenterCropVideo, ToTensorVideo
    )
    from causalvideovae.eval.cal_fvd import calculate_fvd

    class EvalDataset(ValidVideoDataset):
        def __init__(self, real_video_dir, generated_video_dir, resolution=256, num_frames=32):
            self.v_decoder = DecordInit()
            self.video_exts = ["avi", "mp4", "mkv", "mov"]
            self.real_video_files = sorted(sum([glob(os.path.join(real_video_dir, f"*.{ext}")) for ext in self.video_exts], []))
            self.generated_video_files = sorted(sum([glob(os.path.join(generated_video_dir, f"*.{ext}")) for ext in self.video_exts], []))
            if len(self.real_video_files) == 0 or len(self.generated_video_files) == 0:
                raise ValueError(f"Empty video set: real={len(self.real_video_files)}, gen={len(self.generated_video_files)}")
            # Required attributes for _load_video
            self.num_frames = num_frames
            self.sample_rate = 1
            # Transform: numpy [T,H,W,C] -> tensor [T,C,H,W] with preprocessing (FVD expects BTCHW)
            self.transform = Compose([
                Lambda(lambda x: torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0),  # [T,H,W,C] -> [T,C,H,W], [0,255]->[0,1]
                Lambda(lambda x: resize(x, resolution)),
                CenterCropVideo(resolution)
            ])

        def __len__(self):
            return len(self.real_video_files)

        def __getitem__(self, idx):
            try:
                # Load real video with sampling to num_frames
                real_vr = self.v_decoder(self.real_video_files[idx])
                total_frames = len(real_vr)
                if total_frames >= self.num_frames:
                    # Uniform sampling
                    indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
                else:
                    # Repeat frames if too short
                    indices = (np.arange(self.num_frames) % total_frames).astype(int)
                real_v = real_vr.get_batch(indices).asnumpy()  # [T, H, W, C]
                
                # Load generated video, match to num_frames if needed
                gen_vr = self.v_decoder(self.generated_video_files[idx])
                gen_total_frames = len(gen_vr)
                if gen_total_frames != self.num_frames:
                    # Sample to match num_frames
                    if gen_total_frames > self.num_frames:
                        gen_indices = np.linspace(0, gen_total_frames - 1, self.num_frames, dtype=int)
                    else:
                        # Repeat frames if too short
                        gen_indices = (np.arange(self.num_frames) % gen_total_frames).astype(int)
                    gen_v = gen_vr.get_batch(gen_indices).asnumpy()  # [T, H, W, C]
                else:
                    gen_v = gen_vr.get_batch(range(gen_total_frames)).asnumpy()  # [T, H, W, C]
                
                return {"real": self.transform(real_v), "generated": self.transform(gen_v)}
            except Exception as e:
                print(f"[FVD] Failed to load video pair {idx}: {e}, retrying with idx 0")
                if idx != 0:
                    return self.__getitem__(0)
                else:
                    raise Exception(f"Failed to load even fallback video: {e}")

    results = {}
    ds = EvalDataset(real_dir, gen_dir)
    loader = DataLoader(ds, batch_size=1, num_workers=0, pin_memory=True)  # num_workers=0 to avoid multiprocessing issues

    # Load I3D model
    if "fvd" in metrics:
        if fvd_method == 'styleganv':
            from causalvideovae.eval.fvd.styleganv.fvd import load_i3d_pretrained
        else:
            from causalvideovae.eval.fvd.videogpt.fvd import load_i3d_pretrained
        i3d = load_i3d_pretrained(device)
    else:
        i3d = None

    # Compute metrics: accumulate all videos then calculate FVD once
    if "fvd" in metrics:
        real_videos = []
        gen_videos = []
        for batch in loader:
            real_v = rearrange(batch["real"], "b t c h w -> b c t h w").to(device)
            gen_v = rearrange(batch["generated"], "b t c h w -> b c t h w").to(device)
            if real_v.shape != gen_v.shape:
                print(f"[FVD] Real video shape: {real_v.shape}, Generated video shape: {gen_v.shape}")
                continue
            real_videos.append(real_v)
            gen_videos.append(gen_v)
        
        # Concatenate all batches
        real_all = torch.cat(real_videos, dim=0)  # [N, C, T, H, W]
        gen_all = torch.cat(gen_videos, dim=0)
        
        # Calculate FVD once on full dataset
        fvd_result = calculate_fvd(real_all, gen_all, device, i3d=i3d, method=fvd_method)
        results["FVD"] = float(list(fvd_result["value"].values())[0])

    return results


# ------------------------------------------------------------------------------------
# Sampling helper (copied from sample.py logic, adapted to return latents)
# ------------------------------------------------------------------------------------
@torch.no_grad()
def generate_latent_samples_like_sample_py(eval_batch_size, model, args, device):
    """Replicates Latte/sample/sample.py sampling logic to produce latent samples.
    Returns a latent tensor of shape (B, F, C, H, W).
    """
    using_fp16 = bool(getattr(args, "use_fp16", False))
    # use underlying module if DDP-wrapped
    net = model.module if hasattr(model, "module") else model
    if using_fp16:
        print('WARNING: using half percision for inferencing!')
        net.to(dtype=torch.float16)


    # diffusion with respaced timesteps, e.g., "250" or "ddim50"
    sampling_steps = str(getattr(args, "num_sampling_steps", 250))
    sampling_method = str(getattr(args, "sample_method", "ddpm")).lower()
    sampling_diffusion = create_diffusion(sampling_steps)
    #sampling_diffusion = create_diffusion(timestep_respacing="")

    latent_size = int(getattr(args, "latent_size", getattr(args, "image_size", 256) // 8))
    dtype = torch.float16 if using_fp16 else torch.float32

    z = torch.randn(eval_batch_size, args.num_frames, args.in_channels, latent_size, latent_size, dtype=dtype, device=device)
    #[1,1,552,16,16]
    # ----------- 1. 原始噪声 batch -----------
    B = z.shape[0]           # 原始batch
    num_classes = int(getattr(args, "num_classes", 101))
    cfg_scale = float(getattr(args, "cfg_scale", 7.0))
    if args.dataset != "sky":
        neg_id = int(getattr(args, "negative_name", 101))
        cfg_scale = float(getattr(args, "cfg_scale", 7.0))

        # ----------- 2. 构造标签 -----------
        y_pos  = torch.randint(0, num_classes, (B,), device=device)
        y_null = torch.full((B,), neg_id, device=device)

        # ----------- 3. 拼接输入 -----------
        z = torch.cat([z, z], dim=0)          # (2B, ...)
        y = torch.cat([y_pos, y_null], dim=0)  # (2B,)
        model_kwargs = dict(y=y, cfg_scale=cfg_scale, use_fp16=using_fp16)
        sample_fn = getattr(net, "forward_with_cfg", net.forward)
    else:
    # 单类别或无条件生成，不使用CFG
        z=torch.cat([z, z], dim=0)
        y=torch.full((B,), 1, device=device)
        y=torch.cat([y, y], dim=0)
        sample_fn = getattr(net, "forward_with_cfg", net.forward)
        model_kwargs = dict(y=y, cfg_scale=cfg_scale, use_fp16=using_fp16)
    # ----------- 4. 构造kwargs -----------
    samples = sampling_diffusion.p_sample_loop(
        sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    # samples = sampling_diffusion.ddim_sample_loop(
    #     sample_fn, z_cat.shape, z_cat, clip_denoised=True, model_kwargs=model_kwargs, progress=True, device=device
    # )
    return samples  # (B, F, C, H, W) latents
def tensor_to_video(x):
    """[0-1] tensor to video"""
    x = (x * 2 - 1).detach().cpu()
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2
    x = x.permute(1, 0, 2, 3).float().numpy() # c t h w -> t c h w
    x = (255 * x).astype(np.uint8)
    return x

@torch.no_grad()
def _decode_and_save_batch(step,latents,vae, args, device, gen_dir, index_list):
    """Decode a batch of latents and save mp4 files aligned with index_list."""
    # optional temporal decompression
    vae.eval()
    vae.requires_grad_(False)
    B=latents.shape[0]
    latents=latents[:B//2]
    latents=rearrange(latents, "b t c h w -> b c t h w").contiguous()
    if args.fixed_mask_keep_indices is not None:
        keep_idx_tensor = torch.as_tensor(args.fixed_mask_keep_indices, dtype=torch.long, device=device)
        latents=_decompress_with_indices(latents, keep_idx_tensor, args.original_in_channels, device)
    if hasattr(vae, "temporal_processor") and hasattr(vae.temporal_processor, "decompress_backward"):
        latents = vae.temporal_processor.decompress_backward(coeffs=latents)

    b, f, c, h, w = latents.shape
    assert b == len(index_list), "index_list size must match batch size"
    #x = vae.temporal_processor.decompress_backward(coeffs=latents)
    x = vae.decoder(latents)
    x = rearrange(x, 'b c t h w -> b t c h w')
    x=x[:,:16]

    for i in range(b):
        vid = x[i].detach()
        video = ((vid * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        #video = tensor_to_video(vid)
        # Convert to THWC for imageio and expand a batch dimension
        #video = np.transpose(video, (0, 2, 3, 1))  # T, H, W, C
        #video = video[None, ...]  # 1, T, H, W, C
        save_path = os.path.join(gen_dir, f"{index_list[i]}.mp4")
        imageio.mimwrite(save_path, video, fps=8, quality=9)
    wandb.log({f"validation/generated": wandb.Video(save_path, fps=8, format="mp4")}, step=step)


@torch.no_grad()
def distributed_generate_eval_subset(step,ema, vae, args, device, gen_dir, rank, world_size, per_channel_scale):
    """Distributed generation of eval_subset_size videos across ranks.
    Each rank generates indices = range(rank, total, world_size).
    """
    total = int(getattr(args, "eval_subset_size", 0) or 0)
    if total <= 0:
        return
    os.makedirs(gen_dir, exist_ok=True)
    per_batch = int(getattr(args, 'eval_batch_size', 2))
    # build my index list
    my_indices = list(range(rank, total, world_size))
    # chunk into batches
    for i in range(0, len(my_indices), per_batch):
        chunk = my_indices[i:i+per_batch]
        latents = generate_latent_samples_like_sample_py(len(chunk), ema, args, device)
        if args.vae_latent_scale is not None:
            latents=latents/args.vae_latent_scale
        B, F_new, C_new, H_new, W_new = latents.shape

        if args.vae_latent_scale is not None and per_channel_scale is None:
            latents=latents/args.vae_latent_scale


        if per_channel_scale is not None and False:
            latents = rearrange(latents, "b t c h w -> b c t h w").contiguous()
            latents = latents / per_channel_scale.view(1, C_new, 1, 1, 1)
            latents = rearrange(latents, "b c t h w -> b t c h w").contiguous()

        x=latents
        x = x.view(B, 8, 16, 16, 2, 16, 2)      # (B, 8, 16, 16, 2, 16, 2)
        # 反过来恢复到 (B, 8, 16, 2, 2, 16, 16)
        x = x.permute(0, 1, 2, 4, 6, 3, 5).contiguous()         # (B, 8, 16, 2, 2, 16, 16)

        # 合并 8 * 16 * 2 * 2 = 512，并恢复 F=1
        x_latent = x.view(B, 1, 512, 16, 16)          # (B, 1, 512, 16, 16)

        #latents=latents/args.vae_latent_scale
        #latents=latents/0.18215
        # max=4772,min=-4772,mean=1.7 . shape([2, 1, 552, 16, 16])
        _decode_and_save_batch(step,x_latent, vae, args, device, gen_dir, chunk)

# ------------------------------------------------------------------------------------
# Validation: diffusion generation + VAE reconstruction + evaluation
# ------------------------------------------------------------------------------------
@torch.no_grad()
def diff_validation(model, ema, vae, diffusion, args, device, rank, experiment_dir, step, wandb_logger, per_channel_scale):
    """Generate videos (EMA), save to disk, upload ONLY 2 to wandb, evaluate metrics on disk."""
    model.eval(); ema.eval(); vae.eval()

    if rank == 0:
        val_root = os.path.join(experiment_dir, "validation_samples")
        os.makedirs(val_root, exist_ok=True)
        print(f"[Validation] Saving to {val_root}")

    # paths for this validation step
    step_dir = os.path.join(experiment_dir, "validation_samples", f"step_{step:07d}")
    real_dir = os.path.join(step_dir, "real")
    gen_dir  = os.path.join(step_dir, "generated")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(gen_dir,  exist_ok=True)

    # If eval_subset_size is set, generate that many videos distributed across ranks
    eval_subset_size = int(getattr(args, 'eval_subset_size', 0) or 0)
    if eval_subset_size > 0:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        distributed_generate_eval_subset(step,ema, vae, args, device, gen_dir, rank, world_size, per_channel_scale)
        if dist.is_initialized():
            dist.barrier()
        decoded = None
    else:
        # Small preview batch using sample.py logic
        preview_bs = int(getattr(args, 'eval_batch_size', 2))
        samples = generate_latent_samples_like_sample_py(preview_bs, ema, args, device)  # (b, f, c, h, w)
        # optional fixed-mask decompress
        if getattr(args, 'fixed_mask_keep_indices', None) is not None:
            keep_idx_tensor = torch.as_tensor(args.fixed_mask_keep_indices, dtype=torch.long, device=device)
            samples = _decompress_with_indices(
                samples,
                keep_idx_tensor,
                original_channels=getattr(args, 'original_in_channels', args.in_channels),
                device=device
            )
        # decode
        b, f, c, h, w = samples.shape
        samples = rearrange(samples, 'b f c h w -> (b f) c h w')
        samples = vae.decode(samples  ).sample
        samples = rearrange(samples, '(b f) c h w -> b f c h w', b=b)
        decoded = samples

    if rank == 0:
        # save generated preview videos; upload only first 2 to wandb
        if decoded is not None:
            up_to_upload = 2
            uploaded = 0
            for i in range(decoded.shape[0]):
                vid = decoded[i].detach()
                video = ((vid * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
                video_path=os.path.join(gen_dir, f"{i}.mp4")
                imageio.mimwrite(video_path, video, fps=8, quality=9)
                if uploaded < up_to_upload and wandb_logger is not None:
                    wandb_logger.log({f"validation/generated_{i}": wandb.Video(video_path, fps=8, format="mp4")}, step=step)
                    uploaded += 1


        #except Exception as e:
            #print(f"[Validation] Metric evaluation failed: {e}")

    model.train()


@torch.no_grad()
def vae_real_validation(vae, args, device, rank, input_video, step, wandb_logger):
    """VAE reconstruction validation; upload ONLY 1 video to WandB."""
    vae.eval()
    x = input_video.to(device)  # (B, T, C, H, W)
    x = rearrange(x, "b t c h w -> b c t h w").contiguous()

    # forward
    out = vae(x, return_dict=True) if hasattr(vae, "__call__") else {"sample": vae.encode(x)}
    recon = out["sample"] if isinstance(out, dict) else out

    if rank == 0 and recon.shape[0] > 0 and wandb_logger is not None:
        # upload only one recon video
        video = tensor_to_video(recon[0])
        wandb_logger.log({f"validation/vae": wandb.Video(video, fps=8)}, step=step)


def run_validation(model, ema, vae, diffusion, args, device, rank, experiment_dir, step, wandb_logger, per_channel_scale):
    """Unified entry: diffusion generation + VAE recon (with upload limits) + metrics."""
    print(f"\n[Validation] Running @ step {step}")
    diff_validation(model, ema, vae, diffusion, args, device, rank, experiment_dir, step, wandb_logger, per_channel_scale)
    if wandb_logger is not None:
        wandb_logger.log({"validation/step": step})


def find_checkpoints_to_evaluate(checkpoint_dir, eval_interval=20000):
    """
    Find all checkpoints in checkpoint_dir that are multiples of eval_interval.
    Returns sorted list of (step, checkpoint_path) tuples.
    """
    if not os.path.isdir(checkpoint_dir):
        print(f"[Eval] Checkpoint directory does not exist: {checkpoint_dir}")
        return []
    
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pt'):
            try:
                # Extract step number from filename (e.g., "0010000.pt" -> 10000)
                step_str = filename.replace('.pt', '')
                step = int(step_str)
                # Only include checkpoints that are multiples of eval_interval
                if step % eval_interval == 0:
                    checkpoints.append((step, os.path.join(checkpoint_dir, filename)))
            except ValueError:
                # Skip files that don't match the expected format
                continue
    
    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def load_checkpoint_to_model(model, ema, ckpt_path, rank):
    """Load a checkpoint into model and EMA."""
    state_dict = find_model(ckpt_path)
    if "temp_embed" in state_dict:
        if rank == 0:
            print("Drop old temp_embed from checkpoint.")
        state_dict.pop("temp_embed")
    if "pos_embed" in state_dict:
        state_dict.pop("pos_embed")
    
    # Load into model
    missing, unexpected = model.module.load_state_dict(state_dict, strict=False)
    if rank == 0:
        print(f"[Eval] Loaded checkpoint from {ckpt_path}")
        if missing:
            print(f"[Eval] Missing keys: {missing}")
        if unexpected:
            print(f"[Eval] Unexpected keys: {unexpected}")
    
    # Update EMA with model weights
    update_ema(ema, model.module, decay=0)
    
    # Extract step number
    try:
        step = int(os.path.basename(ckpt_path).replace('.pt', ''))
    except:
        step = 0
    
    if dist.is_initialized():
        dist.barrier()
    
    return step


def get_wfvae(args, device, logger=None):
    model_cls = ModelRegistry.get_model(args.vae_model_name)
    if args.vae_from_pretrained and os.path.isfile(args.vae_from_pretrained) and args.vae_from_pretrained.endswith(".ckpt"):

        assert args.vae_model_config is not None and os.path.isfile(args.vae_model_config), "--model_config must be provided when --from_pretrained is a .ckpt file"
        vae = model_cls.from_config(args.vae_model_config)
        checkpoint = torch.load(args.vae_from_pretrained, map_location="cpu")

        ema_state_dict = checkpoint.get("ema_state_dict", None)
        assert ema_state_dict is not None and len(ema_state_dict) > 0, "Invalid checkpoint: missing ema_state_dict"

        print("✓ Loading EMA weights from checkpoint")
        # Remove 'module.' prefix from DDP training (matching model's from_pretrained logic)
        state_dict = {key.replace("module.", ""): value for key, value in ema_state_dict.items()}

        try:
            missing, unexpected = vae.load_state_dict(state_dict, strict=False)
        except Exception:
            # Fallback for models that might be wrapped during training
            missing, unexpected = vae.module.load_state_dict(state_dict, strict=False)
        if len(unexpected) > 0:
            print(f"Warning: unexpected keys when loading ckpt: {unexpected}")
        if len(missing) > 0:
            print(f"Warning: missing keys when loading ckpt: {missing}")
    else:
        print(f"Loading custom WVAE from {args.vae_from_pretrained}")
        vae = model_cls.from_pretrained(args.vae_from_pretrained)
    vae = vae.to(device)
    vae.eval()
    if logger is not None:
        logger.info(f"Loaded custom WVAE from {args.vae_from_pretrained}")
    else:
        print(f"Loaded custom WVAE from {args.vae_from_pretrained}")
    return vae
# ------------------------------------------------------------------------------------
# Training

# ------------------------------------------------------------------------------------



    # ========== VAE Latent Statistics ==========
    # [RAW latent]    mean = 0.003185, std = 0.917545
    # [COMPRESSED latent (with fixed_mask)] mean = 0.005908, std = 1.249699
    # ===========================================

    # Suggested scaling factors (to get std≈1):
    # raw latent:        scale ~= 1.0 / 0.917545 = 1.089865
    # compressed latent: scale ~= 1.0 / 1.249699 = 0.800193

def main(args):
    assert torch.cuda.is_available(), "Training requires GPU."
    setup_distributed()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    # Seed per rank
    seed = getattr(args, "global_seed", 0) + rank
    torch.manual_seed(seed)

    # Experiment setup (rank 0)
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        num_frame_string = f"F{args.num_frames}S{args.frame_interval}"
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-{num_frame_string}-{args.dataset}"
        experiment_dir = get_experiment_dir(experiment_dir, args)
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        logger = create_logger(experiment_dir)
        tb_writer = create_tensorboard(experiment_dir)
        OmegaConf.save(args, os.path.join(experiment_dir, 'config.yaml'))
        logger.info(f"Experiment directory: {experiment_dir}")

        # Ensure W&B uses scratch-backed directories to avoid HOME quota issues
        os.environ.setdefault("WANDB_DIR", experiment_dir)
        os.environ.setdefault("WANDB_CACHE_DIR", os.path.join(experiment_dir, "wandb_cache"))
        os.environ.setdefault("WANDB_ARTIFACTS_DIR", os.path.join(experiment_dir, "wandb_artifacts"))
        os.environ.setdefault("WANDB_CONFIG_DIR", os.path.join(experiment_dir, "wandb_config"))
        os.environ.setdefault("TMPDIR", os.path.join(experiment_dir, "tmp"))
        for _d in [
            os.environ["WANDB_CACHE_DIR"],
            os.environ["WANDB_ARTIFACTS_DIR"],
            os.environ["WANDB_CONFIG_DIR"],
            os.environ["TMPDIR"],
        ]:
            os.makedirs(_d, exist_ok=True)

        wandb.init(
            project=os.getenv("WANDB_PROJECT", "Latte-Training"),
            name=f"{experiment_dir.split('/')[-1]}",
            config=dict(args),
            dir=experiment_dir,
            tags=[args.model, args.dataset, f"F{args.num_frames}"],
            save_code=False,
        )
        wandb_logger = wandb
    else:
        logger = create_logger(None)
        tb_writer = None
        wandb_logger = None
        experiment_dir = None
        checkpoint_dir = None
    # ===== Load fixed mask JSON and broadcast indices =====
    if hasattr(args, "fixed_mask_path") and args.fixed_mask_path:

        mask_data = torch.load(args.fixed_mask_path, map_location='cpu')
        keep_idx = mask_data['selected_indices'].to("cuda")
        print(f"[FixedMask] loaded {keep_idx.numel()} channels from {args.fixed_mask_path}")

        # Store as primitives to avoid OmegaConf tensor assignment error
        args.fixed_mask_keep_indices = keep_idx.detach().cpu().tolist()
    else:
        args.fixed_mask_keep_indices = None
        print(f"[FixedMask] no fixed mask path provided")

    # --------------------------------------------------------------------------------
    # Build models
    # --------------------------------------------------------------------------------
    # NOTE: Replace with your custom VAE if needed; here we assume your project VAE is already available via imports.
    # e.g., from your_registry import MyVideoVAE; vae = MyVideoVAE.from_pretrained(...).to(device)
    from diffusers.models import AutoencoderKL  # fallback to diffusers if not using custom VAE
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    sample_size = args.image_size // 8
    args.latent_size = sample_size
    model = get_models(args)

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        logger.info(f"Model Parameters: {total_params:,} ({total_params * 1.e-9:.3f}B)")
        logger.info(f"Trainable Parameters: {trainable_params:,} ({trainable_params * 1.e-9:.3f}B)")


    # Checkpoint evaluation setup
    eval_checkpoint_dir = getattr(args, "eval_checkpoint_dir", None)
    if eval_checkpoint_dir is None:
        # Fallback to checkpoint_dir if available
        eval_checkpoint_dir = checkpoint_dir if rank == 0 else None
    
    # Broadcast checkpoint directory to all ranks
    if dist.is_initialized():
        checkpoint_info = [eval_checkpoint_dir] if rank == 0 else [None]
        dist.broadcast_object_list(checkpoint_info, src=0)
        if rank != 0:
            eval_checkpoint_dir = checkpoint_info[0]
    
    # If eval_checkpoint_dir is specified, find and evaluate all checkpoints
    checkpoints_to_eval = []
    if eval_checkpoint_dir and os.path.isdir(eval_checkpoint_dir):
        eval_interval = int(getattr(args, "eval_interval", 20000))
        if rank == 0:
            checkpoints_to_eval = find_checkpoints_to_evaluate(eval_checkpoint_dir, eval_interval)
            logger.info(f"[Eval] Found {len(checkpoints_to_eval)} checkpoints to evaluate (interval={eval_interval}):")
            for step, path in checkpoints_to_eval:
                logger.info(f"  - Step {step}: {os.path.basename(path)}")
        else:
            # Non-rank-0 will get the list via broadcast
            pass
        
        # Broadcast checkpoint list to all ranks
        if dist.is_initialized():
            if rank == 0:
                checkpoint_list = checkpoints_to_eval
            else:
                checkpoint_list = []
            dist.broadcast_object_list([checkpoint_list], src=0)
            if rank != 0:
                checkpoints_to_eval = checkpoint_list
        
        if len(checkpoints_to_eval) == 0:
            if rank == 0:
                logger.warning(f"[Eval] No checkpoints found in {eval_checkpoint_dir} matching interval {eval_interval}")
            return
    else:
        # Single checkpoint mode (backward compatibility)
        if args.ckpt is not None:
            if rank == 0:
                checkpoints_to_eval = [(0, args.ckpt)]  # step will be extracted from filename
            else:
                checkpoints_to_eval = []
            # Broadcast checkpoint list to all ranks
            if dist.is_initialized():
                checkpoint_list = checkpoints_to_eval if rank == 0 else []
                dist.broadcast_object_list([checkpoint_list], src=0)
                if rank != 0:
                    checkpoints_to_eval = checkpoint_list
        else:
            if rank == 0:
                logger.warning("[Eval] No checkpoint directory or checkpoint file specified")
            return

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule

    vae = get_wfvae(args, device, logger)
    vae.requires_grad_(False)
    # If original channel count is needed for decompression:
    if getattr(args, 'original_in_channels', None) is None:
        # Default to args.in_channels as original
        args.original_in_channels = int(getattr(args, 'in_channels', 0)) or 256  # fallback
    # EMA & DDP
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[local_rank], find_unused_parameters=True)

    # Optional: load per-channel std stats for fine-grained latent scaling
    per_channel_scale = None
    std_npz_path = getattr(args, 'vae_latent_std_npz', None)
    if std_npz_path is not None and os.path.isfile(std_npz_path):
        try:
            npz = np.load(std_npz_path)
            std_arr = npz.get('std', None)
            if std_arr is not None and std_arr.size > 0:
                std_arr = std_arr.astype('float32')
                std_arr = np.clip(std_arr, 1e-6, None)
                per_channel_scale = torch.from_numpy(1.0 / std_arr).to(device)
                # shape: (C,) -> will reshape to (1,C,1,1,1) when applying
                if rank == 0:
                    logger.info(f"Loaded per-channel std from {std_npz_path}, C={per_channel_scale.numel()}")
            else:
                if rank == 0:
                    logger.info(f"npz at {std_npz_path} missing 'std'; skip per-channel scaling")
        except Exception as e:
            if rank == 0:
                logger.info(f"Failed to load per-channel std npz {std_npz_path}: {e}")

    # Optimizer & Scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=getattr(args, "learning_rate", 1e-4), weight_decay=0)
    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Data
    dataset = get_dataset(args)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=getattr(args, "global_seed", 0)
    )
    loader = DataLoader(
        dataset,
        batch_size=int(getattr(args, "local_batch_size", 1)),
        shuffle=False,
        sampler=sampler,
        num_workers=getattr(args, "num_workers", 4),
        pin_memory=True,
        drop_last=True
    )

    if rank == 0:
        logger.info(f"Dataset contains {len(dataset):,} videos ({getattr(args, 'data_path', '')})")

    # Initialize EMA with current model weights
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    # Train loop states

    log_steps = 0
    running_loss = 0.0
    start_time = time()
    first_epoch = 0

    # derive epochs from steps
    num_update_steps_per_epoch = max(1, math.ceil(len(loader)))
    num_train_epochs = math.ceil(getattr(args, "max_train_steps", 100000) / num_update_steps_per_epoch)
    
    # Evaluation loop: evaluate each checkpoint
    if rank == 0:
        logger.info(f"[Eval] Starting evaluation of {len(checkpoints_to_eval)} checkpoints...")
    
    for ckpt_idx, (expected_step, ckpt_path) in enumerate(reversed(checkpoints_to_eval)):
        if rank == 0:
            logger.info(f"\n{'='*80}")
            logger.info(f"[Eval] [{ckpt_idx+1}/{len(checkpoints_to_eval)}] Evaluating checkpoint: {os.path.basename(ckpt_path)}")
            logger.info(f"{'='*80}")
        
        # Load checkpoint
        train_steps = load_checkpoint_to_model(model, ema, ckpt_path, rank)
        
        # Verify step number matches expected
        if train_steps != expected_step and expected_step != 0:
            if rank == 0:
                logger.warning(f"[Eval] Step mismatch: expected {expected_step}, got {train_steps} from filename")
        
        # Run validation for this checkpoint
        run_validation(model, ema, vae, diffusion, args, device, rank, experiment_dir, train_steps, wandb_logger, per_channel_scale)
        
        if rank == 0:
            logger.info(f"[Eval] Completed evaluation for step {train_steps}\n")
    
    if rank == 0:
        logger.info(f"[Eval] Finished evaluating all {len(checkpoints_to_eval)} checkpoints!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train.yaml")
    cmd_args = parser.parse_args()
    main(OmegaConf.load(cmd_args.config))
