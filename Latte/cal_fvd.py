# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for Latte using PyTorch DDP.
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
 
from torchvision.models import inception_v3, Inception_V3_Weights
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
import torchvision
import torchvision.transforms.functional as TF

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
def evaluate_generated_videos(real_dir, gen_dir, device="cuda", metrics=["fvd"], fvd_method="styleganv", resolution=256, num_frames=32):
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
    ds = EvalDataset(real_dir, gen_dir, resolution=resolution, num_frames=num_frames)
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

    # Compute metrics with low memory footprint: accumulate features, not videos
    if "fvd" in metrics:
        # import feature extractors once
        if fvd_method == 'styleganv':
            from causalvideovae.eval.fvd.styleganv.fvd import get_fvd_feats, frechet_distance
        else:
            from causalvideovae.eval.fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats  # type: ignore
            from causalvideovae.eval.fvd.videogpt.fvd import frechet_distance  # type: ignore

        real_feats_list = []
        gen_feats_list = []
        for batch in tqdm.tqdm(loader, total=len(loader), desc="Extracting FVD features"):
            real_v = rearrange(batch["real"], "b t c h w -> b c t h w").to(device)
            gen_v = rearrange(batch["generated"], "b t c h w -> b c t h w").to(device)
            if real_v.shape != gen_v.shape:
                print(f"[FVD] Real video shape: {real_v.shape}, Generated video shape: {gen_v.shape}")
                continue
            # extract features per-batch to save memory
            with torch.no_grad():
                rf = get_fvd_feats(real_v, i3d=i3d, device=device)
                gf = get_fvd_feats(gen_v, i3d=i3d, device=device)
                # ensure cpu torch tensors for concatenation
                if isinstance(rf, np.ndarray):
                    rf = torch.from_numpy(rf)
                if isinstance(gf, np.ndarray):
                    gf = torch.from_numpy(gf)
                if isinstance(rf, torch.Tensor):
                    rf = rf.detach().cpu()
                if isinstance(gf, torch.Tensor):
                    gf = gf.detach().cpu()
            real_feats_list.append(rf)
            gen_feats_list.append(gf)
            # free batch tensors ASAP
            del real_v, gen_v, rf, gf
            torch.cuda.empty_cache()

        if len(real_feats_list) == 0:
            results["FVD"] = float("nan")
        else:
            real_feats = torch.cat(real_feats_list, dim=0)
            gen_feats = torch.cat(gen_feats_list, dim=0)
            fvd_val = frechet_distance(real_feats, gen_feats)
            results["FVD"] = float(fvd_val)

    # Inception Score on generated videos only
    if "is" in metrics :
        print("Computing IS")
        # Use NVIDIA StyleGAN2-ADA TorchScript Inception (matches original IS)
        detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
        try:
            from torch.hub import download_url_to_file
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "metrics")
            os.makedirs(cache_dir, exist_ok=True)
            detector_path = os.path.join(cache_dir, "inception-2015-12-05.pt")
            if not os.path.exists(detector_path):
                download_url_to_file(detector_url, detector_path, progress=True)
            inception = torch.jit.load(detector_path, map_location=device).eval()
            is_torchscript_inception = True
        except Exception as e:
            # Fallback to torchvision inception if TorchScript download fails
            print(f"[IS] TorchScript inception load failed ({e}), fallback to torchvision weights.")
            inception = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device).eval()
            is_torchscript_inception = False
        # Gather softmax probs across all generated frames
        probs_list = []
        with torch.no_grad():
            for batch in tqdm.tqdm(loader, total=len(loader), desc="Computing IS"):
                gen_v = rearrange(batch["generated"], "b t c h w -> (b t) c h w").to(device)  # [N,3,H,W] in [0,1]
                # resize to 299
                gen_v = torch.nn.functional.interpolate(gen_v, size=(299, 299), mode="bilinear", align_corners=False)
                if is_torchscript_inception:
                    # TorchScript inception (StyleGAN) expects images in [0,255] without mean/std normalization
                    gen_v = (gen_v * 255.0).clamp_(0, 255)
                    outputs = inception(gen_v)
                else:
                    # torchvision inception expects ImageNet-normalized inputs
                    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                    gen_v = (gen_v - mean) / std
                    outputs = inception(gen_v)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                probs = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
                probs_list.append(probs)
                del gen_v, logits, probs
                torch.cuda.empty_cache()
        if len(probs_list) == 0:
            results["IS_mean"] = float("nan")
            results["IS_std"] = float("nan")
        else:
            probs_all = np.concatenate(probs_list, axis=0)
            num_splits = max(1, min(10, len(probs_all) // 100))  # heuristic: ~100 samples per split
            splits = np.array_split(probs_all, num_splits, axis=0)
            scores = []
            for part in splits:
                p_y = np.mean(part, axis=0, keepdims=True)
                kl = part * (np.log(part + 1e-12) - np.log(p_y + 1e-12))
                kl = np.mean(np.sum(kl, axis=1))
                scores.append(np.exp(kl))
            results["IS_mean"] = float(np.mean(scores))
            results["IS_std"] = float(np.std(scores))

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute FVD between folders of real and generated videos.")
    parser.add_argument("--real_dir", type=str, required=True, help="Directory of real/reference videos (mp4/avi/etc).")
    parser.add_argument("--gen_dir", type=str, required=True, help="Directory of generated videos (mp4/avi/etc).")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on, e.g., cuda or cpu.")
    parser.add_argument("--metrics", type=str, nargs="+", default=["is"], help="Metrics to compute (default: fvd).")
    parser.add_argument("--fvd_method", type=str, default="styleganv", choices=["styleganv", "videogpt"], help="FVD backbone.")
    parser.add_argument("--num_frames", type=int, default=32, help="Number of frames to sample per video for evaluation.")
    parser.add_argument("--resolution", type=int, default=256, help="Short side resize/crop resolution before metric.")
    args = parser.parse_args()

    results = evaluate_generated_videos(
        real_dir=args.real_dir,
        gen_dir=args.gen_dir,
        device=args.device,
        metrics=args.metrics,
        fvd_method=args.fvd_method,
        resolution=args.resolution,
        num_frames=args.num_frames,
    )
    print(results)
