"""
Direct VAE evaluation script - Computes PSNR, SSIM, LPIPS, rFVD
Directly computes metrics on tensors, matching training validation logic

Usage:
    python scripts/eval_vae_psnr_fixed.py \
        --real_video_dir /path/to/videos \
        --from_pretrained /path/to/model \
        --model_name WVAE_Compressed_TopK_multi_wavelet \
        --num_frames 32 \
        --resolution 256 \
        --eval_lpips \
        --eval_rfvd
"""

import argparse
from tqdm import tqdm
import torch
import sys
from torch.utils.data import DataLoader, Subset
import os
from contextlib import nullcontext
import numpy as np
from pathlib import Path
import imageio
import subprocess
import tempfile
sys.path.append(".")
from causalvideovae.model import *
from causalvideovae.dataset.video_dataset import ValidVideoDataset
from causalvideovae.eval.cal_psnr import calculate_psnr
from causalvideovae.eval.cal_ssim import calculate_ssim
from causalvideovae.eval.cal_fvd import calculate_fvd
from accelerate import Accelerator
from einops import rearrange

import json
import datetime

def save_video_comparison(real_video, recon_video, save_dir, video_idx, fps=8):
    """
    Save real and reconstructed videos side by side (using imageio like training script)
    Args:
        real_video: [C, T, H, W] tensor in [0, 1]
        recon_video: [C, T, H, W] tensor in [0, 1]
        save_dir: directory to save videos
        video_idx: video index for naming
        fps: frames per second
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert from [C, T, H, W] to [T, H, W, C] and to uint8 (like training script)
    # Input is already in [0, 1], so just multiply by 255
    real_video_uint8 = (real_video.float() * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(1, 2, 3, 0).contiguous()
    recon_video_uint8 = (recon_video.float() * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(1, 2, 3, 0).contiguous()
    
    # Save real video
    real_path = save_dir / f"video_{video_idx:04d}_real.mp4"
    imageio.mimwrite(str(real_path), real_video_uint8, fps=fps, quality=9)
    
    # Save reconstructed video
    recon_path = save_dir / f"video_{video_idx:04d}_recon.mp4"
    imageio.mimwrite(str(recon_path), recon_video_uint8, fps=fps, quality=9)
    
    # Save side-by-side comparison
    comparison_path = save_dir / f"video_{video_idx:04d}_comparison.mp4"
    # Create side-by-side by concatenating along width dimension
    comparison_video = torch.cat([real_video_uint8, recon_video_uint8], dim=2)  # concat along W
    imageio.mimwrite(str(comparison_path), comparison_video, fps=fps, quality=9)


def _tensor_to_uint8_video(tensor_4d: torch.Tensor) -> np.ndarray:
    """
    Convert [C, T, H, W] tensor in [0, 1] (or [-1, 1]) to uint8 [T, H, W, C].
    """
    # Accept [-1, 1] or [0, 1]; clamp to [0,1] first
    if tensor_4d.min() < 0:
        x = (tensor_4d + 1.0) / 2.0
    else:
        x = tensor_4d
    x = x.clamp(0.0, 1.0)
    x_uint8 = (x * 255.0).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8)
    # [C, T, H, W] -> [T, H, W, C]
    x_uint8 = x_uint8.permute(1, 2, 3, 0).contiguous()
    return x_uint8.cpu().numpy()


def _apply_h264_compression_to_clip(
    clip: torch.Tensor,
    qp: int = 30,
    tmp_root: str | None = None,
    fps: int = 8,
) -> torch.Tensor:
    """
    对单个视频 clip 进行 H.264 压缩退化。

    Args:
        clip: [C, T, H, W] in [0,1] or [-1,1]
        qp: quantization parameter, 越大压缩越狠，伪影越明显
        tmp_root: 临时目录（可选），为空则使用系统临时目录
        fps: 写入 / 读取视频的帧率（对内容本身影响不大）

    Returns:
        degraded_clip: [C, T, H, W] float32 in [0,1]
    """
    clip_uint8 = _tensor_to_uint8_video(clip)  # [T, H, W, C], uint8

    # 建立临时目录和文件
    tmp_dir = Path(tmp_root) if tmp_root is not None else Path(tempfile.mkdtemp())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    raw_path = tmp_dir / "raw_input.mp4"
    degraded_path = tmp_dir / "degraded_qp{}.mp4".format(qp)

    # 用 imageio 先写一个“几乎无损”的中间视频
    imageio.mimwrite(str(raw_path), clip_uint8, fps=fps, quality=9)

    # 调用 ffmpeg 做 H.264 压缩
    # 说明：这里假设系统中已安装 ffmpeg
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(raw_path),
        "-c:v",
        "libx264",
        "-qp",
        str(qp),
        "-an",
        str(degraded_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        # 如果压缩失败，直接返回原 clip（不做退化），并打印一次警告
        print(f"[WARN] ffmpeg compression failed ({e}), using original clip without degradation.")
        return clip.float().clamp(0.0, 1.0)

    # 重新读取退化后的视频
    degraded_frames = imageio.mimread(str(degraded_path))
    degraded_arr = np.stack(degraded_frames, axis=0)  # [T, H, W, C], uint8

    # 转回 [C, T, H, W]，并归一化到 [0,1]
    degraded = torch.from_numpy(degraded_arr).permute(3, 0, 1, 2).float() / 255.0

    # 对长度做一个简单对齐（如果因为编码差异导致 T 变化）
    orig_T = clip.shape[1]
    new_T = degraded.shape[1]
    if new_T >= orig_T:
        degraded = degraded[:, :orig_T]
    else:
        # 如果退化视频更短，简单重复最后一帧补齐
        pad_T = orig_T - new_T
        last = degraded[:, -1:].repeat(1, pad_T, 1, 1)
        degraded = torch.cat([degraded, last], dim=1)

    return degraded

@torch.no_grad()
def main(args: argparse.Namespace):
    accelerator = Accelerator()
    device = accelerator.device
    
    print("="*70)
    print("VAE Comprehensive Evaluation (PSNR ↑, SSIM ↑, LPIPS ↓, rFVD ↓)")
    print("="*70)
    
    # ---- Load Model ----
    data_type = torch.bfloat16
    model_cls = ModelRegistry.get_model(args.model_name)
    assert args.from_pretrained and os.path.isfile(args.from_pretrained) and args.from_pretrained.endswith(".ckpt"), "Invalid checkpoint: missing ema_state_dict"

    if args.from_pretrained and os.path.isfile(args.from_pretrained) and args.from_pretrained.endswith(".ckpt"):
        assert args.model_config is not None and os.path.isfile(args.model_config), \
            "--model_config must be provided when --from_pretrained is a .ckpt file"
        vae = model_cls.from_config(args.model_config)
        checkpoint = torch.load(args.from_pretrained, map_location="cpu")
        
        # Check if EMA weights exist, prioritize EMA over non-EMA
        ema_state_dict = checkpoint.get("ema_state_dict", None)
        assert ema_state_dict is not None and len(ema_state_dict) > 0, "Invalid checkpoint: missing ema_state_dict"
        
        print("✓ Loading EMA weights from checkpoint")
        # Remove 'module.' prefix from DDP training (matching model's from_pretrained logic)
        state_dict = {key.replace("module.", ""): value for key, value in ema_state_dict.items()}
        
        try:
            missing, unexpected = vae.load_state_dict(state_dict, strict=False)
        except Exception:
            missing, unexpected = vae.module.load_state_dict(state_dict, strict=False)
        if len(unexpected) > 0:
            print(f"Warning: unexpected keys when loading ckpt: {unexpected}")
        if len(missing) > 0:
            print(f"Warning: missing keys when loading ckpt: {missing}")
    else:
        vae = model_cls.from_pretrained(args.from_pretrained)
    
    vae = vae.to(device)
    vae.eval()
    
    # IMPORTANT: Set temporal_processor to eval mode (like training validation)
    if hasattr(vae, 'temporal_processor'):
        vae.temporal_processor.eval()
        print("✓ temporal_processor set to eval mode")
    
    if args.enable_tiling:
        vae.enable_tiling()
    
    vae.set_training_phase("phase1")
    
    print(f"✓ Model loaded: {args.model_name}")
    
    # ---- Load LPIPS Model ----
    lpips_model = None
    if args.eval_lpips:
        import lpips
        lpips_model = lpips.LPIPS(net="alex", spatial=True)
        lpips_model.to(device)
        lpips_model.requires_grad_(False)
        lpips_model.eval()
        print("✓ LPIPS model loaded")
    
    # ---- Load I3D Model for rFVD ----
    i3d_model = None
    if args.eval_rfvd:
        if args.fvd_method == 'styleganv':
            from causalvideovae.eval.fvd.styleganv.fvd import load_i3d_pretrained
        else:
            from causalvideovae.eval.fvd.videogpt.fvd import load_i3d_pretrained
        i3d_model = load_i3d_pretrained(device)
        print(f"✓ I3D model loaded (method: {args.fvd_method})")
    
    # ---- Prepare Dataset ----
    if args.parquet_path:
        raise NotImplementedError("Parquet dataset support is not available. Please use --real_video_dir instead.")
    
    print(f"Using ValidVideoDataset with directory: {args.real_video_dir}")
    # Use crop_size like training (not crop_size_width/height)
    crop_size = args.crop_size if args.crop_size else None
    dataset = ValidVideoDataset(
        real_video_dir=args.real_video_dir,
        num_frames=args.num_frames,
        sample_rate=args.sample_rate,
        crop_size=crop_size,
        resolution=args.resolution,
    )
    
    if args.subset_size and args.subset_size > 0:
        indices = range(min(args.subset_size, len(dataset)))
        dataset = Subset(dataset, indices=indices)
    
    print(f"✓ Dataset loaded: {len(dataset)} videos")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        pin_memory=True, 
        num_workers=args.num_workers,
        shuffle=False
    )
    dataloader = accelerator.prepare(dataloader)
    
    # ---- Setup autocast ----
    autocast_dtype_str = args.autocast_dtype.lower()
    if autocast_dtype_str == "bf16":
        _dtype = torch.bfloat16
    elif autocast_dtype_str == "fp16":
        _dtype = torch.float16
    else:
        _dtype = None

    if _dtype is None:
        # no autocast
        _autocast_cm = nullcontext()
    else:
        _autocast_cm = torch.amp.autocast(device_type="cuda", dtype=_dtype)
    
    # ---- Setup video saving ----
    video_save_dir = None
    video_counter = 0
    if args.save_videos:
        if args.save_video_dir:
            # Priority 1: Full path specified
            video_save_dir = Path(args.save_video_dir)
        elif args.paper_video_folder:
            # Priority 2: Folder name (create in current directory)
            video_save_dir = Path(args.paper_video_folder)
        else:
            # Priority 3: Auto generate save directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_base = os.path.splitext(os.path.basename(args.from_pretrained))[0]
            video_save_dir = Path("./saved_videos") / f"{args.model_name}_{ckpt_base}_{timestamp}"
        
        video_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Videos will be saved to: {video_save_dir}")
        
        # Save evaluation config
        config_info = {
            'model_name': args.model_name,
            'from_pretrained': args.from_pretrained,
            'num_frames': args.num_frames,
            'resolution': args.resolution,
            'sample_rate': args.sample_rate,
            'max_videos_to_save': args.max_videos_to_save,
        }
        with open(video_save_dir / "config.json", 'w') as f:
            json.dump(config_info, f, indent=2)
    
    # ---- Evaluation Loop ----
    psnr_list = []
    ssim_list = []
    lpips_list = []
    flickering_list = []
    rfvd_list = []  # Collect FVD scores per batch (like eval.py)
    
    print("\n" + "="*70)
    print("Starting evaluation...")
    print("="*70)
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        # clean_inputs: 原始干净视频（作为 GT，用来算 PSNR / SSIM / LPIPS / rFVD）
        clean_inputs = batch["video"].to(device)  # [B, C, T, H, W]

        # degraded_inputs: 退化后的视频，送入 VAE
        if args.enable_degradation:
            degraded_list = []
            for b in range(clean_inputs.shape[0]):
                degraded_clip = _apply_h264_compression_to_clip(
                    clean_inputs[b],
                    qp=args.degradation_qp,
                    tmp_root=args.degradation_tmp_dir,
                    fps=args.save_fps,
                )
                degraded_list.append(degraded_clip.to(device))
            degraded_inputs = torch.stack(degraded_list, dim=0)
        else:
            degraded_inputs = clean_inputs

        # Forward pass: VAE 看到的是 degraded_inputs
        with _autocast_cm:
            output = vae(degraded_inputs)
            video_recon = output.sample  # [B, C, T, H, W]
        
        # Save videos if requested
        if args.save_videos and video_save_dir is not None:
            for i in range(clean_inputs.shape[0]):
                if video_counter >= args.max_videos_to_save:
                    break
                save_video_comparison(
                    clean_inputs[i],
                    video_recon[i], 
                    video_save_dir, 
                    video_counter,
                    fps=args.save_fps
                )
                video_counter += 1
        
        # Convert to float32 once for metrics that need it (avoid redundant conversion)
        # 这里所有指标都和 clean_inputs 对比（即原始干净视频）
        inputs_float = clean_inputs.float()
        video_recon_float = video_recon.float()
        
        # Calculate rFVD per batch (exactly like eval.py)
        if args.eval_rfvd and i3d_model is not None:
            tmp_list = list(
                calculate_fvd(
                    inputs_float, video_recon_float, device, i3d=i3d_model, method=args.fvd_method
                )["value"].values()
            )
            rfvd_list += tmp_list
        
        # Calculate SSIM (requires 5D input: [B, C, T, H, W])
        if args.compute_ssim:
            ssim_result = calculate_ssim(inputs_float, video_recon_float)
            # Extract mean SSIM values from the result dictionary
            ssim_values = list(ssim_result["value"].values())
            ssim_list.extend(ssim_values)
        
        # Reshape for per-frame metrics: [B, C, T, H, W] -> [B*T, C, H, W]
        inputs_frames = rearrange(clean_inputs, "b c t h w -> (b t) c h w").contiguous()
        recon_frames = rearrange(video_recon, "b c t h w -> (b t) c h w").contiguous()
        
        # Calculate PSNR (per-frame)
        mse = torch.mean(torch.square(inputs_frames - recon_frames), dim=(1, 2, 3))
        psnr = 20 * torch.log10(1 / torch.sqrt(mse))
        psnr_list.extend(psnr.detach().cpu().tolist())
        
        # Calculate LPIPS (per-frame)
        if args.eval_lpips and lpips_model is not None:
            lpips_score = lpips_model.forward(inputs_frames, recon_frames).mean().detach().cpu().item()
            lpips_list.append(lpips_score)
        
        # Calculate Flickering (temporal consistency)
        gvideo_dif = recon_frames[1:] - recon_frames[:-1]
        rvideo_dif = inputs_frames[1:] - inputs_frames[:-1]
        flickering = torch.abs(gvideo_dif - rvideo_dif).mean().detach().cpu().item()
        flickering_list.append(flickering)
        
        # Release GPU memory
        torch.cuda.empty_cache()
    
    # Calculate final rFVD (average of all batches, like eval.py)
    rfvd_score = None
    if args.eval_rfvd and len(rfvd_list) > 0:
        rfvd_score = np.mean(rfvd_list)
    
    # Compute final statistics
    psnr_mean = torch.tensor(psnr_list).mean().item() if len(psnr_list) > 0 else None
    psnr_std = torch.tensor(psnr_list).std().item() if len(psnr_list) > 0 else None
    
    ssim_mean = torch.tensor(ssim_list).mean().item() if len(ssim_list) > 0 else None
    ssim_std = torch.tensor(ssim_list).std().item() if len(ssim_list) > 0 else None
    
    lpips_mean = torch.tensor(lpips_list).mean().item() if len(lpips_list) > 0 else None
    lpips_std = torch.tensor(lpips_list).std().item() if len(lpips_list) > 0 else None
    
    flickering_mean = torch.tensor(flickering_list).mean().item() if len(flickering_list) > 0 else None
    flickering_std = torch.tensor(flickering_list).std().item() if len(flickering_list) > 0 else None
    
    # Final results (print and save)
    if accelerator.is_main_process:
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(f"Total videos evaluated: {len(dataloader.dataset)}")
        print(f"Total frames evaluated: {len(psnr_list)}")
        if args.save_videos:
            print(f"Videos saved: {video_counter} videos to {video_save_dir}")
        print("")
        
        # Print metrics with directional indicators
        if psnr_mean is not None:
            print(f"PSNR (↑):        {psnr_mean:.4f} ± {psnr_std:.4f}")
        if ssim_mean is not None:
            print(f"SSIM (↑):        {ssim_mean:.4f} ± {ssim_std:.4f}")
        if lpips_mean is not None:
            print(f"LPIPS (↓):       {lpips_mean:.4f} ± {lpips_std:.4f}")
        if rfvd_score is not None:
            print(f"rFVD (↓):        {rfvd_score:.4f}")
        if flickering_mean is not None:
            print(f"Flickering (↓):  {flickering_mean:.6f} ± {flickering_std:.6f}")
        print("="*70)

        results = {
            'psnr_mean': psnr_mean,
            'psnr_std': psnr_std,
            'ssim_mean': ssim_mean,
            'ssim_std': ssim_std,
            'lpips_mean': lpips_mean,
            'lpips_std': lpips_std,
            'rfvd': rfvd_score,
            'flickering_mean': flickering_mean,
            'flickering_std': flickering_std,
            'num_videos': len(dataloader.dataset),
            'num_frames': len(psnr_list),
            'config': {
                'model_name': args.model_name,
                'from_pretrained': args.from_pretrained,
                'num_frames': args.num_frames,
                'resolution': args.resolution,
                'sample_rate': args.sample_rate,
                'fvd_method': args.fvd_method if args.eval_rfvd else None,
            }
        }

        # Save to user-specified file if provided
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to: {args.output_file}")

        # Also save under metric_result directory
        out_dir = "metric_result"
        os.makedirs(out_dir, exist_ok=True)
        ckpt_base = os.path.splitext(os.path.basename(args.from_pretrained))[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_name = f"{args.model_name}_{ckpt_base}_{args.num_frames}f_sr{args.sample_rate}_rs{args.resolution}_{timestamp}.json"
        out_path = os.path.join(out_dir, auto_name)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results also saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comprehensive VAE Evaluation (PSNR, SSIM, LPIPS, rFVD)')
    parser.add_argument("--autocast_dtype", type=str, default="bf16", 
                        choices=["bf16", "fp16", "none"], 
                        help="autocast dtype for forward")
    
    # Data source
    parser.add_argument("--real_video_dir", type=str, default="", 
                        help="Directory containing video files")
    parser.add_argument("--parquet_path", type=str, default="",
                        help="Path to parquet file containing video paths")
    parser.add_argument("--video_column", type=str, default="video_path",
                        help="Column name in parquet file")
    
    # Model settings
    parser.add_argument("--from_pretrained", type=str, required=True,
                        help="Path to pretrained model or checkpoint")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model class name")
    parser.add_argument("--model_config", type=str, default=None,
                        help="Model config file (required for .ckpt files)")
    parser.add_argument('--enable_tiling', action='store_true',
                        help="Enable tiling for large videos")
    
    # Video parameters
    parser.add_argument("--resolution", type=int, default=256,
                        help="Target resolution")
    parser.add_argument("--crop_size", type=int, default=256,
                        help="Crop size (like training)")
    parser.add_argument("--num_frames", type=int, default=17,
                        help="Number of frames to sample")
    parser.add_argument("--sample_rate", type=int, default=1,
                        help="Frame sampling rate")
    
    # DataLoader settings
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of dataloader workers")
    parser.add_argument("--subset_size", type=int, default=None,
                        help="Evaluate only first N videos (for testing)")
    
    # Evaluation settings
    parser.add_argument('--compute_ssim', action='store_true',
                        help="Compute SSIM metric (↑ higher is better)")
    parser.add_argument('--eval_lpips', action='store_true',
                        help="Compute LPIPS metric (↓ lower is better)")
    parser.add_argument('--eval_rfvd', action='store_true',
                        help="Compute rFVD metric (↓ lower is better)")
    parser.add_argument('--fvd_method', type=str, default='styleganv',
                        choices=['styleganv', 'videogpt'],
                        help="FVD calculation method")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save results JSON")
    
    # Video saving settings
    parser.add_argument('--save_videos', action='store_true',
                        help="Save real and reconstructed videos for paper figures")
    parser.add_argument('--save_video_dir', type=str, default=None,
                        help="Full path to save videos (overrides --paper_video_folder)")
    parser.add_argument('--paper_video_folder', type=str, default=None,
                        help="Folder name for paper videos (created in current dir, e.g. 'paper_videos_32ch')")
    parser.add_argument('--max_videos_to_save', type=int, default=100,
                        help="Maximum number of videos to save")
    parser.add_argument('--save_fps', type=int, default=8,
                        help="FPS for saved videos")
    # Degradation settings
    parser.add_argument('--enable_degradation', action='store_true',
                        help="Apply H.264 compression degradation before feeding videos to VAE")
    parser.add_argument('--degradation_qp', type=int, default=30,
                        help="QP value for H.264 compression (higher = stronger degradation)")
    parser.add_argument('--degradation_tmp_dir', type=str, default=None,
                        help="Optional tmp directory for intermediate compressed videos")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.real_video_dir:
        parser.error("--real_video_dir must be provided")
    
    main(args)
