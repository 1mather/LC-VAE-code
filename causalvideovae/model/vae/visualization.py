import math
import os
from typing import Tuple, Dict, List, Optional
import sys
from pathlib import Path
import subprocess
import shutil

import torch
import torch.nn.functional as F

tvio = None  # do not rely on PyAV-backed torchvision io; use ffmpeg fallback instead

try:
    # When executed as part of the package
    from ..modules import (
        HaarWaveletTransform3D,
        InverseHaarWaveletTransform3D,
    )
except Exception:
    try:
        # When executed from project root with absolute import
        from causalvideovae.model.modules import (
            HaarWaveletTransform3D,
            InverseHaarWaveletTransform3D,
        )
    except Exception:
        # When executed directly as a script from this directory
        # Add project root (parent of 'causalvideovae') to sys.path
        proj_root = Path(__file__).resolve().parents[3]
        if str(proj_root) not in sys.path:
            sys.path.append(str(proj_root))
        from causalvideovae.model.modules import (
            HaarWaveletTransform3D,
            InverseHaarWaveletTransform3D,
        )


def _compute_psnr(x: torch.Tensor, y: torch.Tensor, max_val: Optional[float] = 1.0) -> float:
    mse = torch.mean((x - y) ** 2).item()
    if max_val is None:
        # infer from y range
        y_min = float(y.min().item())
        y_max = float(y.max().item())
        max_val = max(1e-6, y_max - y_min)
    return 10.0 * math.log10((max_val ** 2) / max(mse, 1e-12))


def _mask_topk_global(coeffs: torch.Tensor, keep_ratio: float) -> Tuple[torch.Tensor, int, int]:
    b, c, t, h, w = coeffs.shape
    total = c * t * h * w
    k = max(1, int(keep_ratio * total))
    flat = coeffs.view(b, -1)
    idx = torch.topk(flat.abs(), k=k, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros_like(flat)
    mask.scatter_(1, idx, 1.0)
    masked = (flat * mask).view_as(coeffs)
    return masked, k, total


def _mask_topk_per_channel(coeffs: torch.Tensor, keep_ratio: float) -> Tuple[torch.Tensor, int, int]:
    # For each channel independently, keep a fraction over (T*H*W)
    b, c, t, h, w = coeffs.shape
    per_total = t * h * w
    k_c = max(1, int(keep_ratio * per_total))
    flat = coeffs.view(b, c, -1)
    idx = torch.topk(flat.abs(), k=k_c, dim=2, largest=True, sorted=False).indices  # (B, C, k_c)
    mask = torch.zeros_like(flat)
    # Build gather index for scatter
    expand_idx = idx
    mask.scatter_(2, expand_idx, 1.0)
    masked = (flat * mask).view_as(coeffs)
    kept = k_c * c
    total = per_total * c
    return masked, kept, total


def _mask_topk_per_voxel_channel(coeffs: torch.Tensor, keep_ratio: float) -> Tuple[torch.Tensor, int, int]:
    # For each (t,h,w), keep a fraction of channels
    b, c, t, h, w = coeffs.shape
    n = t * h * w
    k_ch = max(1, int(keep_ratio * c))
    x = coeffs.view(b, c, n)  # (B, C, N)
    idx = torch.topk(x.abs(), k=k_ch, dim=1, largest=True, sorted=False).indices  # (B, k_ch, N)
    mask = torch.zeros_like(x)
    # For scatter along dim=1, idx needs shape (B, k_ch, N)
    mask.scatter_(1, idx, 1.0)
    masked = (x * mask).view_as(coeffs)
    kept = k_ch * n
    total = c * n
    return masked, kept, total


@torch.no_grad()
def two_level_dwt_topk_reconstruct(
    video: torch.Tensor,
    keep_ratio: float = 0.1,
    device: str = "cuda",
    mode: str = "global",  # 'global' | 'per_channel' | 'per_voxel_channel'
    psnr_max_val: Optional[float] = 1.0,
) -> Tuple[torch.Tensor, Dict]:
    """
    Two-level 3D Haar DWT + hard Top-K sparsification + reconstruction.

    Args:
        video: (B, C, T, H, W)
        keep_ratio: fraction in (0,1]
        device: target device
        mode: Top-K mode
        psnr_max_val: max intensity for PSNR; None to infer from data range

    Returns:
        recon, info
    """
    assert video.ndim == 5, "video must be (B, C, T, H, W)"
    assert 0 < keep_ratio <= 1.0
    assert mode in {"global", "per_channel", "per_voxel_channel"}

    video = video.to(device)

    dwt3d = HaarWaveletTransform3D().to(device)
    idwt3d = InverseHaarWaveletTransform3D().to(device)

    coeffs_lv1 = dwt3d(video)
    coeffs_lv2 = dwt3d(coeffs_lv1)

    if mode == "global":
        coeffs_lv2_masked, kept, total = _mask_topk_global(coeffs_lv2, keep_ratio)
    elif mode == "per_channel":
        coeffs_lv2_masked, kept, total = _mask_topk_per_channel(coeffs_lv2, keep_ratio)
    else:
        coeffs_lv2_masked, kept, total = _mask_topk_per_voxel_channel(coeffs_lv2, keep_ratio)

    coeffs_lv1_recon = idwt3d(coeffs_lv2_masked)
    recon = idwt3d(coeffs_lv1_recon)

    mse = torch.mean((recon - video) ** 2).item()
    psnr = _compute_psnr(recon, video, max_val=psnr_max_val)

    info = {
        "input_shape": tuple(video.shape),
        "lvl1_shape": tuple(coeffs_lv1.shape),
        "lvl2_shape": tuple(coeffs_lv2.shape),
        "keep_mode": mode,
        "kept": kept,
        "total": total,
        "keep_ratio": keep_ratio,
        "mse": mse,
        "psnr": psnr,
    }

    return recon, info


def _to_uint8_tchw(video: torch.Tensor) -> torch.Tensor:
    """
    Ensure (T, C, H, W) in [0,1] -> uint8.
    Accepts (B,C,T,H,W) with B=1 or (C,T,H,W); returns (T,C,H,W) uint8 on cpu.
    """
    if video.dim() == 5:
        assert video.shape[0] == 1, "Only batch size 1 supported for saving"
        video = video[0]
    # (C,T,H,W) -> clamp -> (T,C,H,W)
    video = video.detach().cpu()
    video = torch.clamp(video, 0.0, 1.0)
    video = (video * 255.0 + 0.5).to(torch.uint8)
    video = video.permute(1, 0, 2, 3).contiguous()
    return video


def save_video(path: str, video: torch.Tensor, fps: int = 25) -> None:
    """Save video tensor to path.
    Prefers torchvision; otherwise uses ffmpeg (rawvideo pipe). Accepts (B,C,T,H,W) or (C,T,H,W) in [0,1].
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames_tchw = _to_uint8_tchw(video)  # (T,C,H,W)
    if tvio is not None and hasattr(tvio, "write_video"):
        # torchvision expects (T, H, W, C)
        frames_thwc = frames_tchw.permute(0, 2, 3, 1).contiguous()
        try:
            tvio.write_video(path, frames_thwc, fps=fps)
            return
        except Exception:
            # PyAV missing or other error: fall back to ffmpeg
            pass

    # Fallback to ffmpeg via subprocess and rawvideo pipe
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg not found in PATH; please install or load ffmpeg.")
    T, C, H, W = frames_tchw.shape
    assert C == 3, "Expect 3-channel video for ffmpeg writer"
    frames_thwc = frames_tchw.permute(0, 2, 3, 1).contiguous()  # (T,H,W,C)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}",
        "-r", str(fps),
        "-i", "-",  # stdin
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        proc.stdin.write(frames_thwc.numpy().tobytes())
        proc.stdin.close()
        proc.wait()
    except Exception as e:
        proc.kill()
        raise RuntimeError(f"ffmpeg write failed: {e}")


@torch.no_grad()
def evaluate_dataloader_two_level(
    dataloader,
    keep_ratio: float = 0.1,
    modes: Optional[List[str]] = None,
    device: str = "cuda",
    psnr_max_val: Optional[float] = 1.0,
    max_batches: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple Top-K modes over a dataloader that yields (B,C,T,H,W) videos in [0,1].

    Returns a dict: mode -> {mse: avg, psnr: avg}
    """
    if modes is None:
        modes = ["global", "per_channel", "per_voxel_channel"]

    stats = {m: {"mse_sum": 0.0, "psnr_sum": 0.0, "count": 0} for m in modes}

    for b_idx, batch in enumerate(dataloader):
        if isinstance(batch, (list, tuple)):
            video = batch[0]
        else:
            video = batch
        video = video.to(device)

        for m in modes:
            recon, info = two_level_dwt_topk_reconstruct(
                video, keep_ratio=keep_ratio, device=device, mode=m, psnr_max_val=psnr_max_val
            )
            stats[m]["mse_sum"] += info["mse"]
            stats[m]["psnr_sum"] += info["psnr"]
            stats[m]["count"] += 1

        if (max_batches is not None) and (b_idx + 1 >= max_batches):
            break

    results: Dict[str, Dict[str, float]] = {}
    for m, s in stats.items():
        cnt = max(1, s["count"])
        results[m] = {
            "mse": s["mse_sum"] / cnt,
            "psnr": s["psnr_sum"] / cnt,
        }
    return results


if __name__ == "__main__":
    # Demo with random input and a dummy dataloader
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path("./viz_outputs").resolve()

    # Prefer loading one real video sample from dataset if available
    # Priority: VIZ_DATA_ROOT env var -> default path
    default_root = "/scratch/cs/aaltoml/users/guanjr/data/kinetics-dataset/k400/val"
    data_root = os.environ.get("VIZ_DATA_ROOT", default_root)

    loaded_from_dataset = False
    print(f"[INFO] Checking for dataset videos in: {data_root}")
    if os.path.isdir(data_root):
        # Find first video file and load
        def _list_video_files(root: str) -> List[str]:
            exts = {".mp4", ".webm", ".avi", ".mkv", ".mov"}
            paths: List[str] = []
            for r, _, files in os.walk(root):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in exts:
                        paths.append(os.path.join(r, f))
            return sorted(paths)

        paths = _list_video_files(data_root)
        print(f"[INFO] Found {len(paths)} video files")
        if len(paths) > 0:
            vid_path = paths[0]
            print(f"[INFO] Loading video: {vid_path}")
            # Decode using ffmpeg -> raw frames (fallback, no PyAV)
            ffmpeg_bin = shutil.which("ffmpeg")
            if ffmpeg_bin is None:
                print("[WARN] ffmpeg not found; cannot read dataset video without PyAV. Falling back to random.")
            else:
                # Use ffmpeg to decode to raw rgb24 pipe, then reshape
                # We'll sample first target_frames frames at ~25 fps
                target_frames = 25
                proc = subprocess.Popen([
                    ffmpeg_bin,
                    "-i", vid_path,
                    "-vf", f"fps=25,scale=256:256",
                    "-vframes", str(target_frames),
                    "-f", "rawvideo",
                    "-pix_fmt", "rgb24",
                    "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                raw = proc.stdout.read()
                stderr_out = proc.stderr.read()
                proc.stdout.close()
                proc.stderr.close()
                proc.wait()
                if len(raw) > 0:
                    print(f"[INFO] ffmpeg decoded {len(raw)} bytes")
                    import numpy as _np
                    try:
                        frames = _np.frombuffer(raw, dtype=_np.uint8)
                        # Infer number of frames by dividing by per-frame size (256*256*3)
                        per_frame = 256*256*3
                        n = frames.size // per_frame
                        frames = frames[:n*per_frame].reshape(n, 256, 256, 3)
                        # To torch [0,1]
                        video_thwc = torch.from_numpy(frames).float() / 255.0  # (T,H,W,C)
                        # Pad to target_frames if decoded fewer
                        if n < target_frames and n > 0:
                            pad = target_frames - n
                            last = video_thwc[-1:].expand(pad, -1, -1, -1)
                            video_thwc = torch.cat([video_thwc, last], dim=0)
                        # To (1,C,T,H,W)
                        video_tchw = video_thwc.permute(0, 3, 1, 2).contiguous()
                        video = video_tchw.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
                        loaded_from_dataset = True
                        print(f"[INFO] Successfully loaded video from dataset: shape={tuple(video.shape)}")
                    except Exception as e:
                        print(f"[WARN] Failed to parse ffmpeg output: {e}")
                else:
                    print(f"[WARN] ffmpeg returned empty output. stderr: {stderr_out.decode('utf-8', errors='ignore')[:500]}")
    else:
        print(f"[WARN] Data root not found: {data_root}")

    if not loaded_from_dataset:
        # Fallback to random tensor if dataset video unavailable
        print("[INFO] Using random noise as fallback")
        video = torch.rand(1, 3, 25, 256, 256)

    for mode in ["global", "per_channel", "per_voxel_channel"]:
        recon, info = two_level_dwt_topk_reconstruct(video, keep_ratio=0.1, device=device, mode=mode)
        print(f"mode={mode} -> info:", info)
        save_video(str(out_dir / f"input_{mode}.mp4"), video, fps=25)
        save_video(str(out_dir / f"recon_{mode}.mp4"), recon, fps=25)

    # Dataloader demo (random)
    class _DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4
        def __getitem__(self, idx):
            return torch.rand(1, 3, 25, 128, 128).squeeze(0)

    loader = torch.utils.data.DataLoader(_DummyDataset(), batch_size=1, shuffle=False)
    results = evaluate_dataloader_two_level(loader, keep_ratio=0.1, device=device, max_batches=2)
    print("evaluate_dataloader_two_level:", results)

    # Kinetics-400 val loader from cluster path (if available)

    def _list_video_files(root: str) -> List[str]:
        exts = {".mp4", ".webm", ".avi", ".mkv", ".mov"}
        paths: List[str] = []
        for r, _, files in os.walk(root):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in exts:
                    paths.append(os.path.join(r, f))
        return sorted(paths)

    class _KineticsValDataset(torch.utils.data.Dataset):
        def __init__(self, root: str, target_frames: int = 25, size: Tuple[int, int] = (256, 256)):
            self.root = root
            self.size = size
            self.target_frames = target_frames
            self.paths = _list_video_files(root)
            if len(self.paths) == 0:
                print(f"[WARN] No videos found under {root}")

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            path = self.paths[idx]
            if tvio is None:
                raise RuntimeError("torchvision.io is required to load videos.")
            try:
                video, _, info = tvio.read_video(path, pts_unit="sec")  # (T, H, W, C), uint8
            except Exception as e:
                # Fallback to a random tensor if decode fails to keep iteration running
                print(f"[WARN] Failed to read {path}: {e}")
                return torch.rand(3, self.target_frames, self.size[0], self.size[1])

            if video.numel() == 0:
                return torch.rand(3, self.target_frames, self.size[0], self.size[1])

            # Normalize to [0,1]
            video = video.float() / 255.0  # (T,H,W,C)

            # Temporal sampling to target_frames (uniform)
            t = video.shape[0]
            if t >= self.target_frames:
                idxs = torch.linspace(0, t - 1, steps=self.target_frames).round().long()
                video = video.index_select(0, idxs)
            else:
                # Repeat last frame
                pad = self.target_frames - t
                last = video[-1:].expand(pad, -1, -1, -1)
                video = torch.cat([video, last], dim=0)

            # To (T,C,H,W)
            video = video.permute(0, 3, 1, 2).contiguous()
            # Resize spatial to size
            video = F.interpolate(video, size=self.size, mode="bilinear", align_corners=False)
            # To (C,T,H,W)
            video = video.permute(1, 0, 2, 3).contiguous()
            return video



