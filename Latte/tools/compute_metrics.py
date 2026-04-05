#!/usr/bin/env python3
"""
Compute Inception Score (IS) and Fréchet Video Distance (FVD) for video datasets.
This script compares generated videos against real videos using the first 16 frames.
"""

import sys
import os
# Ensure the repository's Latte directory is on sys.path so `tools.*` imports resolve
script_dir = os.path.dirname(os.path.abspath(__file__))
latte_dir = os.path.dirname(script_dir)
if latte_dir not in sys.path:
    sys.path.insert(0, latte_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import argparse
import glob
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
import shutil

# Import metric computation modules
from tools.metrics import metric_utils
from tools.metrics.video_inception_score import compute_isv
from tools.metrics.frechet_video_distance import compute_fvd


def find_videos_recursive(root_dir: str, extensions: List[str] = ['.mp4', '.avi', '.mov', '.mkv']) -> List[str]:
    """
    Recursively find all video files in a directory and its subdirectories.
    
    Args:
        root_dir: Root directory to search
        extensions: List of video file extensions to look for
    
    Returns:
        List of video file paths
    """
    video_files = []
    for ext in extensions:
        pattern = os.path.join(root_dir, '**', f'*{ext}')
        video_files.extend(glob.glob(pattern, recursive=True))
    
    return sorted(video_files)


def extract_frames_from_video(video_path: str, num_frames: int = 16, target_size: Tuple[int, int] = None) -> np.ndarray:
    """
    Extract the first N frames from a video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (default: 16)
        target_size: Optional tuple (width, height) to resize frames
    
    Returns:
        numpy array of shape (num_frames, height, width, 3) or None if failed
    """
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if target_size is not None:
                frame = cv2.resize(frame, target_size)
            
            frames.append(frame)
        
        cap.release()
        
        if len(frames) < num_frames:
            print(f"Warning: {video_path} has only {len(frames)} frames, expected {num_frames}")
            return None
        
        return np.stack(frames, axis=0)
    
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return None


def process_single_video(args_tuple):
    """
    Process a single video and extract frames. This function is designed to be called
    by multiprocessing workers.
    
    Args:
        args_tuple: Tuple of (video_path, output_dir, video_dir, num_frames, resolution)
    
    Returns:
        Tuple of (success: bool, video_path: str, message: str)
    """
    video_path, output_dir, video_dir, num_frames, resolution = args_tuple
    
    try:
        # Create unique folder name for this video
        # Use only the video filename (stem) to flatten the directory structure
        video_name = Path(video_path).stem
        
        # Flatten structure: all video folders directly under output_dir
        video_folder = os.path.join(output_dir, video_name)
        
        # Skip if already processed and verify the frames are valid
        if os.path.exists(video_folder):
            existing_frames = [f for f in os.listdir(video_folder) if f.endswith('.png')]
            if len(existing_frames) >= num_frames:
                # Verify frames can be read to ensure they're not corrupted
                try:
                    from PIL import Image
                    # Check first few frames to ensure they're valid
                    frames_to_check = sorted(existing_frames)[:min(3, num_frames)]
                    for frame_file in frames_to_check:
                        frame_path = os.path.join(video_folder, frame_file)
                        img = Image.open(frame_path)
                        img.load()  # Force loading of image data
                        img.close()
                    return (True, video_path, "Already processed")
                except Exception:
                    # Frames are corrupted, remove and re-extract
                    shutil.rmtree(video_folder, ignore_errors=True)
        
        os.makedirs(video_folder, exist_ok=True)
        
        # Extract frames
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # OpenCV returns frames in BGR; metrics pipeline expects RGB when reading via PIL later.
            # If we save BGR directly, PIL will interpret it as RGB (R/B swapped), corrupting IS/FVD.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Center-crop to square (keeps aspect ratio) then resize to target resolution.
            if resolution is not None:
                h, w = frame.shape[:2]
                min_hw = min(h, w)
                top = (h - min_hw) // 2
                left = (w - min_hw) // 2
                frame = frame[top:top + min_hw, left:left + min_hw]
                frame = cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_AREA)
            
            # Save frame as image
            frame_path = os.path.join(video_folder, f"{i:04d}.png")
            success = cv2.imwrite(frame_path, frame)
            
            if not success:
                cap.release()
                shutil.rmtree(video_folder, ignore_errors=True)
                return (False, video_path, f"Failed to write frame {i}")
            
            # Verify the saved image can be read
            if not os.path.exists(frame_path) or os.path.getsize(frame_path) == 0:
                cap.release()
                shutil.rmtree(video_folder, ignore_errors=True)
                return (False, video_path, f"Frame {i} is empty or corrupted")
            
            frame_idx += 1
        
        cap.release()
        
        if frame_idx >= num_frames:
            return (True, video_path, f"Extracted {frame_idx} frames")
        else:
            # Remove incomplete folder
            shutil.rmtree(video_folder, ignore_errors=True)
            return (False, video_path, f"Only {frame_idx} frames available, expected {num_frames}")
    
    except Exception as e:
        # Clean up on error
        if 'video_folder' in locals():
            shutil.rmtree(video_folder, ignore_errors=True)
        return (False, video_path, f"Error: {str(e)}")


def save_videos_as_frames(video_dir: str, output_dir: str, max_videos: int = 2000, num_frames: int = 16, resolution: int = None, num_workers: int = None, skip_existing: bool = True):
    """
    Convert videos to frame folders for metric computation using multiprocessing.
    
    Args:
        video_dir: Directory containing videos (can have subdirectories)
        output_dir: Directory to save extracted frames
        max_videos: Maximum number of videos to process
        num_frames: Number of frames to extract from each video
        resolution: Target resolution (width and height) for frames
        num_workers: Number of CPU workers for parallel processing (default: cpu_count())
        skip_existing: Skip conversion if output directory already has enough videos (default: True)
    
    Returns:
        Number of successfully processed videos
    """
    if num_workers is None:
        num_workers = min(cpu_count(), 16)  # Cap at 16 to avoid overwhelming the system
    
    # Check if conversion has already been done
    if skip_existing and os.path.exists(output_dir):
        existing_video_folders = [d for d in os.listdir(output_dir) 
                                 if os.path.isdir(os.path.join(output_dir, d))]
        
        # Count valid video folders (those with enough frames)
        valid_count = 0
        for folder in existing_video_folders:
            folder_path = os.path.join(output_dir, folder)
            frame_files = [f for f in os.listdir(folder_path) 
                          if f.endswith('.png') or f.endswith('.jpg')]
            if len(frame_files) >= num_frames:
                valid_count += 1
        
        if valid_count >= max_videos:
            print(f"Found {valid_count} already processed videos in {output_dir}")
            print("Skipping frame extraction (use skip_existing=False to force re-extraction)")
            return valid_count
        elif valid_count > 0:
            print(f"Found {valid_count} already processed videos, will process remaining videos")
    
    print(f"Searching for videos in {video_dir}...")
    video_files = find_videos_recursive(video_dir)
    
    if len(video_files) == 0:
        raise ValueError(f"No video files found in {video_dir}")
    
    print(f"Found {len(video_files)} video files")
    
    # Limit to max_videos
    if len(video_files) > max_videos:
        print(f"Limiting to {max_videos} videos")
        video_files = video_files[:max_videos]
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Using {num_workers} CPU workers for parallel video decoding")
    if resolution:
        print(f"Resizing frames to {resolution}x{resolution}")
    
    # Prepare arguments for multiprocessing
    process_args = [(video_path, output_dir, video_dir, num_frames, resolution) for video_path in video_files]
    
    # Process videos in parallel
    processed_count = 0
    failed_videos = []
    
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for better performance with progress bar
        results = list(tqdm(
            pool.imap_unordered(process_single_video, process_args),
            total=len(video_files),
            desc="Processing videos"
        ))
    
    # Count successful processes and collect failures
    for success, video_path, message in results:
        if success:
            processed_count += 1
        else:
            failed_videos.append((video_path, message))
    
    # Print summary
    print(f"\nSuccessfully processed {processed_count} videos")
    
    if failed_videos:
        print(f"Failed to process {len(failed_videos)} videos:")
        for video_path, message in failed_videos[:10]:  # Show first 10 failures
            print(f"  - {video_path}: {message}")
        if len(failed_videos) > 10:
            print(f"  ... and {len(failed_videos) - 10} more")
    
    return processed_count


def cleanup_corrupted_frames(frames_dir: str, num_frames: int = 16):
    """
    Clean up corrupted or incomplete video frame folders.
    
    Args:
        frames_dir: Directory containing video frame folders
        num_frames: Expected number of frames per video
    
    Returns:
        Number of folders removed
    """
    if not os.path.exists(frames_dir):
        return 0
    
    from PIL import Image
    removed_count = 0
    video_folders = [d for d in os.listdir(frames_dir) 
                    if os.path.isdir(os.path.join(frames_dir, d))]
    
    print(f"Checking {len(video_folders)} video folders for corruption...")
    
    for folder in tqdm(video_folders, desc="Validating frames"):
        folder_path = os.path.join(frames_dir, folder)
        frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
        
        should_remove = False
        
        # Check if folder has enough frames
        if len(frame_files) < num_frames:
            should_remove = True
        else:
            # Verify ALL frames to ensure they're not corrupted
            # This is more thorough than just checking a sample
            for frame_file in frame_files[:num_frames]:
                frame_path = os.path.join(folder_path, frame_file)
                try:
                    # Check file size first (quick check)
                    if os.path.getsize(frame_path) == 0:
                        should_remove = True
                        break
                    
                    # Actually load the image data (not just verify header)
                    # This catches corruption that verify() misses
                    img = Image.open(frame_path)
                    img.load()  # Force loading of image data
                    arr = np.array(img)  # Convert to array to fully validate
                    img.close()
                    
                    # Check that the array has the expected shape
                    if arr.size == 0 or len(arr.shape) != 3:
                        should_remove = True
                        break
                        
                except Exception as e:
                    should_remove = True
                    break
        
        if should_remove:
            shutil.rmtree(folder_path, ignore_errors=True)
            removed_count += 1
    
    if removed_count > 0:
        print(f"Removed {removed_count} corrupted or incomplete video folders")
    else:
        print("All video folders are valid")
    
    return removed_count


def compute_metrics(real_video_dir: str, 
                   generated_video_dir: str,
                   max_videos: int = 2000,
                   num_frames: int = 16,
                   resolution: int = 256,
                   num_gpus: int = 1,
                   num_workers: int = None,
                   skip_existing: bool = True,
                   enable_is: bool = True,
                   enable_fvd: bool = True,
                   fvd_real_subsample_factor: int = 1,
                   fvd_gen_subsample_factor: int = 1):
    """
    Compute IS and FVD metrics between real and generated videos.
    
    Args:
        real_video_dir: Directory containing real videos
        generated_video_dir: Directory containing generated videos
        max_videos: Maximum number of videos to use (default: 2000)
        num_frames: Number of frames to use from each video (default: 16)
        resolution: Resolution for processing (default: 256)
        num_gpus: Number of GPUs to use (default: 1)
        num_workers: Number of CPU workers for video decoding (default: cpu_count())
        skip_existing: Skip frame extraction if already done (default: True)
        enable_is: Whether to compute Inception Score (default: True)
        enable_fvd: Whether to compute FVD (default: True)
    """
    print("="*80)
    print("Video Metrics Computation")
    print("="*80)
    
    # Create temporary directories for frame extraction
    temp_dir = "./temp_frames_for_metrics"
    real_frames_dir = os.path.join(temp_dir, "real")
    gen_frames_dir = os.path.join(temp_dir, "generated")
    
    # Extract frames from videos
    print("\n" + "="*80)
    print("Step 1: Extracting frames from real videos")
    print("="*80)
    real_count = save_videos_as_frames(real_video_dir, real_frames_dir, max_videos, num_frames, resolution, num_workers, skip_existing)
    
    print("\n" + "="*80)
    print("Step 2: Extracting frames from generated videos")
    print("="*80)
    gen_count = save_videos_as_frames(generated_video_dir, gen_frames_dir, max_videos, num_frames, resolution, num_workers, skip_existing)
    
    print("\n" + "="*80)
    print(f"Frame extraction complete:")
    print(f"  Real videos: {real_count}")
    print(f"  Generated videos: {gen_count}")
    print("="*80)
    
    # Clean up any corrupted frames before computing metrics
    print("\n" + "="*80)
    print("Validating extracted frames...")
    print("="*80)
    cleanup_corrupted_frames(real_frames_dir, num_frames)
    cleanup_corrupted_frames(gen_frames_dir, num_frames)
    
    # Import required modules for metric computation
    from tools import dnnlib
    from omegaconf import OmegaConf
    
    # Setup configuration
    dummy_dataset_cfg = OmegaConf.create({'max_num_frames': 10000})
    
    # Dataset options for real data
    dataset_kwargs = dnnlib.EasyDict(
        class_name='tools.utils.dataset.VideoFramesFolderDataset',
        path=real_frames_dir,
        cfg=dummy_dataset_cfg,
        xflip=False,
        resolution=resolution,
        use_labels=False,
        load_n_consecutive=num_frames,
        subsample_factor=1,
        discard_short_videos=True,
    )
    
    # Dataset options for generated data
    gen_dataset_kwargs = dnnlib.EasyDict(
        class_name='tools.utils.dataset.VideoFramesFolderDataset',
        path=gen_frames_dir,
        cfg=dummy_dataset_cfg,
        xflip=False,
        resolution=resolution,
        use_labels=False,
        load_n_consecutive=num_frames,
        subsample_factor=1,
        discard_short_videos=True,
    )
    
    # Create options object
    opts = metric_utils.MetricOptions(
        dataset_kwargs=dataset_kwargs,
        gen_dataset_kwargs=gen_dataset_kwargs,
        generator_as_dataset=True,
        num_gpus=num_gpus,
        rank=0,
        device=torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu'),
        cache=False,
    )
    
    results = {}
    
    # Compute Inception Score
    if enable_is:
        print("\n" + "="*80)
        print("Step 3: Computing Inception Score (IS)")
        print("="*80)
        try:
            # Debug: Verify the dataset configuration
            print(f"IS Configuration:")
            print(f"  Using generator_as_dataset: {opts.generator_as_dataset}")
            print(f"  Generated dataset path: {opts.gen_dataset_kwargs.path}")
            print(f"  Number of generated videos: {gen_count}")
            print(f"  Max videos for IS: {min(gen_count, max_videos)}")
            print(f"  Resolution: {resolution}")
            print(f"  Num frames: {num_frames}")
            
            # Verify the generated frames directory exists and has videos
            if os.path.exists(opts.gen_dataset_kwargs.path):
                gen_video_folders = [d for d in os.listdir(opts.gen_dataset_kwargs.path) 
                                    if os.path.isdir(os.path.join(opts.gen_dataset_kwargs.path, d))]
                print(f"  Found {len(gen_video_folders)} video folders in generated frames directory")
                if len(gen_video_folders) > 0:
                    # Check a sample folder
                    sample_folder = os.path.join(opts.gen_dataset_kwargs.path, gen_video_folders[0])
                    sample_frames = [f for f in os.listdir(sample_folder) if f.endswith('.png')]
                    print(f"  Sample folder '{gen_video_folders[0]}' has {len(sample_frames)} frames")
            else:
                print(f"  WARNING: Generated frames directory does not exist: {opts.gen_dataset_kwargs.path}")
            
            is_mean, is_std = compute_isv(
                opts=opts,
                num_gen=min(gen_count, max_videos),
                num_splits=10,
                backbone='c3d_ucf101'
            )
            results['IS_mean'] = is_mean
            results['IS_std'] = is_std
            print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
            
            # Additional debug info
            if is_mean < 10:
                print(f"\nWARNING: IS score is very low ({is_mean:.4f}). This might indicate:")
                print(f"  - Generated videos are low quality")
                print(f"  - Dataset loading issue (check if correct path is used)")
                print(f"  - Frame preprocessing/normalization issue")
                print(f"  - Not enough diversity in generated videos")
        except Exception as e:
            print(f"Error computing IS: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Compute FVD
    if enable_fvd:
        print("\n" + "="*80)
        print("Step 4: Computing Fréchet Video Distance (FVD)")
        print("="*80)
        try:
            fvd_score = compute_fvd(
                opts=opts,
                max_real=min(real_count, max_videos),
                num_gen=min(gen_count, max_videos),
                num_frames=num_frames,
                realdata_subsample_factor=fvd_real_subsample_factor,
                gendata_subsample_factor=fvd_gen_subsample_factor
            )
            results['FVD'] = fvd_score
            print(f"FVD Score: {fvd_score:.4f}")
        except Exception as e:
            print(f"Error computing FVD: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Print final results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    if 'IS_mean' in results:
        print(f"Inception Score (IS): {results['IS_mean']:.4f} ± {results['IS_std']:.4f}")
    if 'FVD' in results:
        print(f"Fréchet Video Distance (FVD): {results['FVD']:.4f}")
    print("="*80)
    
    # Clean up temporary frame directories
    print("\n" + "="*80)
    print("Cleaning up temporary frame directories...")
    print("="*80)
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Successfully deleted temporary directory: {temp_dir}")
        else:
            print(f"Temporary directory does not exist: {temp_dir}")
    except Exception as e:
        print(f"Warning: Failed to delete temporary directory {temp_dir}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compute IS and FVD metrics for video datasets')
    parser.add_argument('--real_video_dir', type=str, default='/scratch/cs/vidgen/guanjr/UCF-101-test-2048',
                        help='Path to directory containing real videos')
    parser.add_argument('--generated_video_dir', type=str, default='/scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/results/channel4_pretrain_32_100k/validation_samples/step_0095001/generated',
                        help='Path to directory containing generated videos')
    parser.add_argument('--max_videos', type=int, default=2000,
                        help='Maximum number of videos to use (default: 1000)')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames to extract from each video (default: 16)')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Resolution for processing (default: 256)')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to use (default: 1)')
    parser.add_argument('--num_workers', type=int, default=64,
                        help='Number of CPU workers for video decoding (default: auto-detect)')
    parser.add_argument('--fvd_real_subsample_factor', type=int, default=1,
                        help='Temporal subsample factor for REAL videos when computing FVD (default: 1)')
    parser.add_argument('--fvd_gen_subsample_factor', type=int, default=1,
                        help='Temporal subsample factor for GENERATED videos when computing FVD (default: 1)')
    parser.add_argument('--force_reextract', action='store_true',
                        help='Force re-extraction of frames even if they already exist')
    parser.add_argument('--no_is', action='store_true',
                        help='Skip Inception Score computation')
    parser.add_argument('--no_fvd', action='store_true',
                        help='Skip FVD computation')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.real_video_dir):
        raise ValueError(f"Real video directory does not exist: {args.real_video_dir}")
    if not os.path.exists(args.generated_video_dir):
        raise ValueError(f"Generated video directory does not exist: {args.generated_video_dir}")
    
    # Compute metrics
    results = compute_metrics(
        real_video_dir=args.real_video_dir,
        generated_video_dir=args.generated_video_dir,
        max_videos=args.max_videos,
        num_frames=args.num_frames,
        resolution=args.resolution,
        num_gpus=args.num_gpus,
        num_workers=args.num_workers,
        skip_existing=not args.force_reextract,
        enable_is=not args.no_is,
        enable_fvd=not args.no_fvd,
        fvd_real_subsample_factor=args.fvd_real_subsample_factor,
        fvd_gen_subsample_factor=args.fvd_gen_subsample_factor,
    )
    
    return results


if __name__ == "__main__":
    main()
