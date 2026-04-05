"""
Tool to create a parquet file from video files in a directory

Usage:
    python tools/create_parquet_from_videos.py \
        --video_dir /path/to/videos \
        --output_parquet /path/to/output.parquet \
        --recursive
"""

import os
import argparse
from glob import glob
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def get_video_files(video_dir, recursive=True, extensions=None):
    """
    Get all video files from a directory
    
    Args:
        video_dir: Path to the video directory
        recursive: Whether to search recursively
        extensions: List of video extensions to include (default: common video formats)
    
    Returns:
        List of video file paths
    """
    if extensions is None:
        extensions = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv']
    
    video_files = []
    
    if recursive:
        for ext in extensions:
            pattern = os.path.join(video_dir, '**', f'*.{ext}')
            video_files.extend(glob(pattern, recursive=True))
    else:
        for ext in extensions:
            pattern = os.path.join(video_dir, f'*.{ext}')
            video_files.extend(glob(pattern))
    
    return sorted(video_files)


def extract_metadata(video_path, video_dir):
    """
    Extract metadata from video path
    
    Args:
        video_path: Full path to the video file
        video_dir: Base directory for videos
    
    Returns:
        Dictionary with metadata
    """
    path_obj = Path(video_path)
    rel_path = os.path.relpath(video_path, video_dir)
    
    metadata = {
        'video_path': video_path,
        'filename': path_obj.name,
        'relative_path': rel_path,
        'extension': path_obj.suffix[1:],  # Remove the dot
    }
    
    # Try to extract label from parent directory
    parent_dir = path_obj.parent.name
    if parent_dir and parent_dir != Path(video_dir).name:
        metadata['label'] = parent_dir
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Create a parquet file from video files in a directory'
    )
    parser.add_argument(
        '--video_dir',
        type=str,
        required=True,
        help='Path to the directory containing video files'
    )
    parser.add_argument(
        '--output_parquet',
        type=str,
        required=True,
        help='Path to the output parquet file'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Search for videos recursively in subdirectories'
    )
    parser.add_argument(
        '--extensions',
        type=str,
        nargs='+',
        default=None,
        help='Video file extensions to include (default: mp4 avi mov mkv webm flv wmv)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify that video files can be opened (slower but safer)'
    )
    parser.add_argument(
        '--max_files',
        type=int,
        default=None,
        help='Maximum number of files to include (for testing)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.video_dir):
        raise ValueError(f"Video directory does not exist: {args.video_dir}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_parquet)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all video files
    print(f"Searching for videos in: {args.video_dir}")
    print(f"Recursive: {args.recursive}")
    
    video_files = get_video_files(
        args.video_dir,
        recursive=args.recursive,
        extensions=args.extensions
    )
    
    print(f"Found {len(video_files)} video files")
    
    if len(video_files) == 0:
        print("Warning: No video files found!")
        return
    
    # Limit number of files if requested
    if args.max_files is not None:
        video_files = video_files[:args.max_files]
        print(f"Limited to {len(video_files)} files")
    
    # Extract metadata
    print("Extracting metadata...")
    data_list = []
    
    for video_path in tqdm(video_files):
        metadata = extract_metadata(video_path, args.video_dir)
        
        # Verify video if requested
        if args.verify:
            try:
                import decord
                vr = decord.VideoReader(video_path)
                metadata['num_frames'] = len(vr)
                metadata['fps'] = vr.get_avg_fps()
                metadata['valid'] = True
            except Exception as e:
                print(f"Warning: Could not read {video_path}: {e}")
                metadata['valid'] = False
                metadata['num_frames'] = 0
                metadata['fps'] = 0
        
        data_list.append(metadata)
    
    # Create DataFrame
    df = pd.DataFrame(data_list)
    
    # Filter out invalid videos if verification was done
    if args.verify:
        n_invalid = (~df['valid']).sum()
        if n_invalid > 0:
            print(f"Filtering out {n_invalid} invalid videos")
            df = df[df['valid']].reset_index(drop=True)
            df = df.drop(columns=['valid'])
    
    print(f"\nDataFrame shape: {df.shape}")
    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")
    
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Save to parquet
    print(f"\nSaving to: {args.output_parquet}")
    df.to_parquet(args.output_parquet, index=False)
    
    print(f"\n✓ Successfully created parquet file with {len(df)} videos")
    
    # Print some statistics
    print("\nStatistics:")
    if 'label' in df.columns:
        print(f"  Number of unique labels: {df['label'].nunique()}")
        print(f"  Label distribution:")
        print(df['label'].value_counts().head(10))
    
    if 'num_frames' in df.columns:
        print(f"\n  Frame count statistics:")
        print(f"    Min: {df['num_frames'].min()}")
        print(f"    Max: {df['num_frames'].max()}")
        print(f"    Mean: {df['num_frames'].mean():.1f}")
        print(f"    Median: {df['num_frames'].median():.1f}")


if __name__ == '__main__':
    main()

