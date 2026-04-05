"""
Example script for using ParquetVideoDataset

Usage:
    python examples/parquet_dataset_example.py --parquet_path /path/to/videos.parquet
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from torch.utils.data import DataLoader
from causalvideovae.dataset.video_dataset import ParquetVideoDataset


def main():
    parser = argparse.ArgumentParser(description='Test ParquetVideoDataset')
    parser.add_argument('--parquet_path', type=str, required=True,
                        help='Path to the parquet file containing video paths')
    parser.add_argument('--video_column', type=str, default='video_path',
                        help='Name of the column containing video paths')
    parser.add_argument('--sequence_length', type=int, default=16,
                        help='Number of frames to sample from each video')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Target resolution for videos')
    parser.add_argument('--sample_rate', type=int, default=4,
                        help='Maximum frame sampling rate (stride)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for data loader')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--dynamic_sample', action='store_true',
                        help='Enable dynamic sampling (random stride)')
    
    args = parser.parse_args()
    
    # Create dataset
    print("Creating ParquetVideoDataset...")
    dataset = ParquetVideoDataset(
        parquet_path=args.parquet_path,
        sequence_length=args.sequence_length,
        resolution=args.resolution,
        sample_rate=args.sample_rate,
        dynamic_sample=args.dynamic_sample,
        video_column=args.video_column,
        train=True,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Test loading a few batches
    print("\nTesting data loading...")
    for i, batch in enumerate(dataloader):
        video = batch['video']
        label = batch['label']
        
        print(f"\nBatch {i + 1}:")
        print(f"  Video shape: {video.shape}")  # Should be (B, C, T, H, W)
        print(f"  Video dtype: {video.dtype}")
        print(f"  Video range: [{video.min():.3f}, {video.max():.3f}]")
        print(f"  Metadata keys: {label.keys() if isinstance(label, dict) else 'N/A'}")
        
        if i >= 2:  # Test only 3 batches
            break
    
    print("\n✓ ParquetVideoDataset test completed successfully!")


if __name__ == '__main__':
    main()

