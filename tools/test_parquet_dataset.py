"""
Minimal test script for ParquetVideoDataset

Creates a simple parquet file and tests the dataset functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import tempfile
from pathlib import Path


def create_test_parquet():
    """Create a test parquet file with dummy video paths"""
    # This would normally contain actual video paths
    # For testing purposes, you need to replace these with real video paths
    
    data = {
        'video_path': [
            '/path/to/video1.mp4',  # Replace with actual paths
            '/path/to/video2.mp4',
            '/path/to/video3.mp4',
        ],
        'label': ['cat', 'dog', 'bird'],
        'duration': [10.5, 15.2, 8.3],
    }
    
    df = pd.DataFrame(data)
    
    # Create temporary parquet file
    temp_file = tempfile.NamedTemporaryFile(
        mode='w', 
        suffix='.parquet', 
        delete=False
    )
    parquet_path = temp_file.name
    temp_file.close()
    
    df.to_parquet(parquet_path, index=False)
    print(f"Created test parquet file: {parquet_path}")
    print(f"\nContents:")
    print(df)
    
    return parquet_path


def test_dataset_creation(parquet_path):
    """Test creating the dataset"""
    try:
        from causalvideovae.dataset.video_dataset import ParquetVideoDataset
        
        print("\n" + "="*50)
        print("Testing ParquetVideoDataset creation...")
        print("="*50)
        
        dataset = ParquetVideoDataset(
            parquet_path=parquet_path,
            sequence_length=16,
            resolution=256,
            sample_rate=4,
            dynamic_sample=True,
            video_column='video_path',
            train=True,
        )
        
        print(f"✓ Dataset created successfully")
        print(f"  - Length: {len(dataset)}")
        print(f"  - Sequence length: {dataset.sequence_length}")
        print(f"  - Resolution: {dataset.resolution}")
        print(f"  - Sample rate: {dataset.sample_rate}")
        print(f"  - Dynamic sample: {dataset.dynamic_sample}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("ParquetVideoDataset Test Suite")
    print("="*50)
    
    # Create test parquet
    parquet_path = create_test_parquet()
    
    # Test dataset creation
    success = test_dataset_creation(parquet_path)
    
    # Cleanup
    try:
        os.unlink(parquet_path)
        print(f"\nCleaned up test file: {parquet_path}")
    except:
        pass
    
    print("\n" + "="*50)
    if success:
        print("✓ All tests passed!")
        print("\nTo use with real videos:")
        print("1. Create parquet file:")
        print("   python tools/create_parquet_from_videos.py \\")
        print("       --video_dir /path/to/videos \\")
        print("       --output_parquet /path/to/output.parquet \\")
        print("       --recursive")
        print("\n2. Use the dataset:")
        print("   python examples/parquet_dataset_example.py \\")
        print("       --parquet_path /path/to/output.parquet")
    else:
        print("✗ Tests failed")
    print("="*50)


if __name__ == '__main__':
    main()

