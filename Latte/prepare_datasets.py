#!/usr/bin/env python3
"""
数据集下载和准备脚本
支持: UCF101, SkyTimelapse, FaceForensics, Taichi-HD
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def download_ucf101(data_dir):
    """下载 UCF101 数据集"""
    print("\n" + "="*60)
    print("Downloading UCF101 Dataset")
    print("="*60)
    
    ucf_dir = Path(data_dir) / "UCF101"
    ucf_dir.mkdir(parents=True, exist_ok=True)
    
    rar_file = ucf_dir / "UCF101.rar"
    
    # 下载
    if not rar_file.exists():
        print("\n📥 Downloading UCF101.rar (6.5 GB)...")
        url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
        cmd = f"wget --no-check-certificate -P {ucf_dir} {url}"
        subprocess.run(cmd, shell=True, check=True)
        print("✓ Download completed!")
    else:
        print(f"✓ {rar_file} already exists")
    
    # 解压
    extracted_dir = ucf_dir / "UCF-101"
    if not extracted_dir.exists():
        print("\n📦 Extracting UCF101.rar...")
        cmd = f"cd {ucf_dir} && unrar x UCF101.rar"
        subprocess.run(cmd, shell=True, check=True)
        print("✓ Extraction completed!")
    else:
        print(f"✓ {extracted_dir} already exists")
    
    print(f"\n✅ UCF101 dataset ready at: {extracted_dir}")
    return extracted_dir

def download_sky_timelapse(data_dir):
    """下载 SkyTimelapse 数据集"""
    print("\n" + "="*60)
    print("Downloading SkyTimelapse Dataset")
    print("="*60)
    
    sky_dir = Path(data_dir) / "SkyTimelapse"
    
    if sky_dir.exists():
        print(f"✓ {sky_dir} already exists")
        return sky_dir
    
    print("\n📥 Cloning from HuggingFace...")
    # 正确的克隆命令
    cmd = f"git clone https://huggingface.co/datasets/maxin-cn/SkyTimelapse {sky_dir}"
    subprocess.run(cmd, shell=True, check=True)
    
    print(f"\n✅ SkyTimelapse dataset ready at: {sky_dir}")
    return sky_dir

def download_face_forensics(data_dir):
    """下载 FaceForensics 数据集"""
    print("\n" + "="*60)
    print("Downloading FaceForensics Dataset")
    print("="*60)
    
    ff_dir = Path(data_dir) / "FaceForensics"
    
    if ff_dir.exists():
        print(f"✓ {ff_dir} already exists")
        return ff_dir
    
    print("\n📥 Cloning from HuggingFace...")
    cmd = f"git clone https://huggingface.co/datasets/maxin-cn/FaceForensics {ff_dir}"
    subprocess.run(cmd, shell=True, check=True)
    
    print(f"\n✅ FaceForensics dataset ready at: {ff_dir}")
    return ff_dir

def download_taichi(data_dir):
    """下载 Taichi-HD 数据集"""
    print("\n" + "="*60)
    print("Downloading Taichi-HD Dataset")
    print("="*60)
    
    taichi_dir = Path(data_dir) / "Taichi-HD"
    
    if taichi_dir.exists():
        print(f"✓ {taichi_dir} already exists")
        return taichi_dir
    
    print("\n📥 Cloning from HuggingFace...")
    cmd = f"git clone https://huggingface.co/datasets/maxin-cn/Taichi-HD {taichi_dir}"
    subprocess.run(cmd, shell=True, check=True)
    
    print(f"\n✅ Taichi-HD dataset ready at: {taichi_dir}")
    return taichi_dir

def main():
    parser = argparse.ArgumentParser(description="Download datasets for Latte")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ucf101", "sky", "ffs", "taichi", "all"],
        default="ucf101",
        help="Which dataset to download"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./datasets",
        help="Directory to save datasets"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Latte Dataset Preparation Tool")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Save to: {args.data_dir}")
    
    try:
        if args.dataset == "ucf101" or args.dataset == "all":
            download_ucf101(args.data_dir)
        
        if args.dataset == "sky" or args.dataset == "all":
            download_sky_timelapse(args.data_dir)
        
        if args.dataset == "ffs" or args.dataset == "all":
            download_face_forensics(args.data_dir)
        
        if args.dataset == "taichi" or args.dataset == "all":
            download_taichi(args.data_dir)
        
        print("\n" + "="*60)
        print("✅ All requested datasets are ready!")
        print("="*60)
        print("\nNext steps:")
        print("1. Update dataset paths in config files (configs/*/)")
        print("2. Run training: python train.py --config configs/xxx/xxx_train.yaml")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("- For UCF101: Install unrar (sudo apt-get install unrar)")
        print("- For HuggingFace datasets: Check network connection")
        print("- SSL errors: Use wget --no-check-certificate")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()




