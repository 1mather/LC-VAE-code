import os
import re
import json
import time
import torch
import decord
import torchvision
import numpy as np


from PIL import Image
from einops import rearrange
from typing import Dict, List, Tuple
from tqdm import tqdm

class_labels_map = None
cls_sample_cnt = None
import random
import argparse

class_labels_map = None
cls_sample_cnt = None

def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames


def get_filelist(file_path, target_video_len, cache_dir=None):
    """
    Get list of valid video files with caching support for multi-GPU training.
    
    Args:
        file_path: Root directory containing videos
        target_video_len: Minimum number of frames required
        cache_dir: Directory to store cache file (default: same as file_path)
    """
    # Generate cache file path
    if cache_dir is None:
        cache_dir = file_path
    os.makedirs(cache_dir, exist_ok=True)
    
    # Use absolute path for cache file name to avoid conflicts
    cache_file = os.path.join(cache_dir, f'.video_list_cache_{target_video_len}frames.txt')
    
    # Try to load from cache first
    if os.path.exists(cache_file):
        print(f"Loading video list from cache: {cache_file}")
        with open(cache_file, 'r') as f:
            Filelist = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(Filelist)} videos from cache")
        return Filelist
    
    # If cache doesn't exist, scan videos (only main process should do this in DDP)
    import torch.distributed as dist
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    
    if rank == 0:
        print(f"Cache not found, scanning videos in {file_path}...")
        
        # First, collect all file paths
        all_files = []
        for home, dirs, files in os.walk(file_path):
            for filename in files:
                # Only include video file extensions
                if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                    full_path = os.path.join(home, filename)
                    all_files.append(full_path)
        
        # Use Decord to check video length efficiently without loading full video
        Filelist = []
        print(f"Scanning {len(all_files)} video files for length >= {target_video_len} frames...")
        
        # Save cache periodically to avoid losing progress on long scans
        save_interval = max(1000, len(all_files) // 20)  # Save every 5% or every 1000 files
        
        for idx, video_path in enumerate(tqdm(all_files, desc="Checking videos")):
            try:
                # Use decord to quickly get video length without loading frames
                vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
                if len(vr) >= target_video_len:
                    Filelist.append(video_path)
                del vr  # Explicitly release video reader
            except Exception as e:
                # Skip files that can't be read
                continue
            
            # Periodically save cache to avoid losing progress
            if (idx + 1) % save_interval == 0:
                try:
                    with open(cache_file, 'w') as f:
                        for video_path in Filelist:
                            f.write(video_path + '\n')
                    print(f"  [Progress save] Saved {len(Filelist)} valid videos so far...")
                except Exception as e:
                    print(f"  Warning: Failed to save progress: {e}")
        
        print(f"Found {len(Filelist)} valid videos (>= {target_video_len} frames)")
        
        # Save to cache
        try:
            with open(cache_file, 'w') as f:
                for video_path in Filelist:
                    f.write(video_path + '\n')
            print(f"Saved video list to cache: {cache_file}")
        except Exception as e:
            print(f"Warning: Failed to save cache file: {e}")
    
    # In distributed training, wait for rank 0 to finish scanning and saving cache
    # Use file-based synchronization instead of barrier to avoid NCCL timeout
    if is_distributed:
        if rank != 0:
            # Non-rank-0 processes: wait for cache file using file polling
            # This avoids NCCL barrier timeout when rank 0 takes a long time to scan
            print(f"Rank {rank}: Waiting for cache file to be created by rank 0...")
            max_wait_time = 7200  # 2 hours max wait (enough for large scans)
            start_wait = time.time()
            check_interval = 2  # Check every 2 seconds
            
            while not os.path.exists(cache_file):
                elapsed = time.time() - start_wait
                if elapsed > max_wait_time:
                    raise RuntimeError(f"Rank {rank}: Cache file not found after waiting {elapsed:.0f}s: {cache_file}")
                time.sleep(check_interval)
            
            # Small wait to ensure file is fully written
            time.sleep(1)
            
            # Try to read the cache file with retries
            max_read_attempts = 10
            for attempt in range(max_read_attempts):
                try:
                    with open(cache_file, 'r') as f:
                        Filelist = [line.strip() for line in f if line.strip()]
                    if len(Filelist) > 0:
                        print(f"Rank {rank}: Loaded {len(Filelist)} videos from cache")
                        break
                except (IOError, OSError) as e:
                    if attempt < max_read_attempts - 1:
                        time.sleep(1)
                        continue
                    else:
                        raise RuntimeError(f"Rank {rank}: Failed to read cache file after {max_read_attempts} attempts: {e}")
            else:
                raise RuntimeError(f"Rank {rank}: Cache file exists but is empty: {cache_file}")
    
    return Filelist


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(num_class, anno_pth='./k400_classmap.json'):
    global class_labels_map, cls_sample_cnt
    
    if class_labels_map is not None:
        return class_labels_map, cls_sample_cnt
    else:
        cls_sample_cnt = {}
        class_labels_map = load_annotation_data(anno_pth)
        for cls in class_labels_map:
            cls_sample_cnt[cls] = 0
        return class_labels_map, cls_sample_cnt


def load_annotations(ann_file, num_class, num_samples_per_cls):
    dataset = []
    class_to_idx, cls_sample_cnt = get_class_labels(num_class)
    with open(ann_file, 'r') as fin:
        for line in fin:
            line_split = line.strip().split('\t')
            sample = {}
            idx = 0
            # idx for frame_dir
            frame_dir = line_split[idx]
            sample['video'] = frame_dir
            idx += 1
                                
            # idx for label[s]
            label = [x for x in line_split[idx:]]
            assert label, f'missing label in line: {line}'
            assert len(label) == 1
            class_name = label[0]
            class_index = int(class_to_idx[class_name])
            
            # choose a class subset of whole dataset
            if class_index < num_class:
                sample['label'] = class_index
                if cls_sample_cnt[class_name] < num_samples_per_cls:
                    dataset.append(sample)
                    cls_sample_cnt[class_name]+=1

    return dataset


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx



class DecordInit(object):
    """Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

    def __init__(self, num_threads=1):
        self.num_threads = num_threads
        # Don't initialize ctx in __init__ to avoid issues with multiprocessing fork
        self.ctx = None
        
    def __call__(self, filename):
        """Perform the Decord initialization.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        # Initialize ctx lazily in the worker process
        if self.ctx is None:
            self.ctx = decord.cpu(0)
            
        reader = decord.VideoReader(filename,
                                    ctx=self.ctx,
                                    num_threads=self.num_threads)
        return reader

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'num_threads={self.num_threads})')
        return repr_str


class Sky(torch.utils.data.Dataset):


    def __init__(self,
                 configs,
                 transform=None,
                 temporal_sample=None):
        self.configs = configs
        self.data_path = configs.data_path
        self.target_video_len = self.configs.data_num_frames
        
        # Use cache directory if specified, otherwise use data_path
        cache_dir = getattr(configs, 'cache_dir', None)
        self.video_lists = get_filelist(configs.data_path, self.target_video_len, cache_dir=cache_dir)
        
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.v_decoder = DecordInit()
        
        # Print dataset info on rank 0
        try:
            import torch.distributed as dist
            is_distributed = dist.is_available() and dist.is_initialized()
            rank = dist.get_rank() if is_distributed else 0
            if rank == 0:
                print(f"Sky dataset initialized with {len(self.video_lists)} videos")
        except:
            print(f"Sky dataset initialized with {len(self.video_lists)} videos")

    def __getitem__(self, index):
        path = self.video_lists[index]
        vr = None  # Initialize to None for proper cleanup
        
        try:
            # Use decord to efficiently read only needed frames
            vr = self.v_decoder(path)
            total_frames = len(vr)
            
            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            if end_frame_ind - start_frame_ind < self.target_video_len:
                # If not enough frames in sampled range, try another video
                del vr  # Clean up before recursing
                return self.__getitem__(random.randint(0, len(self.video_lists) - 1))
            
            # Calculate frame indices to sample
            frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int)
            
            # Only load the specific frames we need (much more memory efficient!)
            video = vr.get_batch(frame_indice).asnumpy()  # Shape: (T, H, W, C)
            
            # Immediately release video reader after reading frames
            del vr
            vr = None
            
            # Convert to tensor and apply transforms
            video = torch.from_numpy(video)  # Convert to tensor
            video = video.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
            video = self.transform(video)  # T C H W
            
            return {'video': video, 'video_name': 1}
            
        except Exception as e:
            # If video reading fails, try another random video
            print(f"Error reading video {path}: {e}")
            # Clean up resources before recursing
            if vr is not None:
                del vr
            return self.__getitem__(random.randint(0, len(self.video_lists) - 1))

    def __len__(self):
        return len(self.video_lists)


if __name__ == '__main__':

    import argparse
    import video_transforms
    import torch.utils.data as Data
    import torchvision.transforms as transforms
    
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_interval", type=int, default=1)
    # parser.add_argument("--data-path", type=str, default="/nvme/share_data/datasets/UCF101/videos")
    parser.add_argument("--data-path", type=str, default="/path/to/datasets/UCF101/videos/")
    config = parser.parse_args()


    temporal_sample = video_transforms.TemporalRandomCrop(config.num_frames * config.frame_interval)

    transform_ucf101 = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(256),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])


    ffs_dataset = UCF101(config, transform=transform_ucf101, temporal_sample=temporal_sample)
    ffs_dataloader = Data.DataLoader(dataset=ffs_dataset, batch_size=6, shuffle=False, num_workers=1)

    # for i, video_data in enumerate(ffs_dataloader):
    for video_data in ffs_dataloader:
        print(type(video_data))
        video = video_data['video']
        video_name = video_data['video_name']
        print(video.shape)
        print(video_name)
        # print(video_data[2])

        # for i in range(16):
        #     img0 = rearrange(video_data[0][0][i], 'c h w -> h w c')
        #     print('Label: {}'.format(video_data[1]))
        #     print(img0.shape)
        #     img0 = Image.fromarray(np.uint8(img0 * 255))
        #     img0.save('./img{}.jpg'.format(i))
        exit()