#!/usr/bin/env python3
"""
多速率视频数据集：同源多速率视图 + 抗混叠 + 可选两段切片
支持 replay-speed 训练，防止 LLL 被 aliasing 污染
"""

import os
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from PIL import Image
import decord
from decord import VideoReader, cpu
import json

# ========= 抗混叠低通滤波工具 =========

def kaiser_lowpass_1d(Fs_in, Fs_out, atten_db=72, passband_ratio=0.9):
    """
    构造时间维低通FIR滤波器
    Args:
        Fs_in: 原始帧率(单位化只要比值)
        Fs_out: 目标帧率
        atten_db: 阻带衰减(dB)
        passband_ratio: 通带比例(<1给过渡带安全边际)
    Returns:
        h: FIR系数 (1,1,N)
    """
    M = int(round(Fs_in / Fs_out))  # 整倍数降采样
    assert abs(Fs_in / M - Fs_out) < 1e-6, f"建议整倍数降采样: {Fs_in}/{M}≠{Fs_out}"
    
    Fc = (Fs_out * 0.5) * passband_ratio  # 安全通带截止频率
    
    # Kaiser窗参数
    A = atten_db
    if A > 50: 
        beta = 0.1102 * (A - 8.7)
    elif A >= 21: 
        beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21)
    else: 
        beta = 0.0
    
    # 估计滤波器阶数
    trans_bw = (Fs_out * 0.5) * (1 - passband_ratio)
    N = math.ceil((A - 8) / (2.285 * 2 * math.pi * trans_bw / Fs_in))
    if N % 2 == 0: 
        N += 1
    Mhalf = (N - 1) // 2

    n = torch.arange(N, dtype=torch.float32)
    fc = Fc / Fs_in
    
    # 理想sinc低通响应
    h = torch.where(
        n == Mhalf, 
        torch.tensor(2 * fc),
        torch.sin(2 * math.pi * fc * (n - Mhalf)) / (math.pi * (n - Mhalf))
    )
    
    # Kaiser窗函数
    def I0(x):
        """修正贝塞尔函数I0近似"""
        y = torch.ones_like(x)
        t = torch.ones_like(x)
        for k in range(1, 32):
            t = t * (x / 2)**2 / (k**2)
            y = y + t
        return y
    
    w = I0(beta * torch.sqrt(1 - ((2 * n / (N - 1)) - 1)**2)) / I0(torch.tensor(beta))
    h = h * w
    h = h / h.sum()  # 归一化
    
    return h.view(1, 1, -1)  # (out_ch, in_ch/groups, K)


def anti_alias_downsample(video_cthw, sample_rate, atten_db=72):
    """
    抗混叠降采样
    Args:
        video_cthw: (C, T, H, W) 已做ToTensor/Resize/Crop/[-1,1]的视频
        sample_rate: 抽取步长 (1=不变, 2=每2帧取1帧, ...)
        atten_db: 阻带衰减
    Returns:
        (C, T_out, H, W) 其中 T_out = floor(T_in / sample_rate)
    """
    C, T, H, W = video_cthw.shape
    if sample_rate == 1:
        return video_cthw
    
    # 构造低通滤波器
    h = kaiser_lowpass_1d(Fs_in=1.0, Fs_out=1.0/sample_rate, atten_db=atten_db)
    h = h.to(video_cthw.device, video_cthw.dtype)
    
    # 对每个像素沿时间做1D卷积
    # (C,T,H,W) -> (C*H*W, 1, T) -> conv1d -> (C*H*W, 1, T_out) -> (C,T_out,H,W)
    x = video_cthw.view(C * H * W, 1, T)
    x = F.conv1d(x, h, padding=h.shape[-1] // 2)
    x = x[:, :, ::sample_rate]  # 抽取
    T_out = x.shape[-1]
    x = x.view(C, T_out, H, W)
    
    return x


# ========= 视频变换 =========

class VideoTransforms:
    """视频预处理变换"""
    
    def __init__(self, resolution=256, crop_size=None, normalize=True):
        self.resolution = resolution
        self.crop_size = crop_size or resolution
        self.normalize = normalize
        
        # 空间变换
        self.spatial_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.CenterCrop(self.crop_size) if crop_size != resolution else transforms.Lambda(lambda x: x),
        ])
    
    def __call__(self, video_thwc):
        """
        Args:
            video_thwc: (T, H, W, C) numpy array or (T, C, H, W) tensor
        Returns:
            video_tchw: (T, C, H, W) tensor in [-1, 1]
        """
        if isinstance(video_thwc, np.ndarray):
            # numpy (T,H,W,C) -> tensor (T,C,H,W)
            video = torch.from_numpy(video_thwc).permute(0, 3, 1, 2).float()
        else:
            video = video_thwc.float()
        
        # 确保在[0,255]范围
        if video.max() <= 1.0:
            video = video * 255.0
        
        # 逐帧应用空间变换
        T = video.shape[0]
        frames = []
        for t in range(T):
            frame = video[t]  # (C, H, W)
            # 转为PIL进行resize/crop
            frame_pil = transforms.ToPILImage()(frame.byte())
            frame_transformed = self.spatial_transform(frame_pil)
            frame_tensor = transforms.ToTensor()(frame_transformed)
            frames.append(frame_tensor)
        
        video_out = torch.stack(frames, dim=0)  # (T, C, H, W)
        
        # 归一化到[-1, 1]
        if self.normalize:
            video_out = video_out * 2.0 - 1.0
        
        return video_out


# ========= 数据集类 =========

class MultiSpeedVideoDataset(data.Dataset):
    """
    多速率视频数据集
    支持同源多速率视图、抗混叠处理、可选两段切片
    """
    
    def __init__(
        self, 
        video_folder, 
        sequence_length=25, 
        train=True, 
        resolution=256,
        crop_size=None,
        speeds=(1, 2, 4),
        pair_segments=False,
        anchor_same_start=True,
        cache_file=None,
        is_main_process=False,
        max_videos=None
    ):
        """
        Args:
            video_folder: 视频文件夹路径
            sequence_length: 序列长度
            train: 是否训练模式
            resolution: 分辨率
            crop_size: 裁剪尺寸
            speeds: 速率列表 (1, 2, 4) 表示 stride
            pair_segments: 是否返回同速率的两段不重叠片段
            anchor_same_start: 多速率视图从同一起点出发
            cache_file: 缓存文件路径
            is_main_process: 是否主进程
            max_videos: 最大视频数量限制
        """
        self.video_folder = video_folder
        self.sequence_length = sequence_length
        self.train = train
        self.resolution = resolution
        self.crop_size = crop_size or resolution
        self.speeds = list(speeds)
        self.pair_segments = pair_segments
        self.anchor_same_start = anchor_same_start
        self.max_videos = max_videos
        
        # 视频解码器
        decord.bridge.set_bridge('torch')
        self.v_decoder = VideoReader
        
        # 变换
        self.transform = VideoTransforms(resolution, crop_size, normalize=True)
        
        # 收集视频文件
        self._collect_videos(cache_file, is_main_process)
        
        print(f"Dataset initialized: {len(self.samples)} videos, "
              f"speeds={self.speeds}, pair_segments={self.pair_segments}")
    
    def _collect_videos(self, cache_file, is_main_process):
        """收集视频文件路径"""
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached video list from {cache_file}")
            with open(cache_file, 'r') as f:
                self.samples = json.load(f)
        else:
            print(f"Scanning video folder: {self.video_folder}")
            self.samples = []
            
            if os.path.isdir(self.video_folder):
                for root, dirs, files in os.walk(self.video_folder):
                    for file in files:
                        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                            self.samples.append(os.path.join(root, file))
            else:
                raise ValueError(f"Video folder not found: {self.video_folder}")
            
            # 排序保证一致性
            self.samples.sort()
            
            # 限制视频数量
            if self.max_videos and len(self.samples) > self.max_videos:
                self.samples = self.samples[:self.max_videos]
            
            # 保存缓存
            if cache_file and is_main_process:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'w') as f:
                    json.dump(self.samples, f)
                print(f"Cached video list to {cache_file}")
        
        print(f"Found {len(self.samples)} videos")
    
    def decord_read_multi(self, path):
        """
        读取同一视频同一时间窗的多速率视图
        Returns:
            views: dict {rate: (C,T,H,W)} 其中 T = sequence_length
        """
        try:
            decord_vr = self.v_decoder(path, ctx=cpu(0))
            total_frames = len(decord_vr)
        except Exception as e:
            raise ValueError(f"Failed to load video {path}: {e}")
        
        max_rate = max(self.speeds) if len(self.speeds) > 0 else 1
        # 需要的原始连续帧长度（stride=1）
        span = self.sequence_length * max_rate
        
        # 选择起点
        if total_frames <= span:
            # 视频太短，从0开始，后面会重复最后一帧补齐
            start = 0
            frame_ids_full = np.arange(0, min(total_frames, span), dtype=int)
        else:
            start = random.randint(0, total_frames - span) if self.train else 0
            frame_ids_full = np.arange(start, start + span, dtype=int)
        
        # 读取帧
        video_np = decord_vr.get_batch(frame_ids_full).asnumpy()  # (span, H, W, C)
        
        # 应用变换: (T,H,W,C) -> (T,C,H,W)
        video_tchw = self.transform(video_np)  # (T, C, H, W)
        
        # 转换为 (C,T,H,W) 便于时间维处理
        video_cthw = video_tchw.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)
        
        # 对每个速率进行抗混叠降采样
        views = {}
        for rate in self.speeds:
            # 低通滤波 + 抽取
            v_r = anti_alias_downsample(video_cthw, rate)
            
            # 确保长度 >= sequence_length（极端情况下补齐）
            if v_r.shape[1] < self.sequence_length:
                # 重复最后一帧补齐
                pad = self.sequence_length - v_r.shape[1]
                if v_r.shape[1] > 0:
                    last = v_r[:, -1:, ...].repeat(1, pad, 1, 1)
                    v_r = torch.cat([v_r, last], dim=1)
                else:
                    # 如果完全没有帧，创建零帧
                    C, _, H, W = v_r.shape
                    v_r = torch.zeros(C, self.sequence_length, H, W, dtype=video_cthw.dtype)
            
            # 截取到指定长度
            v_r = v_r[:, :self.sequence_length, ...]
            views[rate] = v_r
        
        return views
    
    def __getitem__(self, idx):
        """获取一个样本"""
        video_path = self.samples[idx]
        
        try:
            # 获取多速率视图
            views = self.decord_read_multi(video_path)  # dict: {rate: (C,T,H,W)}
            
            out = {
                "views": views, 
                "label": "", 
                "file": os.path.basename(video_path),
                "path": video_path
            }
            
            # 如果需要片段对，从最慢速率（最小rate）生成两段
            if self.pair_segments:
                anchor_rate = min(self.speeds)
                v = views[anchor_rate]  # (C, T, H, W)
                T = v.shape[1]
                mid = T // 2
                seg1 = v[:, :mid, ...]      # 前半段
                seg2 = v[:, -mid:, ...]     # 后半段（可能与前半段有重叠）
                out["seg_pair"] = (seg1, seg2)
            
            return out
            
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            # 随机选择另一个样本
            return self.__getitem__(random.randint(0, len(self) - 1))
    
    def __len__(self):
        return len(self.samples)


# ========= 批处理整理函数 =========

def multi_speed_collate_fn(batch):
    """
    自定义批处理整理函数
    将各速率的视图整理成统一的张量格式
    
    Args:
        batch: list of dicts from __getitem__
        
    Returns:
        batch_dict: {
            'views': {rate: (B,C,T,H,W)},  # 各速率视图
            'view_tensor': (B, |R|, C, T, H, W),  # 堆叠的多速率张量
            'speeds': list of speeds,
            'files': list of filenames,
            'seg_pairs': (B,C,T//2,H,W) x 2 if pair_segments else None
        }
    """
    if len(batch) == 0:
        return {}
    
    # 收集基本信息
    files = [item['file'] for item in batch]
    speeds = sorted(batch[0]['views'].keys())
    
    # 整理多速率视图
    views = {}
    view_list = []
    
    for rate in speeds:
        rate_batch = torch.stack([item['views'][rate] for item in batch], dim=0)  # (B,C,T,H,W)
        views[rate] = rate_batch
        view_list.append(rate_batch)
    
    # 堆叠成 (B, |R|, C, T, H, W)
    view_tensor = torch.stack(view_list, dim=1)  # (B, |R|, C, T, H, W)
    
    result = {
        'views': views,
        'view_tensor': view_tensor,
        'speeds': speeds,
        'files': files,
        'paths': [item.get('path', '') for item in batch]
    }
    
    # 处理片段对
    if 'seg_pair' in batch[0]:
        seg1_batch = torch.stack([item['seg_pair'][0] for item in batch], dim=0)
        seg2_batch = torch.stack([item['seg_pair'][1] for item in batch], dim=0)
        result['seg_pairs'] = (seg1_batch, seg2_batch)
    else:
        result['seg_pairs'] = None
    
    return result


# ========= 测试和演示 =========

def test_multi_speed_dataset():
    """测试多速率数据集"""
    print("=== 测试多速率视频数据集 ===")
    
    # 创建测试数据集
    dataset = MultiSpeedVideoDataset(
        video_folder="/path/to/your/videos",  # 替换为实际路径
        sequence_length=16,
        resolution=128,
        speeds=(1, 2, 4),
        pair_segments=True,
        train=True,
        max_videos=10
    )
    
    # 测试单个样本
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Views speeds: {list(sample['views'].keys())}")
    
    for rate, view in sample['views'].items():
        print(f"Speed {rate}: {view.shape} [{view.min():.3f}, {view.max():.3f}]")
    
    if 'seg_pair' in sample:
        seg1, seg2 = sample['seg_pair']
        print(f"Segment pair: {seg1.shape}, {seg2.shape}")
    
    # 测试批处理
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=multi_speed_collate_fn,
        num_workers=0
    )
    
    batch = next(iter(dataloader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"View tensor shape: {batch['view_tensor'].shape}")
    print(f"Speeds: {batch['speeds']}")
    
    if batch['seg_pairs'] is not None:
        seg1, seg2 = batch['seg_pairs']
        print(f"Batch segment pairs: {seg1.shape}, {seg2.shape}")


if __name__ == "__main__":
    # 运行测试
    test_multi_speed_dataset()。这是我修改的多速率采样dataset，以下是原来的代码：import os.path as osp
import random
from glob import glob
from torchvision import transforms
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import pickle
import decord
from torch.nn import functional as F
from .transform import ToTensorVideo, CenterCropVideo
from torchvision.transforms._transforms_video import CenterCropVideo as TVCenterCropVideo
from torchvision.transforms import Lambda, Compose, Resize
import torch
import os


class DecordInit(object):
    def __init__(self, num_threads=1):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)

    def __call__(self, filename):
        reader = decord.VideoReader(
            filename, ctx=self.ctx, num_threads=self.num_threads
        )
        return reader

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"sr={self.sr},"
            f"num_threads={self.num_threads})"
        )
        return repr_str

def TemporalRandomCrop(total_frames, size):
    rand_end = max(0, total_frames - size - 1)
    begin_index = random.randint(0, rand_end)
    end_index = min(begin_index + size, total_frames)
    return begin_index, end_index

def _format_video_shape(video, time_compress=4, spatial_compress=8):
    """Prepare video for VAE"""
    time = video.shape[1]
    height = video.shape[2]
    width = video.shape[3]
    new_time = (
        (time - (time - 1) % time_compress) if (time - 1) % time_compress != 0 else time
    )
    new_height = (
        (height - (height) % spatial_compress)
        if height % spatial_compress != 0
        else height
    )
    new_width = (
        (width - (width) % spatial_compress) if width % spatial_compress != 0 else width
    )
    return video[:, :new_time, :new_height, :new_width]


class TrainVideoDataset(data.Dataset):
    video_exts = ["avi", "mp4", "webm"]

    def __init__(
        self,
        video_folder,
        sequence_length,
        train=True,
        resolution=64,
        sample_rate=1,
        dynamic_sample=True,
        cache_file=None,
        is_main_process=False,
    ):

        self.train = train
        self.sequence_length = sequence_length
        self.sample_rate = sample_rate
        self.resolution = resolution
        self.v_decoder = DecordInit()
        self.video_folder = video_folder
        self.dynamic_sample = dynamic_sample
        self.cache_file = cache_file
        self.transform = transforms.Compose(
            [
                ToTensorVideo(),
                Resize(self.resolution),
                CenterCropVideo(self.resolution),
                Lambda(lambda x: 2.0 * x - 1.0),  #图片归一化之后为[0,1]，经过lambd处理范围缩放到[-1,1]
               ]
        )
        print("Building datasets...")
        self.is_main_process = is_main_process
        self.samples = self._make_dataset()

    def _make_dataset(self):
        cache_file = osp.join(self.video_folder, self.cache_file)

        if osp.exists(cache_file):
            with open(cache_file, "rb") as f:
                samples = pickle.load(f)
        else:
            samples = []
            samples += sum(
                [
                    glob(osp.join(self.video_folder, "**", f"*.{ext}"), recursive=True)
                    for ext in self.video_exts
                ],
                [],
            )
            if self.is_main_process:
                with open(cache_file, "wb") as f:
                    pickle.dump(samples, f)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path = self.samples[idx]
        try:
            video = self.decord_read(video_path)
            video = self.transform(video)  # T C H W -> T C H W
            video = video.transpose(0, 1)  # T C H W -> C T H W
            return dict(video=video, label="")
        except Exception as e:
            print(f"Error with {e}, {video_path}")
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def decord_read(self, path): #dynamic_sample:

        """
        动作定义：dynamic_sample=True 时，每次取样会随机选择时间采样步长（stride）。
        当启用时：sample_rate = random.randint(1, self.sample_rate)
        否则：sample_rate = self.sample_rate
        之后按 size = sequence_length * sample_rate 做随机时间裁剪，并用等间隔帧索引取 sequence_length 帧。
        
        作用与好处
        数据增强：随机时间尺度采样（慢/快），提升对不同运动速度的鲁棒性。
        同样输入帧数不变，但覆盖的视频时间范围随步长而变，模型看到的时域变化更多。
        对很短视频：步长大时需要更长时长的视频；代码会退化为在可用范围内等间隔采样（可能出现帧重复）。
        """
        decord_vr = self.v_decoder(path)
        total_frames = len(decord_vr)
        # Sampling video frames
        if self.dynamic_sample:
            sample_rate = random.randint(1, self.sample_rate)
        else:
            sample_rate = self.sample_rate
        size = self.sequence_length * sample_rate
        start_frame_ind, end_frame_ind = TemporalRandomCrop(total_frames, size)
        frame_indice = np.linspace(
            start_frame_ind, end_frame_ind - 1, self.sequence_length, dtype=int
        )

        video_data = decord_vr.get_batch(frame_indice).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)
        return video_data

def resize(x, resolution):
    height, width = x.shape[-2:]
    aspect_ratio = width / height
    if width <= height:
        new_width = resolution
        new_height = int(resolution / aspect_ratio)
    else:
        new_height = resolution
        new_width = int(resolution * aspect_ratio)
    resized_x = F.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners=True, antialias=True)
    return resized_x


class ValidVideoDataset(data.Dataset):
    video_exts = ["avi", "mp4", "webm"]
    
    def __init__(
        self,
        real_video_dir,
        num_frames,
        sample_rate=1,
        crop_size_width=None,
        crop_size_height=None,
        crop_size=None,
        resolution=128,
        is_main_process=False
    ) -> None:
        super().__init__()
        self.is_main_process = is_main_process
        self.real_video_files = self._make_dataset(real_video_dir)
        
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        
        if crop_size is not None:
            self.crop_size_width = crop_size
            self.crop_size_height = crop_size
        else:
            self.crop_size_width = crop_size_width
            self.crop_size_height = crop_size_height
            
        self.short_size = resolution
        self.v_decoder = DecordInit()
        self.transform = Compose(
            [
                ToTensorVideo(),
                Resize(resolution),
                CenterCropVideo((self.crop_size_height, self.crop_size_width)) if self.crop_size_width is not None and self.crop_size_height is not None else Lambda(lambda x: x),
            ]
        )
        
    def _make_dataset(self, real_video_dir):
        cache_file = osp.join(real_video_dir, "idx.pkl")

        if osp.exists(cache_file):
            with open(cache_file, "rb") as f:
                samples = pickle.load(f)
        else:
            samples = []
            samples += sum( #38685
                [
                    glob(osp.join(real_video_dir, "**", f"*.{ext}"), recursive=True)
                    for ext in self.video_exts
                ],
                [],
            )
            if self.is_main_process:
                try:
                    with open(cache_file, "wb") as f:
                        pickle.dump(samples, f)
                except Exception as e:
                    print(f"Error with {e}, {cache_file}")
        return samples
    
    def __len__(self):
        return len(self.real_video_files)

    def __getitem__(self, index):
        try:
            if index >= len(self):
                raise IndexError
            real_video_file = self.real_video_files[index]
            real_video_tensor = self._load_video(real_video_file)
            real_video_tensor = self.transform(real_video_tensor)
            video_name = os.path.basename(real_video_file)
            return {'video': real_video_tensor, 'file_name': video_name }
        except:
            import traceback
            traceback.print_exc()
            print(f"Video error: {self.real_video_files[index]}")
            return self.__getitem__(0)

    def _load_video(self, video_path, sample_rate=None):
        num_frames = self.num_frames
        if not sample_rate:
            sample_rate = self.sample_rate
        try:
            decord_vr = self.v_decoder(video_path)
        except:
            raise Exception(f"fail to load {video_path}.")
        total_frames = len(decord_vr)
        sample_frames_len = sample_rate * num_frames

        if total_frames >= sample_frames_len:
            s = 0
            e = s + sample_frames_len
            num_frames = num_frames
        else:
            raise Exception("video too short!")
            
        frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(3, 0, 1, 2)
        return video_data