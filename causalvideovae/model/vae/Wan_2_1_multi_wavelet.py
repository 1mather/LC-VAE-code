# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


from typing import Any, List, Tuple
from sympy.strategies.core import chain
import torch
import torch.nn as nn
import os
from collections import deque
import math
from ..modules import (
    ResnetBlock2D,
    ResnetBlock3D,
    Conv2d,
    HaarWaveletSpatialTransform,
    InverseHaarWaveletSpatialTransform,
    CausalConv3d,
    Normalize,
    nonlinearity,
)
from ..registry import ModelRegistry
from ..modeling_videobase import VideoBaseAE
from ..utils.module_utils import resolve_str_to_obj
from ..utils.distrib_utils import DiagonalGaussianDistribution
from ..modeling_output import AutoencoderKLOutput, DecoderOutput, ForwardOutput
from diffusers.configuration_utils import register_to_config

__all__ = [
    'Wan2_1_VAE',
    'Wan2_1_VAE_Trainable',
]

CACHE_T = 2
class HaarCompressedLatentProcessor(nn.Module):
    """
    Top-K时间压缩的Latent处理器（三阶段训练）
    """
    def __init__(
        self, 
        dwt3d_cls: str = "HaarWaveletTransform3D", 
        idwt3d_cls: str = "InverseHaarWaveletTransform3D",
        training_phase: str = "phase1",  # "phase1", "phase2", or "phase3"
        temporal_consistency_mode: str = "tv_l1",  # "tv_l1", "temporal_variance", etc.
        keep_ratio: float = 0.5,
        temperature: float = 0.1,
        score_mode: str = "l2",
        hl_boost: float = 1.0,
        var_boost: float = 0.3,
        compress_strategy: str = "ungrouped",  # "ungrouped" or "groupwise"
        dwt_temporal_cls: str = "HaarWaveletTemporalTransform",
        idwt_temporal_cls: str = "InverseHaarWaveletTemporalTransform",
        exact_k: bool = False,
        multi_wavelet: bool = True,
        global_topk: bool = True,
        use_fixed_mask: bool = False,  # 是否使用固定mask
        fixed_mask_path: str = None,   # 固定mask文件路径
        not_compress: bool = False,
    ):
        super().__init__()
        self.not_compress=not_compress

        DWT3D = resolve_str_to_obj(dwt3d_cls)
        IDWT3D = resolve_str_to_obj(idwt3d_cls)
        self.dwt3d = DWT3D()
        self.idwt3d = IDWT3D()
        DWTTemporal = resolve_str_to_obj(dwt_temporal_cls)
        IDWTTemporal = resolve_str_to_obj(idwt_temporal_cls)
        self.dwt_temporal_cls = dwt_temporal_cls
        self.idwt_temporal_cls = idwt_temporal_cls
        self.dwt_temporal = DWTTemporal()
        self.idwt_temporal = IDWTTemporal()

        self.global_topk = global_topk
        
        # 固定mask模式
        self.use_fixed_mask = use_fixed_mask
        self.fixed_mask = None
        if use_fixed_mask and fixed_mask_path:
            self.load_fixed_mask(fixed_mask_path)
        
        # 用于记录通道选择的统计信息
        self.channel_selection_stats = {
            'mask_history': [],  # 记录每次的mask
            'scores_history': [],  # 记录每次的scores
            'selected_channels_count': None,  # 累计每个通道被选中的次数
            'total_calls': 0,  # 总调用次数
        }
        
        # 训练阶段
        self.training_phase = training_phase
        self.temporal_consistency_mode = temporal_consistency_mode
        
        # Top-K 与压缩策略参数
        self.keep_ratio = keep_ratio
        self.temperature = temperature
        self.score_mode = score_mode
        self.hl_boost = hl_boost
        self.var_boost = var_boost
        self.compress_strategy = compress_strategy
        self.exact_k = exact_k
        self.multi_wavelet = multi_wavelet
        # 控制时间小波分解的层数；当前默认做两次（与前向保持对称）
        self.wavelet_levels = 2 if self.multi_wavelet else 0


    
    def set_training_phase(self, phase: str):
        """切换训练阶段"""
        assert phase in ["phase1", "phase2", "phase3"]
        self.training_phase = phase
        try:
            is_main = os.environ.get("LOCAL_RANK", "0") == "0"
        except Exception:
            is_main = True

    
    def lowfreq_consistency_loss(self, LL: torch.Tensor, mode: str = "tv_l1") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算低频一致性损失
        - "temporal_variance": 直接对时间维 T 求方差并平均成标量
        - "tv_l1": 时间差分（TV-L1）平滑损失，抑制相邻帧间的跳跃
        """
        if LL.ndim != 5:
            raise ValueError("LL must be (B,C,T,H,W)")
        
        # 计算方差（用于监控）
        def temporal_var_si(L):
            var_t = L.var(dim=2, unbiased=False)
            scale = (L**2).mean(dim=2, keepdim=False) + 1e-6
            return (var_t / scale).sum()
        
        ll_variance = temporal_var_si(LL)
        

        if mode == "tv_l1":
            B, C, T, H, W = LL.shape
            if T <= 1:
                return (LL * 0.0).sum(), ll_variance  # 保持梯度连接的零损失
            
            # 时间差分 L1范数
            dt = LL[:, :, 1:, :, :] - LL[:, :, :-1, :, :]  # (B,C,T-1,H,W)
            tv = dt.abs()  # L1范数
            return tv.sum(), ll_variance
        else:
            raise ValueError(f"Unknown consistency mode: {mode}")


    @staticmethod
    def hard_topk_mask(scores: torch.Tensor, k: int):
        """
        Hard Top-K 掩码：对每个 batch，在所有 C*T*H*W 位置中取前 k 个置 1
        scores: (B, C, T, H, W)
        返回: (B, C, T, H, W) 掩码
        """
        B, C, T, H, W = scores.shape
        N = C * T * H * W
        
        if k <= 0:
            return torch.zeros_like(scores)
        if k >= N:
            return torch.ones_like(scores)
        
        # 展平所有维度除了 batch
        scores_flat = scores.view(B, -1)  # (B, C*T*H*W)
        idx = torch.topk(scores_flat, k, dim=1).indices  # (B, k)
        mask_flat = torch.zeros_like(scores_flat)
        mask_flat.scatter_(1, idx, 1.0)
        
        return mask_flat.view(B, C, T, H, W)



    @staticmethod
    def score_channels(x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, H, W) → scores: (B, C, T, H, W)
        计算每个位置的重要性分数
        """
        return (x**2)
        # if mode == "l2":
        #     return (x**2).sum(dim=(2,3,4))
        # elif mode == "var":
        #     return x.var(dim=(2,3,4), unbiased=False)
        # else:
        #     e = (x**2).sum(dim=(2,3,4))
        #     v = x.var(dim=(2,3,4), unbiased=False)
        #     return e + var_boost * v
    
    def _record_channel_selection(self, mask: torch.Tensor, scores: torch.Tensor):
        """
        记录每次位置选择的信息
        mask: (B, N) - 每个位置的软/硬掩码值，N可以是C或C*T*H*W
        scores: (B, N) - 每个位置的重要性分数
        """
        # 只在评估模式或需要分析时记录（避免训练时内存爆炸）
        if not hasattr(self, 'enable_channel_recording') or not self.enable_channel_recording:
            return
        
        B, C = mask.shape
        
        # 初始化累计统计（总是在CPU上）
        if self.channel_selection_stats['selected_channels_count'] is None:
            self.channel_selection_stats['selected_channels_count'] = torch.zeros(C, dtype=torch.float32)
        
        # 累计每个通道被选中的情况 (对batch求平均)
        mask_mean = mask.mean(dim=0).detach().cpu()  # (C,) 先转到CPU
        self.channel_selection_stats['selected_channels_count'] += mask_mean
        self.channel_selection_stats['total_calls'] += 1
        
        # 记录最近的mask和scores (限制历史长度避免内存问题)
        max_history = 100
        if len(self.channel_selection_stats['mask_history']) < max_history:
            self.channel_selection_stats['mask_history'].append(mask.detach().cpu())
            self.channel_selection_stats['scores_history'].append(scores.detach().cpu())
    
    def load_fixed_mask(self, mask_path: str):
        """加载固定的通道mask"""
        import torch
        try:
            mask_data = torch.load(mask_path, map_location='cpu')
            self.fixed_mask = mask_data['fixed_mask']  # (C,)
            self.selected_indices=mask_data['selected_indices'].to("cuda")
            num_keep = mask_data.get('num_keep', int(self.fixed_mask.sum().item()))
            print(f"[TemporalProcessor] Loaded fixed mask from {mask_path}")
            print(f"  - Total channels: {len(self.fixed_mask)}")
            print(f"  - Keep channels: {num_keep} ({num_keep/len(self.fixed_mask)*100:.1f}%)")
        except Exception as e:
            print(f"[TemporalProcessor] Failed to load fixed mask: {e}")
            self.use_fixed_mask = False
            self.fixed_mask = None
    
    def set_fixed_mask(self, mask: torch.Tensor):
        """直接设置固定mask（用于动态切换）"""
        self.fixed_mask = mask.cpu() if mask.is_cuda else mask
        self.use_fixed_mask = True
        print(f"[TemporalProcessor] Fixed mask set: keep {int(mask.sum())} / {len(mask)} channels")
    
    def enable_fixed_mask(self, enable: bool = True):
        """启用或禁用固定mask模式"""
        if enable and self.fixed_mask is None:
            print("[TemporalProcessor] Warning: No fixed mask loaded, cannot enable fixed mask mode")
            return
        self.use_fixed_mask = enable
        mode = "enabled" if enable else "disabled"
        print(f"[TemporalProcessor] Fixed mask mode {mode}")
    
    def enable_channel_recording(self, enable: bool = True):
        """启用或禁用通道选择记录"""
        self.enable_channel_recording = enable
        if enable:
            print("[TemporalProcessor] Channel recording enabled")
        else:
            print("[TemporalProcessor] Channel recording disabled")
    
    def reset_channel_stats(self):
        """重置统计信息"""
        self.channel_selection_stats = {
            'mask_history': [],
            'scores_history': [],
            'selected_channels_count': None,
            'total_calls': 0,
        }
        print("[TemporalProcessor] Channel statistics reset")
    
    def get_channel_stats_summary(self):
        """
        获取通道选择统计摘要
        返回一个字典，包含：
        - avg_selection_freq: 每个通道平均被选中的频率 (C,)
        - top_k_channels: 最常被选中的K个通道索引
        - channel_names: 通道名称（根据小波子带命名）
        """
        if self.channel_selection_stats['selected_channels_count'] is None:
            return {"error": "No statistics collected yet"}
        
        total_calls = self.channel_selection_stats['total_calls']
        if total_calls == 0:
            return {"error": "No calls recorded"}
        
        # 计算平均选择频率
        avg_freq = self.channel_selection_stats['selected_channels_count'] / total_calls
        
        # 找出最常被选中的通道
        sorted_indices = torch.argsort(avg_freq, descending=True)
        
        # 为通道生成名称（假设是3D Haar + 可选temporal Haar）
        C = len(avg_freq)
        channel_names = self._generate_channel_names(C)
        
        summary = {
            'total_calls': total_calls,
            'num_channels': C,
            'avg_selection_frequency': avg_freq.numpy(),
            'top_channels_indices': sorted_indices[:50].tolist(),  # 前50个
            'top_channels_freq': avg_freq[sorted_indices[:50]].numpy(),
            'channel_names': channel_names,
            'top_channels_names': [channel_names[i] for i in sorted_indices[:50].tolist()],
        }
        
        return summary
    
    def _generate_channel_names(self, total_channels: int):
        """
        为通道生成名称
        基于3D Haar小波的8个子带 + 可选的temporal多层Haar
        """
        # 3D Haar的8个基础子带
        base_subbands = ['LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH']
        
        if not self.multi_wavelet:
            # 没有额外的temporal小波，直接按子带分组命名
            # total_channels = 8 * C_latent
            channels_per_subband = total_channels // 8
            names = []
            for subband in base_subbands:
                for ch in range(channels_per_subband):
                    names.append(f"{subband}_ch{ch}")
            return names
        else:
            # 有temporal小波，需要更复杂的命名
            # 经过L次temporal Haar后，通道数倍增
            # 简化：直接用通道索引
            return [f"ch_{i}" for i in range(total_channels)]
    
    def save_channel_stats(self, filepath: str):
        """保存统计信息到文件"""
        import json
        import numpy as np
        
        summary = self.get_channel_stats_summary()
        
        # 转换numpy数组为列表以便JSON序列化
        if 'avg_selection_frequency' in summary:
            summary['avg_selection_frequency'] = summary['avg_selection_frequency'].tolist()
        if 'top_channels_freq' in summary:
            summary['top_channels_freq'] = summary['top_channels_freq'].tolist()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[TemporalProcessor] Channel statistics saved to {filepath}")
    
    def print_channel_stats(self, top_k: int = 20):
        """打印通道选择统计信息"""
        summary = self.get_channel_stats_summary()
        
        if 'error' in summary:
            print(f"[TemporalProcessor] {summary['error']}")
            return
        
        print("\n" + "="*80)
        print(f"[TemporalProcessor] Channel Selection Statistics")
        print("="*80)
        print(f"Total calls: {summary['total_calls']}")
        print(f"Total channels: {summary['num_channels']}")
        print(f"\nTop {top_k} most selected channels:")
        print("-"*80)
        print(f"{'Rank':<6} {'Channel':<15} {'Index':<8} {'Avg Frequency':<15}")
        print("-"*80)
        
        for rank, (idx, name, freq) in enumerate(zip(
            summary['top_channels_indices'][:top_k],
            summary['top_channels_names'][:top_k],
            summary['top_channels_freq'][:top_k]
        ), 1):
            print(f"{rank:<6} {name:<15} {idx:<8} {freq:<15.4f}")
        
        print("="*80 + "\n")

    def global_topk_over_coeffs(
        self,
        coeffs: torch.Tensor,
        keep_ratio: float,
        temperature: float,
        score_mode: str,
        var_boost: float,
        exact_k: bool,
        use_hard: bool,
        is_main:bool,
    ):
        """
        在整幅小波系数上做位置级 top-k（在所有 C*T*H*W 位置中选择）。
        支持两种模式：
        1. 动态模式：根据每个样本的score在所有位置上计算top-k
        2. 固定mask模式：所有样本使用相同的预定义mask（通道级）
        
        coeffs: (B, Ctot, T, H, W)
        返回: masked_coeffs (B,Ctot,T,H,W), masks_5d (B,Ctot,T,H,W), meta
        """
        B, Ctot, T, H, W = coeffs.shape
        
        # === 固定mask模式 ===
        if self.use_fixed_mask and self.fixed_mask is not None:
            # 使用预定义的固定mask
            fixed_mask_1d = self.fixed_mask.to(coeffs.device)  # (Ctot,)
            
            # 扩展到batch维度：(B, Ctot)
            mask = fixed_mask_1d.unsqueeze(0).expand(B, -1)
            
            # 扩展到5D：(B, Ctot, 1, 1, 1)
            mask5 = mask.view(B, Ctot, 1, 1, 1)
            # 应用mask
            masked = coeffs * mask5


            reshaped=coeffs.index_select(dim=1, index=self.selected_indices)
            #reshaped[:,512:-1]=0
            original = torch.zeros(B, Ctot, T, H, W, device=coeffs.device, dtype=coeffs.dtype)
            original[:, self.selected_indices, :, :, :] = reshaped


            
            # 元数据
            K_total = int(fixed_mask_1d.sum().item())
            
            meta = {
                "K_total_target": K_total,
                "mask_sum_mean": K_total,  # 固定mask下每个样本都一样
                "mode": "fixed_mask",
            }
            if is_main:
                #pass
                print(
                    f"[TemporalProcessor] global_topk_over_coeffs use_hard={use_hard} K_target={meta.get('K_total_target')} "
                    f"mask_sum_mean={meta.get('mask_sum_mean'):.2f} use_fixed_mask={self.use_fixed_mask}"
                )
            
            return masked, mask5, meta , original
        else:
            #raise ValueError("No fixed mask provided, Currently we only try fixed mask mode")
            print(f"[TemporalProcessor] no fixed mask provided, using dynamic top-k")
        
            #=== 动态top-k模式：在所有 C*T*H*W 位置上做 top-k
            K_total = int(max(0, min(Ctot * T * H * W, round(Ctot * T * H * W * keep_ratio))))
            scores = self.score_channels(coeffs)
            if use_hard:
                mask = self.hard_topk_mask(scores, K_total)  # (B, C, T, H, W)
            else:
                mask = self.soft_topk_mask(scores, K_total, temperature=temperature)  # (B, C, T, H, W)
            
            # 构建临时meta用于打印
            temp_mask_sum_mean = mask.view(B, -1).sum(dim=1).mean().item()
            if is_main:
                print(
                    f"[TemporalProcessor] global_topk_over_coeffs use_hard={use_hard} K_target={K_total} "
                    f"mask_sum_mean={temp_mask_sum_mean:.2f} use_fixed_mask={self.use_fixed_mask} use_hard={use_hard}"
                    f"wflet_K_total={Ctot * T * H * W}, keep_ratio={keep_ratio}"
                )
            
            # 直接使用5D mask，不需要view
            mask5 = mask  # 保持命名一致性，实际上已经是 (B, C, T, H, W)
            masked = coeffs * mask5
        
        # 记录通道选择统计信息（展平后记录）
        mask_flat = mask.view(B, -1)  # (B, C*T*H*W)
        scores_flat = scores.view(B, -1)  # (B, C*T*H*W)
        self._record_channel_selection(mask_flat, scores_flat)
        
        meta = {
            "K_total_target": K_total,
            "mask_sum_mean": mask.view(B, -1).sum(dim=1).mean().item(),
            "mode": "dynamic_topk",
        }
        return masked, mask5, meta


    def compress_forward(self, coeffs: torch.Tensor):

        # 3D Haar小波变换（可选多次变换）
        try:
            is_main = os.environ.get("LOCAL_RANK", "0") == "0"
        except Exception:
            is_main = True

        coeffs=self.dwt3d(coeffs)

        if self.multi_wavelet:
            for _ in range(getattr(self, "wavelet_levels", 2)):
                coeffs = self.dwt_temporal(coeffs)
        B, full_C, T_w, H_w, W_w = coeffs.shape
        masked_coeffs, masks_5d, meta ,_= self.global_topk_over_coeffs(
            coeffs,
            keep_ratio=self.keep_ratio,
            temperature=self.temperature,
            score_mode=self.score_mode,
            var_boost=self.var_boost,
            exact_k=self.exact_k,
            use_hard=True,
            is_main=is_main
        )

        return masked_coeffs, masks_5d, meta

    
    def decompress_backward(
        self, 
        coeffs: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        三阶段反向解压：
        - Phase 1/2: 完整coeffs -> 逆3D Haar -> latent
        - Phase 3: 稀疏coeff_sparse -> 拼接 -> 逆3D Haar -> latent
        
        Args:
            coeffs: (B, 8*C, T, H, W) 完整的小波系数 (Phase 1/2)
            
        Returns:
            z: (B, C, 2*T, 2*H, 2*W) 重构的latent
        """
        if self.training_phase in ["phase1", "phase2"]:
            if self.multi_wavelet and coeffs is not None:
                for _ in range(getattr(self, "wavelet_levels", 2)):
                    coeffs = self.idwt_temporal(coeffs)
        coeffs=self.idwt3d(coeffs)
        return coeffs

class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                                            dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ],
                                            dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -1:, :, :].clone()
                    # if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        #conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  #* 0.5
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        #init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3,
                                         -1).permute(0, 1, 3,
                                                     2).contiguous().chunk(
                                                         3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity


class Encoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale):
        self.clear_cache()
        ## cache
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 32
        ## 对encode输入的x，按时间拆分为1、4、4、4....
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 32 * (i - 1):1 + 32 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    def decode(self, z, scale):
        self.clear_cache()
        # z: [b,c,t,h,w]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
        self.clear_cache()
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        #cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _video_vae(pretrained_path=None, z_dim=None, device='cpu', **kwargs):
    """
    Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.
    """
    # params
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0)
    cfg.update(**kwargs)

    # init model
    with torch.device('meta'):
        model = WanVAE_(**cfg)

    # load checkpoint
    logging.info(f'loading {pretrained_path}')
    model.load_state_dict(
        torch.load(pretrained_path, map_location=device), assign=True)

    return model


class Wan2_1_VAE:

    def __init__(self,
                 z_dim=16,
                 vae_pth='cache/vae_step_411000.pth',
                 dtype=torch.float,
                 device="cuda"):
        self.dtype = dtype
        self.device = device

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = _video_vae(
            pretrained_path=vae_pth,
            z_dim=z_dim,
        ).eval().requires_grad_(False).to(device)

    def encode(self, videos):
        """
        videos: A list of videos each with shape [C, T, H, W].
        """
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0)
                for u in videos
            ]

    def decode(self, zs):
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.decode(u.unsqueeze(0),
                                  self.scale).float().clamp_(-1, 1).squeeze(0)
                for u in zs
            ]


@ModelRegistry.register("Wan2_1_VAE_MultiWavelet")
class Wan2_1_VAE_MultiWavelet(VideoBaseAE):
    """
    Trainable version of Wan2_1_VAE that is compatible with the training infrastructure.
    This class wraps the WanVAE_ model to work with the DDP training script.
    """
    
    @register_to_config
    def __init__(
        self,
        z_channels: int = 16,
        dim: int = 96,
        dim_mult: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List = [],
        temporal_downsample: List[bool] = [False, True, True],
        dropout: float = 0.0,
        # Latent normalization (disabled by default for training)
        use_latent_normalization: bool = False,


        training_phase: str = "phase1",  # "phase1", "phase2", or "phase3"
        temporal_consistency_mode: str = "tv_l1",  # "tv_l1", "temporal_variance", etc.
        keep_ratio: float = 0.5,
        temperature: float = 0.1,
        score_mode: str = "l2",
        hl_boost: float = 1.0,
        var_boost: float = 0.3,
        compress_strategy: str = "ungrouped",  # "ungrouped" or "groupwise"
        exact_k: bool = False,
        multi_wavelet: bool = True,
        global_topk: bool = True,
        use_fixed_mask: bool = True,  # 是否使用固定mask
        fixed_mask_path: str = None,   # 固定mask文件路径
        not_compress: bool = False,

    ):
        super().__init__()
        
        self.z_channels = z_channels
        self.use_latent_normalization = use_latent_normalization
        
        # Only set up normalization if explicitly requested

        
        # Create the encoder and decoder
        self.encoder = Encoder3d(
            dim=dim,
            z_dim=z_channels * 2,  # *2 for mu and log_var
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temporal_downsample,
            dropout=dropout
        )
        
        self.conv1 = CausalConv3d(z_channels * 2, z_channels * 2, 1)
        self.conv2 = CausalConv3d(z_channels, z_channels, 1)
        
        temporal_upsample = temporal_downsample[::-1]
        self.decoder = Decoder3d(
            dim=dim,
            z_dim=z_channels,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_upsample=temporal_upsample,
            dropout=dropout
        )
        
        # Cache for temporal processing
        self._clear_cache()


        self.temporal_processor = HaarCompressedLatentProcessor(
            dwt3d_cls="HaarWaveletTransform3D",
            idwt3d_cls="InverseHaarWaveletTransform3D",
            training_phase=training_phase,
            temporal_consistency_mode=temporal_consistency_mode,
            keep_ratio=keep_ratio,
            temperature=temperature,
            score_mode=score_mode,
            hl_boost=hl_boost,
            var_boost=var_boost,
            compress_strategy=compress_strategy,
            exact_k=exact_k,
            multi_wavelet=multi_wavelet,
            global_topk=global_topk,
            use_fixed_mask=use_fixed_mask,
            fixed_mask_path=fixed_mask_path,
            not_compress=not_compress,
        )
    
    def _clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num
    
    def _encode_with_cache(self, x):
        """Encode with temporal caching (from original implementation)"""
        self._clear_cache()
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 32
        
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx
                )
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 32 * (i - 1):1 + 32 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx
                )
                out = torch.cat([out, out_], 2)
        
        return out
    
    def _decode_with_cache(self, z):
        """Decode with temporal caching (from original implementation)"""
        self._clear_cache()
        iter_ = z.shape[2]
        x = self.conv2(z)
        
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx
                )
            else:
                out_ = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx
                )
                out = torch.cat([out, out_], 2)
        
        self._clear_cache()
        return out
    
    def encode(self, x: torch.Tensor, return_dict: bool = True):
        """
        Encode input video to latent distribution.
        
        Args:
            x: Input video tensor of shape [B, C, T, H, W], values in [0, 1] or [-1, 1]
            return_dict: Whether to return AutoencoderKLOutput
        
        Returns:
            AutoencoderKLOutput with latent_dist
        """
        # Encode
        h = self._encode_with_cache(x)
        
        # Get mu and log_var
        moments = self.conv1(h)
        mu, log_var = moments.chunk(2, dim=1)
        
        # Normalize using learned statistics (only if enabled)

        
        # Create distribution (concatenate mu and log_var)
        parameters = torch.cat([mu, log_var], dim=1)
        posterior = DiagonalGaussianDistribution(parameters)
        
        if not return_dict:
            return (posterior,)
        
        return AutoencoderKLOutput(latent_dist=posterior)
    
    def decode(self, z: torch.Tensor, return_dict: bool = True):
        """
        Decode latent to video.
        
        Args:
            z: Latent tensor of shape [B, C, T, H, W]
            return_dict: Whether to return DecoderOutput
        
        Returns:
            DecoderOutput with sample
        """
        # Denormalize using learned statistics

        
        # Decode
        dec = self._decode_with_cache(z)
        
        if not return_dict:
            return (dec,)
        
        return DecoderOutput(sample=dec)
    
    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = True,
        return_dict: bool = True,
        generator: torch.Generator = None,
    ):
        """
        Forward pass for training.
        
        Args:
            sample: Input video tensor [B, C, T, H, W]
            sample_posterior: Whether to sample from posterior
            return_dict: Whether to return dict
            generator: Random generator for sampling
        
        Returns:
            ForwardOutput with sample and latent_dist
        """
        # Encode
        enc_output = self.encode(sample, return_dict=True)
        posterior = enc_output.latent_dist
        
        # Sample or use mode
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        

        #haar decomposition
        compress_latent=z[:,:,:8]
        result = self.temporal_processor.compress_forward(compress_latent) #z(1,32,9,32,32)
        masked_coeffs, lowfreq_consistency_loss, _ = result #masked_coeffs(1,1024,2,16,16)

        # Decode

        depress_latent = self.temporal_processor.decompress_backward(coeffs=masked_coeffs) #z(1,32,16,32,32)
        z=torch.cat([depress_latent,z[:,:,8:]],dim=2)
        dec_output = self.decode(z, return_dict=True)
        
        if not return_dict:
            return (dec_output.sample, posterior)
        
        return ForwardOutput(
            sample=dec_output.sample,
            latent_dist=posterior,
        )
    
    def get_encoder(self):
        """Return encoder modules for optimizer"""
        return [self.encoder, self.conv1]
    
    def get_decoder(self):
        """Return decoder modules for optimizer"""
        return [self.decoder, self.conv2]
    
    def get_last_layer(self):
        """Return last layer for adaptive weight computation"""
        return self.decoder.head[-1].weight