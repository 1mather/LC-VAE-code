

from typing import Any, List, Tuple, Optional
import torch
import torch.nn as nn
import os
from collections import deque
import math

from ..modules import (
    ResnetBlock2D,
    ResnetBlock3D,
    Conv2d,
    HaarWaveletTransform3D,
    InverseHaarWaveletTransform3D,
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


# ========= 复用原始Encoder和Decoder =========
# 这里直接从原始文件导入，避免重复代码
from .modeling_wfvae import Encoder, Decoder


class Decoder_compressed(VideoBaseAE):

    @register_to_config
    def __init__(
        self,
        latent_dim: int = 8,
        base_channels: int = 128,
        num_resblocks: int = 2,
        dropout: float = 0.0,
        energy_flow_hidden_size: int = 128,
        attention_type: str = "AttnBlock3DFix",
        use_attention: bool = True,
        norm_type: str = "groupnorm",
        t_interpolation: str = "nearest",
        connect_res_layer_num: int = 1,
        l1_upsample_block: str = "Upsample",
        l1_upsample_wavelet: str = "InverseHaarWaveletTransform2D",
        l2_upsample_block: str = "Spatial2xTime2x3DUpsample",
        l2_upsample_wavelet: str = "InverseHaarWaveletTransform3D",
    ) -> None:
        super().__init__()
        self.energy_flow_hidden_size = energy_flow_hidden_size

        self.conv_in = CausalConv3d(
            latent_dim, base_channels * 4, kernel_size=3, stride=1, padding=1
        )
        mid_layers = [
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4,
                dropout=dropout,
                norm_type=norm_type,
            ),
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4 + energy_flow_hidden_size,
                dropout=dropout,
                norm_type=norm_type,
            ),
        ]
        
        if use_attention:
            mid_layers.insert(
                1,
                resolve_str_to_obj(attention_type)(
                    in_channels=base_channels * 4, norm_type=norm_type
                ),
            )

        self.mid = nn.Sequential(*mid_layers)

        upsample_depth = 0
        self.up2 = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=base_channels * 4,
                    out_channels=base_channels * 4,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for _ in range(num_resblocks)
            ],
            resolve_str_to_obj(l2_upsample_block)(
                base_channels * 4, base_channels * 4, t_interpolation=t_interpolation, depth=upsample_depth
            ),
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4 + energy_flow_hidden_size,
                dropout=dropout,
                norm_type=norm_type,
            ),
        )
        upsample_depth += 1
        
        self.up1 = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=base_channels * (4 if i == 0 else 2),
                    out_channels=base_channels * 2,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for i in range(num_resblocks)
            ],
            resolve_str_to_obj(l1_upsample_block)(
                in_channels=base_channels * 2, out_channels=base_channels * 2, t_interpolation=t_interpolation, depth=upsample_depth
            ),
            ResnetBlock3D(
                in_channels=base_channels * 2,
                out_channels=base_channels * 2,
                dropout=dropout,
                norm_type=norm_type,
            ),
        )
        
        self.layer = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=base_channels * (2 if i == 0 else 1),
                    out_channels=base_channels,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for i in range(2)
            ],
        )
        # Connection
        if l1_upsample_block == "Upsample":  # Bad code. For temporal usage.
            l1_channels = 12
        else:
            l1_channels = 24
        self.connect_l1 = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=energy_flow_hidden_size,
                    out_channels=energy_flow_hidden_size,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for _ in range(connect_res_layer_num)
            ],
            Conv2d(
                energy_flow_hidden_size, l1_channels, kernel_size=3, stride=1, padding=1
            ),
        )
        self.connect_l2 = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=energy_flow_hidden_size,
                    out_channels=energy_flow_hidden_size,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for _ in range(connect_res_layer_num)
            ],
            Conv2d(energy_flow_hidden_size, 24, kernel_size=3, stride=1, padding=1),
        )
        # Out
        self.norm_out = Normalize(base_channels, norm_type=norm_type)
        self.conv_out = Conv2d(base_channels, 24, kernel_size=3, stride=1, padding=1)

        self.inverse_wavelet_transform_out = InverseHaarWaveletTransform3D()
        self.inverse_wavelet_transform_l1 = resolve_str_to_obj(l1_upsample_wavelet)()
        self.inverse_wavelet_transform_l2 = resolve_str_to_obj(l2_upsample_wavelet)()

    def forward(self, coeffs_low, coeffs_high):

        h = self.conv_in(coeffs_low) #z: ([1, 16, 7, 32, 32]) h([1, 768, 7, 32, 32])
        h = torch.concat([h, coeffs_high], dim=1)
        h = self.mid(h) #h . torch.Size([1, 896, 7, 32, 32])
        l2_coeffs = self.connect_l2(h[:, -self.energy_flow_hidden_size :])
        l2 = self.inverse_wavelet_transform_l2(l2_coeffs)

        h = self.up2(h[:, : -self.energy_flow_hidden_size]) #torch.Size([1, 896, 13, 64, 64])

        l1_coeffs = h[:, -self.energy_flow_hidden_size :]
        l1_coeffs = self.connect_l1(l1_coeffs)
        l1_coeffs[:, :3] = l1_coeffs[:, :3] + l2
        l1 = self.inverse_wavelet_transform_l1(l1_coeffs)

        h = self.up1(h[:, : -self.energy_flow_hidden_size])#torch.Size([1, 384, 13, 128, 128])

        h = self.layer(h) #([1, 192, 13, 128, 128])
        h = self.norm_out(h) 
        h = nonlinearity(h)
        h = self.conv_out(h) #([1, 24, 13, 128, 128])
        h[:, :3] = h[:, :3] + l1

        dec = self.inverse_wavelet_transform_out(h)
        return dec, (l1_coeffs, l2_coeffs)
# ========= Latent时间压缩处理器 =========

class TemporalCompressedLatentProcessor(nn.Module):
    """
    Top-K时间压缩的Latent处理器（三阶段训练）
    
    功能：
    1. 对latent做3D Haar小波变换
    2. Phase 1: 基础训练，不加任何一致性损失或压缩
    3. Phase 2: 添加 low-frequency-consistency 损失，不压缩
    4. Phase 3: 使用Top-K选择重要的频率系数进行稀疏化压缩
    """
    def __init__(
        self, 
        dwt3d_cls: str = "HaarWaveletTransform3D", 
        idwt3d_cls: str = "InverseHaarWaveletTransform3D",
        training_phase: str = "phase1",  # "phase1", "phase2", or "phase3"
        temporal_consistency_mode: str = "tv_l1",  # "tv_l1", "temporal_variance", etc.
        keep_ratio: float = 0.5,
        temperature: float = 0.5,
        score_mode: str = "mix",
        hl_boost: float = 1.0,
        var_boost: float = 0.3,
        compress_strategy: str = "ungrouped",  # "ungrouped" or "groupwise"
        exact_k: bool = False,
    ):
        super().__init__()
        DWT3D = resolve_str_to_obj(dwt3d_cls)
        IDWT3D = resolve_str_to_obj(idwt3d_cls)
        self.dwt3d = DWT3D()
        self.idwt3d = IDWT3D()
        
        # 训练阶段
        self.training_phase = training_phase
        self.temporal_consistency_mode = temporal_consistency_mode
        
        # Top-K 参数
        self.keep_ratio = keep_ratio
        self.temperature = temperature
        self.score_mode = score_mode
        self.hl_boost = hl_boost
        self.var_boost = var_boost
        self.compression_strategy = compress_strategy
        self.exact_k = exact_k
    
    def set_training_phase(self, phase: str):
        """切换训练阶段"""
        assert phase in ["phase1", "phase2", "phase3"]
        self.training_phase = phase
        print(f"✓ Switched to training {phase}")
    
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
        
        if mode == "temporal_variance":
            # 直接对时间维求方差并平均
            return LL.var(dim=2, unbiased=False).mean(), ll_variance
        elif mode == "tv_l1":
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
        Hard Top-K 掩码：直接取前 k 个为 1，其余为 0
        """
        B, N = scores.shape
        if k <= 0:
            return torch.zeros_like(scores)
        if k >= N:
            return torch.ones_like(scores)
        idx = torch.topk(scores, k, dim=1).indices  # (B,k)
        mask = torch.zeros_like(scores)
        return mask.scatter(1, idx, 1.0)

    @staticmethod
    def soft_topk_mask(scores: torch.Tensor, k: int, temperature: float = 0.5, exact_k: bool = False):
        """
        可微 Top-K 掩码近似（batch 独立）:
        - scores: (B, N)
        - k:      int, 0<=k<=N
        - temperature: 越小越接近硬 top-k；越大越平滑
        - exact_k: 若为 True，会把掩码归一化使 batch 内和 ≈ k（会让掩码可能>1，必要时可再 clamp）
        返回:
        - mask: (B, N) ∈ (0,1)（或经归一化后可能>1）
        """
        B, N = scores.shape
        if k <= 0:
            return torch.zeros_like(scores)
        if k >= N:
            return torch.ones_like(scores)

        # 第 k 大作为阈值（逐 batch）
        # topk 返回前 k 个降序，取第 k 个即阈值
        tau = torch.topk(scores, k, dim=1).values[:, -1].unsqueeze(1)  # (B,1)

        # Sigmoid 软门控：分数高于阈值→接近1，低于→接近0
        mask = torch.sigmoid((scores - tau) / max(1e-6, float(temperature)))  # (B,N)

        if exact_k:
            # 把软权重总和缩放到 k（注意可能>1；如需 [0,1] 可再 clamp）
            sumw = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
            mask = mask * (k / sumw)

        return mask

    @staticmethod
    def ungrouped_soft_topk(
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH,
        keep_ratio=0.5, temperature=0.5,
        score_mode="l2", var_boost=0.3,
        exact_k=False, hl_boost: float = 1.0, use_hard: bool = False
    ):
        """
        不分组：在 8*C 个通道上一次性做 soft-top-k。
        各子带形状: (B, C, T, H, W)
        返回:
          - coeff_sparse: 稀疏后的各子带
          - masks_5d:     (B,C,1,1,1) 掩码
          - meta:         统计信息
        """
        def score(x, mode="l2"):
            # (B,C,T,H,W) → (B,C)
            if mode == "l2":
                s = (x**2).sum(dim=(2,3,4))
            elif mode == "var":
                s = x.var(dim=(2,3,4), unbiased=False)
            else:
                e = (x**2).sum(dim=(2,3,4))
                v = x.var(dim=(2,3,4), unbiased=False)
                s = e + var_boost * v
            return s

        B, C, T, H, W = LLL.shape

        # 1) 计算每子带的通道分数，并拼接成 (B, 8C)
        S_list = [
            score(LLL, score_mode),
            score(LLH, score_mode),
            score(LHL, score_mode),
            score(LHH, score_mode),
            score(HLL, score_mode),
            score(HLH, score_mode),
            score(HHL, score_mode),
            score(HHH, score_mode),
        ]
        # 可选：对时高组加权，提升动态优先级（H** 开头的四个子带）
        if hl_boost != 1.0:
            for i in range(4, 8):
                S_list[i] = S_list[i] * hl_boost
        S_all = torch.cat(S_list, dim=1)  # (B, 8C)

        # 2) 全局 top-k（软/硬）
        total_C = 8 * C
        K_total = int(max(0, min(total_C, round(total_C * keep_ratio))))
        if use_hard:
            mask_all = TemporalCompressedLatentProcessor.hard_topk_mask(S_all, K_total)
        else:
            mask_all = TemporalCompressedLatentProcessor.soft_topk_mask(
                S_all, K_total, temperature=temperature, exact_k=exact_k
            )  # (B, 8C)

        # 3) 切分回 8 个子带的 (B,C) 掩码
        masks = {}
        names = ["LLL","LLH","LHL","LHH","HLL","HLH","HHL","HHH"]
        for i, name in enumerate(names):
            masks[name] = mask_all[:, i*C:(i+1)*C]  # (B,C)

        # 4) 广播到 (B,C,1,1,1) 并应用
        def apply_mask(x, m):
            m5 = m.view(m.shape[0], m.shape[1], 1, 1, 1)
            return x * m5, m5

        coeff_sparse, masks_5d = {}, {}
        for name, x in zip(
            names, [LLL,LLH,LHL,LHH,HLL,HLH,HHL,HHH]
        ):
            xs, m5 = apply_mask(x, masks[name])
            coeff_sparse[name] = xs
            masks_5d[name] = m5

        # 5) 统计信息（近似保留通道数，可用掩码求和或四舍五入）
        ks_per_band = {name: masks[name].sum(dim=1).mean().item() for name in names}  # 平均每样本保留“权重和”
        meta = {
            "K_total_target": K_total,
            "K_total_soft_sum_mean": mask_all.sum(dim=1).mean().item(),
            "ks_per_band_mean": ks_per_band,
        }
        return coeff_sparse, masks_5d, meta

    @staticmethod
    def groupwise_soft_topk(
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH,
        keep_ratio=0.5, temperature=0.5,
        score_mode="mix", hl_boost=1.0, var_boost=0.3,
        alloc = {
            "G1_total": 0.65, "G2_total": 0.35,
            "G1": {"LLL":0.45, "LLH":0.20, "LHL":0.20, "LHH":0.15},
            "G2": {"HLL":0.40, "HLH":0.20, "HHL":0.20, "HHH":0.20},
        },
        use_hard: bool = False,
    ):
        """
        各子带形状: (B, C, T, H, W)
        返回: 掩码 dict（每块一个 (B,C,1,1,1)），以及稀疏后的系数字典
        """
        def score(x, mode="mix"):
            # 计算每通道打分 (B,C)
            if mode == "l2":
                s = (x**2).sum(dim=(2,3,4))                 # 能量
            elif mode == "var":
                s = x.var(dim=(2,3,4), unbiased=False)      # 方差
            else:
                e = (x**2).sum(dim=(2,3,4))
                v = x.var(dim=(2,3,4), unbiased=False)
                s = e + var_boost * v                       # 混合打分
            return s

        B, C, T, H, W = LLL.shape
        total_C = 8*C
        K_total = int(total_C * keep_ratio)

        # 组间预算
        K_G1 = max(1, int(K_total * alloc["G1_total"]))
        K_G2 = max(1, K_total - K_G1)

        # 每块预算（按比例四舍五入）
        def budget_group(Kg, suballoc):
            ks = {name: max(0, int(Kg * w)) for name, w in suballoc.items()}
            # 调整四舍五入误差
            diff = Kg - sum(ks.values())
            if diff > 0:
                # 把剩余分配给权重最大的那些
                sorted_names = sorted(suballoc.keys(), key=lambda n: suballoc[n], reverse=True)
                for i in range(diff): ks[sorted_names[i % len(sorted_names)]] += 1
            return ks

        ks_G1 = budget_group(K_G1, alloc["G1"])
        ks_G2 = budget_group(K_G2, alloc["G2"])

        # 计算打分
        S = {
            "LLL": score(LLL, score_mode),
            "LLH": score(LLH, score_mode),
            "LHL": score(LHL, score_mode),
            "LHH": score(LHH, score_mode),
            "HLL": score(HLL, score_mode),
            "HLH": score(HLH, score_mode),
            "HHL": score(HHL, score_mode),
            "HHH": score(HHH, score_mode),
        }

        # 可选：对时高组加权，提升动态优先级
        if hl_boost != 1.0:
            for name in ["HLL","HLH","HHL","HHH"]:
                S[name] = S[name] * hl_boost

        # 逐块 soft-topk → mask
        masks = {}
        for name, k in {**ks_G1, **ks_G2}.items():
            k = max(0, min(k, C))
            if k == 0:
                masks[name] = torch.zeros_like(S[name])
            elif k >= C:
                masks[name] = torch.ones_like(S[name])
            else:
                if use_hard:
                    masks[name] = TemporalCompressedLatentProcessor.hard_topk_mask(S[name], k)  # (B,C)
                else:
                    masks[name] = TemporalCompressedLatentProcessor.soft_topk_mask(S[name], k, temperature=temperature)  # (B,C)

        # 扩展到 (B,C,1,1,1)，并作用到系数
        def apply_mask(x, m):
            m5 = m.view(m.shape[0], m.shape[1], 1, 1, 1)
            return x * m5, m5

        coeff_sparse = {}
        masks_5d = {}
        for name, x in zip(
            ["LLL","LLH","LHL","LHH","HLL","HLH","HHL","HHH"],
            [LLL,LLH,LHL,LHH,HLL,HLH,HHL,HHH]
        ):
            xs, m5 = apply_mask(x, masks[name])
            coeff_sparse[name] = xs
            masks_5d[name] = m5

        return coeff_sparse, masks_5d, {"K_total":K_total, "K_G1":K_G1, "K_G2":K_G2, "ks_G1":ks_G1, "ks_G2":ks_G2}

    def compress_forward(self, z: torch.Tensor):
        """
        三阶段前向压缩：
        - Phase 1: latent -> 3D Haar -> 不加损失，不压缩（纯基础训练）
        - Phase 2: latent -> 3D Haar -> 添加低频一致性损失，不压缩
        - Phase 3: latent -> 3D Haar -> Top-K稀疏化压缩
        
        Args:
            z: (B, C, T, H, W) 原始latent
            
        Returns:
            Phase 1: (coeffs, None)
                - coeffs: (B, 8*C, T/2, H/2, W/2) 完整的小波系数
                - None: 无一致性损失
            Phase 2: (coeffs, lowfreq_consistency_loss)
                - coeffs: (B, 8*C, T/2, H/2, W/2) 完整的小波系数
                - lowfreq_consistency_loss: (loss, variance) 低频一致性损失
            Phase 3: (coeff_sparse, masks_5d, group_info)
                - coeff_sparse: dict of sparse coefficients {name: (B, C, T, H, W)}
                - masks_5d: dict of masks {name: (B, C, 1, 1, 1)}
                - group_info: dict of group information
        """
        # 3D Haar小波变换
        coeffs = self.dwt3d(z)  # (B, 8*C, T/2, H/2, W/2)
        
        B, full_C, T_w, H_w, W_w = coeffs.shape
        C = z.shape[1]
        
        # 分离8个子带
        # 标准顺序: LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH
        LLL = coeffs[:, 0*C:1*C]
        LLH = coeffs[:, 1*C:2*C]
        LHL = coeffs[:, 2*C:3*C]
        LHH = coeffs[:, 3*C:4*C]
        HLL = coeffs[:, 4*C:5*C]
        HLH = coeffs[:, 5*C:6*C]
        HHL = coeffs[:, 6*C:7*C]
        HHH = coeffs[:, 7*C:8*C]
        
        if self.training_phase == "phase1":
            # Phase 1: 基础训练，不加任何损失，不压缩
            return coeffs, None
            
        elif self.training_phase == "phase2":
            # Phase 2: 添加低频一致性损失，不压缩
            # 计算低频分量(时间低频的4个子带)的一致性损失
            temporal_low_freq = coeffs[:, 0*C:4*C]  # LLL, LLH, LHL, LHH
            lowfreq_consistency_loss = self.lowfreq_consistency_loss(
                temporal_low_freq, mode=self.temporal_consistency_mode
            )
            return coeffs, lowfreq_consistency_loss
            
        elif self.training_phase == "phase3":
            # Phase 3: Top-K稀疏化压缩（可选策略）
            use_hard = (not self.training)
            if self.compression_strategy == "ungrouped":
                coeff_sparse, masks_5d, group_info = TemporalCompressedLatentProcessor.ungrouped_soft_topk(
                    LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH,
                    keep_ratio=self.keep_ratio, 
                    temperature=self.temperature,
                    score_mode=self.score_mode, 
                    var_boost=self.var_boost,
                    exact_k=self.exact_k,
                    hl_boost=self.hl_boost,
                    use_hard=use_hard,
                )
            elif self.compression_strategy == "groupwise":
                coeff_sparse, masks_5d, group_info = TemporalCompressedLatentProcessor.groupwise_soft_topk(
                    LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH,
                    keep_ratio=self.keep_ratio, 
                    temperature=self.temperature,
                    score_mode=self.score_mode, 
                    hl_boost=self.hl_boost, 
                    var_boost=self.var_boost,
                    use_hard=use_hard,
                )
            else:
                raise ValueError(f"Unknown compression_strategy: {self.compression_strategy}")
            return coeff_sparse, masks_5d, group_info
        else:
            raise ValueError(f"Unknown training_phase: {self.training_phase}")
    
    def decompress_backward(
        self, 
        coeffs: torch.Tensor = None,
        coeff_sparse: dict = None,
    ) -> torch.Tensor:
        """
        三阶段反向解压：
        - Phase 1/2: 完整coeffs -> 逆3D Haar -> latent
        - Phase 3: 稀疏coeff_sparse -> 拼接 -> 逆3D Haar -> latent
        
        Args:
            coeffs: (B, 8*C, T, H, W) 完整的小波系数 (Phase 1/2)
            coeff_sparse: dict of sparse coefficients (Phase 3)
            
        Returns:
            z: (B, C, 2*T, 2*H, 2*W) 重构的latent
        """
        if self.training_phase in ["phase1", "phase2"]:
            # Phase 1/2: 使用完整系数
            z = self.idwt3d(coeffs)
        elif self.training_phase == "phase3":
            # Phase 3: 拼接稀疏系数
            order = ["LLL","LLH","LHL","LHH","HLL","HLH","HHL","HHH"]
            coeffs_full = torch.cat([coeff_sparse[k] for k in order], dim=1)
            z = self.idwt3d(coeffs_full)
        else:
            raise ValueError(f"Unknown training_phase: {self.training_phase}")
        return z




# ========= 主模型 =========

@ModelRegistry.register("Latent_WFVAE_TemporalCompressed_TopK")
class WFVAETemporalCompressedModelTopK(VideoBaseAE):
    """
    Top-K时间压缩版WFVAE（三阶段训练）
    
    训练流程：
    1. Phase 1: 基础训练，不加任何损失，不压缩
    2. Phase 2: 添加low-frequency-consistency损失，不压缩  
    3. Phase 3: 使用Top-K稀疏化压缩
    """
    
    @register_to_config
    def __init__(
        self,
        latent_dim: int = 8,
        base_channels: int = 128,
        encoder_num_resblocks: int = 2,
        encoder_energy_flow_hidden_size: int = 64,
        decoder_num_resblocks: int = 2,
        decoder_energy_flow_hidden_size: int = 128,
        attention_type: str = "AttnBlock3DFix",
        use_attention: bool = True,
        dropout: float = 0.0,
        norm_type: str = "groupnorm",
        t_interpolation: str = "nearest",
        connect_res_layer_num: int = 1,
        scale: List[float] = [0.18215, 0.18215, 0.18215, 0.18215],
        shift: List[float] = [0, 0, 0, 0],
        # Module config
        l1_downsample_block: str = "Downsample",
        l1_downsample_wavelet: str = "HaarWaveletTransform2D",
        l2_downsample_block: str = "Spatial2xTime2x3DDownsample",
        l2_downsample_wavelet: str = "HaarWaveletTransform3D",
        l1_upsample_block: str = "Upsample",
        l1_upsample_wavelet: str = "InverseHaarWaveletTransform2D",
        l2_upsample_block: str = "Spatial2xTime2x3DUpsample",
        l2_upsample_wavelet: str = "InverseHaarWaveletTransform3D",
        # Temporal compression config
        training_phase: str = "phase1",  # "phase1", "phase2", "phase3"
        temporal_consistency_mode: str = "tv_l1",  # "tv_l1", "temporal_variance"
        temporal_consistency_weight: float = 5.0,
        # Top-K compression config (for phase3)
        keep_ratio: float = 0.5,
        temperature: float = 0.5,
        score_mode: str = "mix",
        hl_boost: float = 1.0,
        var_boost: float = 0.3,
        compress_strategy: str = "ungrouped",
        exact_k: bool = False,
    ) -> None:
        super().__init__()
        self.use_tiling = False
        self.use_quant_layer = False
        
        self.t_chunk_enc = None
        self.t_chunk_dec = None
        self.temporal_size = None
        
        # 使用原始的Encoder和Decoder
        self.encoder = Encoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            num_resblocks=encoder_num_resblocks,
            energy_flow_hidden_size=encoder_energy_flow_hidden_size,
            dropout=dropout,
            use_attention=use_attention,
            norm_type=norm_type,
            l1_downsample_block=l1_downsample_block,
            l1_downsample_wavelet=l1_downsample_wavelet,
            l2_downsample_block=l2_downsample_block,
            l2_downsample_wavelet=l2_downsample_wavelet,
            attention_type=attention_type,
        )
        
        self.decoder = Decoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            num_resblocks=decoder_num_resblocks,
            energy_flow_hidden_size=decoder_energy_flow_hidden_size,
            dropout=dropout,
            use_attention=use_attention,
            norm_type=norm_type,
            t_interpolation=t_interpolation,
            connect_res_layer_num=connect_res_layer_num,
            l1_upsample_block=l1_upsample_block,
            l1_upsample_wavelet=l1_upsample_wavelet,
            l2_upsample_block=l2_upsample_block,
            l2_upsample_wavelet=l2_upsample_wavelet,
            attention_type=attention_type,
        )
        
        # 时间压缩处理器（三阶段训练版本）
        self.temporal_processor = TemporalCompressedLatentProcessor(
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
        )
        
        self.training_phase = training_phase
        self.temporal_consistency_mode = temporal_consistency_mode
        self.temporal_consistency_weight = temporal_consistency_weight
        
        # Cache offset设置（与原始WFVAE相同）
        if l1_downsample_block == "Downsample":
            self.temporal_uptimes = 4
            self._set_cache_offset(
                [
                    self.decoder.up2,
                    self.decoder.connect_l2,
                    self.decoder.conv_in,
                    self.decoder.mid,
                ],
                1,
            )
            self._set_cache_offset(
                [
                    self.decoder.up2[-2:],
                    self.decoder.up1,
                    self.decoder.connect_l1,
                    self.decoder.layer,
                ],
                2,
            )
        else:
            self.temporal_uptimes = 8
            self._set_cache_offset(
                [
                    self.decoder.up2,
                    self.decoder.connect_l2,
                    self.decoder.conv_in,
                    self.decoder.mid,
                ],
                1,
            )
            self._set_cache_offset(
                [self.decoder.up2[-2:], self.decoder.connect_l1, self.decoder.up1], 2
            )
            self._set_cache_offset([self.decoder.up1[-2:], self.decoder.layer], 4)
        
        print(f"✓ Initialized WFVAE with Top-K 3-Phase Temporal Compression")
        print(f"  Training phase: {training_phase}")
        print(f"  Phase 1: Basic training (no loss, no compression)")
        print(f"  Phase 2: Add low-frequency-consistency loss (no compression)")
        print(f"  Phase 3: Top-K sparse compression (keep_ratio={keep_ratio}, strategy={compress_strategy}, exact_k={exact_k})")
        print(f"  Temporal consistency mode: {temporal_consistency_mode}")
        print(f"  Temporal consistency weight: {temporal_consistency_weight}")
    
    # ========= 辅助方法 =========
    
    def set_training_phase(self, phase: str):
        """切换训练阶段"""
        self.training_phase = phase
        self.temporal_processor.set_training_phase(phase)
    
    def set_consistency_weight(self, weight: float):
        """动态调整一致性损失权重"""
        self.temporal_consistency_weight = weight
    
    def get_encoder(self):
        if self.use_quant_layer:
            return [self.quant_conv, self.encoder]
        return [self.encoder]
    
    def get_decoder(self):
        if self.use_quant_layer:
            return [self.post_quant_conv, self.decoder]
        return [self.decoder]
    
    def _empty_causal_cached(self, parent):
        for name, module in parent.named_modules():
            if hasattr(module, "causal_cached"):
                module.causal_cached = deque()
    
    def _set_causal_cached(self, enable_cached=True):
        for name, module in self.named_modules():
            if hasattr(module, "enable_cached"):
                module.enable_cached = enable_cached
    
    def _set_cache_offset(self, modules, cache_offset=0):
        for module in modules:
            for submodule in module.modules():
                if hasattr(submodule, "cache_offset"):
                    submodule.cache_offset = cache_offset
    
    def _set_first_chunk(self, is_first_chunk=True):
        for module in self.modules():
            if hasattr(module, "is_first_chunk"):
                module.is_first_chunk = is_first_chunk
    
    def build_chunk_start_end(self, t, decoder_mode=False):
        start_end = [[0, 1]]
        start = 1
        end = start
        while True:
            if start >= t:
                break
            end = min(t, end + (self.t_chunk_dec if decoder_mode else self.t_chunk_enc))
            start_end.append([start, end])
            start = end
        return start_end
    
    # ========= 编码/解码接口 =========
    
    def encode(self, x, return_dict=True):
        """
        编码输入视频为潜在表示并进行时间处理
        
        Args:
            x: (B, C, T, H, W) 输入视频
            return_dict: 是否返回字典格式
            
        Returns:
            AutoencoderKLOutput with:
                - latent_dist: DiagonalGaussianDistribution
                - extra_output: 根据阶段不同，返回不同内容
                  Phase 1: (l1, l2, coeffs, None)
                  Phase 2: (l1, l2, coeffs, lowfreq_consistency_loss)
                  Phase 3: (l1, l2, coeff_sparse, masks_5d, group_info)
        """
        self._empty_causal_cached(self.encoder)
        self._set_first_chunk(True)
        
        if self.use_tiling:
            h = self.tile_encode(x)
            l1, l2 = None, None
        else:
            h, (l1, l2) = self.encoder(x)
            if self.use_quant_layer:
                h = self.quant_conv(h)
        
        posterior = DiagonalGaussianDistribution(h)
        z = posterior.sample()
        
        # 根据阶段进行不同处理
        result = self.temporal_processor.compress_forward(z)
        
        if self.training_phase in ["phase1", "phase2"]:
            coeffs, lowfreq_consistency_loss = result
            extra_output = (l1, l2, coeffs, lowfreq_consistency_loss)
        elif self.training_phase == "phase3":
            coeff_sparse, masks_5d, group_info = result
            extra_output = (l1, l2, coeff_sparse, masks_5d, group_info)
        
        if return_dict:
            return AutoencoderKLOutput(
                latent_dist=posterior, 
                extra_output=extra_output
            )
        else:
            return posterior, extra_output

    
    def decode(self, coeffs=None, coeff_sparse=None, return_dict=True):
        """
        从潜在表示解码为视频
        
        Args:
            coeffs: (B, 8*C, T, H, W) 完整系数 (Phase 1/2)
            coeff_sparse: dict of sparse coefficients (Phase 3)
            return_dict: 是否返回字典格式
            
        Returns:
            DecoderOutput with sample and extra_output
        """
        self._empty_causal_cached(self.decoder)
        self._set_first_chunk(True)
        
        # 从压缩格式重构完整latent
        z = self.temporal_processor.decompress_backward(
            coeffs=coeffs,
            coeff_sparse=coeff_sparse
        )
        
        if self.use_tiling:
            dec = self.tile_decode(z)
            l1, l2 = None, None
        else:
            if self.use_quant_layer:
                z = self.post_quant_conv(z)
            dec, (l1, l2) = self.decoder(z)
        
        if return_dict:
            return DecoderOutput(sample=dec, extra_output=(l1, l2))
        else:
            return dec, (l1, l2)
    
    def forward(self, input, sample_posterior=True):
        """
        前向传播：三阶段训练统一
        
        流程：
        1. Encoder: video -> h -> z
        2. 3D Haar变换 + 阶段处理：
           - Phase 1: z -> coeffs (不压缩，无损失)
           - Phase 2: z -> coeffs (不压缩，计算低频一致性损失)
           - Phase 3: z -> coeff_sparse (Top-K压缩)
        3. 反变换 + Decoder: z -> video
        """
        # Encode
        encode_output = self.encode(input, return_dict=True)
        posterior = encode_output.latent_dist
        
        if self.training_phase in ["phase1", "phase2"]:
            # Phase 1/2: 使用完整系数
            enc_l1, enc_l2, coeffs, lowfreq_consistency_loss = encode_output.extra_output
            
            decode_output = self.decode(
                coeffs=coeffs,
                return_dict=True
            )
            dec = decode_output.sample
            dec_l1, dec_l2 = decode_output.extra_output
            
            # 设置输出
            if lowfreq_consistency_loss is None:
                # Phase 1: 无一致性损失
                lowfreq_variance = torch.tensor(0.0, device=dec.device)
                consistency_loss = torch.tensor(0.0, device=dec.device)
            else:
                # Phase 2: 有一致性损失
                consistency_loss, lowfreq_variance = lowfreq_consistency_loss
            
            out = ForwardOutput(
                sample=dec,
                latent_dist=posterior,
                lowfreq_variance=lowfreq_variance,
                extra_output=(enc_l1, dec_l1, enc_l2, dec_l2),
            )
            out.lowfreq_consistency_loss = consistency_loss
            
        elif self.training_phase == "phase3":
            # Phase 3: Top-K稀疏压缩
            enc_l1, enc_l2, coeff_sparse, masks_5d, group_info = encode_output.extra_output

            # 计算低频一致性度量（保持梯度连接）
            names = ["LLL","LLH","LHL","LHH"]
            low_list = [coeff_sparse[n] if n in coeff_sparse else None for n in names]
            if all(v is not None for v in low_list):
                low = torch.cat(low_list, dim=1)
                # 低频一致性损失与方差
                consistency_loss_p3, lowfreq_variance_p3 = self.temporal_processor.lowfreq_consistency_loss(
                    low, mode=self.temporal_consistency_mode
                )
            else:
                device = next(self.parameters()).device
                consistency_loss_p3 = torch.tensor(0.0, device=device, requires_grad=True)
                lowfreq_variance_p3 = torch.tensor(0.0, device=device)

            decode_output = self.decode(
                coeff_sparse=coeff_sparse,
                return_dict=True
            )
            dec = decode_output.sample
            dec_l1, dec_l2 = decode_output.extra_output

            out = ForwardOutput(
                sample=dec,
                latent_dist=posterior,
                lowfreq_variance=lowfreq_variance_p3,
                extra_output=(enc_l1, dec_l1, enc_l2, dec_l2),
            )
            out.lowfreq_consistency_loss = consistency_loss_p3
        
        return out



    
    # ========= Tiling方法（可选，与原始WFVAE相同）=========
    
    def _auto_select_t_chunk(self):
        assert self.temporal_uptimes in [4, 8]
        t_compess_rate = self.temporal_uptimes
        downsample_times = int(math.log(t_compess_rate, 2))
        temporal_size = self.temporal_size
        dec_t_chunk = 2
        enc_t_chunk = t_compess_rate
        
        success_auto_select = False
        while dec_t_chunk < temporal_size and enc_t_chunk < temporal_size:
            T_list = [temporal_size]
            for i in range(downsample_times):
                T_list.append((T_list[-1] - 1) // 2 + 1)
            
            if (T_list[-1] - 1) % dec_t_chunk == 1:
                dec_t_chunk *= 2
                continue
            
            for inner_T in T_list[:-1]:
                if (inner_T - 1) % 2 != 0:
                    enc_t_chunk *= 2
                    continue
                
                if (inner_T - 1) % enc_t_chunk == 1 and (inner_T - 1) / enc_t_chunk > 1:
                    enc_t_chunk *= 2
                    continue
            
            success_auto_select = True
            break
        
        if not success_auto_select:
            raise ValueError(
                "Can't find valid chunk size. Please check your input video length or disable tiling."
            )
        self.t_chunk_enc = enc_t_chunk
        self.t_chunk_dec = dec_t_chunk
        print(f"Auto selected chunk size: {enc_t_chunk} for encoder and {dec_t_chunk} for decoder.")
    
    def tile_encode(self, x):
        b, c, t, h, w = x.shape
        
        if self.temporal_size is None:
            self.temporal_size = t
            self._auto_select_t_chunk()
        
        if self.temporal_size and self.temporal_size != t:
            raise ValueError(
                "Input temporal size is not consistent with the temporal size of the model."
            )
        
        start_end = self.build_chunk_start_end(t)
        result = []
        for idx, (start, end) in enumerate(start_end):
            self._set_first_chunk(idx == 0)
            chunk = x[:, :, start:end, :, :]
            chunk = self.encoder(chunk)[0]
            if self.use_quant_layer:
                chunk = self.quant_conv(chunk)
            result.append(chunk)
        
        return torch.cat(result, dim=2)
    
    def tile_decode(self, x):
        b, c, t_latent, h, w = x.shape
        
        t_upsampled = (t_latent - 1) * self.temporal_uptimes + 1
        if self.temporal_size is None:
            self.temporal_size = t_upsampled
            self._auto_select_t_chunk()
        
        if self.temporal_size and self.temporal_size != t_upsampled:
            raise ValueError(
                "Input temporal size is not consistent with the temporal size of the model."
            )
        
        start_end = self.build_chunk_start_end(t_latent, decoder_mode=True)
        
        result = []
        for idx, (start, end) in enumerate(start_end):
            self._set_first_chunk(idx == 0)
            
            if idx != 0 and end + 1 < t_latent:
                chunk: Any = x[:, :, start : end + 1, :, :]
            else:
                chunk = x[:, :, start:end, :, :]
            
            if self.use_quant_layer:
                chunk = self.post_quant_conv(chunk)
            chunk = self.decoder(chunk)[0]
            if idx != 0 and end + 1 < t_latent:
                chunk = chunk[:, :, : -self.temporal_uptimes]
                result.append(chunk.clone())
            else:
                result.append(chunk.clone())
        
        return torch.cat(result, dim=2)
    
    # ========= 其他工具方法 =========
    
    def get_last_layer(self):
        if hasattr(self.decoder.conv_out, "conv"):
            return self.decoder.conv_out.conv.weight
        else:
            return self.decoder.conv_out.weight
    
    def enable_tiling(self, use_tiling: bool = True):
        self.use_tiling = use_tiling
        self._set_causal_cached(use_tiling)
    
    def disable_tiling(self):
        self.enable_tiling(False)
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        print("init from " + path)
        
        if (
            "ema_state_dict" in sd
            and len(sd["ema_state_dict"]) > 0
            and os.environ.get("NOT_USE_EMA_MODEL", 0) == 0
        ):
            print("Load from ema model!")
            sd = sd["ema_state_dict"]
            sd = {key.replace("module.", ""): value for key, value in sd.items()}
        elif "state_dict" in sd:
            print("Load from normal model!")
            if "gen_model" in sd["state_dict"]:
                sd = sd["state_dict"]["gen_model"]
            else:
                sd = sd["state_dict"]
        
        keys = list(sd.keys())
        
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

