

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
    时间压缩的Latent处理器（三阶段训练版本）
    
    功能：
    1. 对latent做3D Haar小波变换
    2. Phase 1: 仅软一致性约束，不压缩
    3. Phase 2: 软压缩（门控混合）
    4. Phase 3: 硬压缩到1帧，解码用插值+轻低通
    """
    def __init__(
        self, 
        dwt3d_cls: str = "HaarWaveletTransform3D", 
        idwt3d_cls: str = "InverseHaarWaveletTransform3D",
        temporal_consistency_mode: str = "tv_l1",  # "l2_norm", "spatial_variance", "none"
        training_phase: str = "phase1",  # "phase1", "phase2", "phase3"
        phase2_mode: str = "gate",  # "gate" (门控混合) 或 "progressive_frames" (逐步减帧)
        variance_floor: float = 0.02,  # 方差下限阈值
    ):
        super().__init__()
        DWT3D = resolve_str_to_obj(dwt3d_cls)
        IDWT3D = resolve_str_to_obj(idwt3d_cls)
        self.dwt3d = DWT3D()
        self.idwt3d = IDWT3D()
        self.temporal_consistency_mode = temporal_consistency_mode
        self.training_phase = training_phase
        self.phase2_mode = phase2_mode
        self.variance_floor = variance_floor
        
        # Phase 2 门控系数 (0→0.9 的调度)
        self.register_buffer("alpha", torch.tensor(0.0))
        
        # Phase 2 progressive frames 模式的目标帧数
        self.target_frames = None
        self.temporal_scale=nn.Parameter(torch.ones(1))
    
    def set_alpha(self, alpha: float):
        """设置Phase 2的门控系数 (0→0.9)"""
        self.alpha.fill_(alpha)
    
    def set_training_phase(self, phase: str):
        """切换训练阶段"""
        assert phase in ["phase1", "phase2", "phase3"]
        self.training_phase = phase
        print(f"✓ Switched to training {phase}")
    
    def blur_t(self, x: torch.Tensor) -> torch.Tensor:
        """时间维度轻低通滤波 (1×3×1 高斯核)"""
        if x.size(2) <= 1:
            return x
        # 高斯核 [1, 2, 1]
        kernel = torch.tensor([1., 2., 1.], device=x.device, dtype=x.dtype)
        kernel = (kernel / kernel.sum()).view(1, 1, 3, 1, 1)
        
        # 按通道分组卷积
        B, C, T, H, W = x.shape
        x_reshaped = x.view(B * C, 1, T, H, W)
        
        # Padding: 'replicate' 避免边界效应
        x_padded = torch.nn.functional.pad(x_reshaped, (0, 0, 0, 0, 1, 1), mode='replicate')
        x_blurred = torch.nn.functional.conv3d(x_padded, kernel, padding=0)
        
        return x_blurred.view(B, C, T, H, W)
    
    def progressive_temporal_downsample(self, x: torch.Tensor, target_frames: int) -> torch.Tensor:
        """Phase 2 progressive frames模式：逐步减少时间帧数"""
        B, C, T, H, W = x.shape
        if target_frames >= T or target_frames <= 0:
            return x
        
        # 用线性插值降低帧数
        x_down = torch.nn.functional.interpolate(
            x, size=(target_frames, H, W),
            mode='trilinear', align_corners=False
        )
        
        # 轻低通
        #x_down = self.blur_t(x_down)
        
        # 恢复到原始帧数
        x_restored = torch.nn.functional.interpolate(
            x_down, size=(T, H, W),
            mode='trilinear', align_corners=False
        )
        
        return x_restored
    
    def lowfreq_consistency_loss(self, LL: torch.Tensor, clip_len: int = 8, mode: str = "temporal_variance", topk: float = 0.1) -> torch.Tensor:
        """
        目标：让 latent 低频在相邻（或各）片段上的统计一致，抑制时域闪烁与跨片段漂移。
        - "temporal_variance": 直接对时间维 T 求方差并平均成标量（不切片）
        - "mse": 将每片段做时空均值 (B,C) 后对各片段与全局均值做 MSE
        - "kl" : 将每片段看作通道高斯分布 N(μ,σ²)，与全局 N(μ_g,σ²_g) 做 KL
        - "tv_l1": 时间差分（TV-L1）平滑损失，抑制相邻帧间的跳跃，使用top-k选择避免稀释
        """
        if LL.ndim != 5:
            raise ValueError("LL must be (B,C,T,H,W)")
        ll_variance = LL.var(dim=2, unbiased=False).sum()

        # 直接用整段时间序列的方差作为 loss（不切片）
        if mode == "temporal_variance":
            return LL.var(dim=2, unbiased=False).mean(), ll_variance #如果不希望LLL不希望有动态信息，就用方差
        elif mode == "tv_l1":
            # TV-L1 时间差分平滑损失 - 简化版本
            B, C, T, H, W = LL.shape
            
            if T <= 1:
                return (LL * 0.0).sum(), ll_variance  # 保持梯度连接的零损失
            
            # 1) 时间差分
            dt = LL[:, :, 1:, :, :] - LL[:, :, :-1, :, :]  # (B,C,T-1,H,W)
            
            # 2) L1范数
            tv = dt.abs()  # L1范数
            
            # 3) 直接在所有空间维度求和，不做平均
            return tv.sum(),ll_variance  # 标量，数值不会过小
            
        elif mode == "tv_l1_v2":
            # TV-L1 时间差分平滑损失
            B, C, T, H, W = LL.shape
            
            if T <= 1:
                return (LL * 0.0).sum(), ll_variance  # 保持梯度连接的零损失
            
            # 1) 时间差分（TV-L1）
            dt = LL[:, :, 1:, :, :] - LL[:, :, :-1, :, :]  # (B,C,T-1,H,W)
            tv = dt.abs()                                     # L1 比 L2 更稳
            
            # 空间位置聚合前，做 Top-k 以防稀释
            tv_flat = tv.view(B, C, -1)                       # (B,C,(T-1)HW)
            total_elements = tv_flat.size(-1)
            k = max(1, int(total_elements * topk))  # 在计算图外部计算k值
            topk_vals, _ = tv_flat.topk(k, dim=-1)            # (B,C,k)
            tv_l1_v2 = topk_vals.sum()                          # 标量
            
            return tv_l1_v2,ll_variance
    def compress_frames(self, temporal_low_freq: torch.Tensor, remian_last_frame: bool = True) -> torch.Tensor:
        """
        压缩时间维度
        """
        T = temporal_low_freq.size(2)
        H = temporal_low_freq.size(3)
        W = temporal_low_freq.size(4)
        if remian_last_frame:

            # 前 T-1 帧的均值
            LF_T_minus_1_mean = temporal_low_freq[:, :, :-1].mean(dim=2, keepdim=True)  # (B,C,1,H,W)
            
            # 插值到 T-1 帧
            LF_T_minus_1_up = torch.nn.functional.interpolate(
                LF_T_minus_1_mean, size=(T - 1, H, W),
                mode='trilinear', align_corners=False
            )  # (B,C,T-1,H,W)

            # 保留最后一帧
            last_frame = temporal_low_freq[:, :, -1:, :, :]  # (B,C,1,H,W)
            
            # 拼接: 前 T-1 帧(插值) + 最后一帧(保留)
            LF_1_up = torch.cat([LF_T_minus_1_up, last_frame], dim=2)  # (B,C,T,H,W)
        else:
            LF_T_minus_1_mean=temporal_low_freq.mean(dim=2, keepdim=True)
            LF_1_up = torch.nn.functional.interpolate(
                LF_T_minus_1_mean, size=(T, H, W),
                mode='trilinear', align_corners=False
            )  # (B,C,T,H,W)

        
        return LF_1_up
        
    def compress_forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        三阶段前向压缩：latent -> 3D Haar -> [Phase策略] -> 压缩的coeffs
        
        Phase 1: 仅计算一致性损失，不压缩（保持原始T帧）
        Phase 2: 软压缩（门控混合或逐步减帧）
        Phase 3: 硬压缩到1帧
        
        Args:
            z: (B, C, T, H, W) 原始latent
            
        Returns:
            coeffs_low: (B, 4*C, T or 1, H/2, W/2) 时间低频
            coeffs_high: (B, 4*C, T/2, H/2, W/2) 时间高频
            lowfreq_consistency_loss: (loss, variance)
        """
        # 3D Haar小波变换
        coeffs = self.dwt3d(z)  # (B, 8*C, T/2, H/2, W/2)
        
        B, full_C, T_w, H_w, W_w = coeffs.shape
        C = z.shape[1]
        
        # 分离时间低频(前4个子带)和时间高频(后4个子带)
        # 注意：确认你的HaarWaveletTransform3D的通道顺序！
        # 标准顺序应该是: LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH
        temporal_low_freq = coeffs[:, 0*C : 4*C]   # (B, 4*C, T, H, W) - LLL,LLH,LHL,LHH
        temporal_high_freq = coeffs[:, 4*C : 8*C]  # (B, 4*C, T, H, W) - HLL,HLH,HHL,HHH
        
        # ========= 三阶段策略 =========
        
        if self.training_phase == "phase1":
            # Phase 1: 仅软一致性，不压缩
            # 保持原始时间维度
            coeffs_low = temporal_low_freq  # (B, 4*C, T, H, W)
            coeffs_high = temporal_high_freq
            
            # 计算一致性损失
            lowfreq_consistency_loss = self.lowfreq_consistency_loss(
                temporal_low_freq, mode=self.temporal_consistency_mode
            )
            
            # Phase 2: 软压缩
        elif self.training_phase == "phase2":
            if self.phase2_mode == "gate":
                # --- 原有的 ---
                # LF_blurred = self.blur_t(temporal_low_freq)
                # temporal_low_freq_soft = (1 - self.alpha) * temporal_low_freq + self.alpha * LF_blurred

                # --- 改为"前T-1帧用均值插值，最后一帧保持不变" ---
                # LF_1 = temporal_low_freq.mean(dim=2, keepdim=True)             # (B,4C,1,H,W)
                # LF_1_up = torch.nn.functional.interpolate(
                # LF_1, size=(temporal_low_freq.size(2), temporal_low_freq.size(3), temporal_low_freq.size(4)),



                LF_1_up = self.compress_frames(temporal_low_freq, remian_last_frame=False)
                # 可选：用与 phase3 完全一致的后处理（blur/能量补偿）
                # 如果 phase3 在 decompress_backward 里还有 blur_t 或 gamma，请复用同样的调用：
                #LF_1_up = self.blur_t(LF_1_up)*self.temporal_scale  # 若 phase3 有 self.gamma，也在此使用

                temporal_low_freq_soft = (1 - self.alpha) * temporal_low_freq + self.alpha * LF_1_up

                # 用软版本算一致性损失
                lowfreq_consistency_loss = self.lowfreq_consistency_loss(
                    temporal_low_freq_soft, mode=self.temporal_consistency_mode
                )

                # 作为coeffs_low输出（训练中保持T帧）
                coeffs_low = temporal_low_freq_soft
                coeffs_high = temporal_high_freq

                
            elif self.phase2_mode == "progressive_frames":
                # 逐步减帧：T → target_frames → T (通过插值)
                if self.target_frames is not None and self.target_frames < temporal_low_freq.size(2):
                    temporal_low_freq_prog = self.progressive_temporal_downsample(
                        temporal_low_freq, self.target_frames
                    )
                else:
                    temporal_low_freq_prog = temporal_low_freq
                
                lowfreq_consistency_loss = self.lowfreq_consistency_loss(
                    temporal_low_freq_prog, mode=self.temporal_consistency_mode
                )
                
                coeffs_low = temporal_low_freq_prog  # (B, 4*C, T, H, W)
                coeffs_high = temporal_high_freq
            else:
                raise ValueError(f"Unknown phase2_mode: {self.phase2_mode}")
                
        elif self.training_phase == "phase3":
            # Phase 3: 硬压缩到1帧
            # 时间平均压缩到1帧
            temporal_low_freq_compressed = self.compress_frames(temporal_low_freq, remian_last_frame=False)  # (B, 4*C, 1, H, W)
            
            # 一致性损失仍在压缩前的完整序列上计算
            lowfreq_consistency_loss = self.lowfreq_consistency_loss(
                temporal_low_freq, mode=self.temporal_consistency_mode
            )
            
            coeffs_low = temporal_low_freq_compressed  # (B, 4*C, 1, H, W)
            coeffs_high = temporal_high_freq  # (B, 4*C, T, H, W)
            
        else:
            raise ValueError(f"Unknown training_phase: {self.training_phase}")

        return coeffs_low, coeffs_high, lowfreq_consistency_loss
    
    def decompress_backward(
        self, 
        coeffs_low: torch.Tensor, 
        coeffs_high: torch.Tensor, 
    ) -> torch.Tensor:
        """
        三阶段反向解压：压缩的coeffs -> [Phase策略] -> 逆3D Haar -> latent
        
        Phase 1/2: coeffs_low已经是T帧，直接拼接
        Phase 3: 用插值+轻低通扩展，避免expand复制造成的固定相位
        
        Args:
            coeffs_low: (B, 4*C, T or 1, H, W) 时间低频
            coeffs_high: (B, 4*C, T, H, W) 时间高频
            
        Returns:
            z: (B, C, 2*T, 2*H, 2*W) 重构的latent
        """
        B, low_C, T_low, H, W = coeffs_low.shape
        C = low_C // 4
        target_T = coeffs_high.shape[2]
        
        # ========= 三阶段策略 =========
        
        if T_low == target_T:
            # Phase 1/2: 时间维度已匹配，直接拼接
            coeffs_low_expanded = coeffs_low + (0*self.temporal_scale)
            
        elif T_low == 1:
            # Phase 3: 压缩到1帧，需要扩展
            # 使用插值而非expand复制，避免固定相位问题
            
            coeffs_low_up = torch.nn.functional.interpolate(
                coeffs_low,  # (B, 4*C, 1, H, W)
                size=(target_T, H, W),
                mode='trilinear', 
                align_corners=False
            )  # (B, 4*C, T, H, W)
            
            # 轻低通滤波，打散固定相位
            #coeffs_low_expanded = self.blur_t(coeffs_low_up)*self.temporal_scale  # (B, 4*C, T, H, W)
            
        else:
            # 其他情况：通用插值
            if target_T != T_low:
                coeffs_low_expanded = torch.nn.functional.interpolate(
                    coeffs_low, 
                    size=(target_T, H, W),
                    mode='trilinear', 
                    align_corners=False
                )
                coeffs_low_expanded = self.blur_t(coeffs_low_expanded)*self.temporal_scale
            else:
                coeffs_low_expanded = coeffs_low
        
        # 拼接时间低频和时间高频
        coeffs_full = torch.cat([coeffs_low_expanded, coeffs_high], dim=1)  # (B, 8*C, T, H, W)
        
        # 逆3D Haar小波变换
        z = self.idwt3d(coeffs_full)  # (B, C, 2*T, 2*H, 2*W)
        
        return z




# ========= 主模型 =========

@ModelRegistry.register("Latent_WFVAE_TemporalCompressed_V3")
class WFVAETemporalCompressedModelV3(VideoBaseAE):
    """
    时间压缩版WFVAE
    
    与原始WFVAE的区别：
    1. Latent经过3D Haar变换后，时间低频子带压缩到1帧
    2. 训练和推理完全统一
    3. 存储空间减少约50%
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
        temporal_consistency_mode: str = "tv_l1",  # "l2_norm", "spatial_variance", "none"
        temporal_consistency_weight: float = 5,
        training_phase: str = "phase1",  # "phase1", "phase2", "phase3"
        phase2_mode: str = "gate",  # "gate" or "progressive_frames"
        variance_floor: float = 0.02,  # 方差下限
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
            temporal_consistency_mode=temporal_consistency_mode,
            training_phase=training_phase,
            phase2_mode=phase2_mode,
            variance_floor=variance_floor,
        )
        
        self.temporal_consistency_mode = temporal_consistency_mode
        self.temporal_consistency_weight = temporal_consistency_weight
        self.training_phase = training_phase
        
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
        
        print(f"✓ Initialized WFVAE with 3-Phase Temporal Compression")
        print(f"  Training phase: {training_phase}")
        print(f"  Temporal consistency mode: {temporal_consistency_mode}")
        print(f"  Temporal consistency weight: {temporal_consistency_weight}")
        print(f"  Variance floor: {variance_floor}")
        if training_phase == "phase2":
            print(f"  Phase 2 mode: {phase2_mode}")
        print(f"  Expected final compression (Phase 3): ~0.5x (temporal low-freq to 1 frame)")
    
    # ========= 辅助方法 =========
    
    def set_training_phase(self, phase: str):
        """切换训练阶段"""
        self.training_phase = phase
        self.temporal_processor.set_training_phase(phase)
    
    def set_alpha(self, alpha: float):
        """设置Phase 2的门控系数 (0→0.9)"""
        self.temporal_processor.set_alpha(alpha)
    
    def set_target_frames(self, target_frames: int):
        """设置Phase 2 progressive_frames模式的目标帧数"""
        self.temporal_processor.target_frames = target_frames
    
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

    def lowfreq_consistency_loss(self, LL: torch.Tensor, clip_len: int = 8, mode: str = "temporal_variance", topk: float = 0.1) -> torch.Tensor:
        """
        目标：让 latent 低频在相邻（或各）片段上的统计一致，抑制时域闪烁与跨片段漂移。
        - "temporal_variance": 直接对时间维 T 求方差并平均成标量（不切片）
        - "mse": 将每片段做时空均值 (B,C) 后对各片段与全局均值做 MSE
        - "kl" : 将每片段看作通道高斯分布 N(μ,σ²)，与全局 N(μ_g,σ²_g) 做 KL
        - "tv_l1": 时间差分（TV-L1）平滑损失，抑制相邻帧间的跳跃，使用top-k选择避免稀释
        """
        if LL.ndim != 5:
            raise ValueError("LL must be (B,C,T,H,W)")
        def temporal_var_si(L):
            var_t = L.var(dim=2, unbiased=False)
            scale = (L**2).mean(dim=2, keepdim=False) + 1e-6
            return (var_t / scale).sum()
        
        ll_variance = temporal_var_si(LL)


        # 直接用整段时间序列的方差作为 loss（不切片）
        if mode == "temporal_variance":
            return LL.var(dim=2, unbiased=False).mean(), ll_variance #如果不希望LLL不希望有动态信息，就用方差
        elif mode == "tv_l1":
            # TV-L1 时间差分平滑损失 - 简化版本
            B, C, T, H, W = LL.shape
            
            if T <= 1:
                return (LL * 0.0).sum(), ll_variance  # 保持梯度连接的零损失
            
            # 1) 时间差分
            dt = LL[:, :, 1:, :, :] - LL[:, :, :-1, :, :]  # (B,C,T-1,H,W)
            
            # 2) L1范数
            tv = dt.abs()  # L1范数
            
            # 3) 直接在所有空间维度求和，不做平均
            return tv.sum(),ll_variance  # 标量，数值不会过小
            
        elif mode == "tv_l1_v2":
            L_norm = (LL - LL.mean(dim=(2,3,4), keepdim=True)) / (LL.std(dim=(2,3,4), keepdim=True)+1e-6)
            dt = (L_norm[:, :, 1:] - L_norm[:, :, :-1]).abs().sum()
            return dt, ll_variance
        


    
    def encode(self, x, return_dict=True):
        """
        编码输入视频为压缩的latent表示
        
        Args:
            x: (B, C, T, H, W) 输入视频
            return_dict: 是否返回字典格式
            
        Returns:
            如果return_dict=True:
                AutoencoderKLOutput with:
                    - latent_dist: DiagonalGaussianDistribution
                    - extra_output: (enc_l1, enc_l2, coeffs_low, coeffs_high, original_T)
            否则返回元组
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
        
        # 获取latent并做时间压缩
        z = posterior.sample()#([1, 16, 7, 32, 32])
        coeffs_low, coeffs_high ,lowfreq_consistency_loss= self.temporal_processor.compress_forward(z)
        
        if return_dict:
            return AutoencoderKLOutput(
                latent_dist=posterior, 
                extra_output=(l1, l2, coeffs_low, coeffs_high, lowfreq_consistency_loss)
            )
        else:
            return posterior, (l1, l2, coeffs_low, coeffs_high, lowfreq_consistency_loss)
    
    def decode(self, coeffs_low, coeffs_high, return_dict=True):
        """
        从压缩的latent表示解码为视频
        
        Args:
            coeffs_low: (B, 4*C, 1, H, W) 压缩的时间低频
            coeffs_high: (B, 4*C, T, H, W) 时间高频
            original_T: 原始小波域时间维度
            return_dict: 是否返回字典格式
            
        Returns:
            如果return_dict=True:
                DecoderOutput with sample and extra_output
            否则返回元组
        """
        self._empty_causal_cached(self.decoder)
        self._set_first_chunk(True)
        
        # 从压缩格式重构完整latent
        z = self.temporal_processor.decompress_backward(coeffs_low, coeffs_high)
        
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
        前向传播：训练和推理统一
        
        流程：
        1. Encoder: video -> h
        2. 采样: h -> z
        3. 时间压缩: z -> (coeffs_low, coeffs_high)
        4. 计算consistency loss on coeffs_low
        5. 时间解压: (coeffs_low, coeffs_high) -> z
        6. Decoder: z -> video
        """
        # Encode
        encode_output = self.encode(input, return_dict=True)
        posterior = encode_output.latent_dist
        enc_l1, enc_l2, coeffs_low, coeffs_high, lowfreq_consistency_loss = encode_output.extra_output
        
        # Sample latent (训练时已经在encode中处理了)
        # 这里我们直接使用压缩后的coeffs
        # Decode
        decode_output = self.decode(coeffs_low, coeffs_high, return_dict=True)
        dec = decode_output.sample
        dec_l1, dec_l2 = decode_output.extra_output
        
        
        out = ForwardOutput(
            sample=dec,
            latent_dist=posterior,
            lowfreq_variance=lowfreq_consistency_loss[1],
            extra_output=(enc_l1, dec_l1, enc_l2, dec_l2),
        )
        out.lowfreq_consistency_loss = lowfreq_consistency_loss[0]
        
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

