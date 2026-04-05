import torch
import torch.nn.functional as F
import torch.nn as nn
from ..modules import CausalConv3d
from ..modules.ops import video_to_image

from einops import rearrange

class HaarWaveletTransform3D(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.h_conv = CausalConv3d(1, 1, 2, padding=0, stride=2, bias=False)
        self.g_conv = CausalConv3d(1, 1, 2, padding=0, stride=2, bias=False)
        self.hh_conv = CausalConv3d(1, 1, 2, padding=0, stride=2, bias=False)
        self.gh_conv = CausalConv3d(1, 1, 2, padding=0, stride=2, bias=False)
        self.h_v_conv = CausalConv3d(1, 1, 2, padding=0, stride=2, bias=False)
        self.g_v_conv = CausalConv3d(1, 1, 2, padding=0, stride=2, bias=False)
        self.hh_v_conv = CausalConv3d(1, 1, 2, padding=0, stride=2, bias=False)
        self.gh_v_conv = CausalConv3d(1, 1, 2, padding=0, stride=2, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        h = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]) * 0.3536
        g = torch.tensor([[[1, -1], [1, -1]], [[1, -1], [1, -1]]]) * 0.3536
        hh = torch.tensor([[[1, 1], [-1, -1]], [[1, 1], [-1, -1]]]) * 0.3536
        gh = torch.tensor([[[1, -1], [-1, 1]], [[1, -1], [-1, 1]]]) * 0.3536
        h_v = torch.tensor([[[1, 1], [1, 1]], [[-1, -1], [-1, -1]]]) * 0.3536
        g_v = torch.tensor([[[1, -1], [1, -1]], [[-1, 1], [-1, 1]]]) * 0.3536
        hh_v = torch.tensor([[[1, 1], [-1, -1]], [[-1, -1], [1, 1]]]) * 0.3536
        gh_v = torch.tensor([[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]]) * 0.3536
        h = h.view(1, 1, 2, 2, 2)
        g = g.view(1, 1, 2, 2, 2)
        hh = hh.view(1, 1, 2, 2, 2)
        gh = gh.view(1, 1, 2, 2, 2)
        h_v = h_v.view(1, 1, 2, 2, 2)
        g_v = g_v.view(1, 1, 2, 2, 2)
        hh_v = hh_v.view(1, 1, 2, 2, 2)
        gh_v = gh_v.view(1, 1, 2, 2, 2)
        
        with torch.no_grad():
            self.h_conv.conv.weight.copy_(h.to(self.h_conv.conv.weight.device).to(self.h_conv.conv.weight.dtype))
            self.g_conv.conv.weight.copy_(g.to(self.g_conv.conv.weight.device).to(self.g_conv.conv.weight.dtype))
            self.hh_conv.conv.weight.copy_(hh.to(self.hh_conv.conv.weight.device).to(self.hh_conv.conv.weight.dtype))
            self.gh_conv.conv.weight.copy_(gh.to(self.gh_conv.conv.weight.device).to(self.gh_conv.conv.weight.dtype))
            self.h_v_conv.conv.weight.copy_(h_v.to(self.h_v_conv.conv.weight.device).to(self.h_v_conv.conv.weight.dtype))
            self.g_v_conv.conv.weight.copy_(g_v.to(self.g_v_conv.conv.weight.device).to(self.g_v_conv.conv.weight.dtype))
            self.hh_v_conv.conv.weight.copy_(hh_v.to(self.hh_v_conv.conv.weight.device).to(self.hh_v_conv.conv.weight.dtype))
            self.gh_v_conv.conv.weight.copy_(gh_v.to(self.gh_v_conv.conv.weight.device).to(self.gh_v_conv.conv.weight.dtype))
        
        self.h_conv.requires_grad_(False)
        self.g_conv.requires_grad_(False)
        self.hh_conv.requires_grad_(False)
        self.gh_conv.requires_grad_(False)
        self.h_v_conv.requires_grad_(False)
        self.g_v_conv.requires_grad_(False)
        self.hh_v_conv.requires_grad_(False)
        self.gh_v_conv.requires_grad_(False)

    def forward(self, x):
        assert x.dim() == 5
        b = x.shape[0]
        c = x.shape[1]
        t = x.shape[2]
        
        # Pad time dimension if odd
        pad_t = t % 2
        if pad_t > 0:
            # Replicate last frame to make it even
            x = F.pad(x, (0, 0, 0, 0, pad_t, 0), mode='replicate')
        
        x = rearrange(x, "b c t h w -> (b c) 1 t h w") # 3 1 17 256 256
        n_dim = x.shape[0]
        outputs = []
        for i in range(n_dim): #每个channel进行，不能并行
            y = x[i: i+1]#([1, 1, 25, 256, 256])
            outputs.append(self.h_conv(y))
            outputs.append(self.g_conv(y))#([1, 1, 13, 128, 128])
            outputs.append(self.hh_conv(y))
            outputs.append(self.gh_conv(y))
            outputs.append(self.h_v_conv(y))
            outputs.append(self.g_v_conv(y))
            outputs.append(self.hh_v_conv(y))
            outputs.append(self.gh_v_conv(y))
        
        outputs = torch.cat(outputs, dim=0)
        outputs = rearrange(outputs, "(b k c) 1 t h w -> b (c k) t h w", b=b, k=c)
        return outputs
    
class InverseHaarWaveletTransform3D(nn.Module):
    def __init__(self, enable_cached=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.register_buffer('h', 
            torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.register_buffer('g', 
            torch.tensor([[[1, -1], [1, -1]], [[1, -1], [1, -1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.register_buffer('hh', 
            torch.tensor([[[1, 1], [-1, -1]], [[1, 1], [-1, -1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.register_buffer('gh', 
            torch.tensor([[[1, -1], [-1, 1]], [[1, -1], [-1, 1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.register_buffer('h_v', 
            torch.tensor([[[1, 1], [1, 1]], [[-1, -1], [-1, -1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.register_buffer('g_v', 
            torch.tensor([[[1, -1], [1, -1]], [[-1, 1], [-1, 1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.register_buffer('hh_v', 
            torch.tensor([[[1, 1], [-1, -1]], [[-1, -1], [1, 1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.register_buffer('gh_v', 
            torch.tensor([[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.enable_cached = enable_cached
        self.is_first_chunk = True

    def forward(self, coeffs):
        assert coeffs.dim() == 5
        b = coeffs.shape[0]

        (
            low_low_low,
            low_low_high,
            low_high_low,
            low_high_high,
            high_low_low,
            high_low_high,
            high_high_low,
            high_high_high,
        ) = coeffs.chunk(8, dim=1) #[2,128,4,16,16]

        low_low_low = rearrange(low_low_low, "b c t h w -> (b c) 1 t h w")
        low_low_high = rearrange(low_low_high, "b c t h w -> (b c) 1 t h w")
        low_high_low = rearrange(low_high_low, "b c t h w -> (b c) 1 t h w")
        low_high_high = rearrange(low_high_high, "b c t h w -> (b c) 1 t h w")
        high_low_low = rearrange(high_low_low, "b c t h w -> (b c) 1 t h w")
        high_low_high = rearrange(high_low_high, "b c t h w -> (b c) 1 t h w")
        high_high_low = rearrange(high_high_low, "b c t h w -> (b c) 1 t h w")
        high_high_high = rearrange(high_high_high, "b c t h w -> (b c) 1 t h w")

        low_low_low = F.conv_transpose3d(low_low_low, self.h, stride=2)
        low_low_high = F.conv_transpose3d(low_low_high, self.g, stride=2)
        low_high_low = F.conv_transpose3d(low_high_low, self.hh, stride=2)
        low_high_high = F.conv_transpose3d(low_high_high, self.gh, stride=2)
        high_low_low = F.conv_transpose3d(high_low_low, self.h_v, stride=2)
        high_low_high = F.conv_transpose3d(high_low_high, self.g_v, stride=2)
        high_high_low = F.conv_transpose3d(high_high_low, self.hh_v, stride=2)
        high_high_high = F.conv_transpose3d(high_high_high, self.gh_v, stride=2)
        
        if self.enable_cached and not self.is_first_chunk:
            reconstructed = (
                low_low_low
                + low_low_high
                + low_high_low
                + low_high_high
                + high_low_low
                + high_low_high
                + high_high_low
                + high_high_high
            )
        else:
            reconstructed = (
                low_low_low
                + low_low_high
                + low_high_low
                + low_high_high
                + high_low_low
                + high_low_high
                + high_high_low
                + high_high_high
            )
            # reconstructed = (
            #     low_low_low[:, :, 1:]
            #     + low_low_high[:, :, 1:]
            #     + low_high_low[:, :, 1:]
            #     + low_high_high[:, :, 1:]
            #     + high_low_low[:, :, 1:]
            #     + high_low_high[:, :, 1:]
            #     + high_high_low[:, :, 1:]
            #     + high_high_high[:, :, 1:]
            # )
            
            
        reconstructed = rearrange(reconstructed, "(b c) 1 t h w -> b c t h w", b=b)
        return reconstructed


class HaarWaveletTemporalTransform(nn.Module):
    def __init__(self):
        super().__init__()
        # 1D Haar filters along time dimension (normalized)
        self.register_buffer('h_t', torch.tensor([1.0, 1.0]).view(1, 1, 2, 1, 1) * 0.7071)
        self.register_buffer('g_t', torch.tensor([1.0, -1.0]).view(1, 1, 2, 1, 1) * 0.7071)

    def forward(self, x):
        assert x.dim() == 5
        b = x.shape[0]
        c = x.shape[1]
        t = x.shape[2]

        # Pad time dimension to even length
        pad_t = t % 2
        if pad_t > 0:
            x = F.pad(x, (0, 0, 0, 0, pad_t, 0), mode='replicate')

        x = rearrange(x, "b c t h w -> (b c) 1 t h w")
        low = F.conv3d(x, self.h_t, stride=(2, 1, 1), padding=0)
        high = F.conv3d(x, self.g_t, stride=(2, 1, 1), padding=0)
        low = rearrange(low, "(b c) 1 t h w -> b c t h w", b=b)
        high = rearrange(high, "(b c) 1 t h w -> b c t h w", b=b)
        coeffs = torch.cat([low, high], dim=1)  # (B, 2C, T/2, H, W)
        return coeffs


class HaarWaveletSpatialTransform(nn.Module):
    """Haar Wavelet Transform on spatial dimensions (H, W) only.
    
    Input: (B, C, T, H, W)
    Output: (B, 4C, T, H/2, W/2) with subbands [LL, LH, HL, HH]
    """
    def __init__(self):
        super().__init__()
        # 1D Haar filters (normalized)
        # h: low-pass filter, g: high-pass filter
        h = torch.tensor([1.0, 1.0]) * 0.7071
        g = torch.tensor([1.0, -1.0]) * 0.7071
        
        # Filters for H dimension: (out_c, in_c, kT, kH, kW)
        self.register_buffer('h_h', h.view(1, 1, 1, 2, 1))  # (1, 1, 1, 2, 1)
        self.register_buffer('g_h', g.view(1, 1, 1, 2, 1))
        
        # Filters for W dimension: (out_c, in_c, kT, kH, kW)
        self.register_buffer('h_w', h.view(1, 1, 1, 1, 2))  # (1, 1, 1, 1, 2)
        self.register_buffer('g_w', g.view(1, 1, 1, 1, 2))
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, T, H, W)
        
        Returns:
            coeffs: Wavelet coefficients (B, 4C, T, H/2, W/2)
                    Channel order: [LL, LH, HL, HH]
        """
        assert x.dim() == 5, f"Expected 5D input (B, C, T, H, W), got {x.shape}"
        b, c, t, h, w = x.shape
        
        # Pad spatial dimensions to even length if needed
        pad_h = h % 2
        pad_w = w % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        
        # Reshape for processing: (B*C, 1, T, H, W)
        x = rearrange(x, "b c t h w -> (b c) 1 t h w")
        
        # Apply Haar transform on H dimension
        low_h = F.conv3d(x, self.h_h, stride=(1, 2, 1), padding=0)  # (B*C, 1, T, H/2, W)
        high_h = F.conv3d(x, self.g_h, stride=(1, 2, 1), padding=0)  # (B*C, 1, T, H/2, W)
        
        # Apply Haar transform on W dimension to both low_h and high_h
        LL = F.conv3d(low_h, self.h_w, stride=(1, 1, 2), padding=0)   # (B*C, 1, T, H/2, W/2)
        LH = F.conv3d(low_h, self.g_w, stride=(1, 1, 2), padding=0)   # (B*C, 1, T, H/2, W/2)
        HL = F.conv3d(high_h, self.h_w, stride=(1, 1, 2), padding=0)  # (B*C, 1, T, H/2, W/2)
        HH = F.conv3d(high_h, self.g_w, stride=(1, 1, 2), padding=0)  # (B*C, 1, T, H/2, W/2)
        
        # Reshape back and concatenate subbands
        LL = rearrange(LL, "(b c) 1 t h w -> b c t h w", b=b)
        LH = rearrange(LH, "(b c) 1 t h w -> b c t h w", b=b)
        HL = rearrange(HL, "(b c) 1 t h w -> b c t h w", b=b)
        HH = rearrange(HH, "(b c) 1 t h w -> b c t h w", b=b)
        
        # Stack along channel dimension: (B, 4C, T, H/2, W/2)
        coeffs = torch.cat([LL, LH, HL, HH], dim=1)
        return coeffs


class InverseHaarWaveletSpatialTransform(nn.Module):
    """Inverse Haar Wavelet Transform on spatial dimensions (H, W) only.
    
    Input: (B, 4C, T, H, W) with subbands [LL, LH, HL, HH]
    Output: (B, C, T, 2H, 2W)
    """
    def __init__(self):
        super().__init__()
        # 1D Haar filters (normalized)
        h = torch.tensor([1.0, 1.0]) * 0.7071
        g = torch.tensor([1.0, -1.0]) * 0.7071
        
        # Filters for reconstruction: (out_c, in_c, kT, kH, kW)
        self.register_buffer('h_h', h.view(1, 1, 1, 2, 1))
        self.register_buffer('g_h', g.view(1, 1, 1, 2, 1))
        self.register_buffer('h_w', h.view(1, 1, 1, 1, 2))
        self.register_buffer('g_w', g.view(1, 1, 1, 1, 2))
    
    def forward(self, coeffs):
        """
        Args:
            coeffs: Wavelet coefficients (B, 4C, T, H, W)
                    Channel order: [LL, LH, HL, HH]
        
        Returns:
            x: Reconstructed tensor (B, C, T, 2H, 2W)
        """
        assert coeffs.dim() == 5, f"Expected 5D input (B, 4C, T, H, W), got {coeffs.shape}"
        b = coeffs.shape[0]
        c_total = coeffs.shape[1]
        assert c_total % 4 == 0, f"Channel dimension must be divisible by 4, got {c_total}"
        
        c = c_total // 4
        
        # Split into 4 subbands
        LL, LH, HL, HH = coeffs.chunk(4, dim=1)  # Each: (B, C, T, H, W)
        
        # Reshape for processing
        LL = rearrange(LL, "b c t h w -> (b c) 1 t h w")
        LH = rearrange(LH, "b c t h w -> (b c) 1 t h w")
        HL = rearrange(HL, "b c t h w -> (b c) 1 t h w")
        HH = rearrange(HH, "b c t h w -> (b c) 1 t h w")
        
        # Reconstruct W dimension first
        rec_low_h = F.conv_transpose3d(LL, self.h_w, stride=(1, 1, 2)) + \
                    F.conv_transpose3d(LH, self.g_w, stride=(1, 1, 2))
        rec_high_h = F.conv_transpose3d(HL, self.h_w, stride=(1, 1, 2)) + \
                     F.conv_transpose3d(HH, self.g_w, stride=(1, 1, 2))
        
        # Reconstruct H dimension
        reconstructed = F.conv_transpose3d(rec_low_h, self.h_h, stride=(1, 2, 1)) + \
                       F.conv_transpose3d(rec_high_h, self.g_h, stride=(1, 2, 1))
        
        # Reshape back
        reconstructed = rearrange(reconstructed, "(b c) 1 t h w -> b c t h w", b=b)
        return reconstructed


class InverseHaarWaveletTemporalTransform(nn.Module):
    def __init__(self, enable_cached: bool = False):
        super().__init__()
        self.register_buffer('h_t', torch.tensor([1.0, 1.0]).view(1, 1, 2, 1, 1) * 0.7071)
        self.register_buffer('g_t', torch.tensor([1.0, -1.0]).view(1, 1, 2, 1, 1) * 0.7071)
        self.enable_cached = enable_cached
        self.is_first_chunk = True

    def forward(self, coeffs):
        assert coeffs.dim() == 5
        b = coeffs.shape[0]

        low, high = coeffs.chunk(2, dim=1)
        low = rearrange(low, "b c t h w -> (b c) 1 t h w")
        high = rearrange(high, "b c t h w -> (b c) 1 t h w")

        rec_low = F.conv_transpose3d(low, self.h_t, stride=(2, 1, 1))
        rec_high = F.conv_transpose3d(high, self.g_t, stride=(2, 1, 1))

        if self.enable_cached and not self.is_first_chunk:
            reconstructed = rec_low + rec_high
        else:
            reconstructed = rec_low + rec_high
        #reconstructed = reconstructed[:, :, 1:]

        reconstructed = rearrange(reconstructed, "(b c) 1 t h w -> b c t h w", b=b)
        return reconstructed


class HaarWaveletTransform2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('aa', torch.tensor([[1, 1], [1, 1]]).view(1, 1, 2, 2) / 2)
        self.register_buffer('ad', torch.tensor([[1, 1], [-1, -1]]).view(1, 1, 2, 2) / 2)
        self.register_buffer('da', torch.tensor([[1, -1], [1, -1]]).view(1, 1, 2, 2) / 2)
        self.register_buffer('dd', torch.tensor([[1, -1], [-1, 1]]).view(1, 1, 2, 2) / 2)

    @video_to_image
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b * c, 1, h, w)
        low_low = F.conv2d(x, self.aa, stride=2).reshape(b, c, h // 2, w // 2)
        low_high = F.conv2d(x, self.ad, stride=2).reshape(b, c, h // 2, w // 2)
        high_low = F.conv2d(x, self.da, stride=2).reshape(b, c, h // 2, w // 2)
        high_high = F.conv2d(x, self.dd, stride=2).reshape(b, c, h // 2, w // 2)
        coeffs = torch.cat([low_low, low_high, high_low, high_high], dim=1)
        return coeffs

class InverseHaarWaveletTransform2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('aa', torch.tensor([[1, 1], [1, 1]]).view(1, 1, 2, 2) / 2)
        self.register_buffer('ad', torch.tensor([[1, 1], [-1, -1]]).view(1, 1, 2, 2) / 2)
        self.register_buffer('da', torch.tensor([[1, -1], [1, -1]]).view(1, 1, 2, 2) / 2)
        self.register_buffer('dd', torch.tensor([[1, -1], [-1, 1]]).view(1, 1, 2, 2) / 2)

    @video_to_image
    def forward(self, coeffs):
        low_low, low_high, high_low, high_high = coeffs.chunk(4, dim=1)
        b, c, height_half, width_half = low_low.shape
        height = height_half * 2
        width = width_half * 2

        low_low = F.conv_transpose2d(
            low_low.reshape(b * c, 1, height_half, width_half), self.aa, stride=2
        )
        low_high = F.conv_transpose2d(
            low_high.reshape(b * c, 1, height_half, width_half), self.ad, stride=2
        )
        high_low = F.conv_transpose2d(
            high_low.reshape(b * c, 1, height_half, width_half), self.da, stride=2
        )
        high_high = F.conv_transpose2d(
            high_high.reshape(b * c, 1, height_half, width_half), self.dd, stride=2
        )

        return (low_low + low_high + high_low + high_high).reshape(b, c, height, width)






