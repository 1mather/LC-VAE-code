import torch
import torch.nn.functional as F
import torch.nn as nn
from ..modules import CausalConv3d
from einops import rearrange


class HaarWaveletTransform3D:
    def __init__(self, enable_cached=False) -> None:
        self.enable_cached = enable_cached
        self.causal_cached = None
        
    def _causal_conv(self, x, weight, idx):
        if self.causal_cached is None:
            self.causal_cached = {}
        if idx not in self.causal_cached:
            first_frame_pad = x[:, :, :1, :, :]
        else:
            first_frame_pad = self.causal_cached[idx]
        x = torch.concatenate((first_frame_pad, x), dim=2)
        if self.enable_cached:
            self.causal_cached[idx] = x[:, :, 0:0, :, :]
        x = F.conv3d(x, weight, stride=2, padding=0)
        return x

    def __call__(self, x):
        assert x.dim() == 5
        b = x.shape[0]
        device = x.device
        dtype = x.dtype

        h = (
            torch.tensor(
                [[[1, 1], [1, 1]], [[1, 1], [1, 1]]], device=device, dtype=dtype
            ).view(1, 1, 2, 2, 2)
            * 0.3536
        )
        g = (
            torch.tensor(
                [[[1, -1], [1, -1]], [[1, -1], [1, -1]]], device=device, dtype=dtype
            ).view(1, 1, 2, 2, 2)
            * 0.3536
        )
        hh = (
            torch.tensor(
                [[[1, 1], [-1, -1]], [[1, 1], [-1, -1]]], device=device, dtype=dtype
            ).view(1, 1, 2, 2, 2)
            * 0.3536
        )
        gh = (
            torch.tensor(
                [[[1, -1], [-1, 1]], [[1, -1], [-1, 1]]], device=device, dtype=dtype
            ).view(1, 1, 2, 2, 2)
            * 0.3536
        )
        h_v = (
            torch.tensor(
                [[[1, 1], [1, 1]], [[-1, -1], [-1, -1]]], device=device, dtype=dtype
            ).view(1, 1, 2, 2, 2)
            * 0.3536
        )
        g_v = (
            torch.tensor(
                [[[1, -1], [1, -1]], [[-1, 1], [-1, 1]]], device=device, dtype=dtype
            ).view(1, 1, 2, 2, 2)
            * 0.3536
        )
        hh_v = (
            torch.tensor(
                [[[1, 1], [-1, -1]], [[-1, -1], [1, 1]]], device=device, dtype=dtype
            ).view(1, 1, 2, 2, 2)
            * 0.3536
        )
        gh_v = (
            torch.tensor(
                [[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]], device=device, dtype=dtype
            ).view(1, 1, 2, 2, 2)
            * 0.3536
        )

        x = rearrange(x, "b c t h w -> (b c) 1 t h w")
        low_low_low = self._causal_conv(x, h, 0)
        low_low_low = rearrange(low_low_low, "(b c) 1 t h w -> b c t h w", b=b)
        low_low_high = self._causal_conv(x, g, 1)
        low_low_high = rearrange(low_low_high, "(b c) 1 t h w -> b c t h w", b=b)
        low_high_low = self._causal_conv(x, hh, 2)
        low_high_low = rearrange(low_high_low, "(b c) 1 t h w -> b c t h w", b=b)
        low_high_high = self._causal_conv(x, gh, 3)
        low_high_high = rearrange(low_high_high, "(b c) 1 t h w -> b c t h w", b=b)
        high_low_low = self._causal_conv(x, h_v, 4)
        high_low_low = rearrange(high_low_low, "(b c) 1 t h w -> b c t h w", b=b)
        high_low_high = self._causal_conv(x, g_v, 5)
        high_low_high = rearrange(high_low_high, "(b c) 1 t h w -> b c t h w", b=b)
        high_high_low = self._causal_conv(x, hh_v, 6)
        high_high_low = rearrange(high_high_low, "(b c) 1 t h w -> b c t h w", b=b)
        high_high_high = self._causal_conv(x, gh_v, 7)
        high_high_high = rearrange(high_high_high, "(b c) 1 t h w -> b c t h w", b=b)

        output = torch.cat(
            [
                low_low_low,
                low_low_high,
                low_high_low,
                low_high_high,
                high_low_low,
                high_low_high,
                high_high_low,
                high_high_high,
            ],
            dim=1,
        )
        return output


class InverseHaarWaveletTransform3D:
    def __init__(self, enable_cached=False) -> None:
        self.enable_cached = enable_cached
        self.causal_cached = None

    def __call__(self, coeffs):
        assert coeffs.dim() == 5
        b = coeffs.shape[0]

        device = coeffs.device
        dtype = coeffs.dtype
        h = (
            torch.tensor(
                [[[1, 1], [1, 1]], [[1, 1], [1, 1]]], device=device, dtype=dtype
            ).view(1, 1, 2, 2, 2)
            * 0.3536
        )
        g = (
            torch.tensor(
                [[[1, -1], [1, -1]], [[1, -1], [1, -1]]], device=device, dtype=dtype
            ).view(1, 1, 2, 2, 2)
            * 0.3536
        )
        hh = (
            torch.tensor(
                [[[1, 1], [-1, -1]], [[1, 1], [-1, -1]]], device=device, dtype=dtype
            ).view(1, 1, 2, 2, 2)
            * 0.3536
        )
        gh = (
            torch.tensor(
                [[[1, -1], [-1, 1]], [[1, -1], [-1, 1]]], device=device, dtype=dtype
            ).view(1, 1, 2, 2, 2)
            * 0.3536
        )
        h_v = (
            torch.tensor(
                [[[1, 1], [1, 1]], [[-1, -1], [-1, -1]]], device=device, dtype=dtype
            ).view(1, 1, 2, 2, 2)
            * 0.3536
        )
        g_v = (
            torch.tensor(
                [[[1, -1], [1, -1]], [[-1, 1], [-1, 1]]], device=device, dtype=dtype
            ).view(1, 1, 2, 2, 2)
            * 0.3536
        )
        hh_v = (
            torch.tensor(
                [[[1, 1], [-1, -1]], [[-1, -1], [1, 1]]], device=device, dtype=dtype
            ).view(1, 1, 2, 2, 2)
            * 0.3536
        )
        gh_v = (
            torch.tensor(
                [[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]], device=device, dtype=dtype
            ).view(1, 1, 2, 2, 2)
            * 0.3536
        )

        (
            low_low_low,
            low_low_high,
            low_high_low,
            low_high_high,
            high_low_low,
            high_low_high,
            high_high_low,
            high_high_high,
        ) = coeffs.chunk(8, dim=1)

        low_low_low = rearrange(low_low_low, "b c t h w -> (b c) 1 t h w")
        low_low_high = rearrange(low_low_high, "b c t h w -> (b c) 1 t h w")
        low_high_low = rearrange(low_high_low, "b c t h w -> (b c) 1 t h w")
        low_high_high = rearrange(low_high_high, "b c t h w -> (b c) 1 t h w")
        high_low_low = rearrange(high_low_low, "b c t h w -> (b c) 1 t h w")
        high_low_high = rearrange(high_low_high, "b c t h w -> (b c) 1 t h w")
        high_high_low = rearrange(high_high_low, "b c t h w -> (b c) 1 t h w")
        high_high_high = rearrange(high_high_high, "b c t h w -> (b c) 1 t h w")
        low_low_low = F.conv_transpose3d(low_low_low, h, stride=2)
        low_low_high = F.conv_transpose3d(low_low_high, g, stride=2)
        low_high_low = F.conv_transpose3d(low_high_low, hh, stride=2)
        low_high_high = F.conv_transpose3d(low_high_high, gh, stride=2)
        high_low_low = F.conv_transpose3d(high_low_low, h_v, stride=2)
        high_low_high = F.conv_transpose3d(high_low_high, g_v, stride=2)
        high_high_low = F.conv_transpose3d(high_high_low, hh_v, stride=2)
        high_high_high = F.conv_transpose3d(high_high_high, gh_v, stride=2)
        if self.enable_cached and self.causal_cached:
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
                low_low_low[:, :, 1:]
                + low_low_high[:, :, 1:]
                + low_high_low[:, :, 1:]
                + low_high_high[:, :, 1:]
                + high_low_low[:, :, 1:]
                + high_low_high[:, :, 1:]
                + high_high_low[:, :, 1:]
                + high_high_high[:, :, 1:]
            )
            self.causal_cached = True
        reconstructed = rearrange(reconstructed, "(b c) 1 t h w -> b c t h w", b=b)
        return reconstructed

