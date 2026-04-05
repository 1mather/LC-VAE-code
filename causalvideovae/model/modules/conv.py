import torch.nn as nn
from typing import Union, Tuple
import torch
from .ops import cast_tuple
from .ops import video_to_image
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from collections import deque

class Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]] = 3,
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[str, int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    @video_to_image
    def forward(self, x):
        return super().forward(x)


class CausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        enable_cached=False,
        bias=True,
        causal_enable=False,
        **kwargs
    ):
        super().__init__()
        self.causal_enable= False
        self.kernel_size = cast_tuple(kernel_size, 3)
        self.time_kernel_size = self.kernel_size[0]
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = kwargs.pop("stride", 1)
        self.padding = kwargs.pop("padding", 0)
        self.padding = list(cast_tuple(self.padding, 3)) # T,H,W 的apdding
        if self.causal_enable:
            self.padding[0] = 0 #T维度的padding设置为1，手动复制第一帧而不用0来填充来保证casual

        self.stride = cast_tuple(self.stride, 3)#(2,2,2)
        self.conv = nn.Conv3d(
            chan_in,
            chan_out,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=bias
        )
        self.enable_cached = enable_cached #是否在分块推理的时候启用缓存
        self.is_first_chunk = True #第一帧需要用复制第一帧的方式来填充历史
        
        self.causal_cached = deque() #存放下一块需要用的历史帧拼接片段
        self.cache_offset = 0 #用来处理跨块对接的

    def forward(self, x):
        # Ensure input dtype matches conv parameter dtype to avoid AMP mismatch
        if x.dtype is not self.conv.weight.dtype:
            x = x.to(dtype=self.conv.weight.dtype)
        if self.causal_enable:
            if self.is_first_chunk:
                first_frame_pad = x[:, :, :1, :, :].repeat(
                    (1, 1, self.time_kernel_size - 1, 1, 1)
                )
            else:
                first_frame_pad = self.causal_cached.popleft()
                
            x = torch.concatenate((first_frame_pad, x), dim=2)

            if self.enable_cached and self.time_kernel_size != 1:
                if (self.time_kernel_size - 1) // self.stride[0] != 0: #这里(self.time_kernel_size - 1) // self.stride[0]是对历史帧填充的一个近似这样可以减少缓存量
                    if self.cache_offset == 0 or self.is_first_chunk:
                        self.causal_cached.append(x[:, :, -(self.time_kernel_size - 1) // self.stride[0]:].clone())
                    else:
                        self.causal_cached.append(x[:, :, :-self.cache_offset][:, :, -(self.time_kernel_size - 1) // self.stride[0]:].clone()) #当外部设置了偏移（比如跨块重叠对齐时丢弃一部分尾帧），就先裁掉
                else:
                    self.causal_cached.append(x[:, :, 0:0, :, :].clone())
            elif self.enable_cached:
                self.causal_cached.append(x[:, :, 0:0, :, :].clone())
            x = self.conv(x)
        else:
            x = self.conv(x)
        return x

class CausalConv3d_GC(CausalConv3d):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int]],
        init_method="random",
        **kwargs
    ):
        super().__init__(chan_in, chan_out, kernel_size, init_method, **kwargs)

    def forward(self, x):
        # 1 + 16   16 as video, 1 as image
        first_frame_pad = x[:, :, :1, :, :].repeat(
            (1, 1, self.time_kernel_size - 1, 1, 1)
        )  # b c t h w
        x = torch.concatenate((first_frame_pad, x), dim=2)  # 3 + 16
        return checkpoint(self.conv, x)
