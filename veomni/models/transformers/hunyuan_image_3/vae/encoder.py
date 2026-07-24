# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Encoder implementation for the official HunyuanImage 3 VAE.

The encoder modules are adapted from Tencent's ``autoencoder_kl_3d.py`` at
revision ``6e9113a692a27a0751d82aba3b2015a876646c03``.
"""

import math

import numpy as np
import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F


def _swish(tensor: Tensor) -> Tensor:
    return tensor * torch.sigmoid(tensor)


def _forward_with_checkpointing(module: nn.Module, *inputs: Tensor, use_checkpointing: bool = False) -> Tensor:
    if not use_checkpointing:
        return module(*inputs)

    def custom_forward(*args: Tensor) -> Tensor:
        return module(*args)

    return torch.utils.checkpoint.checkpoint(custom_forward, *inputs, use_reentrant=False)


class Conv3d(nn.Conv3d):
    """Chunk large temporal inputs while preserving official Conv3d padding semantics."""

    def forward(self, input: Tensor) -> Tensor:
        _, channels, frames, height, width = input.shape
        memory_gib = channels * frames * height * width * 2 / 1024**3
        if memory_gib <= 2:
            return super().forward(input)

        chunks = torch.chunk(input, chunks=math.ceil(memory_gib / 2), dim=-3)
        padded_chunks = []
        for index, chunk in enumerate(chunks):
            if self.padding[0] == 0:
                padded_chunks.append(chunk)
                continue
            padded = F.pad(
                chunk,
                (0, 0, 0, 0, self.padding[0], self.padding[0]),
                mode="constant" if self.padding_mode == "zeros" else self.padding_mode,
                value=0,
            )
            if index > 0:
                padded[:, :, : self.padding[0]] = chunks[index - 1][:, :, -self.padding[0] :]
            if index < len(chunks) - 1:
                padded[:, :, -self.padding[0] :] = chunks[index + 1][:, :, : self.padding[0]]
            padded_chunks.append(padded)

        original_padding = self.padding
        self.padding = (0, self.padding[1], self.padding[2])
        try:
            outputs = []
            for chunk in padded_chunks:
                outputs.append(super().forward(chunk))
        finally:
            self.padding = original_padding
        return torch.cat(outputs, dim=-3)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, tensor: Tensor) -> Tensor:
        normalized = self.norm(tensor)
        query = self.q(normalized)
        key = self.k(normalized)
        value = self.v(normalized)
        batch, channels, frames, height, width = query.shape
        query = rearrange(query, "b c f h w -> b 1 (f h w) c").contiguous()
        key = rearrange(key, "b c f h w -> b 1 (f h w) c").contiguous()
        value = rearrange(value, "b c f h w -> b 1 (f h w) c").contiguous()
        output = nn.functional.scaled_dot_product_attention(query, key, value)
        output = rearrange(
            output,
            "b 1 (f h w) c -> b c f h w",
            b=batch,
            c=channels,
            f=frames,
            h=height,
            w=width,
        )
        return tensor + self.proj_out(output)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, tensor: Tensor) -> Tensor:
        hidden = self.conv1(_swish(self.norm1(tensor)))
        hidden = self.conv2(_swish(self.norm2(hidden)))
        if self.in_channels != self.out_channels:
            tensor = self.nin_shortcut(tensor)
        return tensor + hidden


class DownsampleDCAE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_temporal_downsample: bool = True) -> None:
        super().__init__()
        factor = 8 if add_temporal_downsample else 4
        if out_channels % factor:
            raise ValueError("DCAE downsample output channels must be divisible by its spatial-temporal factor.")
        self.conv = Conv3d(in_channels, out_channels // factor, kernel_size=3, stride=1, padding=1)
        self.add_temporal_downsample = add_temporal_downsample
        self.group_size = factor * in_channels // out_channels

    def forward(self, tensor: Tensor) -> Tensor:
        temporal_factor = 2 if self.add_temporal_downsample else 1
        hidden = self.conv(tensor)
        hidden = rearrange(
            hidden,
            "b c (f rt) (h rh) (w rw) -> b (rt rh rw c) f h w",
            rt=temporal_factor,
            rh=2,
            rw=2,
        )
        shortcut = rearrange(
            tensor,
            "b c (f rt) (h rh) (w rw) -> b (rt rh rw c) f h w",
            rt=temporal_factor,
            rh=2,
            rw=2,
        )
        batch, _, frames, height, width = shortcut.shape
        shortcut = shortcut.view(batch, hidden.shape[1], self.group_size, frames, height, width).mean(dim=2)
        return hidden + shortcut


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        block_out_channels: tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        downsample_match_channel: bool = True,
    ) -> None:
        super().__init__()
        if block_out_channels[-1] % (2 * z_channels):
            raise ValueError("The final VAE encoder width must be divisible by twice the latent channel count.")
        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks
        self.conv_in = Conv3d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        self.down = nn.ModuleList()
        block_in = block_out_channels[0]
        for level, channels in enumerate(block_out_channels):
            blocks = nn.ModuleList()
            block_out = channels
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = blocks
            add_spatial_downsample = level < np.log2(ffactor_spatial)
            add_temporal_downsample = add_spatial_downsample and level >= np.log2(ffactor_spatial // ffactor_temporal)
            if add_spatial_downsample or add_temporal_downsample:
                if level >= len(block_out_channels) - 1:
                    raise ValueError("VAE downsampling requires another block_out_channels level.")
                next_channels = block_out_channels[level + 1] if downsample_match_channel else block_in
                down.downsample = DownsampleDCAE(block_in, next_channels, add_temporal_downsample)
                block_in = next_channels
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = Conv3d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)
        self.gradient_checkpointing = False

    def forward(self, tensor: Tensor) -> Tensor:
        use_checkpointing = bool(self.training and self.gradient_checkpointing)
        hidden = self.conv_in(tensor)
        for level in range(len(self.block_out_channels)):
            for block in self.down[level].block:
                hidden = _forward_with_checkpointing(block, hidden, use_checkpointing=use_checkpointing)
            if hasattr(self.down[level], "downsample"):
                hidden = _forward_with_checkpointing(
                    self.down[level].downsample,
                    hidden,
                    use_checkpointing=use_checkpointing,
                )
        hidden = _forward_with_checkpointing(self.mid.block_1, hidden, use_checkpointing=use_checkpointing)
        hidden = _forward_with_checkpointing(self.mid.attn_1, hidden, use_checkpointing=use_checkpointing)
        hidden = _forward_with_checkpointing(self.mid.block_2, hidden, use_checkpointing=use_checkpointing)
        group_size = self.block_out_channels[-1] // (2 * self.z_channels)
        shortcut = rearrange(hidden, "b (c r) f h w -> b c r f h w", r=group_size).mean(dim=2)
        hidden = self.conv_out(_swish(self.norm_out(hidden)))
        return hidden + shortcut


__all__ = ["Encoder"]
