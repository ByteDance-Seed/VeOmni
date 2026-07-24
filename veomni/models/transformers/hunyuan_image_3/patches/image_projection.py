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

"""HunyuanImage 3 timestep and image projection modules.

Adapted from Tencent's ``modeling_hunyuan_image_3.py`` at revision
``6e9113a692a27a0751d82aba3b2015a876646c03``. Class and parameter names are
kept compatible with the official checkpoint.
"""

import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * frequencies[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def _conv_nd(dims: int, *args, **kwargs) -> nn.Module:
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    if dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"Unsupported convolution dimensions: {dims}.")


def _avg_pool_nd(dims: int, *args, **kwargs) -> nn.Module:
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    if dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    if dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"Unsupported pooling dimensions: {dims}.")


def _zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


def _normalization(channels: int, **kwargs) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels, **kwargs)


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        act_layer: type[nn.Module] = nn.GELU,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
        out_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"dtype": dtype, "device": device}
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        out_size = hidden_size if out_size is None else out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, **factory_kwargs),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, **factory_kwargs),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        frequency = timestep_embedding(timesteps, self.frequency_embedding_size, self.max_period)
        return self.mlp(frequency.to(dtype=self.mlp[0].weight.dtype))


class Upsample(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = _conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                padding=1,
                device=device,
                dtype=dtype,
            )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {tensor.shape[1]}.")
        if self.dims == 3:
            tensor = F.interpolate(
                tensor,
                (tensor.shape[2], tensor.shape[3] * 2, tensor.shape[4] * 2),
                mode="nearest",
            )
        else:
            tensor = F.interpolate(tensor, scale_factor=2, mode="nearest")
        return self.conv(tensor) if self.use_conv else tensor


class Downsample(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = _conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=1,
                device=device,
                dtype=dtype,
            )
        else:
            if self.channels != self.out_channels:
                raise ValueError("Pooling downsample cannot change channel count.")
            self.op = _avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {tensor.shape[1]}.")
        return self.op(tensor)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        emb_channels: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
        use_conv: bool = False,
        dims: int = 2,
        up: bool = False,
        down: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"dtype": dtype, "device": device}
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        self.in_layers = nn.Sequential(
            _normalization(self.in_channels, **factory_kwargs),
            nn.SiLU(),
            _conv_nd(dims, self.in_channels, self.out_channels, 3, padding=1, **factory_kwargs),
        )
        self.updown = up or down
        if up:
            self.h_upd = Upsample(self.in_channels, False, dims, **factory_kwargs)
            self.x_upd = Upsample(self.in_channels, False, dims, **factory_kwargs)
        elif down:
            self.h_upd = Downsample(self.in_channels, False, dims, **factory_kwargs)
            self.x_upd = Downsample(self.in_channels, False, dims, **factory_kwargs)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels, **factory_kwargs),
        )
        self.out_layers = nn.Sequential(
            _normalization(self.out_channels, **factory_kwargs),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            _zero_module(_conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1, **factory_kwargs)),
        )
        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = _conv_nd(
                dims,
                self.in_channels,
                self.out_channels,
                3,
                padding=1,
                **factory_kwargs,
            )
        else:
            self.skip_connection = _conv_nd(dims, self.in_channels, self.out_channels, 1, **factory_kwargs)

    def forward(self, tensor: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        if self.updown:
            prefix, convolution = self.in_layers[:-1], self.in_layers[-1]
            hidden = convolution(self.h_upd(prefix(tensor)))
            tensor = self.x_upd(tensor)
        else:
            hidden = self.in_layers(tensor)

        embedding_output = self.emb_layers(embedding)
        while embedding_output.ndim < hidden.ndim:
            embedding_output = embedding_output[..., None]
        scale, shift = torch.chunk(embedding_output, 2, dim=1)
        output_norm, output_rest = self.out_layers[0], self.out_layers[1:]
        hidden = output_norm(hidden) * (1.0 + scale) + shift
        hidden = output_rest(hidden)
        return self.skip_connection(tensor) + hidden


class UNetDown(nn.Module):
    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        emb_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if patch_size not in (1, 2, 4, 8):
            raise ValueError("patch_size must be one of 1, 2, 4, or 8.")
        factory_kwargs = {"dtype": dtype, "device": device}
        self.patch_size = patch_size
        self.model = nn.ModuleList(
            [
                _conv_nd(
                    2,
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=3,
                    padding=1,
                    **factory_kwargs,
                )
            ]
        )
        if patch_size == 1:
            self.model.append(
                ResBlock(
                    in_channels=hidden_channels,
                    emb_channels=emb_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    **factory_kwargs,
                )
            )
        else:
            for index in range(patch_size // 2):
                self.model.append(
                    ResBlock(
                        in_channels=hidden_channels,
                        emb_channels=emb_channels,
                        out_channels=hidden_channels if (index + 1) * 2 != patch_size else out_channels,
                        dropout=dropout,
                        down=True,
                        **factory_kwargs,
                    )
                )

    def forward(self, tensor: torch.Tensor, timestep: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if tensor.shape[2] % self.patch_size or tensor.shape[3] % self.patch_size:
            raise ValueError("Image latent dimensions must be divisible by patch_size.")
        # Cached posteriors are stored in fp32; cast the noised latents to the
        # projection's compute dtype (bf16 in training). A no-op when they already
        # match, e.g. the fp32/fp64 official-parity harness.
        tensor = tensor.to(dtype=next(self.parameters()).dtype)
        for module in self.model:
            tensor = module(tensor, timestep) if isinstance(module, ResBlock) else module(tensor)
        _, _, token_height, token_width = tensor.shape
        return rearrange(tensor, "b c h w -> b (h w) c"), token_height, token_width


class UNetUp(nn.Module):
    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        emb_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        out_norm: bool = False,
    ) -> None:
        super().__init__()
        if patch_size not in (1, 2, 4, 8):
            raise ValueError("patch_size must be one of 1, 2, 4, or 8.")
        factory_kwargs = {"dtype": dtype, "device": device}
        self.patch_size = patch_size
        self.model = nn.ModuleList()
        if patch_size == 1:
            self.model.append(
                ResBlock(
                    in_channels=in_channels,
                    emb_channels=emb_channels,
                    out_channels=hidden_channels,
                    dropout=dropout,
                    **factory_kwargs,
                )
            )
        else:
            for index in range(patch_size // 2):
                self.model.append(
                    ResBlock(
                        in_channels=in_channels if index == 0 else hidden_channels,
                        emb_channels=emb_channels,
                        out_channels=hidden_channels,
                        dropout=dropout,
                        up=True,
                        **factory_kwargs,
                    )
                )
        if out_norm:
            self.model.append(
                nn.Sequential(
                    _normalization(hidden_channels, **factory_kwargs),
                    nn.SiLU(),
                    _conv_nd(
                        2,
                        in_channels=hidden_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        **factory_kwargs,
                    ),
                )
            )
        else:
            self.model.append(
                _conv_nd(
                    2,
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    **factory_kwargs,
                )
            )

    def forward(
        self,
        tensor: torch.Tensor,
        timestep: torch.Tensor,
        token_height: int,
        token_width: int,
    ) -> torch.Tensor:
        tensor = rearrange(tensor, "b (h w) c -> b c h w", h=token_height, w=token_width)
        for module in self.model:
            tensor = module(tensor, timestep) if isinstance(module, ResBlock) else module(tensor)
        return tensor


__all__ = ["TimestepEmbedder", "UNetDown", "UNetUp", "timestep_embedding"]
