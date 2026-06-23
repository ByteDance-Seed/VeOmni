"""BAGEL latent VAE module.

Tensor call-sites:
* ``bagel_vae.encode`` maps normalized image tensors to scaled latent grids.
* ``bagel_vae.decode`` maps generated latent grids to decoded image tensors.

The VAE module is a codec boundary only. Flow timestep/noise sampling, latent
patchification, packed MoT indexes, and conversation-carrier mutation belong to
the Bagel VAE module mixin and downstream Bagel nodes.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from transformers import PreTrainedModel

from .configuration import BagelVAEConfig
from .modulemixin import BagelVAEModuleMixin
from .processing import BagelVAEProcessor


class BagelVAE(BagelVAEModuleMixin, PreTrainedModel):
    config_class = BagelVAEConfig
    image_processor_class = BagelVAEProcessor
    base_model_prefix = "bagel_vae"
    main_input_name = "pixel_values"
    _no_split_modules: list[str] = ["ResnetBlock", "AttnBlock"]
    supports_gradient_checkpointing = True

    def __init__(self, config: BagelVAEConfig) -> None:
        super().__init__(config)
        self.encoder = Encoder(
            resolution=config.resolution,
            in_channels=config.in_channels,
            ch=config.ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            z_channels=config.z_channels,
        )
        self.decoder = Decoder(
            resolution=config.resolution,
            in_channels=config.in_channels,
            ch=config.ch,
            out_ch=config.out_ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            z_channels=config.z_channels,
        )
        self.reg = DiagonalGaussian()
        self._image_processor: BagelVAEProcessor | None = None
        self.post_init()

    def freeze_model(self) -> None:
        if self.config.freeze:
            self.eval()
            self.requires_grad_(False)

    @property
    def _encoder_device(self) -> torch.device:
        return self.encoder.conv_in.weight.device

    @property
    def _decoder_device(self) -> torch.device:
        return self.decoder.conv_in.weight.device

    @contextmanager
    def _runtime_context(self, tensor: torch.Tensor):
        grad_context = torch.enable_grad()
        if self.config.freeze:
            grad_context = torch.no_grad()

        autocast_context = nullcontext()
        if tensor.device.type == "cuda" and self.dtype != torch.float32:
            autocast_context = torch.amp.autocast("cuda", enabled=True, dtype=self.dtype)

        with grad_context, autocast_context:
            yield

    def encode(
        self,
        pixel_values: torch.Tensor | None = None,
        **kwargs: object,
    ) -> dict[str, Any]:
        del kwargs
        if pixel_values is not None:
            return self._encode_pixel_values(pixel_values)

        dummy = self.dummy_inputs(kind="encode")
        outputs = self.encode(**dummy)
        outputs["is_dummy"] = True
        return outputs

    def decode(
        self,
        latents: torch.Tensor | None = None,
        **kwargs: object,
    ) -> dict[str, Any]:
        del kwargs
        if latents is not None:
            return self._decode_latents(latents)

        dummy = self.dummy_inputs(kind="decode")
        outputs = self._decode_latents(dummy["latents"])
        outputs["is_dummy"] = True
        return outputs

    def _encode_pixel_values(self, pixel_values: torch.Tensor) -> dict[str, Any]:
        pixel_values = pixel_values.to(device=self._encoder_device, dtype=self.dtype)
        with self._runtime_context(pixel_values):
            latents = self.reg(self.encoder(pixel_values))
            latents = self.config.scale_factor * (latents - self.config.shift_factor)
        return {"latents": latents.to(dtype=self.dtype)}

    def _decode_latents(self, latents: torch.Tensor) -> dict[str, Any]:
        latents = latents.to(device=self._decoder_device, dtype=self.dtype)
        latents = latents / self.config.scale_factor + self.config.shift_factor
        with self._runtime_context(latents):
            pixel_values = self.decoder(latents)
        return {"pixel_values": pixel_values.to(dtype=self.dtype)}


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.norm(hidden_states)
        query = self.q(hidden_states)
        key = self.k(hidden_states)
        value = self.v(hidden_states)

        batch, channels, height, width = query.shape
        query = rearrange(query, "b c h w -> b 1 (h w) c").contiguous()
        key = rearrange(key, "b c h w -> b 1 (h w) c").contiguous()
        value = rearrange(value, "b c h w -> b 1 (h w) c").contiguous()
        hidden_states = nn.functional.scaled_dot_product_attention(query, key, value)
        return rearrange(hidden_states, "b 1 (h w) c -> b c h w", h=height, w=width, c=channels, b=batch)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        hidden_states = self.norm1(x)
        hidden_states = swish(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = swish(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + hidden_states


class Downsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = False
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        hidden_stack = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                block = self.down[i_level].block[i_block]
                hidden_states = (
                    self._gradient_checkpointing_func(block.__call__, hidden_stack[-1])
                    if self.gradient_checkpointing and self.training
                    else block(hidden_stack[-1])
                )
                if len(self.down[i_level].attn) > 0:
                    attn = self.down[i_level].attn[i_block]
                    hidden_states = (
                        self._gradient_checkpointing_func(attn.__call__, hidden_states)
                        if self.gradient_checkpointing and self.training
                        else attn(hidden_states)
                    )
                hidden_stack.append(hidden_states)
            if i_level != self.num_resolutions - 1:
                downsample = self.down[i_level].downsample
                hidden_stack.append(
                    self._gradient_checkpointing_func(downsample.__call__, hidden_stack[-1])
                    if self.gradient_checkpointing and self.training
                    else downsample(hidden_stack[-1])
                )

        hidden_states = hidden_stack[-1]
        if self.gradient_checkpointing and self.training:
            hidden_states = self._gradient_checkpointing_func(self.mid.block_1.__call__, hidden_states)
            hidden_states = self._gradient_checkpointing_func(self.mid.attn_1.__call__, hidden_states)
            hidden_states = self._gradient_checkpointing_func(self.mid.block_2.__call__, hidden_states)
        else:
            hidden_states = self.mid.block_1(hidden_states)
            hidden_states = self.mid.attn_1(hidden_states)
            hidden_states = self.mid.block_2(hidden_states)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = swish(hidden_states)
        return self.conv_out(hidden_states)


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = False
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        hidden_states = self.conv_in(z)

        if self.gradient_checkpointing and self.training:
            hidden_states = self._gradient_checkpointing_func(self.mid.block_1.__call__, hidden_states)
            hidden_states = self._gradient_checkpointing_func(self.mid.attn_1.__call__, hidden_states)
            hidden_states = self._gradient_checkpointing_func(self.mid.block_2.__call__, hidden_states)
        else:
            hidden_states = self.mid.block_1(hidden_states)
            hidden_states = self.mid.attn_1(hidden_states)
            hidden_states = self.mid.block_2(hidden_states)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                block = self.up[i_level].block[i_block]
                hidden_states = (
                    self._gradient_checkpointing_func(block.__call__, hidden_states)
                    if self.gradient_checkpointing and self.training
                    else block(hidden_states)
                )
                if len(self.up[i_level].attn) > 0:
                    attn = self.up[i_level].attn[i_block]
                    hidden_states = (
                        self._gradient_checkpointing_func(attn.__call__, hidden_states)
                        if self.gradient_checkpointing and self.training
                        else attn(hidden_states)
                    )
            if i_level != 0:
                upsample = self.up[i_level].upsample
                hidden_states = (
                    self._gradient_checkpointing_func(upsample.__call__, hidden_states)
                    if self.gradient_checkpointing and self.training
                    else upsample(hidden_states)
                )

        hidden_states = self.norm_out(hidden_states)
        hidden_states = swish(hidden_states)
        return self.conv_out(hidden_states)


class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1) -> None:
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        return mean


__all__ = [
    "AttnBlock",
    "BagelVAE",
    "BagelVAEConfig",
    "Decoder",
    "DiagonalGaussian",
    "Downsample",
    "Encoder",
    "ResnetBlock",
    "Upsample",
]
