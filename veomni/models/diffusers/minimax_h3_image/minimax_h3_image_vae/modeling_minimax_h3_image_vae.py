"""MiniMax-H3-Image VAE.

Adapted from the MiniMax-H3-Image reference implementation in diffusers:
``diffusers/models/autoencoders/autoencoder_kl_minimax_h3_image.py``.
The encoder is the MiniMax-H3 causal video encoder over one frame; the
decoder is a non-causal 2D ViT. The state dict follows ``vae_diffusers/`` in
the released checkpoint.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from .configuration_minimax_h3_image_vae import MiniMaxH3ImageVAEConfig


def _parameter_dtype(module: nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


class MiniMaxH3ImageCausalConv3d(nn.Conv3d):
    """3D convolution with symmetric spatial and causal temporal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        spatial_padding: int = 0,
        temporal_padding: int = 0,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.spatial_padding = spatial_padding
        self.temporal_padding = temporal_padding
        self.spatial_padding_mode = spatial_padding_mode

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.spatial_padding > 0:
            padding = self.spatial_padding
            hidden_states = F.pad(
                hidden_states, (padding, padding, padding, padding, 0, 0), mode=self.spatial_padding_mode
            )
        if self.temporal_padding > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, 0, self.temporal_padding, 0), mode="constant")
        return F.conv3d(hidden_states, self.weight, self.bias, stride=self.stride, padding=0, dilation=self.dilation)


class MiniMaxH3ImageGroupNorm(nn.GroupNorm):
    """Group norm applied to each latent frame in isolation."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous()
        hidden_states = hidden_states.view(batch_size * num_frames, num_channels, 1, height, width)
        hidden_states = super().forward(hidden_states)
        hidden_states = hidden_states.view(batch_size, num_frames, num_channels, height, width)
        return hidden_states.permute(0, 2, 1, 3, 4).contiguous()


class MiniMaxH3ImageResnetBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = MiniMaxH3ImageGroupNorm(norm_num_groups, in_channels, eps=norm_eps, affine=True)
        self.conv1 = MiniMaxH3ImageCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.norm2 = MiniMaxH3ImageGroupNorm(norm_num_groups, out_channels, eps=norm_eps, affine=True)
        self.conv2 = MiniMaxH3ImageCausalConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = MiniMaxH3ImageCausalConv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = F.silu(self.norm1(hidden_states))
        hidden_states = self.conv1(hidden_states)
        hidden_states = F.silu(self.norm2(hidden_states))
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return residual + hidden_states


class MiniMaxH3ImageDownsample3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_stride: int = 1,
        spatial_stride: int = 2,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.spatial_stride = spatial_stride
        self.spatial_padding_mode = spatial_padding_mode
        self.conv = MiniMaxH3ImageCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=(temporal_stride, spatial_stride, spatial_stride),
            spatial_padding=0,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.spatial_stride == 2:
            hidden_states = F.pad(hidden_states, (0, 1, 0, 1, 0, 0), mode=self.spatial_padding_mode)
        return self.conv(hidden_states)


class MiniMaxH3ImageDownBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        temporal_downsample_factor: int,
        spatial_downsample_factor: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                MiniMaxH3ImageResnetBlock3d(
                    in_channels=in_channels if index == 0 else out_channels,
                    out_channels=out_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for index in range(num_layers)
            ]
        )
        self.downsamplers = None
        if temporal_downsample_factor * spatial_downsample_factor > 1:
            self.downsamplers = nn.ModuleList(
                [
                    MiniMaxH3ImageDownsample3d(
                        out_channels,
                        out_channels,
                        temporal_stride=temporal_downsample_factor,
                        spatial_stride=spatial_downsample_factor,
                        spatial_padding_mode=spatial_padding_mode,
                    )
                ]
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class MiniMaxH3ImageEncoder3d(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 48,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024),
        layers_per_block: int = 2,
        spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.conv_in = MiniMaxH3ImageCausalConv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        block_in_channels = (block_out_channels[0],) + tuple(block_out_channels[:-1])
        self.down_blocks = nn.ModuleList(
            [
                MiniMaxH3ImageDownBlock3d(
                    in_channels=block_in_channels[index],
                    out_channels=block_out_channels[index],
                    num_layers=layers_per_block,
                    temporal_downsample_factor=temporal_downsample_factors[index],
                    spatial_downsample_factor=spatial_downsample_factors[index],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for index in range(len(block_out_channels))
            ]
        )
        self.norm_out = MiniMaxH3ImageGroupNorm(norm_num_groups, block_out_channels[-1], eps=norm_eps, affine=True)
        self.conv_out = MiniMaxH3ImageCausalConv3d(
            block_out_channels[-1],
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)
        hidden_states = F.silu(self.norm_out(hidden_states))
        return self.conv_out(hidden_states)


class MiniMaxH3ImageRotaryPosEmbed(nn.Module):
    def __init__(self, dim: int, theta: float = 100.0, num_axes: int = 3) -> None:
        super().__init__()
        if dim % (2 * num_axes) != 0:
            raise ValueError(f"dim {dim} must be divisible by 2 * num_axes {2 * num_axes}.")
        inv_freq = 1.0 / theta ** torch.arange(0, 1, 2 * num_axes / dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = 2.0 * math.pi * position_ids[:, :, :, None] * self.inv_freq[None, None, None, :]
        angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
        return angles.cos(), angles.sin()


class MiniMaxH3ImageVAEAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-5, bias: bool = True) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=bias), nn.Dropout(0.0)])

    def forward(
        self, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        query = self.to_q(hidden_states).unflatten(2, (self.heads, -1))
        key = self.to_k(hidden_states).unflatten(2, (self.heads, -1))
        value = self.to_v(hidden_states).unflatten(2, (self.heads, -1))

        query = self.norm_q(query.float()).to(query.dtype)
        key = self.norm_k(key.float()).to(key.dtype)

        if rotary_emb is not None:
            cos, sin = rotary_emb
            cos = cos.to(query.dtype)
            sin = sin.to(query.dtype)
            rotary_dim = cos.shape[-1]
            query_rotary, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
            key_rotary, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
            query_first, query_second = query_rotary.chunk(2, dim=-1)
            key_first, key_second = key_rotary.chunk(2, dim=-1)
            query_rotated = torch.cat([-query_second, query_first], dim=-1)
            key_rotated = torch.cat([-key_second, key_first], dim=-1)
            query = torch.cat([query_rotary * cos + query_rotated * sin, query_pass], dim=-1)
            key = torch.cat([key_rotary * cos + key_rotated * sin, key_pass], dim=-1)

        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        return self.to_out[0](hidden_states)


class MiniMaxH3ImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ffn_mult: int = 4,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.attn = MiniMaxH3ImageVAEAttention(dim=dim, heads=heads, dim_head=dim_head, eps=eps, bias=bias)
        self.scale1 = nn.Parameter(torch.zeros(dim))
        self.norm2 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.ff = FeedForward(dim, mult=ffn_mult, activation_fn="swiglu", bias=bias)
        self.scale2 = nn.Parameter(torch.zeros(dim))

    def forward(
        self, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        compute_dtype = _parameter_dtype(self.attn.to_q)
        norm_hidden_states = self.norm1(hidden_states.float()).to(compute_dtype)
        hidden_states = hidden_states + self.attn(norm_hidden_states, rotary_emb) * self.scale1
        norm_hidden_states = self.norm2(hidden_states.float()).to(compute_dtype)
        hidden_states = hidden_states + self.ff(norm_hidden_states) * self.scale2
        return hidden_states


class MiniMaxH3ImageViTDecoder2d(nn.Module):
    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 3,
        patch_size: int = 16,
        num_layers: int = 24,
        num_attention_heads: int = 16,
        attention_head_dim: int = 64,
        num_register_tokens: int = 4,
        ffn_mult: int = 4,
        rope_theta: float = 100.0,
        rope_dim_ratio: float = 0.5,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens
        self.rope = MiniMaxH3ImageRotaryPosEmbed(
            int(attention_head_dim * rope_dim_ratio), theta=rope_theta, num_axes=2
        )
        self.proj_in = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, dim))
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3ImageTransformerBlock(
                    dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    ffn_mult=ffn_mult,
                    eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True, eps=norm_eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size * patch_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, num_channels)
        hidden_states = self.proj_in(hidden_states)
        num_patches = hidden_states.shape[1]

        register_tokens = self.register_tokens.expand(batch_size, -1, -1).to(hidden_states.dtype)
        cls_token = torch.zeros_like(hidden_states[:, :1, :])
        hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)

        grids = [
            2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden_states.device) / size) - 1.0
            for size in (height, width)
        ]
        position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 1)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
        suffix_ids = position_ids.new_zeros((batch_size, self.num_register_tokens + 1, 2))
        position_ids = torch.cat([position_ids, suffix_ids], dim=1)
        rotary_emb = self.rope(position_ids)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary_emb)

        hidden_states = self.norm_out(hidden_states).to(_parameter_dtype(self.proj_out))
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states[:, :num_patches, :]

        patch_size = self.patch_size
        hidden_states = hidden_states.view(batch_size, height, width, self.out_channels, patch_size, patch_size)
        hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5).contiguous()
        return hidden_states.reshape(batch_size, self.out_channels, height * patch_size, width * patch_size)


class MiniMaxH3ImageVAE(nn.Module):
    """Released MiniMax-H3-Image VAE with tiled encode/decode."""

    def __init__(self, config: MiniMaxH3ImageVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_compression_ratio = math.prod(config.spatial_downsample_factors)
        self.encoder = MiniMaxH3ImageEncoder3d(
            in_channels=config.in_channels,
            out_channels=2 * config.latent_channels,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
            spatial_downsample_factors=config.spatial_downsample_factors,
            temporal_downsample_factors=config.temporal_downsample_factors,
            norm_num_groups=config.norm_num_groups,
            norm_eps=config.norm_eps,
            spatial_padding_mode=config.spatial_padding_mode,
        )
        self.quant_conv = nn.Conv3d(2 * config.latent_channels, 2 * config.latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(config.latent_channels, config.latent_channels, kernel_size=1)
        self.decoder = MiniMaxH3ImageViTDecoder2d(
            in_channels=config.latent_channels,
            out_channels=config.out_channels,
            patch_size=self.spatial_compression_ratio,
            num_layers=config.decoder_num_layers,
            num_attention_heads=config.decoder_num_attention_heads,
            attention_head_dim=config.decoder_attention_head_dim,
            num_register_tokens=config.decoder_num_register_tokens,
            ffn_mult=config.decoder_ffn_mult,
            rope_theta=config.decoder_rope_theta,
            rope_dim_ratio=config.decoder_rope_dim_ratio,
            norm_eps=config.decoder_norm_eps,
        )
        self.use_slicing = False
        self.use_tiling = True
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_overlap_height = 64
        self.tile_sample_min_overlap_width = 64

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def enable_tiling(
        self,
        tile_sample_min_height: int | None = None,
        tile_sample_min_width: int | None = None,
        tile_sample_min_overlap_height: int | None = None,
        tile_sample_min_overlap_width: int | None = None,
    ) -> None:
        height = self.tile_sample_min_height if tile_sample_min_height is None else tile_sample_min_height
        width = self.tile_sample_min_width if tile_sample_min_width is None else tile_sample_min_width
        overlap_height = (
            self.tile_sample_min_overlap_height
            if tile_sample_min_overlap_height is None
            else tile_sample_min_overlap_height
        )
        overlap_width = (
            self.tile_sample_min_overlap_width
            if tile_sample_min_overlap_width is None
            else tile_sample_min_overlap_width
        )
        if height <= 0 or width <= 0 or overlap_height <= 0 or overlap_width <= 0:
            raise ValueError(
                f"Tiled VAE sizes and overlaps must be positive: got height={height}, width={width}, "
                f"overlap_height={overlap_height}, overlap_width={overlap_width}."
            )
        if height <= overlap_height or width <= overlap_width:
            raise ValueError(
                "Tiled VAE sampling requires each tile size to be greater than its overlap: "
                f"got height={height}, overlap_height={overlap_height}, width={width}, overlap_width={overlap_width}."
            )
        self.use_tiling = True
        self.tile_sample_min_height = height
        self.tile_sample_min_width = width
        self.tile_sample_min_overlap_height = overlap_height
        self.tile_sample_min_overlap_width = overlap_width

    def disable_tiling(self) -> None:
        self.use_tiling = False

    def _split_tiles(self, length: int, tile_size: int, min_overlap: int) -> tuple[list[int], list[int], list[int]]:
        if tile_size <= 0 or min_overlap <= 0 or tile_size <= min_overlap:
            raise ValueError(
                f"Tile size and overlap must be positive, with tile size greater than the overlap: got tile_size={tile_size}, "
                f"min_overlap={min_overlap}."
            )
        if tile_size >= length:
            return [0], [length], []
        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
            num_tiles += 1
        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for index in range(remaining // self.spatial_compression_ratio):
            overlaps[index % (num_tiles - 1)] += self.spatial_compression_ratio
        indices = [0]
        for index in range(num_tiles - 1):
            indices.append(indices[-1] + tile_size - overlaps[index])
        return indices, [tile_size] * num_tiles, overlaps

    @staticmethod
    def _blend(a: torch.Tensor, b: torch.Tensor, blend_extent: int, dim: int) -> torch.Tensor:
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = (1 - positions / blend_extent).view(shape)
        weight_b = (positions / blend_extent).view(shape)
        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)
        blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b
        if blend_extent == b.shape[dim]:
            return blended
        slice_rest = [slice(None)] * b.ndim
        slice_rest[dim] = slice(blend_extent, None)
        return torch.cat([blended, b[tuple(slice_rest)]], dim=dim)

    def _stitch_tiles(
        self,
        tiles: list[list[torch.Tensor]],
        height_overlaps: list[int],
        width_overlaps: list[int],
    ) -> torch.Tensor:
        result_rows = []
        for row_index, row in enumerate(tiles):
            result_row = []
            for column_index, tile in enumerate(row):
                if row_index > 0:
                    tile = self._blend(tiles[row_index - 1][column_index], tile, height_overlaps[row_index - 1], -2)
                if column_index > 0:
                    tile = self._blend(row[column_index - 1], tile, width_overlaps[column_index - 1], -1)
                if row_index < len(tiles) - 1:
                    tile = tile[..., : -height_overlaps[row_index], :]
                if column_index < len(row) - 1:
                    tile = tile[..., :, : -width_overlaps[column_index]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_tiling:
            return self.quant_conv(self.encoder(x[:, :, None]))[:, :, -1]
        height, width = x.shape[-2], x.shape[-1]
        y_indices, y_lengths, y_overlaps = self._split_tiles(
            height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = self._split_tiles(
            width, self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )
        rows = []
        for y_pos, y_len in zip(y_indices, y_lengths):
            row = []
            for x_pos, x_len in zip(x_indices, x_lengths):
                tile = x[..., y_pos : y_pos + y_len, x_pos : x_pos + x_len]
                row.append(self.quant_conv(self.encoder(tile[:, :, None]))[:, :, -1])
            rows.append(row)
        latent_y_overlaps = [overlap // self.spatial_compression_ratio for overlap in y_overlaps]
        latent_x_overlaps = [overlap // self.spatial_compression_ratio for overlap in x_overlaps]
        return self._stitch_tiles(rows, latent_y_overlaps, latent_x_overlaps)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        if not self.use_tiling:
            return self.decoder(self.post_quant_conv(z))
        height = z.shape[-2] * self.spatial_compression_ratio
        width = z.shape[-1] * self.spatial_compression_ratio
        y_indices, y_lengths, y_overlaps = self._split_tiles(
            height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = self._split_tiles(
            width, self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )
        ratio = self.spatial_compression_ratio
        rows = []
        for y_pos, y_len in zip(y_indices, y_lengths):
            row = []
            for x_pos, x_len in zip(x_indices, x_lengths):
                tile = z[
                    ...,
                    y_pos // ratio : y_pos // ratio + y_len // ratio,
                    x_pos // ratio : x_pos // ratio + x_len // ratio,
                ]
                row.append(self.decoder(self.post_quant_conv(tile)))
            rows.append(row)
        return self._stitch_tiles(rows, y_overlaps, x_overlaps)

    def encode(self, x: torch.Tensor, return_dict: bool = True) -> AutoencoderKLOutput | tuple:
        x = x.to(_parameter_dtype(self.encoder))
        if self.use_slicing and x.shape[0] > 1:
            moments = torch.cat([self._encode(x_slice) for x_slice in x.split(1)])
        else:
            moments = self._encode(x)
        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | tuple:
        z = z.to(_parameter_dtype(self.post_quant_conv))
        if self.use_slicing and z.shape[0] > 1:
            decoded = torch.cat([self._decode(z_slice) for z_slice in z.split(1)])
        else:
            decoded = self._decode(z)
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> DecoderOutput | tuple:
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        return self.decode(z, return_dict=return_dict)
