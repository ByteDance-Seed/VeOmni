"""3-D interleaved mRoPE for the Cosmos3 towers.

Reasoner tokens carry the same increasing id on all three axes; generation tokens
carry the patched latent grid. The generation temporal axis is real time: one latent
frame advances it by ``base_fps / fps`` (4.8 for 5 fps clips at ``base_fps=24``), and
it starts after the sample's prompt (``temporal_offset``). Spatial ids restart per
sample. The three axes fold into one table by Qwen3-VL's interleaved rule.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


def build_text_position_ids(num_tokens: int, *, temporal_offset: float = 0, device=None) -> Tensor:
    """``(3, num_tokens)`` ids, identical on all three axes."""
    ids = torch.arange(num_tokens, dtype=torch.float32, device=device) + temporal_offset
    return ids.unsqueeze(0).expand(3, -1).contiguous()


def build_latent_grid(latent_shape: Sequence[int], patch_size: Sequence[int] = (1, 2, 2)) -> tuple[int, int, int]:
    """Patch grid ``(t, h, w)`` of a ``(C, F, H, W)`` latent."""
    _, frames, height, width = (int(size) for size in latent_shape)
    pt, ph, pw = patch_size
    if frames % pt or height % ph or width % pw:
        raise ValueError(f"latent {tuple(latent_shape)} is not divisible by patch {tuple(patch_size)}.")
    return frames // pt, height // ph, width // pw


def latent_temporal_positions(
    grid_t: int,
    *,
    fps: Optional[float] = None,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    enable_fps_modulation: Optional[bool] = None,
    temporal_offset: float = 0.0,
    device=None,
) -> Tensor:
    """``i * base_fps / fps + temporal_offset`` for ``i < grid_t``.

    ``enable_fps_modulation=None`` modulates only when ``fps`` is given; ``True``
    requires an ``fps``; ``False`` keeps the integer layout.
    """
    modulated = fps is not None if enable_fps_modulation is None else bool(enable_fps_modulation)
    if modulated and fps is None:
        raise ValueError("enable_fps_modulation=True needs the clip's fps.")
    if fps is not None and (fps <= 0 or not math.isfinite(float(fps))):
        raise ValueError(f"fps must be positive and finite, got {fps!r}.")

    index = torch.arange(grid_t, dtype=torch.float32, device=device)
    if not modulated:
        return index + float(temporal_offset)
    # Written in tokens per second like the reference; the compression factor cancels.
    tps = float(fps) / float(temporal_compression_factor)
    base_tps = float(base_fps) / float(temporal_compression_factor)
    return index / tps * base_tps + float(temporal_offset)


def build_latent_position_ids(grid_t: int, grid_h: int, grid_w: int, *, device=None, **temporal_kwargs) -> Tensor:
    """``(3, t*h*w)`` ids in the patchifier's ``t, h, w`` order."""
    t_index = latent_temporal_positions(grid_t, device=device, **temporal_kwargs).view(-1, 1, 1)
    h_index = torch.arange(grid_h, dtype=torch.float32, device=device).view(1, -1, 1)
    w_index = torch.arange(grid_w, dtype=torch.float32, device=device).view(1, 1, -1)
    return torch.stack(
        [
            t_index.expand(grid_t, grid_h, grid_w).reshape(-1),
            h_index.expand(grid_t, grid_h, grid_w).reshape(-1),
            w_index.expand(grid_t, grid_h, grid_w).reshape(-1),
        ],
        dim=0,
    )


def build_segmented_text_position_ids(num_tokens_per_segment: Sequence[int], *, device=None) -> Tensor:
    """Text ids restarting at zero for every sample of a pack."""
    return torch.cat([build_text_position_ids(int(length), device=device) for length in num_tokens_per_segment], dim=1)


def build_segmented_latent_position_ids(
    grids: Sequence[Sequence[int]],
    *,
    fps: Sequence[Optional[float]],
    temporal_offsets: Sequence[float],
    device=None,
    **temporal_kwargs,
) -> Tensor:
    """Latent ids per sample of a pack, each with its own fps and temporal start."""
    return torch.cat(
        [
            build_latent_position_ids(
                int(t), int(h), int(w), fps=fps_i, temporal_offset=offset_i, device=device, **temporal_kwargs
            )
            for (t, h, w), fps_i, offset_i in zip(grids, fps, temporal_offsets)
        ],
        dim=1,
    )


def interleave_mrope_frequencies(axis_frequencies: Tensor, mrope_section: Sequence[int], *, interleaved: bool = True):
    """Fold ``(3, L, F)`` per-axis angles into ``(L, F)``.

    Interleaved (Qwen3-VL): start from the temporal axis and overwrite the H/W stripes
    ``1, 4, 7, ...`` / ``2, 5, 8, ...``. Otherwise a contiguous section split.
    """
    if not interleaved:
        chunks, offset = [], 0
        for axis, size in enumerate(mrope_section):
            chunks.append(axis_frequencies[axis, :, offset : offset + size])
            offset += size
        return torch.cat(chunks, dim=-1)
    mixed = axis_frequencies[0].clone()
    for axis, offset in ((1, 1), (2, 2)):
        index = slice(offset, mrope_section[axis] * 3, 3)
        mixed[:, index] = axis_frequencies[axis][:, index]
    return mixed


class MotRope(nn.Module):
    """``(3, L)`` position ids -> ``(cos, sin)`` tables of shape ``(L, head_dim)``."""

    def __init__(self, head_dim: int, *, rope_theta: float, mrope_section: Sequence[int], interleaved: bool = True):
        super().__init__()
        if sum(mrope_section) != head_dim // 2:
            raise ValueError(f"mrope_section {tuple(mrope_section)} must sum to head_dim/2 = {head_dim // 2}.")
        self.mrope_section = tuple(int(size) for size in mrope_section)
        self.interleaved = interleaved
        steps = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        self.register_buffer("inv_freq", 1.0 / (rope_theta**steps), persistent=False)

    def forward(self, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        inv_freq = self.inv_freq.to(device=position_ids.device, dtype=torch.float32)
        angles = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq.view(1, 1, -1)
        mixed = interleave_mrope_frequencies(angles, self.mrope_section, interleaved=self.interleaved)
        emb = torch.cat((mixed, mixed), dim=-1)  # rotate-half layout
        return emb.cos(), emb.sin()
