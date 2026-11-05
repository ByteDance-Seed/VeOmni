"""Per-sample token cost for token-budget batching (the reference's ``PackingDataLoader``).

Mirrors the reference's ``SampleTokenCounter``: ``und = prompt tokens + 1`` (eos)
and ``gen = ceil(h_lat / p) * ceil(w_lat / p) * t_lat + 2`` (vision markers). The
prompt is tokenized only inside the condition model, so before that its length is
estimated from characters (``CHARS_PER_TOKEN``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Optional

import torch


CHARS_PER_TOKEN = 4.0


def _tokens_for(text_tokens: int, latent_grid: tuple[int, int, int], latent_patch_size: int) -> int:
    latent_t, latent_h, latent_w = latent_grid
    patches = math.ceil(latent_h / latent_patch_size) * math.ceil(latent_w / latent_patch_size)
    return (text_tokens + 1) + patches * latent_t + 2


def _prompt_tokens(sample: Mapping[str, Any]) -> int:
    """Exact once ``input_ids`` exist, else estimated; an empty (CFG-dropped) prompt costs at least one."""
    input_ids = sample.get("input_ids")
    if isinstance(input_ids, torch.Tensor) and input_ids.numel():
        return int(input_ids.numel())
    if isinstance(input_ids, (list, tuple)) and input_ids and isinstance(input_ids[0], torch.Tensor):
        return int(sum(tensor.numel() for tensor in input_ids))
    for key in ("inputs", "prompt"):
        value = sample.get(key)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        if isinstance(value, str):
            return max(1, math.ceil(len(value) / CHARS_PER_TOKEN))
    raise ValueError(f"cannot price a Cosmos3 sample without `input_ids` or a prompt; keys are {sorted(sample)}.")


def _frame_geometry(sample: Mapping[str, Any]) -> tuple[int, int, int]:
    """``(T, H, W)`` of the sample's decoded clip."""
    videos = sample.get("videos")
    if isinstance(videos, (list, tuple)):
        videos = videos[0]
    if isinstance(videos, torch.Tensor) and videos.dim() == 4:  # [T, C, H, W]
        return int(videos.shape[0]), int(videos.shape[2]), int(videos.shape[3])
    raise ValueError(f"cannot price a Cosmos3 sample: expected one decoded [T, C, H, W] clip, got {type(videos)}.")


def _latent_geometry(sample: Mapping[str, Any]) -> Optional[tuple[int, int, int]]:
    """``(T_lat, H_lat, W_lat)`` of an ``offline_training`` sample, which carries latents instead of frames."""
    latents = sample.get("latents")
    if isinstance(latents, (list, tuple)):
        latents = latents[0] if latents else None
    if not isinstance(latents, torch.Tensor):
        return None
    return int(latents.shape[1]), int(latents.shape[2]), int(latents.shape[3])


def cosmos3_packed_length(
    sample: Mapping[str, Any],
    *,
    spatial_compression_factor: int,
    temporal_compression_factor: int,
    latent_patch_size: int,
) -> int:
    text_tokens = _prompt_tokens(sample)
    # ``offline_training`` samples carry latents, the other tasks raw frames.
    latent_grid = _latent_geometry(sample)
    if latent_grid is None:
        frames, height, width = _frame_geometry(sample)
        latent_grid = (
            1 + (frames - 1) // temporal_compression_factor,
            height // spatial_compression_factor,
            width // spatial_compression_factor,
        )
    return _tokens_for(text_tokens, latent_grid, latent_patch_size)
