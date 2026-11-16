"""Packed sequence layout for MiniMax-H3-Image."""

from __future__ import annotations

import math

import numpy as np
import torch


MINIMAX_H3_IMAGE_TAG = 0
MINIMAX_H3_TEXT_TAG = 1
_ROPE_SPATIAL_SCALE = 32


def patchify_image_latents(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """Pack ``(batch, channels, 1, height, width)`` latents into transformer rows."""
    patch_t, patch_h, patch_w = patch_size
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"Latents of shape {tuple(latents.shape)} are not divisible by patch {patch_size}.")
    latents = latents.reshape(
        batch_size,
        channels,
        num_frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(-1, channels * patch_t * patch_h * patch_w).contiguous()


def unpatchify_image_rows(
    rows: torch.Tensor,
    latent_height: int,
    latent_width: int,
    channels: int,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """Inverse of :func:`patchify_image_latents` for a single image."""
    _, patch_h, patch_w = patch_size
    rows = rows.reshape(-1, latent_height // patch_h, latent_width // patch_w, channels, patch_h, patch_w)
    rows = rows.permute(0, 3, 1, 4, 2, 5)
    return rows.reshape(-1, channels, latent_height, latent_width).contiguous()


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def _frame_position_grid(
    latent_height: int, latent_width: int, patch_h: int, patch_w: int
) -> tuple[torch.Tensor, torch.Tensor]:
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def get_image_canvas_size(height: int, width: int, min_generation_area: int, multiple: int) -> tuple[int, int]:
    """Canvas size used for generation before resizing back to the request size."""
    scale = max(1.0, math.sqrt(min_generation_area / (width * height)))
    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )


def build_image_packed_sequence(
    text_token_tags: torch.Tensor,
    reference_grids: list[tuple[int, int]],
    latent_height: int,
    latent_width: int,
    patch_size: tuple[int, int, int],
    image_tag: int = MINIMAX_H3_IMAGE_TAG,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build the ``[text | reference images | generated image]`` layout."""
    _, patch_h, patch_w = patch_size
    num_text_tokens = text_token_tags.shape[0]
    num_reference_rows = sum((height // patch_h) * (width // patch_w) for height, width in reference_grids)
    num_target_rows = (latent_height // patch_h) * (latent_width // patch_w)
    sequence_length = num_text_tokens + num_reference_rows + num_target_rows

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    cursor = num_text_tokens
    rotary_time = float(num_text_tokens)
    for height, width in reference_grids:
        rows = slice(cursor, cursor + (height // patch_h) * (width // patch_w))
        cursor = rows.stop
        frame_grid, _ = _frame_position_grid(height, width, patch_h, patch_w)
        position_ids[rows, 0] = rotary_time
        position_ids[rows, 1:] = frame_grid
        rotary_time += 1.0
    frame_grid, _ = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)
    position_ids[cursor:, 0] = rotary_time
    position_ids[cursor:, 1:] = frame_grid

    text_indices = torch.arange(num_text_tokens)
    image_indices = torch.arange(num_text_tokens, sequence_length)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[image_indices] = image_tag

    return position_ids, token_tags, image_indices, text_indices, num_reference_rows
