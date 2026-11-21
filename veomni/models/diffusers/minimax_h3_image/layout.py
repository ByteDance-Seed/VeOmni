"""Packed sequence layout for MiniMax-H3-Image."""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image, ImageOps


MINIMAX_H3_IMAGE_TAG = 0
MINIMAX_H3_TEXT_TAG = 1
DEFAULT_SIGMAS = (1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0)
REFERENCE_NOISE_AUG = 0.999
_ROPE_SPATIAL_SCALE = 32


def resize_reference(image: Image.Image, min_short_edge: int, max_short_edge: int) -> Image.Image:
    """Match inference reference sizing for both training encoders."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    if min(image.size) < 32:
        raise ValueError(f"A reference image needs at least 32 pixels on each side, got {image.size}.")
    short_edge = min(image.size)
    if min_short_edge <= short_edge <= max_short_edge:
        return image
    scale = (min_short_edge if short_edge < min_short_edge else max_short_edge) / short_edge
    width, height = (max(32, round(size * scale / 32) * 32) for size in image.size)
    return image.resize((width, height), Image.Resampling.BICUBIC)


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

    device = text_token_tags.device
    text_indices = torch.arange(num_text_tokens, device=device)
    image_indices = torch.arange(num_text_tokens, sequence_length, device=device)
    token_tags = torch.empty(sequence_length, dtype=torch.long, device=device)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[image_indices] = image_tag

    return position_ids, token_tags, image_indices, text_indices, num_reference_rows


def pack_image_samples(samples: list[dict]) -> tuple[dict, list[int]]:
    """Concatenate image samples, retaining independent attention, positions and heads."""
    if not samples:
        raise ValueError("H3-Image requires a nonempty microbatch.")
    chunks = {
        key: []
        for key in (
            "x",
            "prompt_embeds",
            "unique_timesteps",
            "timestep_indices",
            "token_tags",
            "position_ids",
            "image_indices",
            "text_indices",
        )
    }
    cu, refiner_cu, row_counts, heads = [0], [0], [], []
    time_offset = 0
    for sample in samples:
        if sample["unique_timesteps"].dtype != torch.float32 or sample["position_ids"].dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise ValueError(
                "H3-Image packing must retain timestep/position precision; use cast_forward_inputs=false."
            )
        length = sample["position_ids"].shape[0]
        text = sample["prompt_embeds"]
        text = text[0] if text.dim() == 3 else text
        for key in chunks:
            value = text if key == "prompt_embeds" else sample[key]
            if key in ("image_indices", "text_indices"):
                value = value + cu[-1]
            elif key == "timestep_indices":
                value = value + time_offset
            chunks[key].append(value)
        cu.append(cu[-1] + length)
        refiner_cu.append(refiner_cu[-1] + text.shape[0])
        time_offset += sample["unique_timesteps"].numel()
        row_counts.append(sample["image_indices"].numel())
        heads.append((sample["pdd_step_index"], length))
    result = {key: torch.cat(values, dim=0) for key, values in chunks.items()}
    device = result["x"].device
    for key, bounds in (("packed_seq_params", cu), ("refiner_packed_seq_params", refiner_cu)):
        result[key] = {
            "cu_seqlens_q": torch.tensor(bounds, dtype=torch.int32, device=device),
            "cu_seqlens_host": tuple(bounds),
            "max_seqlen_q": max(stop - start for start, stop in zip(bounds, bounds[1:])),
        }
    result["pdd_step_index"] = heads
    return result, row_counts
