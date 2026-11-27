"""Packed sequence layout for MiniMax-H3-Image."""

from __future__ import annotations

import math

import torch

from ..minimax_h3.minimax_h3_core.batch_packing import _validate_fp32_metadata
from ..minimax_h3.minimax_h3_core.minimax_h3_dit import patchify_video, unpatchify_video
from ..minimax_h3.minimax_h3_core.packed_sequence import build_packed_fl2va, build_packed_ref2va


MINIMAX_H3_IMAGE_TAG = 0
MINIMAX_H3_TEXT_TAG = 1


def _validate_patch_size(patch_size: tuple[int, int, int]) -> None:
    if tuple(patch_size) != (1, 2, 2):
        raise ValueError(f"MiniMax-H3 uses patch size (1, 2, 2), got {patch_size}.")


def patchify_image_latents(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """Pack ``(batch, channels, 1, height, width)`` latents into transformer rows."""
    _validate_patch_size(patch_size)
    if any(size % patch for size, patch in zip(latents.shape[-3:], patch_size)):
        raise ValueError(f"Latents of shape {tuple(latents.shape)} are not divisible by patch {patch_size}.")
    return patchify_video(latents)


def unpatchify_image_rows(
    rows: torch.Tensor,
    latent_height: int,
    latent_width: int,
    channels: int,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """Inverse of :func:`patchify_image_latents` for a single image."""
    _validate_patch_size(patch_size)
    return unpatchify_video(rows, 1, latent_height, latent_width, channel=channels).squeeze(2)


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
    """Build an audio-free H3 layout with one target latent frame."""
    _validate_patch_size(patch_size)
    num_text_tokens = text_token_tags.shape[0]
    geometry = dict(text_len=num_text_tokens, latent_t=1, latent_h=latent_height, latent_w=latent_width, audio_t=0)
    if reference_grids:
        ref_blocks = [
            {"kind": "image", "latent_t": 1, "latent_h": height, "latent_w": width}
            for height, width in reference_grids
        ]
        packed = build_packed_ref2va(**geometry, ref_blocks=ref_blocks)
    else:
        packed = build_packed_fl2va(**geometry, keyframe_indices=[])

    device = text_token_tags.device
    text_indices = packed["text_pos"].to(device, non_blocking=True)
    image_indices = packed["img_pos"].to(device, non_blocking=True)
    token_tags = packed["token_tags"].to(device, non_blocking=True)
    token_tags[:num_text_tokens] = text_token_tags
    if image_tag != MINIMAX_H3_IMAGE_TAG:
        token_tags[image_indices] = image_tag

    return packed["img_position_ids"].squeeze(0), token_tags, image_indices, text_indices, packed["cond_rows"]


def pack_image_samples(samples: list[dict]) -> tuple[dict, list[int]]:
    """Concatenate compact visual inputs without mixing sample positions or heads."""
    chunks = {
        key: []
        for key in ("x", "prompt_embeds", "unique_timesteps", "inverse_indices", "token_tags", "img_position_ids")
    }
    placements = {key: [] for key in ("img_pos_info", "text_pos_info", "img_pos_for_infer_output_info")}
    cu, refiner_cu, row_counts, heads = [0], [0], [], []
    time_offset = 0
    for sample in samples:
        _validate_fp32_metadata(sample["unique_timesteps"], sample["img_position_ids"])
        if sample.get("update_mask") is not None and not sample.get("skip_mask_out_condition", False):
            raise ValueError("H3-Image packing requires explicit condition-row output cropping.")
        length = sample["img_position_ids"].shape[1]
        for key in chunks:
            chunks[key].append(sample[key] + time_offset if key == "inverse_indices" else sample[key])
        for key in placements:
            placements[key].append(sample[key]["position_ids"] + cu[-1])
        cu.append(cu[-1] + length)
        refiner_cu.append(refiner_cu[-1] + sample["prompt_embeds"].shape[0])
        time_offset += sample["unique_timesteps"].numel()
        row_counts.append(sample["img_pos_for_infer_output_info"]["position_ids"].numel())
        heads.append((sample["pdd_step_index"], length))
    result = {key: torch.cat(values, dim=1 if key == "img_position_ids" else 0) for key, values in chunks.items()}
    result.update({key: {"position_ids": torch.cat(values)} for key, values in placements.items()})
    device = result["x"].device
    for key, bounds in (("packed_seq_params", cu), ("refiner_packed_seq_params", refiner_cu)):
        result[key] = {
            "cu_seqlens_q": torch.tensor(bounds, dtype=torch.int32, device="cpu").to(device, non_blocking=True),
            "cu_seqlens_host": tuple(bounds),
            "max_seqlen_q": max(stop - start for start, stop in zip(bounds, bounds[1:])),
        }
    result.update(pdd_step_index=heads, skip_mask_out_condition=True)
    return result, row_counts
