"""Qwen3-VL packed-sequence helpers (CPU pack + GPU mask scatter).

The conversation carrier path still walks ``conversation_list`` between
modules. The packed path builds tokens / M-RoPE / visual masks once on the
CPU preprocessor, then each module only:

1. gathers the embeddings it owns via a boolean mask, and
2. ``masked_scatter``s them back onto the packed sequence.

Dummy FSDP-anchor images stay *out* of the packed sequence (same as the
conversation packer skipping ``role='dummy'``) and are folded in with
``mean() * 0`` so the ViT still participates in FSDP collectives.

Vision token counts are variable (image / video ``grid_thw``), unlike Janus's
fixed 576-token slots. DeepStack features are produced on GPU by the vision
tower and passed as a sibling tensor — they are not CPU-packed.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F

from veomni.utils.constants import IGNORE_INDEX

from ...utils.conversation import ConversationItem, is_dummy
from .llm.modeling import qwen3vl_vision_position_ids
from .vision.processing import _OMNI_GRID


VISION_SOURCE = "qwen3vl_vision"

PACKED_INPUT_IDS = "packed_input_ids"
PACKED_LABELS = "packed_labels"
PACKED_ATTENTION_MASK = "packed_attention_mask"
PACKED_POSITION_IDS = "packed_position_ids"
PACKED_FEATURES = "packed_features"
PACKED_HIDDEN = "packed_hidden"
PACKED_CU_SEQLENS = "packed_cu_seqlens"
PACKED_MAX_LENGTH = "packed_max_length"
VISUAL_POS_MASK = "visual_pos_mask"
PIXEL_VALUES = "pixel_values"
IMAGE_GRID_THW = "image_grid_thw"
VISUAL_NUM_REAL = "visual_num_real"
DEEPSTACK_VISUAL_EMBEDS = "deepstack_visual_embeds"


def _as_1d_long(value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.long)
    if tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)
    return tensor.reshape(-1).to(dtype=torch.long)


def _grid_from_item(item: ConversationItem) -> list[int]:
    grid = item.meta.get(_OMNI_GRID, item.meta.get("grid_thw"))
    if grid is None:
        raise ValueError(
            "pack_qwen3vl_conversations: image/video item is missing grid_thw (run the vision preprocessor first)."
        )
    if isinstance(grid, torch.Tensor):
        return [int(x) for x in grid.reshape(-1).tolist()]
    return [int(x) for x in grid]


def _merged_tokens(grid: list[int], spatial_merge_size: int) -> int:
    t, h, w = grid
    return int(t * h * w) // (spatial_merge_size * spatial_merge_size)


def _patches_2d(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            "pack_qwen3vl_conversations: image/video item.value must be a patch tensor "
            f"(run the vision preprocessor first); got {type(value).__name__}."
        )
    tensor = value
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 2:
        raise ValueError(f"pack_qwen3vl_conversations: expected 2-D patches (N, C), got shape {tuple(tensor.shape)}.")
    return tensor


def pack_qwen3vl_conversations(
    conversation_list: Iterable[list[ConversationItem]],
    *,
    pad_token_id: int,
    spatial_merge_size: int = 2,
) -> dict[str, Any]:
    """Build packed ids / 3-row M-RoPE / visual mask / concat patches.

    ``conversation_list`` is consumed exactly once, so the caller may pass a
    generator that tokenizes each sample on demand.
    """
    ids_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    position_chunks: list[torch.Tensor] = []
    visual_chunks: list[torch.Tensor] = []
    sample_lengths: list[int] = []
    real_pixels: list[torch.Tensor] = []
    real_grids: list[list[int]] = []
    dummy_pixels: list[torch.Tensor] = []
    dummy_grids: list[list[int]] = []

    merge = int(spatial_merge_size)
    pad_id = int(pad_token_id)

    for sample in conversation_list:
        sample_ids: list[torch.Tensor] = []
        sample_labels: list[torch.Tensor] = []
        sample_visual: list[torch.Tensor] = []
        sample_pos: list[torch.Tensor] = []
        current_pos = 0
        for part in sample:
            if is_dummy(part):
                if isinstance(part.value, torch.Tensor):
                    dummy_pixels.append(_patches_2d(part.value))
                    dummy_grids.append(_grid_from_item(part))
                continue
            if part.type == "text":
                token_ids = _as_1d_long(part.value)
                labels = _as_1d_long(part.meta.get("labels", torch.full_like(token_ids, IGNORE_INDEX)))
                length = int(token_ids.numel())
                sample_ids.append(token_ids)
                sample_labels.append(labels)
                sample_visual.append(torch.zeros(length, dtype=torch.bool))
                seg_pos = torch.arange(length, dtype=torch.long).view(1, -1).expand(3, -1) + current_pos
                sample_pos.append(seg_pos)
                current_pos += length
                continue
            if part.type not in ("image", "video"):
                continue
            grid = _grid_from_item(part)
            n_tok = _merged_tokens(grid, merge)
            if n_tok <= 0:
                raise ValueError(f"pack_qwen3vl_conversations: non-positive merged tokens for grid {grid}.")
            sample_ids.append(torch.full((n_tok,), pad_id, dtype=torch.long))
            sample_labels.append(torch.full((n_tok,), IGNORE_INDEX, dtype=torch.long))
            sample_visual.append(torch.ones(n_tok, dtype=torch.bool))
            grid_thw = torch.tensor(grid, dtype=torch.long)
            sample_pos.append(qwen3vl_vision_position_ids(current_pos, grid_thw, merge))
            current_pos += int(max(int(grid[1]), int(grid[2])) // merge)
            real_pixels.append(_patches_2d(part.value))
            real_grids.append(grid)

        if not sample_ids:
            continue
        cat_ids = torch.cat(sample_ids, dim=0)
        cat_labels = torch.cat(sample_labels, dim=0)
        # ``shift_packed_labels`` shifts globally, so sample k's first label would
        # become the target of sample k-1's last position -- a token that
        # ``packed_cu_seqlens`` already cut attention at. Mask it, exactly as
        # ``PackingCollator`` does on the single-model path, instead of relying on
        # the chat template happening to emit a masked leading marker.
        # ``torch.cat`` copied, so the conversation carrier's labels stay intact.
        if ids_chunks:
            cat_labels[0] = IGNORE_INDEX
        ids_chunks.append(cat_ids)
        label_chunks.append(cat_labels)
        mask_chunks.append(torch.ones(cat_ids.numel(), dtype=torch.long))
        position_chunks.append(torch.cat(sample_pos, dim=1))
        visual_chunks.append(torch.cat(sample_visual, dim=0))
        sample_lengths.append(int(cat_ids.numel()))

    if not ids_chunks:
        raise ValueError("pack_qwen3vl_conversations: no non-dummy tokens to pack.")

    packed_ids = torch.cat(ids_chunks, dim=0).unsqueeze(0)
    packed_labels = torch.cat(label_chunks, dim=0).unsqueeze(0)
    packed_attention = torch.cat(mask_chunks, dim=0).unsqueeze(0)
    packed_position = torch.cat(position_chunks, dim=1).unsqueeze(1)
    visual_mask = torch.cat(visual_chunks, dim=0).unsqueeze(0)
    cu = [0]
    for length in sample_lengths:
        cu.append(cu[-1] + length)
    packed_cu = torch.tensor(cu, dtype=torch.int32)
    max_length = max(sample_lengths)

    visual_num_real = len(real_pixels)
    pixel_values, image_grid_thw = _concat_patches(real_pixels, real_grids, dummy_pixels, dummy_grids)

    return {
        PACKED_INPUT_IDS: packed_ids,
        PACKED_LABELS: packed_labels,
        PACKED_ATTENTION_MASK: packed_attention,
        PACKED_POSITION_IDS: packed_position,
        PACKED_CU_SEQLENS: packed_cu,
        PACKED_MAX_LENGTH: max_length,
        VISUAL_POS_MASK: visual_mask,
        PIXEL_VALUES: pixel_values,
        IMAGE_GRID_THW: image_grid_thw,
        VISUAL_NUM_REAL: visual_num_real,
    }


def _concat_patches(
    real_pixels: list[torch.Tensor],
    real_grids: list[list[int]],
    dummy_pixels: list[torch.Tensor],
    dummy_grids: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = real_pixels + dummy_pixels
    grids = real_grids + dummy_grids
    if not pixels:
        raise ValueError(
            "pack_qwen3vl_conversations: training requires at least one real or dummy "
            "visual so FSDP ranks stay symmetric."
        )
    return torch.cat(pixels, dim=0), torch.tensor(grids, dtype=torch.long)


def ensure_packed_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    """``[T, D]`` → ``[1, T, D]``; leave ``[1, T, ...]`` unchanged."""
    if tensor.dim() == 2:
        return tensor.unsqueeze(0)
    return tensor


def masked_scatter_embeds(
    packed_features: torch.Tensor,
    mask: torch.Tensor,
    embeds: torch.Tensor,
) -> torch.Tensor:
    """Write ``embeds`` onto ``packed_features`` where ``mask`` is True.

    ``embeds`` is ``[N, S, D]`` or ``[N * S, D]``. ``mask`` is ``[1, T]`` / ``[T]``.
    """
    packed = ensure_packed_batch_dim(packed_features)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    hidden = embeds
    if hidden.dim() == 3:
        hidden = hidden.reshape(-1, hidden.size(-1))
    mask_3d = mask.unsqueeze(-1).expand_as(packed)
    n_true = int(mask.sum().item()) if mask.device.type == "cpu" else hidden.size(0)
    if hidden.size(0) != n_true and mask.device.type == "cpu":
        raise ValueError(f"masked_scatter_embeds: mask selects {n_true} tokens but embeds has {hidden.size(0)} rows.")
    return packed.masked_scatter(mask_3d, hidden.to(device=packed.device, dtype=packed.dtype))


def fold_dummy_anchor(target: torch.Tensor, dummy: torch.Tensor | None) -> torch.Tensor:
    """Keep dummy activations on the autograd graph without changing values."""
    if dummy is None:
        return target
    anchor = dummy.mean().to(device=target.device, dtype=target.dtype) * 0
    return target + anchor


def shift_packed_labels(packed_labels: torch.Tensor) -> torch.Tensor:
    """Causal LM shift: drop token 0, pad IGNORE_INDEX at the end."""
    labels = packed_labels[..., 1:].contiguous()
    return F.pad(labels, (0, 1), "constant", IGNORE_INDEX)


def visual_token_count(grid_thw: torch.Tensor, spatial_merge_size: int, num_real: int) -> int:
    """Merged-token count for the first ``num_real`` visual items."""
    if num_real <= 0:
        return 0
    merge_area = int(spatial_merge_size) ** 2
    return int(grid_thw[:num_real].prod(dim=1).sum().item()) // merge_area


__all__ = [
    "VISION_SOURCE",
    "PACKED_INPUT_IDS",
    "PACKED_LABELS",
    "PACKED_ATTENTION_MASK",
    "PACKED_POSITION_IDS",
    "PACKED_FEATURES",
    "PACKED_HIDDEN",
    "PACKED_CU_SEQLENS",
    "PACKED_MAX_LENGTH",
    "VISUAL_POS_MASK",
    "PIXEL_VALUES",
    "IMAGE_GRID_THW",
    "VISUAL_NUM_REAL",
    "DEEPSTACK_VISUAL_EMBEDS",
    "pack_qwen3vl_conversations",
    "ensure_packed_batch_dim",
    "masked_scatter_embeds",
    "fold_dummy_anchor",
    "shift_packed_labels",
    "visual_token_count",
]
