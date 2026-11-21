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

from veomni.utils.constants import IGNORE_INDEX

from ...utils.conversation import ConversationItem
from ..base.packing import (
    PACKED_CU_SEQLENS,
    PACKED_FEATURES,
    PACKED_HIDDEN,
    PACKED_INPUT_IDS,
    PACKED_LABELS,
    PACKED_MAX_LENGTH,
    PACKED_POSITION_IDS,
    as_1d_long,
    ensure_packed_batch_dim,
    fold_dummy_anchor,
    masked_scatter_embeds,
    shift_packed_labels,
)
from .llm.modeling import qwen3vl_vision_position_advance, qwen3vl_vision_position_ids
from .vision.processing import _OMNI_GRID


VISUAL_POS_MASK = "visual_pos_mask"
PIXEL_VALUES = "pixel_values"
IMAGE_GRID_THW = "image_grid_thw"
# One bool per visual row: True for a real image/video, False for an FSDP-anchor
# placeholder. Position in the batch is *not* load-bearing — a consumer selects
# with `image_grid_thw[visual_is_real]` rather than "the first N rows", so the
# collator can join samples with a plain `torch.cat` in any order.
VISUAL_IS_REAL = "visual_is_real"
DEEPSTACK_VISUAL_EMBEDS = "deepstack_visual_embeds"


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
    position_chunks: list[torch.Tensor] = []
    visual_chunks: list[torch.Tensor] = []
    sample_lengths: list[int] = []
    visual_pixels: list[torch.Tensor] = []
    visual_grids: list[list[int]] = []
    visual_is_real: list[bool] = []

    merge = int(spatial_merge_size)
    pad_id = int(pad_token_id)

    for sample in conversation_list:
        sample_ids: list[torch.Tensor] = []
        sample_labels: list[torch.Tensor] = []
        sample_visual: list[torch.Tensor] = []
        sample_pos: list[torch.Tensor] = []
        current_pos = 0
        for part in sample:
            if part.is_dummy:
                if isinstance(part.value, torch.Tensor):
                    visual_pixels.append(_patches_2d(part.value))
                    visual_grids.append(_grid_from_item(part))
                    visual_is_real.append(False)
                continue
            if part.type == "text":
                token_ids = as_1d_long(part.value)
                labels = as_1d_long(part.meta.get("labels", torch.full_like(token_ids, IGNORE_INDEX)))
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
            current_pos += qwen3vl_vision_position_advance(grid_thw, merge)
            visual_pixels.append(_patches_2d(part.value))
            visual_grids.append(grid)
            visual_is_real.append(True)

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
        position_chunks.append(torch.cat(sample_pos, dim=1))
        visual_chunks.append(torch.cat(sample_visual, dim=0))
        sample_lengths.append(int(cat_ids.numel()))

    if not ids_chunks:
        raise ValueError("pack_qwen3vl_conversations: no non-dummy tokens to pack.")

    packed_ids = torch.cat(ids_chunks, dim=0).unsqueeze(0)
    packed_labels = torch.cat(label_chunks, dim=0).unsqueeze(0)
    packed_position = torch.cat(position_chunks, dim=1).unsqueeze(1)
    visual_mask = torch.cat(visual_chunks, dim=0).unsqueeze(0)
    cu = [0]
    for length in sample_lengths:
        cu.append(cu[-1] + length)
    packed_cu = torch.tensor(cu, dtype=torch.int32)
    max_length = max(sample_lengths)

    if not visual_pixels:
        raise ValueError(
            "pack_qwen3vl_conversations: training requires at least one real or dummy "
            "visual so FSDP ranks stay symmetric."
        )
    pixel_values = torch.cat(visual_pixels, dim=0)
    image_grid_thw = torch.tensor(visual_grids, dtype=torch.long)
    is_real = torch.tensor(visual_is_real, dtype=torch.bool)

    return {
        PACKED_INPUT_IDS: packed_ids,
        PACKED_LABELS: packed_labels,
        PACKED_POSITION_IDS: packed_position,
        PACKED_CU_SEQLENS: packed_cu,
        PACKED_MAX_LENGTH: max_length,
        VISUAL_POS_MASK: visual_mask,
        PIXEL_VALUES: pixel_values,
        IMAGE_GRID_THW: image_grid_thw,
        VISUAL_IS_REAL: is_real,
    }


def visual_row_mask(grid_thw: torch.Tensor, spatial_merge_size: int, is_real: torch.Tensor) -> torch.Tensor:
    """Per-*token* bool over the ViT's output rows: True where the row is real.

    ``is_real`` is one flag per visual item; the ViT emits ``prod(grid) / merge²``
    rows per item, so expand each flag to its row count. Returned on the same
    device as ``grid_thw``.
    """
    rows = grid_thw.prod(dim=1) // (int(spatial_merge_size) ** 2)
    return torch.repeat_interleave(is_real.to(grid_thw.device), rows)


def visual_token_count(grid_thw: torch.Tensor, spatial_merge_size: int, is_real: torch.Tensor) -> int:
    """Merged-token count over the real visual rows of a packed batch."""
    if not bool(is_real.any()):
        return 0
    merge_area = int(spatial_merge_size) ** 2
    return int(grid_thw[is_real].prod(dim=1).sum().item()) // merge_area


__all__ = [
    "PACKED_INPUT_IDS",
    "PACKED_LABELS",
    "PACKED_POSITION_IDS",
    "PACKED_FEATURES",
    "PACKED_HIDDEN",
    "PACKED_CU_SEQLENS",
    "PACKED_MAX_LENGTH",
    "VISUAL_POS_MASK",
    "PIXEL_VALUES",
    "IMAGE_GRID_THW",
    "VISUAL_IS_REAL",
    "DEEPSTACK_VISUAL_EMBEDS",
    "pack_qwen3vl_conversations",
    "ensure_packed_batch_dim",
    "masked_scatter_embeds",
    "fold_dummy_anchor",
    "shift_packed_labels",
    "visual_token_count",
    "visual_row_mask",
]
