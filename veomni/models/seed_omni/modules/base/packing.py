"""Packed-sequence vocabulary and tensor helpers shared by every packed graph.

A family's ``packing.py`` owns the part that is actually family-specific: how a
conversation becomes one flat sequence (which placeholder a modality expands to,
how many rows it takes, what its position ids look like). What every family then
does with that sequence is the same, so the keys naming the packed batch and the
four tensor operations over it live here rather than being copied per family:

1. ``masked_scatter_embeds`` — a module writes the rows it owns onto the packed
   sequence, selected by its own boolean mask.
2. ``fold_dummy_anchor`` — an FSDP-anchor input contributes ``mean() * 0`` so its
   module still produces a gradient without occupying a position.
3. ``shift_packed_labels`` — the causal shift, applied once over the whole packed
   batch (per-sample boundaries are masked by the family's packer).
4. ``ensure_packed_batch_dim`` — the packed sequence is always ``[1, T, D]``.

Modality keys stay with the family: janus's generation-image masks and
qwen3omni's audio keys mean nothing to each other, and a shared spelling for
them would only invite a shared assumption that isn't there.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from veomni.utils.constants import IGNORE_INDEX


# Keys a family's packed preprocessor writes onto the training batch, and that
# the packed graph nodes read back. They are strings rather than a dataclass
# because the graph YAML passes node outputs by name.
PACKED_INPUT_IDS = "packed_input_ids"
PACKED_LABELS = "packed_labels"
PACKED_ATTENTION_MASK = "packed_attention_mask"
PACKED_POSITION_IDS = "packed_position_ids"
PACKED_FEATURES = "packed_features"
PACKED_HIDDEN = "packed_hidden"
PACKED_CU_SEQLENS = "packed_cu_seqlens"
PACKED_MAX_LENGTH = "packed_max_length"


def as_1d_long(value: Any) -> torch.Tensor:
    """Token ids as a flat int64 tensor, accepting a scalar or a python list."""
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.long)
    if tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)
    return tensor.reshape(-1).to(dtype=torch.long)


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

    Surplus rows in ``embeds`` are consumed silently by ``masked_scatter`` — it
    takes the first N source elements and drops the rest. Counting the mask's
    True values here would catch that, but by the time a training step reaches
    this the mask is on the device, and counting a device mask costs a stream
    sync every step. So the row count is checked upstream instead, host-side,
    where each tower compares the packer's stamped count against its own config;
    the check below only fires on the CPU path (tests, offline packing).
    """
    packed = ensure_packed_batch_dim(packed_features)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    hidden = embeds
    if hidden.dim() == 3:
        hidden = hidden.reshape(-1, hidden.size(-1))
    mask_3d = mask.unsqueeze(-1).expand_as(packed)
    if mask.device.type == "cpu":
        n_true = int(mask.sum().item())
        if hidden.size(0) != n_true:
            raise ValueError(
                f"masked_scatter_embeds: mask selects {n_true} tokens but embeds has {hidden.size(0)} rows."
            )
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


__all__ = [
    "PACKED_ATTENTION_MASK",
    "PACKED_CU_SEQLENS",
    "PACKED_FEATURES",
    "PACKED_HIDDEN",
    "PACKED_INPUT_IDS",
    "PACKED_LABELS",
    "PACKED_MAX_LENGTH",
    "PACKED_POSITION_IDS",
    "as_1d_long",
    "ensure_packed_batch_dim",
    "fold_dummy_anchor",
    "masked_scatter_embeds",
    "shift_packed_labels",
]
