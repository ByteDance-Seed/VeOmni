# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Packed-sequence vocabulary and tensor helpers shared by every packed graph."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from veomni.utils.constants import IGNORE_INDEX


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
    """Write ``embeds`` onto ``packed_features`` where ``mask`` is True."""
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
