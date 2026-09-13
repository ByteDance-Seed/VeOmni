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
# See the License for the specific language governing limitations
# under the License.

"""standard cross-entropy chunked impl."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .....registry import SavedState
from . import eager as _eager


def _ce_loss_func(
    hidden_states: Tensor,
    weight: Tensor,
    labels: Tensor,
    num_items_in_batch: int | Tensor,
    ignore_index: int,
) -> Tensor:
    """Per-chunk body: linear then eager CE."""
    labels_flat = labels.reshape(-1)
    hidden_flat = hidden_states.reshape(-1, hidden_states.size(-1))
    logits = F.linear(hidden_flat, weight).float()
    return _eager.cross_entropy_from_logits(
        logits, labels_flat, ignore_index=ignore_index, num_items_in_batch=num_items_in_batch
    )


def forward(
    hidden: Tensor,
    labels: Tensor,
    weight: Tensor,
    *,
    ignore_index: int = -100,
    num_items_in_batch: int | None = None,
    chunk_size: int = 1024,
) -> tuple[Tensor, SavedState]:
    """Chunked fused linear + CE. ``weight`` must be present.

    Split the sequence, run ``torch.func.grad_and_value`` on ``F.linear``
    plus eager CE, and accumulate. Does not shift labels or reduce across
    SP. When ``num_items_in_batch`` is omitted, the shared denominator is
    the valid-token count.
    """
    if weight.numel() == 0:
        raise RuntimeError("chunk_loss requires a nonempty ``weight`` (fused-linear path)")

    labels_flat = labels.reshape(-1)
    if hidden.numel() == 0 or labels_flat.numel() == 0:
        loss = hidden.sum() * 0 if hidden.numel() else torch.zeros((), device=hidden.device, dtype=torch.float32)
        return loss, SavedState((torch.zeros_like(hidden), torch.zeros_like(weight)))

    denom: int | Tensor
    if num_items_in_batch is None:
        denom = (labels_flat != ignore_index).sum().clamp(min=1)
    else:
        denom = num_items_in_batch
    split_dim = 1 if hidden.ndim >= 3 else 0

    accumulated_loss = torch.zeros((), device=hidden.device, dtype=torch.float32)
    grad_hidden = torch.empty_like(hidden)
    grad_weight = torch.zeros_like(weight)

    for hidden_chunk, label_chunk, grad_chunk in zip(
        hidden.split(chunk_size, dim=split_dim),
        labels.split(chunk_size, dim=split_dim),
        grad_hidden.split(chunk_size, dim=split_dim),
        strict=True,
    ):
        (chunk_grad_hidden, chunk_grad_weight), chunk_loss = torch.func.grad_and_value(_ce_loss_func, argnums=(0, 1))(
            hidden_chunk, weight, label_chunk, denom, ignore_index
        )
        accumulated_loss = accumulated_loss + chunk_loss
        grad_chunk.copy_(chunk_grad_hidden)
        grad_weight = grad_weight + chunk_grad_weight

    return accumulated_loss, SavedState((grad_hidden, grad_weight))


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, None, Tensor]:
    """Return ``(grad_hidden, None, grad_weight)``."""
    grad_hidden, grad_weight = saved.tensors
    return grad_hidden * grad_output, None, grad_weight * grad_output
