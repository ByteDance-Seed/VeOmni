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

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .....registry import SavedState
from . import eager as _eager


@dataclass(frozen=True)
class _Meta:
    """Which input grads were materialized."""

    hidden_needs_grad: bool
    weight_needs_grad: bool


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
    grad_enabled: bool | None = None,
) -> tuple[Tensor, SavedState]:
    """Chunked fused linear + CE. ``weight`` must be present.

    Split the sequence, run ``torch.func.grad_and_value`` on ``F.linear``
    plus eager CE, and accumulate. Does not shift labels or reduce across
    SP. When ``num_items_in_batch`` is omitted, the shared denominator is
    the valid-token count. Unused ``V×H`` grads follow the eager caller
    ``grad_enabled`` contract because ``Function.forward`` runs under
    ``no_grad``.
    """
    if weight.numel() == 0:
        raise RuntimeError("chunk_loss requires a nonempty ``weight`` (fused-linear path)")

    compute_grads = torch.is_grad_enabled() if grad_enabled is None else grad_enabled
    hidden_needs_grad = _eager._needs_input_grad(hidden, compute_grads)
    weight_needs_grad = _eager._needs_input_grad(weight, compute_grads)
    meta = _Meta(hidden_needs_grad, weight_needs_grad)

    hidden_token_count = hidden.shape[:-1].numel()
    label_token_count = labels.numel()
    if hidden_token_count != label_token_count:
        raise ValueError(f"token count {hidden_token_count} != labels {label_token_count}")
    if hidden.numel() == 0 or label_token_count == 0:
        loss = hidden.sum() * 0 if hidden.numel() else torch.zeros((), device=hidden.device, dtype=torch.float32)
        return loss, SavedState(
            (
                torch.zeros_like(hidden) if hidden_needs_grad else hidden.new_empty(0),
                torch.zeros_like(weight) if weight_needs_grad else weight.new_empty(0),
            ),
            meta,
        )

    denom: int | Tensor
    if num_items_in_batch is None:
        denom = (labels != ignore_index).sum().clamp(min=1)
    else:
        denom = num_items_in_batch
    split_dim = 1 if hidden.ndim >= 3 else 0

    accumulated_loss = torch.zeros((), device=hidden.device, dtype=torch.float32)
    hidden_chunks = hidden.split(chunk_size, dim=split_dim)
    label_chunks = labels.split(chunk_size, dim=split_dim)
    if hidden_needs_grad:
        grad_hidden = torch.empty_like(hidden)
        grad_chunks = grad_hidden.split(chunk_size, dim=split_dim)
    else:
        grad_hidden = hidden.new_empty(0)
        grad_chunks = (None,) * len(hidden_chunks)
    grad_weight = torch.zeros_like(weight) if weight_needs_grad else weight.new_empty(0)

    for hidden_chunk, label_chunk, grad_chunk in zip(hidden_chunks, label_chunks, grad_chunks, strict=True):
        if hidden_needs_grad and weight_needs_grad:
            (chunk_grad_hidden, chunk_grad_weight), chunk_loss = torch.func.grad_and_value(
                _ce_loss_func, argnums=(0, 1)
            )(hidden_chunk, weight, label_chunk, denom, ignore_index)
            grad_chunk.copy_(chunk_grad_hidden)
            grad_weight.add_(chunk_grad_weight)
        elif hidden_needs_grad:
            (chunk_grad_hidden,), chunk_loss = torch.func.grad_and_value(_ce_loss_func, argnums=(0,))(
                hidden_chunk, weight, label_chunk, denom, ignore_index
            )
            grad_chunk.copy_(chunk_grad_hidden)
        elif weight_needs_grad:
            (chunk_grad_weight,), chunk_loss = torch.func.grad_and_value(_ce_loss_func, argnums=(1,))(
                hidden_chunk, weight, label_chunk, denom, ignore_index
            )
            grad_weight.add_(chunk_grad_weight)
        else:
            chunk_loss = _ce_loss_func(hidden_chunk, weight, label_chunk, denom, ignore_index)
        accumulated_loss = accumulated_loss + chunk_loss

    return accumulated_loss, SavedState((grad_hidden, grad_weight), meta)


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor | None, None, Tensor | None]:
    """Return ``(grad_hidden, None, grad_weight)``, omitting unused grads."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    grad_hidden, grad_weight = saved.tensors
    return (
        grad_hidden * grad_output if meta.hidden_needs_grad else None,
        None,
        grad_weight * grad_output if meta.weight_needs_grad else None,
    )
