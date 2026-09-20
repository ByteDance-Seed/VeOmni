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

"""standard cross-entropy eager math (HF ``fixed_cross_entropy``)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .....registry import SavedState


@dataclass(frozen=True)
class _Meta:
    """Whether ``weight`` was a real projection matrix, and which grads to keep."""

    has_weight: bool
    hidden_needs_grad: bool = True
    weight_needs_grad: bool = True


def flatten_tokens(hidden: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Flatten leading dims to ``[tokens, dim]`` / ``[tokens]``."""
    labels_flat = labels.reshape(-1)
    hidden_flat = hidden.reshape(-1, hidden.shape[-1])
    if hidden_flat.shape[0] != labels_flat.shape[0]:
        raise ValueError(f"token count {hidden_flat.shape[0]} != labels {labels_flat.shape[0]}")
    return hidden_flat, labels_flat


def cross_entropy_from_logits(
    logits: Tensor,
    labels: Tensor,
    *,
    ignore_index: int,
    num_items_in_batch: int | Tensor | None,
) -> Tensor:
    """Same reduction as HuggingFace ``fixed_cross_entropy``.

    ``mean`` over non-ignored tokens, or ``sum / num_items_in_batch``. A zero
    explicit count uses one so all-ignored and empty labels return a
    graph-connected zero. Valid-token count stays on device; do not ``.item()``
    it.
    """
    if labels.numel() == 0:
        return logits.float().sum()
    loss = F.cross_entropy(logits.float(), labels, ignore_index=ignore_index, reduction="sum")
    if num_items_in_batch is not None:
        if isinstance(num_items_in_batch, Tensor):
            denominator = num_items_in_batch.to(dtype=loss.dtype, device=loss.device)
            denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        else:
            denominator = 1 if num_items_in_batch == 0 else num_items_in_batch
        return loss / denominator
    n_valid = (labels != ignore_index).sum().to(dtype=loss.dtype)
    return loss / n_valid.clamp(min=1)


def _loss_hidden_weight(
    hidden: Tensor,
    weight: Tensor,
    labels: Tensor,
    ignore_index: int,
    num_items_in_batch: int | Tensor | None,
) -> Tensor:
    """Project ``hidden`` with ``F.linear`` then ``fixed_cross_entropy``."""
    hidden_flat, labels_flat = flatten_tokens(hidden, labels)
    logits = F.linear(hidden_flat, weight).float()
    return cross_entropy_from_logits(
        logits, labels_flat, ignore_index=ignore_index, num_items_in_batch=num_items_in_batch
    )


def _loss_logits(
    hidden: Tensor,
    labels: Tensor,
    ignore_index: int,
    num_items_in_batch: int | Tensor | None,
) -> Tensor:
    """Token CE when ``hidden`` is already logits."""
    hidden_flat, labels_flat = flatten_tokens(hidden, labels)
    return cross_entropy_from_logits(
        hidden_flat, labels_flat, ignore_index=ignore_index, num_items_in_batch=num_items_in_batch
    )


def _needs_input_grad(tensor: Tensor, grad_enabled: bool) -> bool:
    """``grad_and_value`` ignores outer ``no_grad``; both flags must agree."""
    return bool(tensor.requires_grad and grad_enabled)


def forward(
    hidden: Tensor,
    labels: Tensor,
    weight: Tensor,
    *,
    ignore_index: int = -100,
    num_items_in_batch: int | Tensor | None = None,
    grad_enabled: bool | None = None,
) -> tuple[Tensor, SavedState]:
    """Token-level CE. Empty ``weight`` means ``hidden`` is already logits.

    This logits path is eager-only. ``liger_kernel`` and ``chunk_loss`` reject
    empty ``weight``. Label shift and SP reduction stay in the caller.
    ``torch.func.grad_and_value`` builds its own graph and ignores ``no_grad``,
    so unused ``V×H`` grads require both ``requires_grad`` and the caller's
    ``is_grad_enabled()``. The generated wrapper passes that flag because
    ``Function.forward`` itself always runs under ``no_grad``.
    """
    has_weight = weight.numel() > 0
    compute_grads = torch.is_grad_enabled() if grad_enabled is None else grad_enabled
    hidden_needs_grad = _needs_input_grad(hidden, compute_grads)
    weight_needs_grad = has_weight and _needs_input_grad(weight, compute_grads)
    if has_weight:
        if hidden_needs_grad and weight_needs_grad:
            (grad_hidden, grad_weight), loss = torch.func.grad_and_value(_loss_hidden_weight, argnums=(0, 1))(
                hidden, weight, labels, ignore_index, num_items_in_batch
            )
        elif hidden_needs_grad:
            (grad_hidden,), loss = torch.func.grad_and_value(_loss_hidden_weight, argnums=(0,))(
                hidden, weight, labels, ignore_index, num_items_in_batch
            )
            grad_weight = None
        elif weight_needs_grad:
            (grad_weight,), loss = torch.func.grad_and_value(_loss_hidden_weight, argnums=(1,))(
                hidden, weight, labels, ignore_index, num_items_in_batch
            )
            grad_hidden = None
        else:
            loss = _loss_hidden_weight(hidden, weight, labels, ignore_index, num_items_in_batch)
            grad_hidden = None
            grad_weight = None
        return loss, SavedState(
            (
                grad_hidden if grad_hidden is not None else hidden.new_empty(0),
                grad_weight if grad_weight is not None else weight.new_empty(0),
            ),
            _Meta(True, hidden_needs_grad, weight_needs_grad),
        )

    if hidden_needs_grad:
        (grad_hidden,), loss = torch.func.grad_and_value(_loss_logits, argnums=(0,))(
            hidden, labels, ignore_index, num_items_in_batch
        )
    else:
        loss = _loss_logits(hidden, labels, ignore_index, num_items_in_batch)
        grad_hidden = None
    return loss, SavedState(
        (grad_hidden if grad_hidden is not None else hidden.new_empty(0),),
        _Meta(False, hidden_needs_grad, False),
    )


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor | None, None, Tensor | None]:
    """Return ``(grad_hidden, None, grad_weight_or_None)``. Labels are constants."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    if meta.has_weight:
        grad_hidden, grad_weight = saved.tensors
        return (
            grad_hidden * grad_output if meta.hidden_needs_grad else None,
            None,
            grad_weight * grad_output if meta.weight_needs_grad else None,
        )
    (grad_hidden,) = saved.tensors
    return (grad_hidden * grad_output if meta.hidden_needs_grad else None), None, None
