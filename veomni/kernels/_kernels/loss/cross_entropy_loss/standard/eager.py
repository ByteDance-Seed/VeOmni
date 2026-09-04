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
    """Whether ``weight`` was a real projection matrix."""

    has_weight: bool


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

    ``mean`` over non-ignored tokens, or ``sum / num_items_in_batch``. All-ignored
    or empty labels return a graph-connected zero. Valid-token count stays on
    device; do not ``.item()`` it.
    """
    if labels.numel() == 0:
        return logits.sum() * 0
    loss = F.cross_entropy(logits.float(), labels, ignore_index=ignore_index, reduction="sum")
    connected = loss + logits.sum() * 0
    if num_items_in_batch is not None:
        return connected / num_items_in_batch
    n_valid = (labels != ignore_index).sum().to(dtype=connected.dtype)
    return connected / n_valid.clamp(min=1)


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


def forward(
    hidden: Tensor,
    labels: Tensor,
    weight: Tensor,
    *,
    ignore_index: int = -100,
    num_items_in_batch: int | None = None,
) -> tuple[Tensor, SavedState]:
    """Token-level CE. Empty ``weight`` means ``hidden`` is already logits.

    Label shift and SP reduction stay in the caller. Grads are taken through
    ``F.linear`` + ``F.cross_entropy``, then scaled in backward.
    """
    has_weight = weight.numel() > 0
    if has_weight:
        (grad_hidden, grad_weight), loss = torch.func.grad_and_value(_loss_hidden_weight, argnums=(0, 1))(
            hidden, weight, labels, ignore_index, num_items_in_batch
        )
        return loss, SavedState((grad_hidden, grad_weight), _Meta(True))

    (grad_hidden,), loss = torch.func.grad_and_value(_loss_logits, argnums=(0,))(
        hidden, labels, ignore_index, num_items_in_batch
    )
    return loss, SavedState((grad_hidden,), _Meta(False))


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, None, Tensor | None]:
    """Return ``(grad_hidden, None, grad_weight_or_None)``. Labels are constants."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    if meta.has_weight:
        grad_hidden, grad_weight = saved.tensors
        return grad_hidden * grad_output, None, grad_weight * grad_output
    (grad_hidden,) = saved.tensors
    return grad_hidden * grad_output, None, None
