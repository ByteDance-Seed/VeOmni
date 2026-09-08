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

"""standard cross-entropy Liger adapter (fused linear + CE)."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .....registry import SavedState
from . import eager as _eager


@dataclass(frozen=True)
class _Meta:
    """Whether the empty-tensor path ran."""

    empty: bool


def forward(
    hidden: Tensor,
    labels: Tensor,
    weight: Tensor,
    *,
    ignore_index: int = -100,
    num_items_in_batch: int | None = None,
) -> tuple[Tensor, SavedState]:
    """Liger fused linear + CE. ``weight`` must be present.

    Calls ``fused_linear_cross_entropy_forward`` / ``_backward`` the same way
    ``LigerFusedLinearCrossEntropyFunction`` does. Reduction matches eager /
    HF ``fixed_cross_entropy``: mean over non-ignored tokens, or
    ``sum / num_items_in_batch``. Empty ``hidden`` falls back to the eager pair.
    """
    if weight.numel() == 0:
        raise RuntimeError("liger_kernel requires a nonempty ``weight`` (fused-linear path)")
    if hidden.numel() == 0:
        output, saved = _eager.forward(
            hidden, labels, weight, ignore_index=ignore_index, num_items_in_batch=num_items_in_batch
        )
        return output, SavedState(saved.tensors, _Meta(True))

    from liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_forward

    hidden_flat, labels_flat = _eager.flatten_tokens(hidden, labels)
    # ``Function.forward`` runs with autograd disabled. A fresh ``contiguous()``
    # copy then has ``requires_grad=False``, and Liger skips grad buffers.
    hidden_needs_grad = hidden.requires_grad
    weight_needs_grad = weight.requires_grad
    hidden_flat = hidden_flat.contiguous()
    weight_c = weight.contiguous()
    if hidden_needs_grad and not hidden_flat.requires_grad:
        hidden_flat.requires_grad_(True)
    if weight_needs_grad and not weight_c.requires_grad:
        weight_c.requires_grad_(True)
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss, _z_loss, _token_accuracy, grad_hidden, grad_weight, _grad_bias = fused_linear_cross_entropy_forward(
        _input=hidden_flat,
        weight=weight_c,
        target=labels_flat.contiguous(),
        bias=None,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    if grad_hidden is None or grad_weight is None:
        raise RuntimeError("liger fused CE did not allocate input/weight grads")
    if num_items_in_batch is not None:
        scale = 1.0 / num_items_in_batch
        loss = loss * scale
        grad_hidden = grad_hidden * scale
        grad_weight = grad_weight * scale
    return loss, SavedState((hidden, grad_hidden.detach(), grad_weight.detach()), _Meta(False))


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, None, Tensor]:
    """Return ``(grad_hidden, None, grad_weight)``. Empty inputs reuse eager."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    if meta.empty:
        return _eager.backward(grad_output, SavedState(saved.tensors, _eager._Meta(True)))

    from liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_backward

    hidden, grad_hidden, grad_weight = saved.tensors
    grad_hidden, grad_weight, _grad_bias = fused_linear_cross_entropy_backward(
        grad_output, grad_hidden, grad_weight, None
    )
    return grad_hidden.view_as(hidden), None, grad_weight
