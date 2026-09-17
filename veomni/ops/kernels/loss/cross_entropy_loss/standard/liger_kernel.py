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

import torch
from torch import Tensor

from .....registry import SavedState
from . import eager as _eager


@dataclass(frozen=True)
class _Meta:
    """Execution path and requested input gradients."""

    empty: bool
    hidden_needs_grad: bool
    weight_needs_grad: bool


def forward(
    hidden: Tensor,
    labels: Tensor,
    weight: Tensor,
    *,
    ignore_index: int = -100,
    num_items_in_batch: int | Tensor | None = None,
    grad_enabled: bool | None = None,
) -> tuple[Tensor, SavedState]:
    """Liger fused linear + CE. ``weight`` must be present.

    Calls ``fused_linear_cross_entropy_forward`` / ``_backward`` the same way
    ``LigerFusedLinearCrossEntropyFunction`` does. Reduction matches eager /
    HF ``fixed_cross_entropy``: mean over non-ignored tokens, or
    ``sum / num_items_in_batch``. A zero explicit count uses one, matching the
    eager row's connected-zero behavior. Empty ``hidden`` falls back to eager.
    Unused grads require both ``requires_grad`` and the caller's
    ``is_grad_enabled()``.
    """
    if weight.numel() == 0:
        raise RuntimeError("liger_kernel requires a nonempty ``weight`` (fused-linear path)")
    compute_grads = torch.is_grad_enabled() if grad_enabled is None else grad_enabled
    hidden_needs_grad = _eager._needs_input_grad(hidden, compute_grads)
    weight_needs_grad = _eager._needs_input_grad(weight, compute_grads)
    if hidden.numel() == 0:
        output, saved = _eager.forward(
            hidden,
            labels,
            weight,
            ignore_index=ignore_index,
            num_items_in_batch=num_items_in_batch,
            grad_enabled=compute_grads,
        )
        return output, SavedState(saved.tensors, _Meta(True, hidden_needs_grad, weight_needs_grad))

    from liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_forward

    hidden_flat, labels_flat = _eager.flatten_tokens(hidden, labels)
    # Liger reads tensor ``requires_grad``, not the caller ``no_grad``. A
    # contiguous trainable leaf is a no-op copy and would still allocate the
    # ``V×H`` weight buffer. Detach first, then re-enable only requested grads.
    hidden_flat = hidden_flat.detach().contiguous()
    weight_c = weight.detach().contiguous()
    if hidden_needs_grad or weight_needs_grad:
        hidden_flat.requires_grad_(True)
    if weight_needs_grad:
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
    if (hidden_needs_grad or weight_needs_grad) and grad_hidden is None:
        raise RuntimeError("liger fused CE did not allocate the input grad buffer")
    if weight_needs_grad and grad_weight is None:
        raise RuntimeError("liger fused CE did not allocate the requested weight grad buffer")
    if num_items_in_batch is not None:
        if isinstance(num_items_in_batch, Tensor):
            denominator = num_items_in_batch.to(dtype=loss.dtype, device=loss.device)
            denominator = denominator.masked_fill(denominator == 0, 1)
            scale = denominator.reciprocal()
        else:
            scale = 1.0 / (1 if num_items_in_batch == 0 else num_items_in_batch)
        loss = loss * scale
        if grad_hidden is not None:
            grad_hidden = grad_hidden * scale
        if grad_weight is not None:
            grad_weight = grad_weight * scale

    grad_hidden_saved = grad_hidden.detach() if grad_hidden is not None else hidden.new_empty(0)
    grad_weight_saved = grad_weight.detach() if grad_weight is not None else weight.new_empty(0)
    return loss, SavedState(
        (hidden, grad_hidden_saved, grad_weight_saved),
        _Meta(False, hidden_needs_grad, weight_needs_grad),
    )


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor | None, None, Tensor | None]:
    """Return ``(grad_hidden, None, grad_weight)``. Empty inputs reuse eager."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    if meta.empty:
        return _eager.backward(
            grad_output,
            SavedState(
                saved.tensors,
                _eager._Meta(True, meta.hidden_needs_grad, meta.weight_needs_grad),
            ),
        )

    from liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_backward

    hidden, grad_hidden, grad_weight_saved = saved.tensors
    grad_weight = grad_weight_saved if meta.weight_needs_grad else None
    grad_hidden, grad_weight, _grad_bias = fused_linear_cross_entropy_backward(
        grad_output, grad_hidden, grad_weight, None
    )
    return grad_hidden.view_as(hidden) if meta.hidden_needs_grad else None, None, grad_weight
