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

"""standard cross-entropy chunked impl."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .....registry import SavedState
from . import eager as _eager
from .eager import _grad_and_value, _Meta, _needs_input_grad


def _ce_loss_func(
    hidden_states: Tensor,
    weight: Tensor,
    labels: Tensor,
    num_items_in_batch: int | Tensor,
    ignore_index: int,
    logits_bias: Tensor | None = None,
) -> Tensor:
    """Per-chunk body: linear (+ optional logits_bias) then eager CE."""
    labels_flat = labels.reshape(-1)
    hidden_flat = hidden_states.reshape(-1, hidden_states.size(-1))
    logits = F.linear(hidden_flat, weight).float()
    if logits_bias is not None:
        logits = logits + logits_bias.reshape(-1, logits_bias.size(-1)).float()
    return _eager.cross_entropy_from_logits(
        logits, labels_flat, ignore_index=ignore_index, num_items_in_batch=num_items_in_batch
    )


def forward(
    hidden: Tensor,
    labels: Tensor,
    weight: Tensor,
    logits_bias: Tensor | None = None,
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
    the valid-token count. Unused ``V×H`` grads require both
    ``requires_grad`` and the caller's ``is_grad_enabled()``.

    When ``logits_bias`` is provided (shaped ``hidden.shape[:-1] + (V,)``),
    only supervised rows (``labels != ignore_index``) are projected,
    ``chunk_size`` rows at a time, adding ``logits_bias`` to each chunk's
    fp32 logits.
    """
    if weight.numel() == 0:
        raise RuntimeError("chunk_loss requires a nonempty ``weight`` (fused-linear path)")
    if logits_bias is not None and logits_bias.shape[:-1] != hidden.shape[:-1]:
        raise ValueError(
            f"logits_bias rows {tuple(logits_bias.shape[:-1])} do not match hidden_states rows "
            f"{tuple(hidden.shape[:-1])}."
        )

    compute_grads = torch.is_grad_enabled() if grad_enabled is None else grad_enabled
    hidden_needs_grad = _needs_input_grad(hidden, compute_grads)
    weight_needs_grad = _needs_input_grad(weight, compute_grads)
    bias_needs_grad = logits_bias is not None and _needs_input_grad(logits_bias, compute_grads)
    hidden_token_count = hidden.shape[:-1].numel()
    label_token_count = labels.numel()
    if hidden_token_count != label_token_count:
        raise ValueError(f"token count {hidden_token_count} != labels {label_token_count}")
    if hidden.numel() == 0 or label_token_count == 0:
        loss = hidden.sum() * 0 if hidden.numel() else torch.zeros((), device=hidden.device, dtype=torch.float32)
        saved_tensors: tuple[Tensor, ...] = (
            torch.zeros_like(hidden) if hidden_needs_grad else hidden.new_empty(0),
            torch.zeros_like(weight) if weight_needs_grad else weight.new_empty(0),
            *((torch.zeros_like(logits_bias),) if bias_needs_grad and logits_bias is not None else ()),
        )
        return loss, SavedState(
            saved_tensors,
            _Meta(True, hidden_needs_grad, weight_needs_grad, bias_needs_grad),
        )

    denom: int | Tensor
    if num_items_in_batch is None:
        denom = (labels != ignore_index).sum().clamp(min=1)
    else:
        denom = num_items_in_batch

    split_dim = 1 if hidden.ndim >= 3 else 0
    hidden_chunks = hidden.split(chunk_size, dim=split_dim)
    label_chunks = labels.split(chunk_size, dim=split_dim)
    bias_chunks = (
        logits_bias.split(chunk_size, dim=split_dim) if logits_bias is not None else (None,) * len(hidden_chunks)
    )
    accumulated_loss = torch.zeros((), device=hidden.device, dtype=torch.float32)
    if not (hidden_needs_grad or weight_needs_grad or bias_needs_grad):
        for hidden_chunk, label_chunk, bias_chunk in zip(hidden_chunks, label_chunks, bias_chunks, strict=True):
            accumulated_loss = accumulated_loss + _ce_loss_func(
                hidden_chunk, weight, label_chunk, denom, ignore_index, bias_chunk
            )
        return accumulated_loss, SavedState(
            (hidden.new_empty(0), weight.new_empty(0)),
            _Meta(True, False, False, False),
        )

    argnums = tuple(
        idx for idx, flag in ((0, hidden_needs_grad), (1, weight_needs_grad), (5, bias_needs_grad)) if flag
    )
    grad_hidden = torch.empty_like(hidden) if hidden_needs_grad else None
    grad_weight = torch.zeros_like(weight) if weight_needs_grad else None
    grad_bias = torch.empty_like(logits_bias) if bias_needs_grad and logits_bias is not None else None
    grad_hidden_chunks = (
        grad_hidden.split(chunk_size, dim=split_dim) if grad_hidden is not None else (None,) * len(hidden_chunks)
    )
    grad_bias_chunks = (
        grad_bias.split(chunk_size, dim=split_dim) if grad_bias is not None else (None,) * len(hidden_chunks)
    )
    for hidden_chunk, label_chunk, bias_chunk, grad_hidden_chunk, grad_bias_chunk in zip(
        hidden_chunks, label_chunks, bias_chunks, grad_hidden_chunks, grad_bias_chunks, strict=True
    ):
        grads, chunk_loss = _grad_and_value(
            _ce_loss_func, argnums, hidden_chunk, weight, label_chunk, denom, ignore_index, bias_chunk
        )
        accumulated_loss = accumulated_loss + chunk_loss
        grad_iter = iter(grads)
        if hidden_needs_grad and grad_hidden_chunk is not None:
            grad_hidden_chunk.copy_(next(grad_iter))
        if weight_needs_grad and grad_weight is not None:
            grad_weight.add_(next(grad_iter))
        if bias_needs_grad and grad_bias_chunk is not None:
            grad_bias_chunk.copy_(next(grad_iter))

    saved_tensors = (
        grad_hidden if grad_hidden is not None else hidden.new_empty(0),
        grad_weight if grad_weight is not None else weight.new_empty(0),
        *((grad_bias,) if grad_bias is not None else ()),
    )
    return accumulated_loss, SavedState(
        saved_tensors,
        _Meta(True, hidden_needs_grad, weight_needs_grad, bias_needs_grad),
    )


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor | None, None, Tensor | None, Tensor | None]:
    """Return ``(grad_hidden, None, grad_weight, grad_logits_bias)`` for the requested inputs."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    grad_hidden, grad_weight, *grad_bias = saved.tensors
    return (
        grad_hidden * grad_output if meta.hidden_needs_grad else None,
        None,
        grad_weight * grad_output if meta.weight_needs_grad else None,
        grad_bias[0] * grad_output if meta.bias_needs_grad else None,
    )
