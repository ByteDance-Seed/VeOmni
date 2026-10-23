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

"""standard cross-entropy eager math (HF ``fixed_cross_entropy``)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

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
    bias_needs_grad: bool = False


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
    logits_bias: Tensor | None = None,
) -> Tensor:
    """Project ``hidden`` with ``F.linear`` (+ optional ``logits_bias``) then ``fixed_cross_entropy``."""
    hidden_flat, labels_flat = flatten_tokens(hidden, labels)
    logits = F.linear(hidden_flat, weight).float()
    if logits_bias is not None:
        logits = logits + logits_bias.reshape(-1, logits_bias.size(-1)).float()
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


@contextmanager
def _suspend_saved_tensor_hooks() -> Iterator[None]:
    """Pop saved-tensor hooks for one ``torch.func`` call, then push them back.

    ``grad_and_value`` raises while a pack/unpack pair is installed. Sync
    activation offload installs that pair for the whole forward. The outer
    ``save_for_backward`` still runs after this block, so offload packs the
    precomputed gradients rather than the ``V×H`` logits.
    """
    stack: list[tuple[Callable[..., Any], Callable[..., Tensor]]] = []
    top = torch._C._autograd._top_saved_tensors_default_hooks(False)
    while top is not None:
        stack.append(top)
        torch._C._autograd._pop_saved_tensors_default_hooks()
        top = torch._C._autograd._top_saved_tensors_default_hooks(False)
    try:
        yield
    finally:
        for pack_hook, unpack_hook in reversed(stack):
            torch._C._autograd._push_saved_tensors_default_hooks(pack_hook, unpack_hook)


def _grad_and_value(
    function: Callable[..., Tensor], argnums: tuple[int, ...], *args: Any
) -> tuple[tuple[Tensor, ...], Tensor]:
    """``torch.func.grad_and_value`` that tolerates an outer saved-tensor hook."""
    with _suspend_saved_tensor_hooks():
        grads, loss = torch.func.grad_and_value(function, argnums=argnums)(*args)
    return grads, loss


def forward(
    hidden: Tensor,
    labels: Tensor,
    weight: Tensor,
    logits_bias: Tensor | None = None,
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
    ``Function.forward`` itself always runs under ``no_grad``. The func call
    suspends saved-tensor hooks so sync activation offload can stay installed.
    """
    has_weight = weight.numel() > 0
    if logits_bias is not None:
        if not has_weight:
            raise RuntimeError("logits_bias requires a nonempty ``weight`` (fused-linear path)")
        if logits_bias.shape[:-1] != hidden.shape[:-1]:
            raise ValueError(
                f"logits_bias rows {tuple(logits_bias.shape[:-1])} do not match hidden_states rows "
                f"{tuple(hidden.shape[:-1])}."
            )
    compute_grads = torch.is_grad_enabled() if grad_enabled is None else grad_enabled
    hidden_needs_grad = _needs_input_grad(hidden, compute_grads)
    weight_needs_grad = has_weight and _needs_input_grad(weight, compute_grads)
    bias_needs_grad = has_weight and logits_bias is not None and _needs_input_grad(logits_bias, compute_grads)
    if has_weight:
        argnums = tuple(
            idx for idx, flag in ((0, hidden_needs_grad), (1, weight_needs_grad), (5, bias_needs_grad)) if flag
        )
        grad_hidden: Tensor | None = None
        grad_weight: Tensor | None = None
        grad_bias: Tensor | None = None
        if argnums:
            grads, loss = _grad_and_value(
                _loss_hidden_weight,
                argnums,
                hidden,
                weight,
                labels,
                ignore_index,
                num_items_in_batch,
                logits_bias,
            )
            grad_iter = iter(grads)
            if hidden_needs_grad:
                grad_hidden = next(grad_iter)
            if weight_needs_grad:
                grad_weight = next(grad_iter)
            if bias_needs_grad:
                grad_bias = next(grad_iter)
        else:
            loss = _loss_hidden_weight(hidden, weight, labels, ignore_index, num_items_in_batch, logits_bias)
        saved_tensors: tuple[Tensor, ...] = (
            grad_hidden if grad_hidden is not None else hidden.new_empty(0),
            grad_weight if grad_weight is not None else weight.new_empty(0),
            *((grad_bias,) if grad_bias is not None else ()),
        )
        return loss, SavedState(
            saved_tensors,
            _Meta(True, hidden_needs_grad, weight_needs_grad, bias_needs_grad),
        )

    if hidden_needs_grad:
        (grad_hidden,), loss = _grad_and_value(_loss_logits, (0,), hidden, labels, ignore_index, num_items_in_batch)
    else:
        loss = _loss_logits(hidden, labels, ignore_index, num_items_in_batch)
        grad_hidden = None
    return loss, SavedState(
        (grad_hidden if grad_hidden is not None else hidden.new_empty(0),),
        _Meta(False, hidden_needs_grad, False, False),
    )


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor | None, None, Tensor | None, Tensor | None]:
    """Return ``(grad_hidden, None, grad_weight_or_None, grad_logits_bias)``. Labels are constants."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    if meta.has_weight:
        grad_hidden, grad_weight, *grad_bias = saved.tensors
        return (
            grad_hidden * grad_output if meta.hidden_needs_grad else None,
            None,
            grad_weight * grad_output if meta.weight_needs_grad else None,
            grad_bias[0] * grad_output if meta.bias_needs_grad else None,
        )
    (grad_hidden,) = saved.tensors
    return (grad_hidden * grad_output if meta.hidden_needs_grad else None), None, None, None
