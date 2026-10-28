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

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .....registry import SavedState
from . import eager as _eager
from .eager import _grad_and_value, _Meta, _needs_input_grad


# Channel-loss capture is opt-in and scoped by the trainer. Keeping this state
# in the op preserves standalone use without importing the trainer package.
_CAPTURE: ContextVar[dict[str, Any] | None] = ContextVar("chunk_loss_capture", default=None)


def _ce_loss_func(
    hidden_states: Tensor,
    weight: Tensor,
    labels: Tensor,
    num_items_in_batch: int | Tensor,
    ignore_index: int,
    logits_bias: Tensor | None = None,
    logits_bias_weight: Tensor | None = None,
    observe: Callable[[Tensor, Tensor], None] | None = None,
) -> Tensor:
    """Per-chunk projection and CE, optionally recording detached token losses."""
    labels_flat = labels.reshape(-1)
    hidden_flat = hidden_states.reshape(-1, hidden_states.size(-1))
    logits = F.linear(hidden_flat, weight).float()
    if logits_bias is not None:
        bias_flat = logits_bias.reshape(-1, logits_bias.size(-1))
        if logits_bias_weight is not None:
            logits = logits + F.linear(bias_flat, logits_bias_weight).float()
        else:
            logits = logits + bias_flat.float()
    loss = _eager.cross_entropy_from_logits(
        logits, labels_flat, ignore_index=ignore_index, num_items_in_batch=num_items_in_batch
    )
    if observe is not None:
        observe(logits, labels)
    return loss


def forward(
    hidden: Tensor,
    labels: Tensor,
    weight: Tensor,
    logits_bias: Tensor | None = None,
    logits_bias_weight: Tensor | None = None,
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

    When ``logits_bias`` is provided, ``hidden``, ``labels``, and
    ``logits_bias`` are split into ``chunk_size`` slices along the sequence
    dimension. If ``logits_bias_weight`` (shape ``(V, rank)``) is also
    provided, ``F.linear(bias_chunk, logits_bias_weight)`` is projected inside
    each chunk so no full ``[B, S, V]`` bias tensor or gradient is
    materialized; otherwise ``logits_bias`` has shape ``hidden.shape[:-1] + (V,)``
    and is added directly to each chunk's fp32 logits.
    """
    if weight.numel() == 0:
        raise RuntimeError("chunk_loss requires a nonempty ``weight`` (fused-linear path)")
    if logits_bias_weight is not None and logits_bias is None:
        raise ValueError("logits_bias_weight requires logits_bias to be provided.")
    if logits_bias is not None:
        if logits_bias.shape[:-1] != hidden.shape[:-1]:
            raise ValueError(
                f"logits_bias rows {tuple(logits_bias.shape[:-1])} do not match hidden_states rows "
                f"{tuple(hidden.shape[:-1])}."
            )
        if logits_bias_weight is not None:
            if logits_bias_weight.ndim != 2 or logits_bias_weight.shape != (weight.shape[0], logits_bias.shape[-1]):
                raise ValueError(
                    f"logits_bias_weight shape {tuple(logits_bias_weight.shape)} must match "
                    f"(vocab_size={weight.shape[0]}, rank={logits_bias.shape[-1]})."
                )
        elif logits_bias.shape[-1] != weight.shape[0]:
            raise ValueError(
                f"logits_bias last dim {logits_bias.shape[-1]} does not match vocab_size {weight.shape[0]}."
            )

    compute_grads = torch.is_grad_enabled() if grad_enabled is None else grad_enabled
    hidden_needs_grad = _needs_input_grad(hidden, compute_grads)
    weight_needs_grad = _needs_input_grad(weight, compute_grads)
    bias_needs_grad = logits_bias is not None and _needs_input_grad(logits_bias, compute_grads)
    bias_weight_needs_grad = logits_bias_weight is not None and _needs_input_grad(logits_bias_weight, compute_grads)
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
            *(
                (torch.zeros_like(logits_bias_weight),)
                if bias_weight_needs_grad and logits_bias_weight is not None
                else ()
            ),
        )
        return loss, SavedState(
            saved_tensors,
            _Meta(True, hidden_needs_grad, weight_needs_grad, bias_needs_grad, bias_weight_needs_grad),
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
    # ContextVar.get cannot be traced by Dynamo. A compiled op therefore uses
    # the unchanged training path; the trainer can recompute detached metrics.
    capture = None if torch.compiler.is_compiling() else _CAPTURE.get()
    if capture is not None:
        capture["op_calls"] = capture.get("op_calls", 0) + 1
    token_chunks = [] if capture is not None else None

    def observe(logits: Tensor, chunk_labels: Tensor) -> None:
        if capture is None or capture.get("error") is not None:
            return
        try:
            with torch.no_grad():
                token_chunks.append(
                    F.cross_entropy(
                        logits.detach(), chunk_labels.reshape(-1), ignore_index=ignore_index, reduction="none"
                    ).reshape_as(chunk_labels)
                )
        except Exception as error:
            capture["error"] = f"{type(error).__name__}: {error}"
            token_chunks.clear()

    def finish_capture() -> None:
        if capture is not None and capture.get("error") is None:
            if "per_token" in capture:
                capture["error"] = "multiple chunk-loss observations in one causal-loss call"
                capture.pop("per_token", None)
                return
            try:
                capture["per_token"] = torch.cat(token_chunks, dim=split_dim).detach()
            except Exception as error:
                capture["error"] = f"{type(error).__name__}: {error}"

    observe_chunk = observe if capture is not None else None
    accumulated_loss = torch.zeros((), device=hidden.device, dtype=torch.float32)
    if not (hidden_needs_grad or weight_needs_grad or bias_needs_grad or bias_weight_needs_grad):
        for hidden_chunk, label_chunk, bias_chunk in zip(hidden_chunks, label_chunks, bias_chunks, strict=True):
            chunk_loss = _ce_loss_func(
                hidden_chunk, weight, label_chunk, denom, ignore_index, bias_chunk, logits_bias_weight, observe_chunk
            )
            accumulated_loss = accumulated_loss + chunk_loss
        finish_capture()
        return accumulated_loss, SavedState(
            (hidden.new_empty(0), weight.new_empty(0)),
            _Meta(True, False, False, False, False),
        )

    argnums = tuple(
        idx
        for idx, flag in (
            (0, hidden_needs_grad),
            (1, weight_needs_grad),
            (5, bias_needs_grad),
            (6, bias_weight_needs_grad),
        )
        if flag
    )
    grad_hidden = torch.empty_like(hidden) if hidden_needs_grad else None
    grad_weight = torch.zeros_like(weight) if weight_needs_grad else None
    grad_bias = torch.empty_like(logits_bias) if bias_needs_grad and logits_bias is not None else None
    grad_bias_weight = (
        torch.zeros_like(logits_bias_weight) if bias_weight_needs_grad and logits_bias_weight is not None else None
    )
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
            _ce_loss_func,
            argnums,
            hidden_chunk,
            weight,
            label_chunk,
            denom,
            ignore_index,
            bias_chunk,
            logits_bias_weight,
            observe_chunk,
        )
        accumulated_loss = accumulated_loss + chunk_loss
        grad_iter = iter(grads)
        if hidden_needs_grad and grad_hidden_chunk is not None:
            grad_hidden_chunk.copy_(next(grad_iter))
        if weight_needs_grad and grad_weight is not None:
            grad_weight.add_(next(grad_iter))
        if bias_needs_grad and grad_bias_chunk is not None:
            grad_bias_chunk.copy_(next(grad_iter))
        if bias_weight_needs_grad and grad_bias_weight is not None:
            grad_bias_weight.add_(next(grad_iter))

    finish_capture()
    saved_tensors = (
        grad_hidden if grad_hidden is not None else hidden.new_empty(0),
        grad_weight if grad_weight is not None else weight.new_empty(0),
        *((grad_bias,) if bias_needs_grad and grad_bias is not None else ()),
        *((grad_bias_weight,) if bias_weight_needs_grad and grad_bias_weight is not None else ()),
    )
    return accumulated_loss, SavedState(
        saved_tensors,
        _Meta(True, hidden_needs_grad, weight_needs_grad, bias_needs_grad, bias_weight_needs_grad),
    )


def backward(
    grad_output: Tensor, saved: SavedState
) -> tuple[Tensor | None, None, Tensor | None, Tensor | None, Tensor | None]:
    """Return ``(grad_hidden, None, grad_weight, grad_logits_bias, grad_logits_bias_weight)``."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    grad_hidden, grad_weight, *rest = saved.tensors
    rest_iter = iter(rest)
    grad_bias = next(rest_iter) * grad_output if meta.bias_needs_grad else None
    grad_bias_weight = next(rest_iter) * grad_output if meta.bias_weight_needs_grad else None
    return (
        grad_hidden * grad_output if meta.hidden_needs_grad else None,
        None,
        grad_weight * grad_output if meta.weight_needs_grad else None,
        grad_bias,
        grad_bias_weight,
    )
