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

"""Model-facing chunked fused linear log-probs and entropy policy for PPO-style RL.

Handles sequence-parallel awareness, causal label shifting
(``labels[..., 1:]`` / ``hidden_states[..., :-1, :]``), and trailing zero
padding while delegating the tensor-level chunked projection and
device-dispatched per-token log-probability kernels to
``veomni.ops.kernels.loss.token_logprobs``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import torch

from veomni.distributed.parallel_state import get_parallel_state
from veomni.ops.kernels.loss.token_logprobs.chunked import (
    _ChunkedLinearLogProbs,
    chunk_linear_logprobs,
    resolve_token_logprobs_op,
)
from veomni.ops.kernels.loss.token_logprobs.standard.eager import token_entropy_from_logits


def _per_token_log_probs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    *,
    op: Callable[..., torch.Tensor] | None = None,
    impl: str | None = None,
) -> torch.Tensor:
    """Compute per-token ``log p(y_t)`` from fp32 logits via the ``token_logprobs`` ``VeomniOp``."""
    token_op = op if op is not None else resolve_token_logprobs_op(impl, device_type=logits.device.type)
    return token_op(logits, labels, ignore_index=ignore_index)


_per_token_entropy_from_logits = token_entropy_from_logits


def _prepare_causal_chunk_inputs(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    shift_labels: Optional[torch.Tensor],
    *,
    sp_enabled: bool,
    label_aligned_tensors: tuple[torch.Tensor, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...], bool]:
    """Align ``hidden_states``, ``labels``, and optional label-aligned tensors for causal prediction.

    Three modes for choosing the per-position target (matching ``ForCausalLMLoss``):
    1. Caller passes pre-shifted labels via ``shift_labels`` -> trust them and run
       ``hidden_states`` unchanged. Output keeps the input's seq length.
    2. SP enabled, ``shift_labels`` not provided -> ``SequenceParallelCollator`` has
       already globally shifted ``labels``; do not shift again.
    3. SP disabled, ``shift_labels`` not provided -> apply the causal
       ``labels[..., 1:]`` / ``hidden_states[..., :-1, :]`` shift (plus
       ``tensor[..., 1:, :]`` on any label-aligned auxiliary tensors) and mark the
       outputs for a trailing zero pad so the returned shape matches ``labels``.
    """
    if shift_labels is not None:
        return hidden_states, shift_labels, label_aligned_tensors, False
    if sp_enabled:
        return hidden_states, labels, label_aligned_tensors, False
    return (
        hidden_states[..., :-1, :].contiguous(),
        labels[..., 1:].contiguous(),
        tuple(tensor[..., 1:, :].contiguous() for tensor in label_aligned_tensors),
        True,
    )


def _pad_causal_chunk_outputs(
    outputs: tuple[torch.Tensor, ...],
    *,
    needs_trailing_pad: bool,
) -> tuple[torch.Tensor, ...]:
    """Pad each output tensor with one trailing ``0.0`` along the sequence dim when needed."""
    if not needs_trailing_pad:
        return outputs
    return tuple(torch.nn.functional.pad(output, (0, 1), value=0.0) for output in outputs)


def chunk_logprobs_function(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 1024,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    *,
    op: Callable[..., torch.Tensor] | None = None,
    impl: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token log-probabilities and entropy via a chunked fused linear.

    Args:
        hidden_states: ``[B, L, H]`` (or ``[L, H]`` for already-packed
            inputs).
        weights: lm_head weight, ``[V, H]``. Bias is not supported.
        labels: integer label tensor with shape matching the leading
            dims of ``hidden_states``. Positions equal to
            ``ignore_index`` produce ``0.0`` in both outputs (no
            gradient flows through them either).
        chunk_size: token-dim chunk for the streamed projection.
        ignore_index: label value to mask (default ``-100`` /
            ``IGNORE_INDEX``).
        shift_labels: pre-shifted target labels. When provided, the
            kernel uses them as-is and skips the internal causal
            shift, so callers can supply custom alignment (e.g. the
            ``ForCausalLMLoss`` SP path that pre-shifts via padding).
            The output tensors' seq length matches ``shift_labels``
            (no trailing pad). When ``None``, this function applies
            the causal ``labels[..., 1:]`` /
            ``hidden_states[..., :-1, :]`` shift internally and pads
            the trailing seq slot with ``0.0`` so the output shape
            matches the input ``labels``.
        temperature: divides logits before log_softmax (PPO actor
            path). Defaults to 1.0 (no-op).
        op: optional ``VeomniOp("token_logprobs", "standard", ...)`` handle
            owned by the consuming model.
        impl: optional ``token_logprobs`` implementation name override
            (``"eager"``, ``"flash_attn"``, or ``"torch_npu"``) when ``op``
            is not supplied. When ``None``, resolves the active device
            default via ``VeomniOp``.

    Returns:
        ``(log_probs, entropy)`` — both with the same shape as the
        input ``labels``.

        - ``log_probs``: per-token ``log p(y_t)`` (non-positive). Zero
          at IGNORE_INDEX positions and at the trailing pad slot.
        - ``entropy``: per-token softmax entropy
          ``H[p] = -Σ_v p_v log p_v`` (non-negative). Zero at
          IGNORE_INDEX positions and at the trailing pad slot.

        Sign conventions match HF / verl — no negation needed at the
        call site.
    """
    hidden_states, labels_shifted, _, needs_trailing_pad = _prepare_causal_chunk_inputs(
        hidden_states,
        labels,
        shift_labels,
        sp_enabled=get_parallel_state().sp_enabled,
    )

    token_logprobs_fn = (
        op if op is not None else resolve_token_logprobs_op(impl, device_type=hidden_states.device.type)
    )
    log_probs, entropy = chunk_linear_logprobs(
        hidden_states,
        weights,
        labels_shifted,
        temperature=temperature,
        chunk_size=chunk_size,
        ignore_index=ignore_index,
        token_logprobs_fn=token_logprobs_fn,
    )

    log_probs, entropy = _pad_causal_chunk_outputs(
        (log_probs, entropy),
        needs_trailing_pad=needs_trailing_pad,
    )
    return log_probs, entropy


__all__ = [
    "_ChunkedLinearLogProbs",
    "_per_token_entropy_from_logits",
    "_per_token_log_probs_from_logits",
    "chunk_logprobs_function",
]
