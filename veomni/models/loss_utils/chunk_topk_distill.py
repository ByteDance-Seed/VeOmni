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

"""Model-facing chunked fused linear top-k forward-KL distillation policy.

Handles sequence-parallel awareness, causal label and teacher top-k shifting,
and trailing zero padding while delegating the tensor-level chunked top-k
distillation autograd Function and device-dispatched per-token log-probability
kernels to ``veomni.ops.kernels.loss.token_logprobs``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import torch

from veomni.distributed.parallel_state import get_parallel_state
from veomni.ops.kernels.loss.token_logprobs.chunked import (
    _ChunkedLinearTopkDistill,
    chunk_linear_topk_distill,
    resolve_token_logprobs_op,
)

from .chunk_logprobs import _pad_causal_chunk_outputs, _prepare_causal_chunk_inputs


def chunk_topk_distill_function(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    chunk_size: int = 1024,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    log_prob_min_clamp: Optional[float] = None,
    *,
    op: Callable[..., torch.Tensor] | None = None,
    impl: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunked fused-linear top-k forward-KL distillation + log-probs + entropy.

    Args:
        hidden_states: ``[B, L, H]`` (or ``[L, H]`` for packed inputs).
        weights: lm_head weight ``[V, H]``. Bias is not supported.
        labels: integer label tensor with shape matching the leading dims of
            ``hidden_states``. Positions equal to ``ignore_index`` produce
            ``0.0`` on every output (no gradient flows through them).
        teacher_topk_ids: ``[B, L, K]`` int64 — teacher's top-k vocabulary
            indices per position. Caller is expected to densify any nested
            tensors upstream (verl's FSDP path does
            ``.values().unsqueeze(0)`` before this function is reached).
        teacher_topk_log_probs: ``[B, L, K]`` — teacher's log-probabilities
            at those indices. May be bf16 from a nested storage; cast to
            fp32 internally for the KL reduction.
        chunk_size: token-dim chunk for the streamed projection.
        ignore_index: label value to mask (default ``-100``).
        shift_labels: pre-shifted labels (SP path). Same semantics as
            ``chunk_logprobs_function``. When the caller (verl) is already
            in remove-padding + use_fused_kernels mode, it passes the
            already-rolled labels as ``shift_labels`` and the kernel skips
            its internal causal shift; teacher tensors must align with
            those shifted labels.
        temperature: divides logits before log_softmax. Defaults to 1.0.
        log_prob_min_clamp: if not None, clamp both teacher and student
            top-k log-probabilities to this minimum before the KL
            reduction. Matches verl's
            ``DistillationLossConfig.log_prob_min_clamp`` semantics.
        op: optional ``VeomniOp("token_logprobs", "standard", ...)`` handle
            owned by the consuming model.
        impl: optional ``token_logprobs`` implementation name override
            (``"eager"``, ``"flash_attn"``, or ``"torch_npu"``) when ``op``
            is not supplied. When ``None``, resolves the active device
            default via ``VeomniOp``.

    Returns:
        ``(log_probs, entropy, distillation_losses, student_mass, teacher_mass)``
        — all five tensors share ``labels``' shape, all are fp32. The
        last two are ``detach()``-ed (metrics only); the first three
        flow gradients into ``hidden_states`` and ``weights`` through
        the closed-form backward.
    """
    (
        hidden_states,
        labels_shifted,
        (teacher_topk_ids, teacher_topk_log_probs),
        needs_trailing_pad,
    ) = _prepare_causal_chunk_inputs(
        hidden_states,
        labels,
        shift_labels,
        sp_enabled=get_parallel_state().sp_enabled,
        label_aligned_tensors=(teacher_topk_ids, teacher_topk_log_probs),
    )

    token_logprobs_fn = (
        op if op is not None else resolve_token_logprobs_op(impl, device_type=hidden_states.device.type)
    )
    log_probs, entropy, distill, student_mass, teacher_mass = chunk_linear_topk_distill(
        hidden_states,
        weights,
        labels_shifted,
        teacher_topk_ids,
        teacher_topk_log_probs,
        temperature=temperature,
        chunk_size=chunk_size,
        ignore_index=ignore_index,
        log_prob_min_clamp=log_prob_min_clamp,
        token_logprobs_fn=token_logprobs_fn,
    )

    log_probs, entropy, distill, student_mass, teacher_mass = _pad_causal_chunk_outputs(
        (log_probs, entropy, distill, student_mass, teacher_mass),
        needs_trailing_pad=needs_trailing_pad,
    )
    # ``apply`` attaches ``grad_fn`` even after the inner detach. Mass
    # outputs are metrics-only on every branch, including explicit shift
    # and sequence-parallel.
    return log_probs, entropy, distill, student_mass.detach(), teacher_mass.detach()


__all__ = [
    "_ChunkedLinearTopkDistill",
    "chunk_topk_distill_function",
]
