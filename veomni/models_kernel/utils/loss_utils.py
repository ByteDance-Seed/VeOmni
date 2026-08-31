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

"""Modeling helpers around token-level ``cross_entropy_loss``.

Shift, SP reduce, and the log-probs / distill side paths live here. The
registry row is only ``(hidden, labels, weight, *, ignore_index,
num_items_in_batch)``.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, nn

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import reduce_sequence_parallel_loss
from veomni.kernels import VeomniKernel
from veomni.utils import logging
from veomni.utils.model_outputs import FusedLinearAuxOutput

from .chunk_logprobs import chunk_logprobs_function
from .chunk_topk_distill import chunk_topk_distill_function


logger = logging.get_logger(__name__)


def _default_kernel() -> VeomniKernel:
    return VeomniKernel("cross_entropy_loss", "standard", "eager")


def _select_hidden_weight(
    logits: Tensor | None,
    hidden_states: Tensor | None,
    weights: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Return ``(hidden, weight, out_logits)``.

    Hidden+weight is the fused path. ``out_logits`` is the flattened caller
    logits when those were passed, otherwise ``None``.
    """
    if hidden_states is not None and weights is not None:
        out_logits = logits.float() if logits is not None else None
        return hidden_states, weights, out_logits
    if logits is not None:
        logits = logits.float()
        return logits, logits.new_empty(0), logits
    raise ValueError("Provide hidden_states and weights, or logits.")


def ForCausalLMLoss(
    logits: Tensor | None = None,
    labels: Tensor | None = None,
    vocab_size: int | None = None,
    num_items_in_batch: int | None = None,
    ignore_index: int = -100,
    shift_labels: Tensor | None = None,
    *,
    kernel: Callable | None = None,
    **kwargs,
) -> tuple[Tensor | None, Tensor | None, FusedLinearAuxOutput | None]:
    """Causal LM helper: shift, flatten, token CE, SP reduce.

    ``kernel`` is a token-level ``cross_entropy_loss`` handle.
    """
    del vocab_size
    hidden_states = kwargs.pop("hidden_states", None)
    weights = kwargs.pop("weights", None)
    return_log_probs = kwargs.pop("return_log_probs", False)
    temperature = kwargs.pop("temperature", 1.0)
    teacher_topk_ids = kwargs.pop("teacher_topk_ids", None)
    teacher_topk_log_probs = kwargs.pop("teacher_topk_log_probs", None)
    log_prob_min_clamp = kwargs.pop("log_prob_min_clamp", None)
    chunk_size = kwargs.pop("chunk_size", 1024)

    if hidden_states is None and logits is None:
        raise ValueError("hidden_states or logits must be provided.")

    if return_log_probs:
        if hidden_states is None:
            raise ValueError("return_log_probs=True requires hidden_states (fused-linear path).")
        if weights is None:
            raise ValueError("return_log_probs=True requires weights (lm_head weight).")
        if (teacher_topk_ids is None) != (teacher_topk_log_probs is None):
            raise ValueError(
                "teacher_topk_ids and teacher_topk_log_probs must be provided together for "
                "the top-k distillation path."
            )
        if teacher_topk_ids is not None:
            log_probs, entropy, distill, student_mass, teacher_mass = chunk_topk_distill_function(
                hidden_states,
                weights,
                labels,
                teacher_topk_ids,
                teacher_topk_log_probs,
                chunk_size=chunk_size,
                ignore_index=ignore_index,
                shift_labels=shift_labels,
                temperature=temperature,
                log_prob_min_clamp=log_prob_min_clamp,
            )
            return (
                None,
                None,
                FusedLinearAuxOutput(
                    log_probs=log_probs,
                    entropy=entropy,
                    distillation_losses=distill,
                    student_mass=student_mass,
                    teacher_mass=teacher_mass,
                ),
            )
        log_probs, entropy = chunk_logprobs_function(
            hidden_states,
            weights,
            labels,
            chunk_size=chunk_size,
            ignore_index=ignore_index,
            shift_labels=shift_labels,
            temperature=temperature,
        )
        return None, None, FusedLinearAuxOutput(log_probs=log_probs, entropy=entropy)

    token_kernel = kernel if kernel is not None else _default_kernel()
    device = logits.device if logits is not None else hidden_states.device
    sp_enabled = get_parallel_state().sp_enabled

    if not sp_enabled:
        if shift_labels is None:
            labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:].contiguous()
    else:
        if shift_labels is not None:
            logger.warning_once("labels have been shifted in dataloader when `sp_enabeld=True`, ignore shift_labels.")
        shift_labels = labels

    shift_labels = shift_labels.view(-1)
    if hidden_states is not None:
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
    if logits is not None:
        logits = logits.view(-1, logits.size(-1))
    shift_labels = shift_labels.to(device)

    hidden, weight, out_logits = _select_hidden_weight(logits, hidden_states, weights)
    loss = token_kernel(
        hidden,
        shift_labels,
        weight,
        ignore_index=ignore_index,
        num_items_in_batch=num_items_in_batch,
    )
    if sp_enabled:
        num_valid_tokens = (labels != ignore_index).sum()
        loss = reduce_sequence_parallel_loss(loss, num_valid_tokens)
    return loss, out_logits, None


def ForSequenceClassificationLoss(
    logits: Tensor | None = None,
    labels: Tensor | None = None,
    num_labels: int | None = None,
    num_items_in_batch: int | None = None,
    ignore_index: int = -100,
    *,
    kernel: Callable | None = None,
    **kwargs,
) -> tuple[Tensor, Tensor | None, None]:
    """Seq-cls helper: flatten, token CE, SP reduce. No label shift."""
    hidden_states = kwargs.pop("hidden_states", None)
    weights = kwargs.pop("weights", None)
    kwargs.pop("return_log_probs", None)
    kwargs.pop("temperature", None)
    kwargs.pop("teacher_topk_ids", None)
    kwargs.pop("teacher_topk_log_probs", None)
    kwargs.pop("log_prob_min_clamp", None)
    kwargs.pop("chunk_size", None)

    if hidden_states is None and logits is None:
        raise ValueError("Either hidden_states or logits must be provided.")
    if labels is None:
        raise ValueError("labels must be provided for sequence classification loss.")
    if num_labels is None:
        raise ValueError("num_labels must be provided.")

    token_kernel = kernel if kernel is not None else _default_kernel()
    device = logits.device if logits is not None else hidden_states.device
    target = labels.view(-1)
    if hidden_states is not None:
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
    if logits is not None:
        logits = logits.float().view(-1, num_labels)
    target = target.to(device)

    hidden, weight, out_logits = _select_hidden_weight(logits, hidden_states, weights)
    loss = token_kernel(
        hidden,
        target,
        weight,
        ignore_index=ignore_index,
        num_items_in_batch=num_items_in_batch,
    )
    if get_parallel_state().sp_enabled:
        num_valid_tokens = (target != ignore_index).sum()
        loss = reduce_sequence_parallel_loss(loss, num_valid_tokens)
    return loss, out_logits, None
