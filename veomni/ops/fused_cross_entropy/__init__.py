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

import os
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.loss.loss_utils import LOSS_MAPPING

from ...distributed.parallel_state import get_parallel_state
from ...distributed.sequence_parallel import reduce_sequence_parallel_loss
from ...utils import logging
from ...utils.env import get_env
from ...utils.import_utils import is_liger_kernel_available, is_torch_npu_available
from .eager import eager_cross_entropy


logger = logging.get_logger(__name__)


_cross_entropy = None


def ForCausalLMLoss(
    logits: torch.Tensor = None,
    labels: torch.Tensor = None,
    vocab_size: int = None,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    # pop fused loss kwargs
    hidden_states = kwargs.pop("hidden_states", None)
    weights = kwargs.pop("weights", None)

    assert hidden_states is not None or logits is not None, "hidden_states or logits must be provided."

    device = logits.device if logits is not None else hidden_states.device
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    if logits is not None:
        logits = logits.float()

    sp_enabled = get_parallel_state().sp_enabled

    # veomni sp patch
    if not sp_enabled:
        # Shift so that tokens < n predict n
        if shift_labels is None:
            labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:].contiguous()
    else:
        if shift_labels is not None:
            logger.warning_once("labels have been shifted in dataloader when `sp_enabeld=True`, ignore shift_labels.")
        shift_labels = labels

    # Flatten the tokens
    shift_labels = shift_labels.view(-1)
    if hidden_states is not None:
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
    if logits is not None:
        logits = logits.view(-1, vocab_size)
    # Enable model parallelism
    shift_labels = shift_labels.to(device)

    if hidden_states is None or weights is None:
        logger.warning_once(
            "hidden_states or weights is None, use eager loss implementation."
            "To enable fused linear cross entropy loss, please patch modeling.py `forward` function "
            "to pass `hidden_states` and `weights` to `loss_function`."
        )
        loss_func = eager_cross_entropy
    else:
        loss_func = _cross_entropy
    loss, logits = loss_func(
        logits,
        shift_labels,
        vocab_size,
        num_items_in_batch,
        ignore_index,
        shift_labels,
        hidden_states=hidden_states,
        weights=weights,
        **kwargs,
    )

    # Reduce loss when using sp
    if sp_enabled:
        num_valid_tokens = (labels != ignore_index).sum()
        loss = reduce_sequence_parallel_loss(loss, num_valid_tokens)
    return loss, logits


def ForSequenceClassificationLoss(
    logits: torch.Tensor = None,
    labels: torch.Tensor = None,
    num_labels: int = None,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    r"""
    Token-level loss for sequence classification.

    This loss follows the "token-level labels" convention:
    `labels` has the same layout as the token sequence,
    with all positions set to `ignore_index` except the supervised tokens (the last valid token of each sample).
    No shifting is applied.
    When SP is enabled, the loss is reduced across SP ranks using the number of non-ignored tokens.

    Args:
        logits (`torch.Tensor`):
            Classification logits.
        labels (`torch.Tensor`):
            Token-level labels with `ignore_index` marking non-supervised positions.
        num_labels (`int`):
            Number of classes.
        num_items_in_batch (`int`):
            Used to accurately calculate the average loss for each sample.
        ignore_index (`int`, defaults to `-100`):
            Label value to ignore when computing the loss.
        hidden_states (`torch.Tensor`):
            Hidden states, used for fused linear cross-entropy.
        weights (`torch.Tensor`):
            Classification head weights, used for fused linear cross-entropy.

    Returns:
        loss (`torch.Tensor`):
            Scalar classification loss.
        logits (`torch.Tensor`):
            Flattened logits.
    """

    # pop fused loss kwargs
    hidden_states = kwargs.pop("hidden_states", None)
    weights = kwargs.pop("weights", None)

    if hidden_states is None and logits is None:
        raise ValueError("Either hidden_states or logits must be provided.")

    if labels is None:
        raise ValueError("labels must be provided for sequence classification loss.")

    if num_labels is None:
        raise ValueError("num_labels must be provided.")

    device = logits.device if logits is not None else hidden_states.device
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    if logits is not None:
        logits = logits.float()

    sp_enabled = get_parallel_state().sp_enabled
    target = labels

    # Flatten the tokens
    target = target.view(-1)
    if hidden_states is not None:
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
    if logits is not None:
        logits = logits.view(-1, num_labels)
    # Enable model parallelism
    target = target.to(device)

    if hidden_states is None or weights is None:
        logger.warning_once(
            "hidden_states or weights is None, use eager loss implementation."
            "To enable fused linear cross entropy loss, please patch modeling.py `forward` function "
            "to pass `hidden_states` and `weights` to `loss_function`."
        )
        loss_func = eager_cross_entropy
    else:
        loss_func = _cross_entropy

    loss, logits = loss_func(
        logits,
        target,
        num_labels,
        num_items_in_batch,
        ignore_index,
        target,
        hidden_states=hidden_states,
        weights=weights,
        **kwargs,
    )

    # Reduce loss when using sp
    if sp_enabled:
        num_valid_tokens = (target != ignore_index).sum()
        loss = reduce_sequence_parallel_loss(loss, num_valid_tokens)
    return loss, logits


class ChunkLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_forward: Callable,
        loss_kwargs_chunks: list[Any],
        chunk_size: int,
    ):
        if head_bias is not None:
            raise NotImplementedError("head_bias is not supported in ChunkLoss")

        device = hidden_states.device
        accumulated_loss = torch.tensor(0.0, device=device)
        grad_inputs = torch.empty_like(hidden_states)
        grad_weight = torch.zeros_like(head_weight)

        grad_inputs_chunks = torch.split(grad_inputs, chunk_size, dim=1)

        hidden_states_chunks = torch.split(hidden_states, chunk_size, dim=1)

        for i in range(len(hidden_states_chunks)):
            hidden_states_chunk = hidden_states_chunks[i]
            grad_inputs_chunk = grad_inputs_chunks[i]
            (chunk_grad_input, chunk_grad_weight), (chunk_loss, _) = torch.func.grad_and_value(
                loss_forward, argnums=(0, 1), has_aux=True
            )(hidden_states_chunk, head_weight, None, **loss_kwargs_chunks[i])

            accumulated_loss.add_(chunk_loss)
            grad_inputs_chunk.copy_(chunk_grad_input)
            grad_weight.add_(chunk_grad_weight)

        ctx.save_for_backward(grad_inputs, grad_weight)
        return accumulated_loss

    @staticmethod
    def backward(ctx, *grad_output):
        grad_input, grad_weight = ctx.saved_tensors
        if torch.ne(grad_output[0], torch.tensor(1.0, device=grad_output[0].device)):
            grad_input = grad_input * grad_output[0]
            grad_weight = grad_weight * grad_output[0]
        return grad_input, grad_weight, None, None, None, None


def chunk_loss_function(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 1024,
    vocab_size: Optional[int] = None,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    if not get_parallel_state().sp_enabled:
        labels = labels[..., 1:].contiguous()
        hidden_states = hidden_states[..., :-1, :].contiguous()

    def ce_loss_func(hidden_states, weight, bias, labels, num_items_in_batch, ignore_index=-100, **kwargs):
        # Flatten the labels and hidden_states
        labels = labels.view(-1)
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        logits = F.linear(hidden_states, weight).float()
        loss_func = _cross_entropy
        loss, logits = loss_func(
            logits,
            labels,
            vocab_size,
            num_items_in_batch,
            ignore_index,
            shift_labels,
            hidden_states=hidden_states,
            weights=weights,
            **kwargs,
        )
        return loss, logits

    chunk_labels = torch.split(labels, chunk_size, dim=1)

    loss_kwargs_chunks = [
        {"labels": chunk_labels[i], "ignore_index": ignore_index, "num_items_in_batch": (labels != ignore_index).sum()}
        for i in range(len(chunk_labels))
    ]

    chunk_loss = ChunkLoss.apply(hidden_states, weights, None, ce_loss_func, loss_kwargs_chunks, chunk_size)
    return chunk_loss, None


def apply_veomni_loss_patch():
    LOSS_MAPPING["ForCausalLM"] = ForCausalLMLoss
    LOSS_MAPPING["ForConditionalGeneration"] = ForCausalLMLoss
    LOSS_MAPPING["ForSequenceClassification"] = ForSequenceClassificationLoss
    global _cross_entropy
    if is_torch_npu_available():
        if os.environ.get("VEOMNI_ENABLE_CHUNK_LOSS", "0") == "1":
            LOSS_MAPPING["ForCausalLM"] = chunk_loss_function
        _cross_entropy = eager_cross_entropy
    elif is_liger_kernel_available() and get_env("USE_LIGER_KERNEL") == "1":
        from .liger_kernel import fused_liger_kernel_cross_entropy

        _cross_entropy = fused_liger_kernel_cross_entropy
    else:
        _cross_entropy = eager_cross_entropy
