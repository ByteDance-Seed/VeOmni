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

"""Chunked cross-entropy loss for causal LM heads on NPU.

Used when ``OpsImplementationConfig.cross_entropy_loss_implementation == "npu"``.
The outer ``chunk_loss_function`` splits the sequence into chunks and calls the
eager cross-entropy on each chunk, accumulating gradients via a custom autograd
``Function``. The chunked path always uses eager — it is installed directly into
``LOSS_MAPPING["ForCausalLM"]`` / ``LOSS_MAPPING["ForConditionalGeneration"]``
by ``install_loss_mapping("npu")`` and never reaches ``ForCausalLMLoss``.

Causal-only: the function hard-codes a causal label shift, so it cannot back
``ForSequenceClassification`` (token-level labels, no shift). SP reduction is
applied here so VLMs with SP enabled produce correct losses.
"""

from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from ....distributed.parallel_state import get_parallel_state
from ....distributed.sequence_parallel import reduce_sequence_parallel_loss
from .eager import eager_cross_entropy


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
    sp_enabled = get_parallel_state().sp_enabled
    # Snapshot the pre-shift labels for the SP denominator — the non-SP branch
    # below rewrites `labels` in place with the shifted view.
    sp_reduction_labels = labels

    if not sp_enabled:
        labels = labels[..., 1:].contiguous()
        hidden_states = hidden_states[..., :-1, :].contiguous()

    def ce_loss_func(hidden_states, weight, bias, labels, num_items_in_batch, ignore_index=-100, **kwargs):
        labels = labels.view(-1)
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        logits = F.linear(hidden_states, weight).float()
        loss, logits = eager_cross_entropy(
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

    # Match ``ForCausalLMLoss`` SP behavior so chunk_loss can back both
    # ForCausalLM and ForConditionalGeneration heads when SP is enabled.
    if sp_enabled:
        num_valid_tokens = (sp_reduction_labels != ignore_index).sum()
        chunk_loss = reduce_sequence_parallel_loss(chunk_loss, num_valid_tokens)
    return chunk_loss, None
