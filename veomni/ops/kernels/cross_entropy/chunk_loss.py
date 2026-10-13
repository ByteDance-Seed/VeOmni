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

"""Chunked cross-entropy loss for causal LM heads.

Hardware-agnostic: composes ``F.linear`` and ``eager_cross_entropy`` and runs
on both CUDA and Ascend NPU without any device-specific calls. Selected via
``OpsImplementationConfig.cross_entropy_loss_implementation == "chunk_loss"``
(the default); ``"npu"`` is kept as a back-compat alias for the same kernel.

The outer ``chunk_loss_function`` splits the sequence into chunks and calls
eager CE on each chunk, accumulating gradients via a custom autograd
``Function``. It is installed directly into ``LOSS_MAPPING["ForCausalLM"]`` /
``LOSS_MAPPING["ForConditionalGeneration"]`` by
``install_loss_mapping("chunk_loss")`` and never reaches ``ForCausalLMLoss``.

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
        logits_bias: torch.Tensor | None = None,
        row_chunks: list[torch.Tensor] | None = None,
    ):
        if head_bias is not None:
            raise NotImplementedError("head_bias is not supported in ChunkLoss")

        device = hidden_states.device
        accumulated_loss = torch.tensor(0.0, device=device)
        grad_weight = torch.zeros_like(head_weight)
        need_bias_grad = logits_bias is not None and ctx.needs_input_grad[6]
        argnums = (0, 1, 3) if need_bias_grad else (0, 1)

        if row_chunks is not None:
            # Chunks are index sets over the flattened rows; rows in none of them
            # are never projected and get a zero gradient.
            flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
            flat_bias = logits_bias.reshape(-1, logits_bias.size(-1)) if logits_bias is not None else None
            grad_inputs = hidden_states.new_zeros(flat_hidden.shape)
            grad_logits_bias = logits_bias.new_zeros(flat_bias.shape) if need_bias_grad else None
            for rows, loss_kwargs in zip(row_chunks, loss_kwargs_chunks):
                bias_rows = flat_bias.index_select(0, rows) if flat_bias is not None else None
                chunk_grads, (chunk_loss, _) = torch.func.grad_and_value(loss_forward, argnums=argnums, has_aux=True)(
                    flat_hidden.index_select(0, rows), head_weight, None, bias_rows, **loss_kwargs
                )
                accumulated_loss.add_(chunk_loss)
                grad_inputs.index_copy_(0, rows, chunk_grads[0])
                grad_weight.add_(chunk_grads[1])
                if need_bias_grad:
                    grad_logits_bias.index_copy_(0, rows, chunk_grads[2])
            grad_inputs = grad_inputs.view(hidden_states.shape)
            if need_bias_grad:
                grad_logits_bias = grad_logits_bias.view(logits_bias.shape)
        else:
            grad_inputs = torch.empty_like(hidden_states)
            grad_logits_bias = torch.zeros_like(logits_bias) if need_bias_grad else None
            grad_inputs_chunks = torch.split(grad_inputs, chunk_size, dim=1)
            hidden_states_chunks = torch.split(hidden_states, chunk_size, dim=1)
            num_chunks = len(hidden_states_chunks)
            bias_chunks = (
                torch.split(logits_bias, chunk_size, dim=1) if logits_bias is not None else [None] * num_chunks
            )
            grad_bias_chunks = torch.split(grad_logits_bias, chunk_size, dim=1) if need_bias_grad else None
            for i in range(num_chunks):
                chunk_grads, (chunk_loss, _) = torch.func.grad_and_value(loss_forward, argnums=argnums, has_aux=True)(
                    hidden_states_chunks[i], head_weight, None, bias_chunks[i], **loss_kwargs_chunks[i]
                )
                accumulated_loss.add_(chunk_loss)
                grad_inputs_chunks[i].copy_(chunk_grads[0])
                grad_weight.add_(chunk_grads[1])
                if need_bias_grad:
                    grad_bias_chunks[i].copy_(chunk_grads[2])

        ctx.need_bias_grad = need_bias_grad
        ctx.save_for_backward(grad_inputs, grad_weight, *((grad_logits_bias,) if need_bias_grad else ()))
        return accumulated_loss

    @staticmethod
    def backward(ctx, *grad_output):
        grad_input, grad_weight, *grad_logits_bias = ctx.saved_tensors
        grad_logits_bias = grad_logits_bias[0] if ctx.need_bias_grad else None
        if torch.ne(grad_output[0], torch.tensor(1.0, device=grad_output[0].device)):
            grad_input = grad_input * grad_output[0]
            grad_weight = grad_weight * grad_output[0]
            if grad_logits_bias is not None:
                grad_logits_bias = grad_logits_bias * grad_output[0]
        return grad_input, grad_weight, None, None, None, None, grad_logits_bias, None


def chunk_loss_function(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 1024,
    vocab_size: Optional[int] = None,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    loss_reduction_group: Optional[Any] = None,
    logits_bias: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Chunked ``CE(hidden_states @ weights.T [+ logits_bias], labels)``.

    ``logits_bias`` is an optional additive term per row, shaped like the logits
    (``hidden_states.shape[:-1] + (vocab,)``) and differentiable. It is sliced
    along with the hidden states, so only one chunk of fp32 logits is ever live;
    the bias and its gradient are the only full-size tensors. With a bias only
    the supervised rows are projected, ``chunk_size`` of them at a time: a
    caller cannot drop the others itself without copying the bias.
    """
    sp_enabled = get_parallel_state().sp_enabled
    if logits_bias is not None and logits_bias.shape[:-1] != hidden_states.shape[:-1]:
        raise ValueError(
            f"logits_bias rows {tuple(logits_bias.shape[:-1])} do not match hidden_states rows "
            f"{tuple(hidden_states.shape[:-1])}."
        )
    if shift_labels is not None:
        labels = shift_labels
    elif not sp_enabled:
        labels = labels[..., 1:].contiguous()
        hidden_states = hidden_states[..., :-1, :].contiguous()
        if logits_bias is not None:
            logits_bias = logits_bias[..., :-1, :].contiguous()
    # Flat ``[N, H]`` rows with ``[N]`` targets, as a caller that has already
    # dropped its unsupervised rows passes them, are one sequence of ``N``.
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
        hidden_states = hidden_states.unsqueeze(0)
        if logits_bias is not None:
            logits_bias = logits_bias.unsqueeze(0)
    if hidden_states.shape[:-1] != labels.shape:
        raise ValueError(
            f"hidden_states rows {tuple(hidden_states.shape[:-1])} do not match labels {tuple(labels.shape)}."
        )

    def ce_loss_func(
        hidden_states, weight, bias, logits_bias, labels, num_items_in_batch, ignore_index=-100, **kwargs
    ):
        # Use ``reshape`` instead of ``view`` because the per-chunk tensors come
        # from ``torch.split(..., dim=1)`` on contiguous parents, which yields
        # non-contiguous views (parent stride is preserved on dim 0).
        labels = labels.reshape(-1)
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        logits = F.linear(hidden_states, weight).float()
        if logits_bias is not None:
            logits = logits + logits_bias.reshape(-1, logits_bias.size(-1)).float()
        loss, logits = eager_cross_entropy(
            logits,
            labels,
            vocab_size,
            num_items_in_batch,
            ignore_index,
            hidden_states=hidden_states,
            weights=weights,
            **kwargs,
        )
        return loss, logits

    num_valid_tokens = (labels != ignore_index).sum()
    # A rank with nothing supervised scores 0.0 rather than 0 / 0, so its NaN
    # cannot survive the token-weighted reduction below.
    num_items_in_batch = num_valid_tokens.clamp(min=1)

    row_chunks = None
    if logits_bias is not None:
        flat_labels = labels.reshape(-1)
        row_chunks = list(torch.split((flat_labels != ignore_index).nonzero().squeeze(-1), chunk_size))
        chunk_labels = [flat_labels.index_select(0, rows) for rows in row_chunks]
    else:
        chunk_labels = torch.split(labels, chunk_size, dim=1)

    loss_kwargs_chunks = [
        {"labels": chunk_labels[i], "ignore_index": ignore_index, "num_items_in_batch": num_items_in_batch}
        for i in range(len(chunk_labels))
    ]

    chunk_loss = ChunkLoss.apply(
        hidden_states, weights, None, ce_loss_func, loss_kwargs_chunks, chunk_size, logits_bias, row_chunks
    )

    # Match ``ForCausalLMLoss``: reduce over the SP group when SP is enabled, or
    # over the caller's explicit group (SeedOmni V2 passes its dp_sp FSDP group).
    if sp_enabled or loss_reduction_group is not None:
        chunk_loss = reduce_sequence_parallel_loss(chunk_loss, num_valid_tokens, group=loss_reduction_group)
    return chunk_loss, None
