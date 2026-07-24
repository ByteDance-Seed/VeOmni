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
from torch.autograd.graph import saved_tensors_hooks

from ....distributed.parallel_state import get_parallel_state
from ....distributed.sequence_parallel import reduce_sequence_parallel_loss
from ....utils.device import empty_cache, stream_synchronize
from .eager import eager_cross_entropy


def _keep_saved_tensor_on_device(tensor: torch.Tensor) -> torch.Tensor:
    """Identity hook for the short-lived, chunk-local CE graph."""
    return tensor


def _release_accelerator_cache(device: torch.device) -> None:
    """Release transient chunk-CE allocator blocks before model backward.

    ``ChunkLoss.apply`` returns only after the outer saved-tensor hook has
    packed the precomputed input and lm-head gradients. Synchronizing at that
    boundary makes the short-lived logits/autograd allocations reusable, and
    ``empty_cache`` returns their now-unoccupied blocks to the accelerator
    allocator. CPU execution has no accelerator cache to release.
    """
    if device.type == "cpu":
        return
    stream_synchronize()
    empty_cache()


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
            # ``torch.func`` transforms reject any active saved-tensor hooks,
            # including the hooks used by activation offloading. A custom
            # autograd ``Function.forward`` runs under no-grad, so explicitly
            # re-enable grad and differentiate detached leaves for this chunk.
            # This computes the same first-order VJP that the custom backward
            # stores below while remaining compatible with saved-tensor hooks.
            # The chunk-local graph is differentiated immediately, so moving
            # its vocab-sized FP32 logits to CPU only adds transfers and can
            # retain asynchronous copies across chunks. Shadow the outer
            # activation-offload hook with an identity hook here; once this
            # scope exits, ``ctx.save_for_backward`` below again uses the outer
            # hook for the long-lived precomputed gradients.
            with torch.enable_grad(), saved_tensors_hooks(_keep_saved_tensor_on_device, _keep_saved_tensor_on_device):
                hidden_states_leaf = hidden_states_chunk.detach().requires_grad_(True)
                head_weight_leaf = head_weight.detach().requires_grad_(True)
                chunk_loss, chunk_aux = loss_forward(
                    hidden_states_leaf, head_weight_leaf, None, **loss_kwargs_chunks[i]
                )
                # The auxiliary value is normally the FP32 logits. Drop the
                # direct reference before differentiation so consecutive
                # chunks cannot retain an extra vocab-sized buffer.
                del chunk_aux
                chunk_grad_input, chunk_grad_weight = torch.autograd.grad(
                    chunk_loss,
                    (hidden_states_leaf, head_weight_leaf),
                    create_graph=False,
                    retain_graph=False,
                )

            accumulated_loss.add_(chunk_loss.detach())
            grad_inputs_chunk.copy_(chunk_grad_input)
            grad_weight.add_(chunk_grad_weight)
            # Assignment evaluates the next iteration's RHS before rebinding
            # these names. Delete them now so the previous FP32 logits graph
            # and full lm-head gradient cannot overlap the next projection.
            del hidden_states_leaf, head_weight_leaf, chunk_loss, chunk_grad_input, chunk_grad_weight

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
    release_cache: bool = False,
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
        # Use ``reshape`` instead of ``view`` because the per-chunk tensors come
        # from ``torch.split(..., dim=1)`` on contiguous parents, which yields
        # non-contiguous views (parent stride is preserved on dim 0).
        labels = labels.reshape(-1)
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
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

    # Keep this cleanup after ``apply`` rather than inside ``ChunkLoss.forward``:
    # saved-tensor pack hooks run as the custom Function finishes applying.
    # Before this boundary the NPU originals of ``grad_inputs``/``grad_weight``
    # may still be live, so an allocator flush cannot reclaim the full
    # short-lived CE working set. This is opt-in because synchronization and
    # allocator churn can trade throughput for lower peak memory.
    if release_cache:
        _release_accelerator_cache(hidden_states.device)

    # Match ``ForCausalLMLoss`` SP behavior so chunk_loss can back both
    # ForCausalLM and ForConditionalGeneration heads when SP is enabled.
    if sp_enabled:
        num_valid_tokens = (sp_reduction_labels != ignore_index).sum()
        chunk_loss = reduce_sequence_parallel_loss(chunk_loss, num_valid_tokens)
    return chunk_loss, None
