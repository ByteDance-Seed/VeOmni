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

"""Pure-PyTorch implementation of the MoE load balancing auxiliary loss.

This module provides a deterministic, device-agnostic fallback that uses only
standard PyTorch operations. It is used as the eager backend on non-CUDA
devices (e.g. NPU) and can also serve as the reference implementation for
testing.
"""

from typing import Optional, Union

import torch
import torch.distributed as dist


class _DifferentiableAllReduceSum(torch.autograd.Function):
    """Sum a tensor across ``group`` with the matching autograd collective.

    Every sequence-parallel rank consumes the same global auxiliary loss.  The
    backward therefore has to sum the loss gradient from every rank before the
    later FSDP gradient average.  An in-place, non-autograd ``all_reduce`` here
    would under-scale the router gradient by the sequence-parallel world size.
    """

    @staticmethod
    def forward(ctx, value: torch.Tensor, group) -> torch.Tensor:
        ctx.group = group
        result = value.clone()
        dist.all_reduce(result, group=group)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = grad_output.contiguous().clone()
        dist.all_reduce(grad_input, group=ctx.group)
        return grad_input, None


def _global_sequence_parallel_stats(
    expert_count: torch.Tensor,
    router_prob_sum: torch.Tensor,
    total_weight: torch.Tensor,
    group,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce the sufficient statistics for the nonlinear MoE aux loss."""
    if group is None or not dist.is_available() or not dist.is_initialized():
        return expert_count, router_prob_sum, total_weight
    if dist.get_world_size(group) == 1:
        return expert_count, router_prob_sum, total_weight

    global_expert_count = expert_count.detach().clone()
    global_total_weight = total_weight.detach().clone()
    dist.all_reduce(global_expert_count, group=group)
    dist.all_reduce(global_total_weight, group=group)
    global_router_prob_sum = _DifferentiableAllReduceSum.apply(router_prob_sum, group)
    return global_expert_count, global_router_prob_sum, global_total_weight


def load_balancing_loss_pytorch(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k: int = 2,
    attention_mask: Optional[torch.Tensor] = None,
    group=None,
) -> Union[torch.Tensor, int]:
    """Pure-PyTorch load balancing loss for Mixture-of-Experts models.

    Computes the auxiliary load balancing loss from the Switch Transformer paper
    (Fedus et al., 2021; https://arxiv.org/abs/2101.03961), equations (4)-(6)::

        loss = num_experts * sum_e(f_e * P_e)

    where ``f_e`` is the fraction of tokens routed to expert *e* and ``P_e`` is
    the average router probability assigned to expert *e* across all tokens.

    Compared to the HuggingFace reference (``transformers.models.qwen3_moe.
    modeling_qwen3_moe.load_balancing_loss_func``), this implementation avoids
    materializing the ``[N, top_k, num_experts]`` one-hot tensor by using
    ``scatter_add_`` instead. It loops over layers so peak memory per layer is
    ``O(N_layer * E)``. The HF reference is used as the ground-truth in tests.

    Args:
        gate_logits: Tuple of per-layer gate logits, each shaped
            ``[batch_size * seq_len, num_experts]``. May be float16, bfloat16,
            or float32; softmax is computed in float32. Returns ``0`` if
            ``None`` or not a tuple.
        num_experts: Total number of experts ``E``.
        top_k: Number of experts selected per token.
        attention_mask: Optional ``[batch_size, seq_len]`` binary mask where
            ``1`` indicates a real token and ``0`` indicates padding. When
            provided, padded tokens are excluded from both the expert count
            and the router probability sum. Named ``attention_mask`` (rather
            than ``loss_mask``) for compatibility with the HuggingFace
            ``load_balancing_loss_func`` API.
        group: Optional unified sequence-parallel process group.  When its
            world size is greater than one, the expert counts, router
            probability sums, and token count are reduced before evaluating
            the nonlinear auxiliary loss.

    Returns:
        Scalar float32 loss tensor, or ``0`` when *gate_logits* is ``None``
        or not a tuple.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    compute_device = gate_logits[0].device

    # Prepare per-token mask weights if attention_mask is provided.
    if attention_mask is not None:
        batch_size, sequence_length = attention_mask.shape
        # Expand mask to cover all layers: [num_layers, batch_size, seq_len] -> [num_layers * batch_size * seq_len]
        mask_flat = attention_mask.to(compute_device, dtype=torch.float32)
    else:
        mask_flat = None

    expert_count = torch.zeros(num_experts, device=compute_device, dtype=torch.float32)
    router_prob_sum = torch.zeros(num_experts, device=compute_device, dtype=torch.float32)
    total_weight = torch.tensor(0.0, device=compute_device)

    for layer_logits in gate_logits:
        layer_logits = layer_logits.to(compute_device)
        probs = torch.softmax(layer_logits.float(), dim=-1)  # [N_layer, E]
        _, selected = torch.topk(probs, top_k, dim=-1)  # [N_layer, top_k]

        if mask_flat is not None:
            # mask_flat is [batch_size, seq_len], layer_logits is [batch_size * seq_len, E]
            mask = mask_flat.reshape(-1)  # [N_layer]
            w = mask.unsqueeze(-1)  # [N_layer, 1]
            router_prob_sum += (probs * w).sum(dim=0)

            flat_selected = selected.reshape(-1)  # [N_layer * top_k]
            weights = mask.unsqueeze(-1).expand_as(selected).reshape(-1)
            expert_count.scatter_add_(0, flat_selected, weights)

            total_weight = total_weight + mask.sum()
        else:
            router_prob_sum += probs.sum(dim=0)

            flat_selected = selected.reshape(-1)
            ones = torch.ones_like(flat_selected, dtype=torch.float32)
            expert_count.scatter_add_(0, flat_selected, ones)

            total_weight = total_weight + layer_logits.shape[0]

    expert_count, router_prob_sum, total_weight = _global_sequence_parallel_stats(
        expert_count,
        router_prob_sum,
        total_weight,
        group,
    )

    if total_weight == 0:
        return torch.tensor(0.0, device=compute_device)

    loss = torch.dot(expert_count, router_prob_sum) * (num_experts / (total_weight * total_weight))
    return loss
