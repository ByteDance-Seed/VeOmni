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

"""Quack GEMM backend for fused MoE forward/backward.

Uses ``quack.gemm_interface.gemm`` (CUTLASS/CuTe DSL, SM90+) for all GEMM
operations: forward, dgrad, and wgrad.

- Forward/dgrad: ``cu_seqlens_m`` mode — batches over the token (M) dimension.
- Wgrad: ``cu_seqlens_k`` mode — batches over the token (K) dimension,
  computing ``grad.T @ input`` per expert group.

EP (Expert Parallelism) is supported for both split and merged fc1 weights.
"""

import torch
from quack.gemm_interface import gemm

from ....distributed.moe import preprocess, token_pre_all2all, tokens_post_all2all
from ....distributed.parallel_state import get_parallel_state
from ._kernels.kernel.moe import expert_histogram, moe_gather, moe_scatter


def _build_moe_indices(expert_index: torch.Tensor, num_experts: int):
    """Build cu_seqlens_m, A_idx, and scatter_index from expert routing.

    Args:
        expert_index: [T, topk] expert assignments.
        num_experts: total number of experts.

    Returns:
        cu_seqlens_m: [E+1] cumulative token counts per expert (int32).
        A_idx: [T*topk] token indices sorted by expert assignment (int32).
        scatter_index: [T, topk] indices for moe_gather/moe_scatter (int32).
    """
    topk = expert_index.shape[1]
    flat = expert_index.flatten()
    sorted_order = flat.argsort(stable=True)
    scatter_index = sorted_order.argsort().int().view(expert_index.shape)
    # A_idx maps expert-sorted positions to original token indices (0..T-1).
    # sorted_order values are flat indices (t*topk + k), so integer-divide by topk.
    A_idx = (sorted_order // topk).int()

    splits = expert_histogram(expert_index, num_experts)
    cu_seqlens_m = torch.zeros(num_experts + 1, dtype=torch.int32, device=expert_index.device)
    cu_seqlens_m[1:] = torch.cumsum(splits, dim=0).int()

    return cu_seqlens_m, A_idx, scatter_index


class QuackFusedMoeExpertFunction(torch.autograd.Function):
    """Fused MoE with split fc1 weights using quack GEMM."""

    @staticmethod
    def forward(
        ctx,
        num_experts,
        gate_weights,
        expert_index,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
    ):
        cu_seqlens_m, A_idx, scatter_index = _build_moe_indices(expert_index, num_experts)

        # Transpose weights for forward: [E, N, K] -> [E, K, N] (view, no copy).
        # quack internally transposes back before calling CUTLASS kernels,
        # so no .contiguous() call is needed here.
        fc1_1_w_t = fc1_1_weight.transpose(1, 2)
        fc1_2_w_t = fc1_2_weight.transpose(1, 2)
        fc2_w_t = fc2_weight.transpose(1, 2)

        # fc1_1: [T*topk, I] (expert-sorted via A_idx)
        fc1_1_output = gemm(hidden_states, fc1_1_w_t, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx)

        # fc1_2: [T*topk, I]
        fc1_2_output = gemm(hidden_states, fc1_2_w_t, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx)

        # SiLU activation + gate multiply
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_activation = fc1_1_activation * fc1_2_output

        # Apply routing weights.
        # Note: A_idx alone cannot replace scatter_index here because A_idx only
        # carries token indices (sorted_order // topk) but not the topk-slot index,
        # so it cannot address into gate_weights[T, topk] without extra bookkeeping.
        reshaped_gate_weight = gate_weights.reshape(-1, 1)
        scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
        scattered_gate_weight[scatter_index.flatten()] = reshaped_gate_weight

        fc1_weighted_output = fc1_activation * scattered_gate_weight

        # fc2: input is already expert-sorted, no A_idx needed
        fc2_output = gemm(fc1_weighted_output, fc2_w_t, cu_seqlens_m=cu_seqlens_m)

        # Gather output tokens back to original order
        expert_output = moe_gather(fc2_output, scatter_index)
        del fc2_output
        output = expert_output.reshape(hidden_states.shape)

        ctx.num_experts = num_experts
        ctx.save_for_backward(
            gate_weights,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            cu_seqlens_m,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            gate_weights,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            cu_seqlens_m,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
        ) = ctx.saved_tensors
        hidden_dim = grad_output.shape[-1]
        grad_output = grad_output.view(-1, hidden_dim)

        # Step 10: scatter grad to expert-sorted order
        grad_fc2_output = moe_scatter(grad_output, scatter_index)

        # Step 9 dgrad: grad @ fc2_weight (original layout [E, H, I] is already [K, N] for quack)
        grad_fc1_weighted_output = gemm(grad_fc2_output, fc2_weight, cu_seqlens_m=cu_seqlens_m)

        # Step 9 wgrad: grad_fc2_output.T @ fc1_weighted_output → [E, H, I]
        # cu_seqlens_k mode: A=[M, total_K] @ B=[total_K, N] → [L, M, N] per expert group.
        # Pass .T view (not .T.contiguous()) — quack varlen_k requires A to be m-major.
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = gemm(grad_fc2_output.T, fc1_weighted_output, cu_seqlens_k=cu_seqlens_m)
        del fc1_weighted_output

        # Step 8-2: routing weight backward
        grad_fc1_activation = grad_fc1_weighted_output * scattered_gate_weight
        del scattered_gate_weight

        # Step 8-1: gate weight backward
        grad_scattered_gate_weight = torch.sum(fc1_activation * grad_fc1_weighted_output, dim=-1)
        del fc1_activation
        grad_gate_weight = grad_scattered_gate_weight[scatter_index.flatten()]
        del grad_scattered_gate_weight
        grad_gate_weight = grad_gate_weight.reshape(gate_weights.shape)

        # Recompute SiLU activation
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

        # Step 7
        grad_fc1_1_activation = grad_fc1_activation * fc1_2_output
        del fc1_2_output
        grad_fc1_2_output = fc1_1_activation * grad_fc1_activation
        del grad_fc1_activation, fc1_1_activation

        # Step 6 dgrad: fc1_2_weight [E, I, H] is already [K, N] for quack
        grad_scatter_output_2 = gemm(grad_fc1_2_output, fc1_2_weight, cu_seqlens_m=cu_seqlens_m)

        # Step 5: SiLU backward
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)
        del fc1_1_output

        # Step 4 dgrad: fc1_1_weight [E, I, H] is already [K, N] for quack
        grad_scatter_output_1 = gemm(grad_fc1_1_output, fc1_1_weight, cu_seqlens_m=cu_seqlens_m)

        # Recompute scatter_output for wgrad
        scatter_output = moe_scatter(hidden_states, scatter_index)

        # Step 6 wgrad: grad_fc1_2_output.T @ scatter_output → [E, I, H]
        grad_fc1_2_weight = None
        if fc1_2_weight.requires_grad:
            grad_fc1_2_weight = gemm(grad_fc1_2_output.T, scatter_output, cu_seqlens_k=cu_seqlens_m)
        del grad_fc1_2_output

        # Step 4 wgrad: grad_fc1_1_output.T @ scatter_output → [E, I, H]
        grad_fc1_1_weight = None
        if fc1_1_weight.requires_grad:
            grad_fc1_1_weight = gemm(grad_fc1_1_output.T, scatter_output, cu_seqlens_k=cu_seqlens_m)
        del grad_fc1_1_output, scatter_output

        # Step 3: gather gradients back to original token order
        grad_scatter_output = grad_scatter_output_1 + grad_scatter_output_2
        del grad_scatter_output_1, grad_scatter_output_2
        grad_hidden_states = moe_gather(grad_scatter_output, scatter_index)
        grad_hidden_states = grad_hidden_states.reshape(hidden_states.shape)

        return (
            None,  # num_experts
            grad_gate_weight,  # gate_weights
            None,  # expert_index
            grad_hidden_states,  # hidden_states
            grad_fc1_1_weight,  # fc1_1_weight
            grad_fc1_2_weight,  # fc1_2_weight
            grad_fc2_weight,  # fc2_weight
        )


class MergedFc1QuackFusedMoeExpertFunction(torch.autograd.Function):
    """Fused MoE with merged fc1_1_2 weight [E, 2I, H] using quack GEMM."""

    @staticmethod
    def forward(
        ctx,
        num_experts,
        gate_weights,
        expert_index,
        hidden_states,
        fc1_1_2_weight,
        fc2_weight,
    ):
        cu_seqlens_m, A_idx, scatter_index = _build_moe_indices(expert_index, num_experts)

        # Transpose weights for forward: [E, N, K] -> [E, K, N] (view, no copy).
        # quack internally transposes back before calling CUTLASS kernels,
        # so no .contiguous() call is needed here.
        fc1_1_2_w_t = fc1_1_2_weight.transpose(1, 2)
        fc2_w_t = fc2_weight.transpose(1, 2)

        # Single fc1 GEMM: output [T*topk, 2I]
        fc1_output = gemm(hidden_states, fc1_1_2_w_t, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx)

        fc1_1_output, fc1_2_output = fc1_output.chunk(2, dim=-1)

        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_activation = fc1_1_activation * fc1_2_output

        reshaped_gate_weight = gate_weights.reshape(-1, 1)
        scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
        scattered_gate_weight[scatter_index.flatten()] = reshaped_gate_weight

        fc1_weighted_output = fc1_activation * scattered_gate_weight

        fc2_output = gemm(fc1_weighted_output, fc2_w_t, cu_seqlens_m=cu_seqlens_m)

        expert_output = moe_gather(fc2_output, scatter_index)
        del fc2_output
        output = expert_output.reshape(hidden_states.shape)

        ctx.num_experts = num_experts
        ctx.save_for_backward(
            gate_weights,
            fc1_1_2_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            cu_seqlens_m,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            gate_weights,
            fc1_1_2_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            cu_seqlens_m,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
        ) = ctx.saved_tensors
        hidden_dim = grad_output.shape[-1]
        grad_output = grad_output.view(-1, hidden_dim)

        # Step 10
        grad_fc2_output = moe_scatter(grad_output, scatter_index)

        # Step 9 dgrad
        grad_fc1_weighted_output = gemm(grad_fc2_output, fc2_weight, cu_seqlens_m=cu_seqlens_m)

        # Step 9 wgrad: grad_fc2_output.T @ fc1_weighted_output → [E, H, I]
        # cu_seqlens_k mode: A=[M, total_K] @ B=[total_K, N] → [L, M, N] per expert group.
        # Pass .T view (not .T.contiguous()) — quack varlen_k requires A to be m-major.
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = gemm(grad_fc2_output.T, fc1_weighted_output, cu_seqlens_k=cu_seqlens_m)
        del fc1_weighted_output

        # Step 8-2
        grad_fc1_activation = grad_fc1_weighted_output * scattered_gate_weight
        del scattered_gate_weight

        # Step 8-1
        grad_scattered_gate_weight = torch.sum(fc1_activation * grad_fc1_weighted_output, dim=-1)
        del fc1_activation
        grad_gate_weight = grad_scattered_gate_weight[scatter_index.flatten()]
        del grad_scattered_gate_weight
        grad_gate_weight = grad_gate_weight.reshape(gate_weights.shape)

        # Recompute
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

        # Step 7
        grad_fc1_1_activation = grad_fc1_activation * fc1_2_output
        del fc1_2_output
        grad_fc1_2_output = fc1_1_activation * grad_fc1_activation
        del grad_fc1_activation, fc1_1_activation

        # Step 5
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)
        del fc1_1_output

        # Merge grads back to [T, 2I]
        grad_fc1_output = torch.cat([grad_fc1_1_output, grad_fc1_2_output], dim=-1)
        del grad_fc1_1_output, grad_fc1_2_output

        # Step 4 dgrad: fc1_1_2_weight [E, 2I, H] is [K, N] for quack
        grad_scatter_output = gemm(grad_fc1_output, fc1_1_2_weight, cu_seqlens_m=cu_seqlens_m)

        # Step 4 wgrad: grad_fc1_output.T @ scatter_output → [E, 2I, H]
        grad_fc1_1_2_weight = None
        if fc1_1_2_weight.requires_grad:
            scatter_output = moe_scatter(hidden_states, scatter_index)
            grad_fc1_1_2_weight = gemm(grad_fc1_output.T, scatter_output, cu_seqlens_k=cu_seqlens_m)
            del scatter_output
        del grad_fc1_output

        # Step 3
        grad_hidden_states = moe_gather(grad_scatter_output, scatter_index)
        del grad_scatter_output
        grad_hidden_states = grad_hidden_states.reshape(hidden_states.shape)

        return (
            None,  # num_experts
            grad_gate_weight,  # gate_weights
            None,  # expert_index
            grad_hidden_states,  # hidden_states
            grad_fc1_1_2_weight,  # fc1_1_2_weight
            grad_fc2_weight,  # fc2_weight
        )


def _cumsum_to_cu_seqlens(cumsum: torch.Tensor) -> torch.Tensor:
    """Convert [E] cumsum to [E+1] cu_seqlens_m with leading zero (int32)."""
    zero = torch.zeros(1, dtype=torch.int32, device=cumsum.device)
    return torch.cat([zero, cumsum.int()])


class EPQuackGroupGemm(torch.autograd.Function):
    """EP autograd function with split fc1 weights using quack GEMM."""

    @staticmethod
    def forward(
        ctx,
        permute_tokens,
        cumsum,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
    ):
        cu_seqlens_m = _cumsum_to_cu_seqlens(cumsum)

        fc1_1_w_t = fc1_1_weight.transpose(1, 2)
        fc1_2_w_t = fc1_2_weight.transpose(1, 2)
        fc2_w_t = fc2_weight.transpose(1, 2)

        fc1_1_output = gemm(permute_tokens, fc1_1_w_t, cu_seqlens_m=cu_seqlens_m)
        fc1_2_output = gemm(permute_tokens, fc1_2_w_t, cu_seqlens_m=cu_seqlens_m)

        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_result = fc1_1_activation * fc1_2_output

        fc2_output = gemm(fc1_result, fc2_w_t, cu_seqlens_m=cu_seqlens_m)

        ctx.save_for_backward(
            permute_tokens,
            cumsum,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
        )

        return fc2_output

    @staticmethod
    def backward(ctx, grad_output):
        (
            permute_tokens,
            cumsum,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
        ) = ctx.saved_tensors

        cu_seqlens_m = _cumsum_to_cu_seqlens(cumsum)

        # recompute
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_result = fc1_1_activation * fc1_2_output

        # dgrad fc2
        grad_fc1_result = gemm(grad_output, fc2_weight, cu_seqlens_m=cu_seqlens_m)

        # wgrad fc2
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = gemm(grad_output.T, fc1_result, cu_seqlens_k=cu_seqlens_m)
        del fc1_result

        # gate gradients
        grad_fc1_2_output = fc1_1_activation * grad_fc1_result
        grad_fc1_1_activation = grad_fc1_result * fc1_2_output
        del fc1_1_activation

        # dgrad fc1_2
        grad_scatter_output_2 = gemm(grad_fc1_2_output, fc1_2_weight, cu_seqlens_m=cu_seqlens_m)

        # wgrad fc1_2
        grad_fc1_2_weight = None
        if fc1_2_weight.requires_grad:
            grad_fc1_2_weight = gemm(grad_fc1_2_output.T, permute_tokens, cu_seqlens_k=cu_seqlens_m)
        del grad_fc1_2_output

        # silu backward
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)

        # dgrad fc1_1
        grad_scatter_output_1 = gemm(grad_fc1_1_output, fc1_1_weight, cu_seqlens_m=cu_seqlens_m)

        # wgrad fc1_1
        grad_fc1_1_weight = None
        if fc1_1_weight.requires_grad:
            grad_fc1_1_weight = gemm(grad_fc1_1_output.T, permute_tokens, cu_seqlens_k=cu_seqlens_m)
        del grad_fc1_1_output

        grad_permute_tokens = grad_scatter_output_1 + grad_scatter_output_2
        del grad_scatter_output_1, grad_scatter_output_2

        return (
            grad_permute_tokens,  # permute_tokens
            None,  # cumsum
            grad_fc1_1_weight,  # fc1_1_weight
            grad_fc1_2_weight,  # fc1_2_weight
            grad_fc2_weight,  # fc2_weight
        )


class EPMergedFc1QuackGroupGemm(torch.autograd.Function):
    """EP autograd function with merged fc1_1_2 weight [E, 2I, H] using quack GEMM."""

    @staticmethod
    def forward(
        ctx,
        permute_tokens,
        cumsum,
        fc1_1_2_weight,
        fc2_weight,
    ):
        assert fc1_1_2_weight.shape[1] % 2 == 0, (
            f"Merged fc1_1_2_weight dim 1 must be even, got {fc1_1_2_weight.shape[1]}"
        )
        cu_seqlens_m = _cumsum_to_cu_seqlens(cumsum)

        fc1_1_2_w_t = fc1_1_2_weight.transpose(1, 2)
        fc2_w_t = fc2_weight.transpose(1, 2)

        # Single fc1 GEMM: output [T, 2I]
        fc1_output = gemm(permute_tokens, fc1_1_2_w_t, cu_seqlens_m=cu_seqlens_m)

        # chunk is a view, no copy
        fc1_1_output, fc1_2_output = fc1_output.chunk(2, dim=-1)

        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_result = fc1_1_activation * fc1_2_output

        # fc2
        fc2_output = gemm(fc1_result, fc2_w_t, cu_seqlens_m=cu_seqlens_m)

        ctx.save_for_backward(
            permute_tokens,
            cumsum,
            fc1_1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
        )

        return fc2_output

    @staticmethod
    def backward(ctx, grad_output):
        (
            permute_tokens,
            cumsum,
            fc1_1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
        ) = ctx.saved_tensors

        cu_seqlens_m = _cumsum_to_cu_seqlens(cumsum)

        # recompute
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_result = fc1_1_activation * fc1_2_output

        # dgrad fc2: fc2_weight [E, H, I] is [K, N] for quack
        grad_fc1_result = gemm(grad_output, fc2_weight, cu_seqlens_m=cu_seqlens_m)

        # wgrad fc2
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = gemm(grad_output.T, fc1_result, cu_seqlens_k=cu_seqlens_m)
        del fc1_result

        # gate gradients
        grad_fc1_2_output = fc1_1_activation * grad_fc1_result
        grad_fc1_1_activation = grad_fc1_result * fc1_2_output
        del fc1_1_activation, fc1_2_output

        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)
        del fc1_1_output

        # Merge grads back to [T, 2I]
        grad_fc1_output = torch.cat([grad_fc1_1_output, grad_fc1_2_output], dim=-1)
        del grad_fc1_1_output, grad_fc1_2_output

        # single dgrad for merged fc1: fc1_1_2_weight [E, 2I, H] is [K, N] for quack
        grad_permute_tokens = gemm(grad_fc1_output, fc1_1_2_weight, cu_seqlens_m=cu_seqlens_m)

        # single wgrad for merged fc1
        grad_fc1_1_2_weight = None
        if fc1_1_2_weight.requires_grad:
            grad_fc1_1_2_weight = gemm(grad_fc1_output.T, permute_tokens, cu_seqlens_k=cu_seqlens_m)
        del grad_fc1_output

        return (
            grad_permute_tokens,  # permute_tokens
            None,  # cumsum
            grad_fc1_1_2_weight,  # fc1_1_2_weight
            grad_fc2_weight,  # fc2_weight
        )


def quack_gemm_fused_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor | None,
    fc1_2_weight: torch.Tensor | None,
    fc2_weight: torch.Tensor,
    fc1_1_2_weight: torch.Tensor | None = None,
):
    """Quack GEMM fused MoE forward pass.

    Same interface as ``group_gemm_fused_moe_forward``. Supports both split
    and merged fc1 weight layouts, including EP (Expert Parallelism).
    """
    if get_parallel_state().ep_enabled:
        if fc1_1_2_weight is not None:
            if fc1_1_weight is not None or fc1_2_weight is not None:
                raise ValueError("Provide either split fc1 weights or merged fc1_1_2_weight, not both.")
        else:
            if fc1_1_weight is None or fc1_2_weight is None:
                raise ValueError("EP requires split fc1 weights (fc1_1_weight and fc1_2_weight).")

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
        input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert = (
            preprocess(
                expert_mask=expert_mask,
                num_experts=num_experts,
                ep_group=get_parallel_state().ep_group,
            )
        )
        permute_tokens, routing_map, local_input_permutation_mapping, org_hidden_states_shape = token_pre_all2all(
            hidden_states=hidden_states,
            expert_mask=expert_mask,
            num_experts=num_experts,
            input_splits=input_splits,
            output_splits=output_splits,
            num_global_tokens_per_local_expert=num_global_tokens_per_local_expert,
            ep_group=get_parallel_state().ep_group,
        )

        cumsum = torch.cumsum(num_global_sum_tokens_per_local_expert, dim=0).to(permute_tokens.device)

        if fc1_1_2_weight is not None:
            final_permute_tokens = EPMergedFc1QuackGroupGemm.apply(
                permute_tokens,
                cumsum,
                fc1_1_2_weight,
                fc2_weight,
            )
        else:
            final_permute_tokens = EPQuackGroupGemm.apply(
                permute_tokens,
                cumsum,
                fc1_1_weight,
                fc1_2_weight,
                fc2_weight,
            )

        final_hidden_states = tokens_post_all2all(
            expert_outputs=final_permute_tokens,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            num_experts=num_experts,
            input_splits=input_splits,
            output_splits=output_splits,
            num_global_tokens_per_local_expert=num_global_tokens_per_local_expert,
            routing_map=routing_map,
            local_input_permutation_mapping=local_input_permutation_mapping,
            org_hidden_states_shape=org_hidden_states_shape,
            ep_group=get_parallel_state().ep_group,
        )
        return final_hidden_states

    if fc1_1_2_weight is not None:
        if fc1_1_weight is not None or fc1_2_weight is not None:
            raise ValueError("Provide either split fc1 weights or merged fc1_1_2_weight, not both.")
        return MergedFc1QuackFusedMoeExpertFunction.apply(
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            fc1_1_2_weight,
            fc2_weight,
        )
    else:
        if fc1_1_weight is None or fc1_2_weight is None:
            raise ValueError("Split fc1 mode requires both fc1_1_weight and fc1_2_weight.")
        return QuackFusedMoeExpertFunction.apply(
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
        )
