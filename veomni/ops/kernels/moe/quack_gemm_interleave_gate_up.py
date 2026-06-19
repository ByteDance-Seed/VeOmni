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

"""Quack GEMM backend for GPT-OSS interleaved gate/up expert layout."""

import torch
from quack.gemm_interface import gemm

from ....distributed.moe import preprocess, token_pre_all2all, tokens_post_all2all
from ....distributed.parallel_state import get_parallel_state
from ._kernels.kernel.moe import moe_gather, moe_scatter
from .quack_gemm import _build_moe_indices, _cumsum_to_cu_seqlens


def _segment_sum(values: torch.Tensor, cu_seqlens_m: torch.Tensor) -> torch.Tensor:
    return torch.segment_reduce(values, "sum", axis=0, offsets=cu_seqlens_m.to(torch.long))


def _assert_contiguous(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_contiguous():
        raise RuntimeError(f"{name} must be contiguous before passing a transposed view to quack GEMM")


def _gpt_oss_mlp_activation(gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    gate = gate_up[..., ::2].clamp(max=limit)
    up = gate_up[..., 1::2].clamp(min=-limit, max=limit)
    return ((up + 1) * (gate * torch.sigmoid(gate * alpha))).contiguous()


def _gpt_oss_mlp_activation_backward(
    grad_activation: torch.Tensor,
    gate_up: torch.Tensor,
    alpha: float,
    limit: float,
) -> torch.Tensor:
    gate_raw = gate_up[..., ::2]
    up_raw = gate_up[..., 1::2]
    gate = gate_raw.clamp(max=limit)
    up = up_raw.clamp(min=-limit, max=limit)

    sigmoid_gate = torch.sigmoid(gate * alpha)
    grad_gate = grad_activation * (up + 1) * (sigmoid_gate + alpha * gate * sigmoid_gate * (1 - sigmoid_gate))
    grad_gate = grad_gate.masked_fill(gate_raw > limit, 0)

    grad_up = grad_activation * gate * sigmoid_gate
    grad_up = grad_up.masked_fill((up_raw < -limit) | (up_raw > limit), 0)

    grad_gate_up = torch.empty_like(gate_up)
    grad_gate_up[..., ::2] = grad_gate
    grad_gate_up[..., 1::2] = grad_up
    return grad_gate_up


class GptOssQuackFusedMoeExpertFunction(torch.autograd.Function):
    """GPT-OSS MoE with interleaved gate/up weights using quack GEMM."""

    @staticmethod
    def forward(
        ctx,
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        gate_up_proj,
        gate_up_proj_bias,
        down_proj,
        down_proj_bias,
        alpha,
        limit,
    ):
        cu_seqlens_m, A_idx, scatter_index = _build_moe_indices(selected_experts, num_experts)

        gate_up = gemm(
            hidden_states,
            gate_up_proj,
            bias=gate_up_proj_bias,
            cu_seqlens_m=cu_seqlens_m,
            A_idx=A_idx,
        )
        fc1_activation = _gpt_oss_mlp_activation(gate_up, alpha, limit)

        fc2_output = gemm(fc1_activation, down_proj, bias=down_proj_bias, cu_seqlens_m=cu_seqlens_m)

        reshaped_gate_weight = routing_weights.to(hidden_states.dtype).reshape(-1, 1)
        scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
        scattered_gate_weight[scatter_index.flatten()] = reshaped_gate_weight
        fc2_weighted_output = fc2_output * scattered_gate_weight

        expert_output = moe_gather(fc2_weighted_output, scatter_index)
        output = expert_output.reshape(hidden_states.shape)

        ctx.num_experts = num_experts
        ctx.alpha = alpha
        ctx.limit = limit
        ctx.save_for_backward(
            routing_weights,
            selected_experts,
            hidden_states,
            gate_up_proj,
            down_proj,
            scatter_index,
            cu_seqlens_m,
            gate_up,
            fc1_activation,
            fc2_output,
            scattered_gate_weight,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            routing_weights,
            selected_experts,
            hidden_states,
            gate_up_proj,
            down_proj,
            scatter_index,
            cu_seqlens_m,
            gate_up,
            fc1_activation,
            fc2_output,
            scattered_gate_weight,
        ) = ctx.saved_tensors
        hidden_dim = grad_output.shape[-1]
        grad_output = grad_output.view(-1, hidden_dim)

        grad_fc2_weighted_output = moe_scatter(grad_output, scatter_index)

        grad_scattered_gate_weight = torch.sum(fc2_output * grad_fc2_weighted_output, dim=-1)
        grad_routing_weights = grad_scattered_gate_weight[scatter_index.flatten()].reshape(routing_weights.shape)

        grad_fc2_output = grad_fc2_weighted_output * scattered_gate_weight
        grad_down_proj_bias = _segment_sum(grad_fc2_output, cu_seqlens_m)

        _assert_contiguous(down_proj, "down_proj")
        grad_fc1_activation = gemm(grad_fc2_output, down_proj.transpose(1, 2), cu_seqlens_m=cu_seqlens_m)
        grad_down_proj = None
        if down_proj.requires_grad:
            _assert_contiguous(fc1_activation, "fc1_activation")
            grad_down_proj = gemm(fc1_activation.T, grad_fc2_output, cu_seqlens_k=cu_seqlens_m)

        grad_gate_up = _gpt_oss_mlp_activation_backward(grad_fc1_activation, gate_up, ctx.alpha, ctx.limit)
        grad_gate_up_proj_bias = _segment_sum(grad_gate_up, cu_seqlens_m)

        scatter_output = moe_scatter(hidden_states, scatter_index)
        _assert_contiguous(gate_up_proj, "gate_up_proj")
        grad_scatter_output = gemm(grad_gate_up, gate_up_proj.transpose(1, 2), cu_seqlens_m=cu_seqlens_m)
        grad_gate_up_proj = None
        if gate_up_proj.requires_grad:
            _assert_contiguous(scatter_output, "scatter_output")
            grad_gate_up_proj = gemm(scatter_output.T, grad_gate_up, cu_seqlens_k=cu_seqlens_m)
        del scatter_output

        grad_hidden_states = moe_gather(grad_scatter_output, scatter_index).reshape(hidden_states.shape)

        return (
            None,  # num_experts
            grad_routing_weights,
            None,  # selected_experts
            grad_hidden_states,
            grad_gate_up_proj,
            grad_gate_up_proj_bias,
            grad_down_proj,
            grad_down_proj_bias,
            None,  # alpha
            None,  # limit
        )


class EPGptOssQuackGroupGemm(torch.autograd.Function):
    """EP GPT-OSS MoE expert GEMMs with quack."""

    @staticmethod
    def forward(
        ctx,
        permute_tokens,
        cumsum,
        gate_up_proj,
        gate_up_proj_bias,
        down_proj,
        down_proj_bias,
        alpha,
        limit,
    ):
        cu_seqlens_m = _cumsum_to_cu_seqlens(cumsum)

        gate_up = gemm(permute_tokens, gate_up_proj, bias=gate_up_proj_bias, cu_seqlens_m=cu_seqlens_m)
        fc1_activation = _gpt_oss_mlp_activation(gate_up, alpha, limit)
        fc2_output = gemm(fc1_activation, down_proj, bias=down_proj_bias, cu_seqlens_m=cu_seqlens_m)

        ctx.alpha = alpha
        ctx.limit = limit
        ctx.save_for_backward(
            permute_tokens,
            cumsum,
            gate_up_proj,
            down_proj,
            gate_up,
            fc1_activation,
        )
        return fc2_output

    @staticmethod
    def backward(ctx, grad_output):
        (
            permute_tokens,
            cumsum,
            gate_up_proj,
            down_proj,
            gate_up,
            fc1_activation,
        ) = ctx.saved_tensors
        cu_seqlens_m = _cumsum_to_cu_seqlens(cumsum)

        grad_down_proj_bias = _segment_sum(grad_output, cu_seqlens_m)
        _assert_contiguous(down_proj, "down_proj")
        grad_fc1_activation = gemm(grad_output, down_proj.transpose(1, 2), cu_seqlens_m=cu_seqlens_m)
        grad_down_proj = None
        if down_proj.requires_grad:
            _assert_contiguous(fc1_activation, "fc1_activation")
            grad_down_proj = gemm(fc1_activation.T, grad_output, cu_seqlens_k=cu_seqlens_m)

        grad_gate_up = _gpt_oss_mlp_activation_backward(grad_fc1_activation, gate_up, ctx.alpha, ctx.limit)
        grad_gate_up_proj_bias = _segment_sum(grad_gate_up, cu_seqlens_m)

        _assert_contiguous(gate_up_proj, "gate_up_proj")
        grad_permute_tokens = gemm(grad_gate_up, gate_up_proj.transpose(1, 2), cu_seqlens_m=cu_seqlens_m)
        grad_gate_up_proj = None
        if gate_up_proj.requires_grad:
            _assert_contiguous(permute_tokens, "permute_tokens")
            grad_gate_up_proj = gemm(permute_tokens.T, grad_gate_up, cu_seqlens_k=cu_seqlens_m)

        return (
            grad_permute_tokens,
            None,  # cumsum
            grad_gate_up_proj,
            grad_gate_up_proj_bias,
            grad_down_proj,
            grad_down_proj_bias,
            None,  # alpha
            None,  # limit
        )


def quack_gemm_gpt_oss_fused_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    gate_up_proj_bias: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_bias: torch.Tensor,
    alpha: float = 1.702,
    limit: float = 7.0,
):
    """GPT-OSS quack MoE forward with native interleaved gate/up layout."""
    if get_parallel_state().ep_enabled:
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
        final_permute_tokens = EPGptOssQuackGroupGemm.apply(
            permute_tokens,
            cumsum,
            gate_up_proj,
            gate_up_proj_bias,
            down_proj,
            down_proj_bias,
            alpha,
            limit,
        )

        return tokens_post_all2all(
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

    return GptOssQuackFusedMoeExpertFunction.apply(
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        gate_up_proj,
        gate_up_proj_bias,
        down_proj,
        down_proj_bias,
        alpha,
        limit,
    )
