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

"""standard MoE experts MLU Apex group-gemm implementation.

Apex grouped GEMM. Apex is imported lazily so GPU hosts can register
this row without ``apex`` / ``torch_mlu``. See ByteDance-Seed/VeOmni#903.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .....distributed.parallel_state import get_parallel_state
from ..shared.swiglu import apply_swiglu_clamp


def _apex_gmm(**kwargs):
    """Call Apex grouped GEMM. Lazy-imports ``apex``."""
    from apex.contrib.grouped_gemm.ops import backend as apex_backend

    return apex_backend.gmm(**kwargs)


class MLUGroupGemm(torch.autograd.Function):
    """Split-fc1 MLU grouped GEMM with SwiGLU and fc2."""

    @staticmethod
    def forward(ctx, permute_tokens, cumsum, fc1_1_weight, fc1_2_weight, fc2_weight, swiglu_limit=None):
        """Permute tokens through split fc1, SwiGLU, and fc2."""
        batch_sizes = torch.cat([cumsum[:1], cumsum[1:] - cumsum[:-1]])
        fc1_1_output = _apex_gmm(
            a=permute_tokens, b=fc1_1_weight, batch_sizes=batch_sizes, trans_a=False, trans_b=True
        )
        fc1_2_output = _apex_gmm(
            a=permute_tokens, b=fc1_2_weight, batch_sizes=batch_sizes, trans_a=False, trans_b=True
        )
        fc1_1_output, fc1_2_output, mask_fc1_1, mask_fc1_2 = apply_swiglu_clamp(
            fc1_1_output, fc1_2_output, swiglu_limit
        )
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_output = fc1_1_activation * fc1_2_output
        fc2_output = _apex_gmm(a=fc1_output, b=fc2_weight, batch_sizes=batch_sizes, trans_a=False, trans_b=True)
        ctx.swiglu_limit = swiglu_limit
        ctx.save_for_backward(
            permute_tokens,
            cumsum,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            mask_fc1_1 if mask_fc1_1 is not None else torch.empty(0, device=permute_tokens.device),
            mask_fc1_2 if mask_fc1_2 is not None else torch.empty(0, device=permute_tokens.device),
        )
        return fc2_output

    @staticmethod
    def backward(ctx, grad_output):
        """Return grads for permute tokens and the three weight tensors."""
        (
            permute_tokens,
            cumsum,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            mask_fc1_1,
            mask_fc1_2,
        ) = ctx.saved_tensors
        swiglu_limit = ctx.swiglu_limit
        batch_sizes = torch.cat([cumsum[:1], cumsum[1:] - cumsum[:-1]])
        grad_fc1_output = _apex_gmm(a=grad_output, b=fc2_weight, batch_sizes=batch_sizes, trans_a=False, trans_b=False)
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_output = fc1_1_activation * fc1_2_output
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = _apex_gmm(
                a=grad_output, b=fc1_output, batch_sizes=batch_sizes, trans_a=True, trans_b=False
            )
        grad_fc1_2_output = fc1_1_activation * grad_fc1_output
        grad_fc1_1_activation = grad_fc1_output * fc1_2_output
        if swiglu_limit is not None:
            grad_fc1_2_output.masked_fill_(~mask_fc1_2, 0)
        grad_scatter_output_2 = _apex_gmm(
            a=grad_fc1_2_output, b=fc1_2_weight, batch_sizes=batch_sizes, trans_a=False, trans_b=False
        )
        grad_fc1_2_weight = None
        if fc1_2_weight.requires_grad:
            grad_fc1_2_weight = _apex_gmm(
                a=grad_fc1_2_output, b=permute_tokens, batch_sizes=batch_sizes, trans_a=True, trans_b=False
            )
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)
        if swiglu_limit is not None:
            grad_fc1_1_output.masked_fill_(~mask_fc1_1, 0)
        grad_scatter_output_1 = _apex_gmm(
            a=grad_fc1_1_output, b=fc1_1_weight, batch_sizes=batch_sizes, trans_a=False, trans_b=False
        )
        grad_fc1_1_weight = None
        if fc1_1_weight.requires_grad:
            grad_fc1_1_weight = _apex_gmm(
                a=grad_fc1_1_output, b=permute_tokens, batch_sizes=batch_sizes, trans_a=True, trans_b=False
            )
        return (
            grad_scatter_output_1 + grad_scatter_output_2,
            None,
            grad_fc1_1_weight,
            grad_fc1_2_weight,
            grad_fc2_weight,
            None,
        )


class MLUMergedFc1GroupGemm(torch.autograd.Function):
    """Merged-fc1 MLU grouped GEMM with SwiGLU and fc2."""

    @staticmethod
    def forward(ctx, permute_tokens, cumsum, fc1_1_2_weight, fc2_weight, swiglu_limit=None):
        """Permute tokens through merged fc1, SwiGLU, and fc2."""
        if fc1_1_2_weight.shape[1] % 2 != 0:
            raise ValueError(f"Merged fc1_1_2_weight dim 1 must be even, got {fc1_1_2_weight.shape[1]}")
        batch_sizes = torch.cat([cumsum[:1], cumsum[1:] - cumsum[:-1]])
        fc1_output = _apex_gmm(
            a=permute_tokens, b=fc1_1_2_weight, batch_sizes=batch_sizes, trans_a=False, trans_b=True
        )
        fc1_1_output, fc1_2_output = fc1_output.chunk(2, dim=-1)
        fc1_1_output, fc1_2_output, mask_fc1_1, mask_fc1_2 = apply_swiglu_clamp(
            fc1_1_output, fc1_2_output, swiglu_limit
        )
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_result = fc1_1_activation * fc1_2_output
        fc2_output = _apex_gmm(a=fc1_result, b=fc2_weight, batch_sizes=batch_sizes, trans_a=False, trans_b=True)
        ctx.swiglu_limit = swiglu_limit
        ctx.save_for_backward(
            permute_tokens,
            cumsum,
            fc1_1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            mask_fc1_1 if mask_fc1_1 is not None else torch.empty(0, device=permute_tokens.device),
            mask_fc1_2 if mask_fc1_2 is not None else torch.empty(0, device=permute_tokens.device),
        )
        return fc2_output

    @staticmethod
    def backward(ctx, grad_output):
        """Return grads for permute tokens and the two weight tensors."""
        (
            permute_tokens,
            cumsum,
            fc1_1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            mask_fc1_1,
            mask_fc1_2,
        ) = ctx.saved_tensors
        swiglu_limit = ctx.swiglu_limit
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_result = fc1_1_activation * fc1_2_output
        batch_sizes = torch.cat([cumsum[:1], cumsum[1:] - cumsum[:-1]])
        grad_fc1_result = _apex_gmm(a=grad_output, b=fc2_weight, batch_sizes=batch_sizes, trans_b=False)
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = _apex_gmm(
                a=grad_output, b=fc1_result, batch_sizes=batch_sizes, trans_a=True, trans_b=False
            )
        grad_fc1_2_output = fc1_1_activation * grad_fc1_result
        grad_fc1_1_activation = grad_fc1_result * fc1_2_output
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)
        if swiglu_limit is not None:
            grad_fc1_1_output.masked_fill_(~mask_fc1_1, 0)
            grad_fc1_2_output.masked_fill_(~mask_fc1_2, 0)
        grad_fc1_output = torch.cat([grad_fc1_1_output, grad_fc1_2_output], dim=-1)
        grad_permute_tokens = _apex_gmm(a=grad_fc1_output, b=fc1_1_2_weight, batch_sizes=batch_sizes, trans_b=False)
        grad_fc1_1_2_weight = None
        if fc1_1_2_weight.requires_grad:
            grad_fc1_1_2_weight = _apex_gmm(
                a=grad_fc1_output, b=permute_tokens, batch_sizes=batch_sizes, trans_a=True, trans_b=False
            )
        return grad_permute_tokens, None, grad_fc1_1_2_weight, grad_fc2_weight, None


def mlu_group_gemm_fused_moe_forward(
    num_experts: int,
    routing_weights: Tensor,
    selected_experts: Tensor,
    hidden_states: Tensor,
    fc1_1_weight: Tensor | None,
    fc1_2_weight: Tensor | None,
    fc2_weight: Tensor,
    fc1_1_2_weight: Tensor | None = None,
    swiglu_limit: float | None = None,
) -> Tensor:
    """MLU grouped-gemm fused MoE. Empty weights are ``None``.

    Split or merged fc1. EP comm stays outside the Function.
    """
    if get_parallel_state().ep_enabled:
        from .....distributed.moe import preprocess, token_pre_all2all, tokens_post_all2all

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
    else:
        from apex.contrib.permute import permute
        from apex.contrib.unpermute import unpermute

        permute_tokens, sorted_indice = permute(hidden_states, selected_experts, -1)
        splits = torch.bincount(selected_experts.view(-1), minlength=num_experts)
        cumsum = torch.cumsum(splits, dim=0)

    if fc1_1_2_weight is not None:
        if fc1_1_weight is not None or fc1_2_weight is not None:
            raise ValueError("Provide either split fc1 weights or merged fc1_1_2_weight, not both.")
        final_permute_tokens = MLUMergedFc1GroupGemm.apply(
            permute_tokens, cumsum, fc1_1_2_weight, fc2_weight, swiglu_limit
        )
    else:
        if fc1_1_weight is None or fc1_2_weight is None:
            raise ValueError("EP requires split fc1 weights (fc1_1_weight and fc1_2_weight).")
        final_permute_tokens = MLUGroupGemm.apply(
            permute_tokens, cumsum, fc1_1_weight, fc1_2_weight, fc2_weight, swiglu_limit
        )

    if get_parallel_state().ep_enabled:
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
    return unpermute(final_permute_tokens, sorted_indice, routing_weights)


def wrapper(
    hidden_states: Tensor,
    routing_weights: Tensor,
    selected_experts: Tensor,
    fc1_1_weight: Tensor,
    fc1_2_weight: Tensor,
    fc2_weight: Tensor,
    fc1_1_2_weight: Tensor,
    *,
    num_experts: int,
    swiglu_limit: float | None = None,
) -> Tensor:
    """Call the MLU fused MoE path. Empty weights are ``None``."""
    return mlu_group_gemm_fused_moe_forward(
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_weight if fc1_1_weight.numel() else None,
        fc1_2_weight if fc1_2_weight.numel() else None,
        fc2_weight,
        fc1_1_2_weight if fc1_1_2_weight.numel() else None,
        swiglu_limit=swiglu_limit,
    )
