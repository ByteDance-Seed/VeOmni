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


"""standard MoE experts npu implementation."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from .....distributed.parallel_state import get_parallel_state


class GmmFunction(torch.autograd.Function):
    """NPU grouped matmul with dgrad and wgrad."""

    @staticmethod
    def forward(ctx, x, weight, group_list):
        """Grouped matmul ``x @ weight`` per expert."""
        import torch_npu

        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list

        fwd_output = torch_npu.npu_grouped_matmul(
            [x], [weight], bias=None, group_list=group_list, split_item=2, group_type=0, group_list_type=1
        )[0]
        return fwd_output

    @staticmethod
    def backward(ctx, grad_output):
        """Grouped dgrad and wgrad via ``npu_grouped_matmul``."""
        import torch_npu

        input_tensor, weight = ctx.saved_tensors
        group_list = ctx.group_list

        weight = torch.transpose(weight, 1, 2)
        grad_input = torch_npu.npu_grouped_matmul(
            [grad_output], [weight], bias=None, group_list=group_list, split_item=2, group_type=0, group_list_type=1
        )[0]

        grad_weight = torch_npu.npu_grouped_matmul(
            [input_tensor.T],
            [grad_output],
            bias=None,
            group_list=group_list,
            split_item=3,
            group_type=2,
            group_list_type=1,
        )[0]

        return grad_input, grad_weight, None


def npu_group_gemm(x, weight, group_list):
    """Apply ``GmmFunction`` to ``(x, weight, group_list)``."""
    output = GmmFunction.apply(x, weight, group_list)
    return output


def _clamped_swiglu(x: torch.Tensor, limit: float) -> torch.Tensor:
    """gpt-oss-style clamped SwiGLU (DeepSeek-V4).

    ``torch_npu.npu_swiglu`` is a fused kernel with no clamp support, so this
    manual (unfused) path is only taken when ``swiglu_limit`` is set -- today
    that is exclusively DeepSeek-V4's ``PatchedDeepseekV4Experts`` (see
    ``deepseek_v4_gpu_patch_gen_config.py``). Chunk convention and clamp
    bounds mirror that class's eager ``_apply_gate`` exactly: first half of
    the last dim is ``gate`` (clamped to ``max=limit``), second half is ``up``
    (clamped to ``[-limit, limit]``), activation is SiLU (DeepSeek-V4's
    ``config.hidden_act``).
    """
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


def _fc1_weight(
    fc1_1_weight: torch.Tensor | None,
    fc1_2_weight: torch.Tensor | None,
    fc1_1_2_weight: torch.Tensor | None,
) -> torch.Tensor:
    """Resolve split vs merged fc1. Same contract as the eager wrapper."""
    has_merged = fc1_1_2_weight is not None
    has_split = fc1_1_weight is not None or fc1_2_weight is not None
    if has_split == has_merged:
        raise ValueError("Provide either split fc1 weights or merged fc1_1_2_weight, not both or neither.")
    if has_merged:
        return fc1_1_2_weight
    if fc1_1_weight is None or fc1_2_weight is None:
        raise ValueError("Split fc1 mode requires both fc1_1_weight and fc1_2_weight.")
    return torch.cat([fc1_1_weight, fc1_2_weight], dim=1)


def _swiglu(x: torch.Tensor, swiglu_limit: float | None) -> torch.Tensor:
    """NPU SwiGLU, or the clamped path when ``swiglu_limit`` is set."""
    import torch_npu

    if swiglu_limit is None:
        return torch_npu.npu_swiglu(x, dim=-1)
    return _clamped_swiglu(x, swiglu_limit)


def _npu_fused_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor | None,
    fc1_2_weight: torch.Tensor | None,
    fc2_weight: torch.Tensor,
    fc1_1_2_weight: torch.Tensor | None = None,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """NPU single-device fused MoE forward pass (non-EP).

    Accepts either split fc1 weights or a merged fc1_1_2_weight tensor.
    Weights are merged and transposed for the NPU group-gemm kernel.
    """
    import torch_npu

    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(
        hidden_states, selected_experts.to(torch.int32)
    )
    tokens_per_expert = torch.histc(selected_experts, bins=num_experts, min=0, max=num_experts)

    fc1_weight = _fc1_weight(fc1_1_weight, fc1_2_weight, fc1_1_2_weight).transpose(1, 2)
    intermediate_hidden_states = npu_group_gemm(permuted_hidden_states, fc1_weight, tokens_per_expert)
    intermediate_activations = _swiglu(intermediate_hidden_states, swiglu_limit)
    output = npu_group_gemm(intermediate_activations, fc2_weight.transpose(1, 2), tokens_per_expert)
    # NPU unpermute applies routing after fc2. Equivalent for a biasless down
    # projection; bf16 rounding can differ from the pre-fc2 fused order.
    hidden_states = torch_npu.npu_moe_token_unpermute(output, row_ids_map, probs=routing_weights)
    return hidden_states


def npu_ep_fused_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor | None,
    fc1_2_weight: torch.Tensor | None,
    fc2_weight: torch.Tensor,
    fc1_1_2_weight: torch.Tensor | None = None,
    ep_group: dist.ProcessGroup | None = None,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """NPU expert-parallel fused MoE forward pass.

    Accepts either split fc1 weights or a merged fc1_1_2_weight tensor.
    Handles alltoall dispatch/combine for expert parallelism.
    """
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert = (
        dispatch_preprocess(selected_experts, num_experts, ep_group)
    )
    hidden_states, unpermute_indices = alltoall_dispatch(
        hidden_states,
        selected_experts,
        input_splits,
        output_splits,
        num_experts,
        num_global_tokens_per_local_expert,
        ep_group,
    )

    fc1_weight = _fc1_weight(fc1_1_weight, fc1_2_weight, fc1_1_2_weight).transpose(1, 2)
    intermediate_hidden_states = npu_group_gemm(hidden_states, fc1_weight, num_global_sum_tokens_per_local_expert)
    intermediate_activations = _swiglu(intermediate_hidden_states, swiglu_limit)
    hidden_states = npu_group_gemm(
        intermediate_activations, fc2_weight.transpose(1, 2), num_global_sum_tokens_per_local_expert
    )

    hidden_states = alltoall_combine(
        hidden_states,
        routing_weights,
        unpermute_indices,
        input_splits,
        output_splits,
        num_experts,
        num_global_tokens_per_local_expert,
        ep_group,
    )
    return hidden_states


def dispatch_preprocess(
    selected_experts: torch.Tensor,
    num_global_experts: int,
    ep_group: dist.ProcessGroup | None = None,
):
    """Build EP split sizes and per-expert token counts."""
    if ep_group is None:
        ep_size = 1
        ep_rank = 0
    else:
        ep_size = dist.get_world_size(ep_group)
        ep_rank = dist.get_rank(ep_group)
    assert num_global_experts % ep_size == 0, (
        f"Number of experts ({num_global_experts}) must be divisible by expert parallel size ({ep_size})."
    )
    num_local_experts = num_global_experts // ep_size

    num_local_tokens_per_expert = torch.bincount(selected_experts.view(-1), minlength=num_global_experts)

    if ep_group is None or ep_size <= 1:
        num_global_tokens_per_expert = num_local_tokens_per_expert.view(1, -1)
    else:
        num_global_tokens_per_expert = torch.zeros(
            ep_size,
            num_global_experts,
            dtype=num_local_tokens_per_expert.dtype,
            device=num_local_tokens_per_expert.device,
        )
        dist.all_gather_into_tensor(num_global_tokens_per_expert, num_local_tokens_per_expert, group=ep_group)

    start_idx, end_idx = ep_rank * num_local_experts, (ep_rank + 1) * num_local_experts
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, start_idx:end_idx].contiguous()

    input_splits = num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1).tolist()
    output_splits = num_global_tokens_per_local_expert.sum(dim=1).tolist()

    num_global_sum_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0)
    num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.to(torch.device("cpu"), non_blocking=True)
    return input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert


def alltoall_dispatch(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    input_splits: list,
    output_splits: list,
    num_global_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    ep_group: dist.ProcessGroup | None = None,
):
    """Permute tokens and all-to-all them to local experts."""
    import torch_npu

    from .....distributed.moe.comm import all_to_all
    from .....distributed.moe.moe_utils import sort_chunks_by_idxs
    from .....utils.device import stream_synchronize

    hidden_states, unpermute_indices = torch_npu.npu_moe_token_permute(hidden_states, selected_experts.to(torch.int32))
    hidden_states = all_to_all(ep_group, hidden_states, output_splits, input_splits)

    stream_synchronize()
    ep_size = 1 if ep_group is None else dist.get_world_size(ep_group)
    num_local_experts = num_global_experts // ep_size
    assert num_global_experts % ep_size == 0, (
        f"Number of experts ({num_global_experts}) must be divisible by expert parallel size ({ep_size})."
    )
    permute_order = torch.arange(num_global_experts).reshape(-1, num_local_experts).T.ravel().tolist()
    hidden_states = sort_chunks_by_idxs(
        hidden_states,
        num_global_tokens_per_local_expert.ravel(),
        permute_order,
    )
    return hidden_states, unpermute_indices


def alltoall_combine(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    unpermute_indices: torch.Tensor,
    input_splits: list,
    output_splits: list,
    num_global_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    ep_group: dist.ProcessGroup | None = None,
):
    """Reverse all-to-all and unpermute expert outputs."""
    import torch_npu

    from .....distributed.moe.comm import all_to_all
    from .....distributed.moe.moe_utils import sort_chunks_by_idxs

    ep_size = 1 if ep_group is None else dist.get_world_size(ep_group)
    num_local_experts = num_global_experts // ep_size
    assert num_global_experts % ep_size == 0, (
        f"Number of experts ({num_global_experts}) must be divisible by expert parallel size ({ep_size})."
    )
    unpermute_order = torch.arange(num_global_experts).reshape(num_local_experts, -1).T.ravel().tolist()
    hidden_states = sort_chunks_by_idxs(
        hidden_states,
        num_global_tokens_per_local_expert.T.ravel(),
        unpermute_order,
    )

    hidden_states = all_to_all(ep_group, hidden_states, input_splits, output_splits)
    hidden_states = torch_npu.npu_moe_token_unpermute(hidden_states, unpermute_indices, probs=routing_weights)
    return hidden_states


def npu_fused_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor | None,
    fc1_2_weight: torch.Tensor | None,
    fc2_weight: torch.Tensor,
    fc1_1_2_weight: torch.Tensor | None = None,
    swiglu_limit: float | None = None,
):
    """NPU fused MoE entry. Branches on EP."""
    # EP comm is outside the Function so all2all is not under no_grad.
    if get_parallel_state().ep_enabled:
        final_hidden_states = npu_ep_fused_moe_forward(
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_2_weight,
            ep_group=get_parallel_state().ep_group,
            swiglu_limit=swiglu_limit,
        )
    else:
        final_hidden_states = _npu_fused_moe_forward(
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_2_weight,
            swiglu_limit=swiglu_limit,
        )
    return final_hidden_states


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
    """Call the NPU fused MoE path. Empty weights are ``None``."""
    # Empty tensor = unused layout. Registry args stay tensors.
    return npu_fused_moe_forward(
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
