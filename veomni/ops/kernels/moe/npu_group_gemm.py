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


from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional

import torch
import torch.distributed as dist
import torch_npu

from ....distributed.moe.comm import (
    AsyncCollectiveHandle,
    all_to_all,
    all_to_all_async,
    register_expert_backward_boundary,
)
from ....distributed.moe.moe_utils import sort_chunks_by_idxs
from ....distributed.parallel_state import get_parallel_state
from ....utils.device import stream_synchronize
from ._kernels.kernel.npu_group_gemm import npu_group_gemm


@dataclass
class NpuEpDispatchPlan:
    num_global_experts: int
    input_splits: List
    output_splits: List
    num_global_tokens_per_local_expert: torch.Tensor
    num_global_sum_tokens_per_local_expert: torch.Tensor
    routing_shape: torch.Size
    ep_group: Optional[dist.ProcessGroup]


NpuEpDispatchPlanCallback = Callable[[NpuEpDispatchPlan], None]
_npu_ep_dispatch_plan_callback: ContextVar[Optional[NpuEpDispatchPlanCallback]] = ContextVar(
    "npu_ep_dispatch_plan_callback",
    default=None,
)
NpuEpDispatchLaunchCallback = Callable[[], None]
_npu_ep_dispatch_launch_callback: ContextVar[Optional[NpuEpDispatchLaunchCallback]] = ContextVar(
    "npu_ep_dispatch_launch_callback",
    default=None,
)


@contextmanager
def npu_ep_dispatch_plan_callback(
    callback: Optional[NpuEpDispatchPlanCallback],
) -> Iterator[None]:
    token = _npu_ep_dispatch_plan_callback.set(callback)
    try:
        yield
    finally:
        _npu_ep_dispatch_plan_callback.reset(token)


@contextmanager
def npu_ep_dispatch_launch_callback(
    callback: Optional[NpuEpDispatchLaunchCallback],
) -> Iterator[None]:
    token = _npu_ep_dispatch_launch_callback.set(callback)
    try:
        yield
    finally:
        _npu_ep_dispatch_launch_callback.reset(token)


def _npu_fused_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor | None,
    fc1_2_weight: torch.Tensor | None,
    fc2_weight: torch.Tensor,
    fc1_1_2_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """NPU single-device fused MoE forward pass (non-EP).

    Accepts either split fc1 weights or a merged fc1_1_2_weight tensor.
    Weights are merged and transposed for the NPU group-gemm kernel.
    """
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(
        hidden_states, selected_experts.to(torch.int32)
    )
    tokens_per_expert = torch.histc(selected_experts, bins=num_experts, min=0, max=num_experts)

    if fc1_1_2_weight is not None:
        fc1_weight = fc1_1_2_weight
    else:
        fc1_weight = torch.cat([fc1_1_weight, fc1_2_weight], dim=1)
    fc1_weight = fc1_weight.transpose(1, 2)
    intermediate_hidden_states = npu_group_gemm(permuted_hidden_states, fc1_weight, tokens_per_expert)
    intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
    output = npu_group_gemm(intermediate_activations, fc2_weight.transpose(1, 2), tokens_per_expert)
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
    ep_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """NPU expert-parallel fused MoE forward pass.

    Accepts either split fc1 weights or a merged fc1_1_2_weight tensor.
    Handles alltoall dispatch/combine for expert parallelism.
    """
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    dispatch_plan = npu_ep_dispatch_prepare(selected_experts, num_experts, ep_group)
    callback = _npu_ep_dispatch_plan_callback.get()
    if callback is not None:
        callback(dispatch_plan)
    launch_callback = _npu_ep_dispatch_launch_callback.get()
    if launch_callback is not None:
        launch_callback()
    hidden_states, unpermute_indices = alltoall_dispatch(
        hidden_states,
        selected_experts,
        dispatch_plan.input_splits,
        dispatch_plan.output_splits,
        num_experts,
        dispatch_plan.num_global_tokens_per_local_expert,
        ep_group,
    )
    hidden_states = register_expert_backward_boundary(hidden_states)

    if fc1_1_2_weight is not None:
        fc1_weight = fc1_1_2_weight
    else:
        fc1_weight = torch.cat([fc1_1_weight, fc1_2_weight], dim=1)
    fc1_weight = fc1_weight.transpose(1, 2)
    intermediate_hidden_states = npu_group_gemm(
        hidden_states,
        fc1_weight,
        dispatch_plan.num_global_sum_tokens_per_local_expert,
    )
    intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
    hidden_states = npu_group_gemm(
        intermediate_activations,
        fc2_weight.transpose(1, 2),
        dispatch_plan.num_global_sum_tokens_per_local_expert,
    )

    hidden_states = alltoall_combine(
        hidden_states,
        routing_weights,
        unpermute_indices,
        dispatch_plan.input_splits,
        dispatch_plan.output_splits,
        num_experts,
        dispatch_plan.num_global_tokens_per_local_expert,
        ep_group,
    )
    return hidden_states


def dispatch_preprocess(
    selected_experts: torch.Tensor,
    num_global_experts: int,
    ep_group: Optional[dist.ProcessGroup] = None,
):
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
    input_splits: List,
    output_splits: List,
    num_global_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup] = None,
):
    hidden_states, unpermute_indices = torch_npu.npu_moe_token_permute(hidden_states, selected_experts.to(torch.int32))
    hidden_states = all_to_all(ep_group, hidden_states, output_splits, input_splits, phase="dispatch")

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
    input_splits: List,
    output_splits: List,
    num_global_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup] = None,
):
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

    hidden_states = all_to_all(ep_group, hidden_states, input_splits, output_splits, phase="combine")
    hidden_states = torch_npu.npu_moe_token_unpermute(hidden_states, unpermute_indices, probs=routing_weights)
    return hidden_states


@dataclass
class NpuEpDispatchState(NpuEpDispatchPlan):
    unpermute_indices: torch.Tensor
    output: torch.Tensor
    handle: AsyncCollectiveHandle


@dataclass
class NpuEpDispatchInput:
    plan: NpuEpDispatchPlan
    permuted: torch.Tensor
    unpermute_indices: torch.Tensor


@dataclass
class NpuEpCombineState:
    routing_weights: torch.Tensor
    unpermute_indices: torch.Tensor
    output: torch.Tensor
    handle: AsyncCollectiveHandle


def npu_ep_dispatch_prepare(
    selected_experts: torch.Tensor,
    num_global_experts: int,
    ep_group: Optional[dist.ProcessGroup],
) -> NpuEpDispatchPlan:
    """Prepare token counts and split metadata before the overlap boundary."""
    input_splits, output_splits, tokens_per_local_expert, summed_tokens_per_local_expert = dispatch_preprocess(
        selected_experts, num_global_experts, ep_group
    )
    return NpuEpDispatchPlan(
        num_global_experts=num_global_experts,
        input_splits=input_splits,
        output_splits=output_splits,
        num_global_tokens_per_local_expert=tokens_per_local_expert,
        num_global_sum_tokens_per_local_expert=summed_tokens_per_local_expert,
        routing_shape=selected_experts.shape,
        ep_group=ep_group,
    )


def npu_ep_dispatch_input_prepare(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    plan: NpuEpDispatchPlan,
) -> NpuEpDispatchInput:
    """Run replay token permutation before its communication ticket is granted."""
    permuted, unpermute_indices = torch_npu.npu_moe_token_permute(
        hidden_states,
        selected_experts.to(torch.int32),
    )
    return NpuEpDispatchInput(
        plan=plan,
        permuted=permuted,
        unpermute_indices=unpermute_indices,
    )


def npu_ep_dispatch_async(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    num_global_experts: int,
    ep_group: Optional[dist.ProcessGroup],
    plan: Optional[NpuEpDispatchPlan] = None,
    prepared_input: Optional[NpuEpDispatchInput] = None,
) -> NpuEpDispatchState:
    """Launch replay dispatch and defer its host/device synchronization."""
    if plan is None:
        plan = npu_ep_dispatch_prepare(selected_experts, num_global_experts, ep_group)
    elif plan.num_global_experts != num_global_experts or plan.ep_group is not ep_group:
        raise ValueError("NPU EP dispatch plan does not match the requested expert topology.")
    if prepared_input is None:
        prepared_input = npu_ep_dispatch_input_prepare(hidden_states, selected_experts, plan)
    elif prepared_input.plan is not plan:
        raise ValueError("NPU EP prepared dispatch input does not match the requested dispatch plan.")
    permuted = prepared_input.permuted
    unpermute_indices = prepared_input.unpermute_indices
    output, handle = all_to_all_async(ep_group, permuted, plan.output_splits, plan.input_splits)
    return NpuEpDispatchState(
        num_global_experts=plan.num_global_experts,
        input_splits=plan.input_splits,
        output_splits=plan.output_splits,
        num_global_tokens_per_local_expert=plan.num_global_tokens_per_local_expert,
        num_global_sum_tokens_per_local_expert=plan.num_global_sum_tokens_per_local_expert,
        unpermute_indices=unpermute_indices,
        routing_shape=plan.routing_shape,
        output=output,
        handle=handle,
        ep_group=plan.ep_group,
    )


def npu_ep_dispatch_wait(state: NpuEpDispatchState, synchronize: bool = True) -> torch.Tensor:
    state.handle.wait()
    if synchronize:
        stream_synchronize()
    ep_size = 1 if state.ep_group is None else dist.get_world_size(state.ep_group)
    num_local_experts = state.num_global_experts // ep_size
    permute_order = torch.arange(state.num_global_experts).reshape(-1, num_local_experts).T.ravel().tolist()
    return sort_chunks_by_idxs(
        state.output,
        state.num_global_tokens_per_local_expert.ravel(),
        permute_order,
    )


def npu_ep_dispatch_input_backward(
    grad_dispatched: torch.Tensor,
    state: NpuEpDispatchState,
) -> torch.Tensor:
    """Map local expert-input gradients back to the pre-dispatch token order."""
    ep_size = 1 if state.ep_group is None else dist.get_world_size(state.ep_group)
    num_local_experts = state.num_global_experts // ep_size
    unpermute_order = torch.arange(state.num_global_experts).reshape(num_local_experts, -1).T.ravel().tolist()
    grad_dispatched = sort_chunks_by_idxs(
        grad_dispatched,
        state.num_global_tokens_per_local_expert.T.ravel(),
        unpermute_order,
    )
    grad_permuted, handle = all_to_all_async(
        state.ep_group,
        grad_dispatched,
        state.input_splits,
        state.output_splits,
    )
    handle.wait()
    return torch_npu.npu_moe_token_unpermute(
        grad_permuted,
        state.unpermute_indices,
        probs=torch.ones(state.routing_shape, dtype=grad_permuted.dtype, device=grad_permuted.device),
    )


def npu_ep_expert_forward(
    hidden_states: torch.Tensor,
    dispatch_state: NpuEpDispatchState,
    fc2_weight: torch.Tensor,
    fc1_1_2_weight: torch.Tensor,
) -> torch.Tensor:
    fc1_weight = fc1_1_2_weight.transpose(1, 2)
    intermediate = npu_group_gemm(
        hidden_states,
        fc1_weight,
        dispatch_state.num_global_sum_tokens_per_local_expert,
    )
    activations = torch_npu.npu_swiglu(intermediate, dim=-1)
    return npu_group_gemm(
        activations,
        fc2_weight.transpose(1, 2),
        dispatch_state.num_global_sum_tokens_per_local_expert,
    )


def npu_ep_combine_async(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    dispatch_state: NpuEpDispatchState,
    backward_phase_callback=None,
) -> NpuEpCombineState:
    ep_size = 1 if dispatch_state.ep_group is None else dist.get_world_size(dispatch_state.ep_group)
    num_local_experts = dispatch_state.num_global_experts // ep_size
    unpermute_order = torch.arange(dispatch_state.num_global_experts).reshape(num_local_experts, -1).T.ravel().tolist()
    hidden_states = sort_chunks_by_idxs(
        hidden_states,
        dispatch_state.num_global_tokens_per_local_expert.T.ravel(),
        unpermute_order,
    )
    output, handle = all_to_all_async(
        dispatch_state.ep_group,
        hidden_states,
        dispatch_state.input_splits,
        dispatch_state.output_splits,
        phase="combine",
        callback=backward_phase_callback,
    )
    return NpuEpCombineState(
        routing_weights=routing_weights,
        unpermute_indices=dispatch_state.unpermute_indices,
        output=output,
        handle=handle,
    )


def npu_ep_combine_wait(state: NpuEpCombineState) -> torch.Tensor:
    state.handle.wait()
    return torch_npu.npu_moe_token_unpermute(
        state.output,
        state.unpermute_indices,
        probs=state.routing_weights,
    )


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
    if swiglu_limit is not None:
        raise NotImplementedError("NPU fused MoE does not support swiglu_limit clamp semantics.")

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
        )
    return final_hidden_states
