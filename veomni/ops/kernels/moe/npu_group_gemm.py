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


from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu

from ....distributed.moe.comm import all_to_all
from ....distributed.moe.ep_dispatch import _record_ep_balance, _validate_ep_balance_dispatch
from ....distributed.moe.ep_load_balance import ExpertReplicaTransfer
from ....distributed.moe.moe_utils import sort_chunks_by_idxs
from ....distributed.parallel_state import get_parallel_state
from ....utils.device import stream_synchronize
from . import npu_ep_memory
from ._kernels.kernel.npu_group_gemm import npu_group_gemm


def _eager_clamped_swiglu(x: torch.Tensor, limit: float) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


def _is_triton_ascend_available() -> bool:
    try:
        from triton._C import libtriton
    except ImportError:
        return False
    return hasattr(libtriton, "ascend")


def _clamped_swiglu(x: torch.Tensor, limit: float) -> torch.Tensor:
    """Use Ascend Triton when available and preserve the eager training fallback."""
    if _is_triton_ascend_available():
        from ._kernels.kernel.npu_clamped_swiglu import npu_triton_clamped_swiglu

        return npu_triton_clamped_swiglu(x, limit)

    return _eager_clamped_swiglu(x, limit)


def _swiglu(x: torch.Tensor, swiglu_limit: float | None) -> torch.Tensor:
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
    intermediate_activations = _swiglu(intermediate_hidden_states, swiglu_limit)
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
    swiglu_limit: float | None = None,
    load_balancer=None,
) -> torch.Tensor:
    """NPU expert-parallel fused MoE forward pass.

    Accepts either split fc1 weights or a merged fc1_1_2_weight tensor.
    Handles alltoall dispatch/combine for expert parallelism.
    """
    if load_balancer is not None:
        return _npu_ep_balanced_moe_forward(
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_2_weight,
            ep_group,
            swiglu_limit,
            load_balancer,
        )

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

    if fc1_1_2_weight is not None:
        fc1_weight = fc1_1_2_weight
    else:
        fc1_weight = torch.cat([fc1_1_weight, fc1_2_weight], dim=1)
    fc1_weight = fc1_weight.transpose(1, 2)
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


def _npu_ep_balanced_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor | None,
    fc1_2_weight: torch.Tensor | None,
    fc2_weight: torch.Tensor,
    fc1_1_2_weight: torch.Tensor | None,
    ep_group: Optional[dist.ProcessGroup],
    swiglu_limit: float | None,
    load_balancer,
) -> torch.Tensor:
    if fc1_1_2_weight is None or fc1_1_weight is not None or fc1_2_weight is not None:
        raise ValueError("NPU EP load balancing requires merged fc1_1_2_weight without split fc1 weights.")
    if ep_group is None or getattr(load_balancer, "ep_group", None) is not ep_group:
        raise ValueError(
            "NPU EP load balancing requires the entry point and load balancer to use the same process group."
        )

    fc1_weight = fc1_1_2_weight
    plan = _validate_ep_balance_dispatch(
        num_experts,
        selected_experts,
        fc1_weight,
        fc2_weight,
        load_balancer,
    )
    _record_ep_balance(load_balancer, plan)

    gate_up_transfer = None
    down_transfer = None
    optimize_memory = getattr(load_balancer, "optimize_memory", False) and bool(plan.replicas)
    if plan.replicas:
        transfer_options = {"pool": load_balancer.buffer_pool} if optimize_memory else {}
        gate_up_transfer = ExpertReplicaTransfer.start(fc1_weight, plan, ep_group, **transfer_options)
        down_transfer = ExpertReplicaTransfer.start(fc2_weight, plan, ep_group, **transfer_options)

    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    physical_slots_per_rank = plan.tokens_per_local_physical_expert.shape[1]
    physical_num_experts = plan.ep_size * physical_slots_per_rank if plan.replicas else num_experts
    hidden_states, unpermute_indices = alltoall_dispatch(
        hidden_states,
        plan.selected_physical_experts,
        plan.input_splits,
        plan.output_splits,
        physical_num_experts,
        plan.tokens_per_local_physical_expert,
        ep_group,
        host_counts_ready=(
            optimize_memory and hidden_states.device.type == "npu" and dist.get_backend(ep_group) == "hccl"
        ),
    )

    host_counts = plan.tokens_per_local_physical_expert.sum(dim=0)
    tokens_per_local_physical_expert = host_counts.to(
        hidden_states.device,
        dtype=torch.int64,
        non_blocking=True,
    )
    if gate_up_transfer is not None and down_transfer is not None:
        replica_fc1_weight = gate_up_transfer.wait()
        replica_fc2_weight = down_transfer.wait()
        try:
            intermediate_hidden_states = npu_ep_memory.balanced_group_gemm(
                hidden_states,
                fc1_weight,
                replica_fc1_weight,
                tokens_per_local_physical_expert,
                host_counts,
                plan,
                ep_group,
                gate_up_transfer,
                optimize_memory,
                npu_group_gemm,
            )
            hidden_states = npu_ep_memory.balanced_group_gemm(
                _swiglu(intermediate_hidden_states, swiglu_limit),
                fc2_weight,
                replica_fc2_weight,
                tokens_per_local_physical_expert,
                host_counts,
                plan,
                ep_group,
                down_transfer,
                optimize_memory,
                npu_group_gemm,
            )
        finally:
            try:
                gate_up_transfer.record_stream_use()
            finally:
                down_transfer.record_stream_use()
    else:
        intermediate_hidden_states = npu_group_gemm(
            hidden_states,
            fc1_weight.transpose(1, 2),
            tokens_per_local_physical_expert,
        )
        intermediate_activations = _swiglu(intermediate_hidden_states, swiglu_limit)
        hidden_states = npu_group_gemm(
            intermediate_activations,
            fc2_weight.transpose(1, 2),
            tokens_per_local_physical_expert,
        )

    return alltoall_combine(
        hidden_states,
        routing_weights,
        unpermute_indices,
        plan.input_splits,
        plan.output_splits,
        physical_num_experts,
        plan.tokens_per_local_physical_expert,
        ep_group,
    )


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
    *,
    host_counts_ready: bool = False,
):
    hidden_states, unpermute_indices = torch_npu.npu_moe_token_permute(hidden_states, selected_experts.to(torch.int32))
    hidden_states = all_to_all(ep_group, hidden_states, output_splits, input_splits)

    # The collective already orders device consumers. Only an unfinished
    # host count transfer requires waiting before the CPU reads its values.
    if not host_counts_ready or num_global_tokens_per_local_expert.device.type != "cpu":
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
    assume_distinct_experts: bool = False,  # intentionally unused; see the del below
    load_balancer=None,
):
    # ``assume_distinct_experts`` only tightens the grouped-GEMM ``max_M``
    # launch bound in the Triton backend. The NPU group GEMM has no such launch
    # bound to tighten, so the flag is a no-op here. It stays in the signature
    # only so the unified ``fused_moe_forward`` dispatch and the OpSlot adapter
    # can forward it to any backend uniformly.
    del assume_distinct_experts
    if load_balancer is not None:
        parallel_state = get_parallel_state()
        if not parallel_state.ep_enabled:
            raise ValueError("NPU EP load balancing requires expert parallelism to be enabled.")
        return npu_ep_fused_moe_forward(
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_2_weight,
            ep_group=parallel_state.ep_group,
            swiglu_limit=swiglu_limit,
            load_balancer=load_balancer,
        )

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
