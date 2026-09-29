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
# See the License for the specific language governing limitations
# under the License.

"""Cross-variant helpers for ``moe_experts_lora``.

Not a registered kernel family. ``shared/`` here is the LoRA variant name,
not a helper package. NPU permute / EP / SwiGLU / combine lives here so
each variant ``npu.py`` can own its own LoRA deltas.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist
from torch import Tensor


GateUpDelta = Callable[[Tensor, Tensor], Tensor]
DownDelta = Callable[[Tensor, Tensor], Tensor]


def npu_non_ep_forward(
    num_experts: int,
    routing_weights: Tensor,
    selected_experts: Tensor,
    hidden_states: Tensor,
    fc1_1_2_weight: Tensor,
    fc2_weight: Tensor,
    gate_up_delta: GateUpDelta,
    down_delta: DownDelta,
) -> Tensor:
    """NPU single-device fused MoE plus caller-supplied LoRA deltas."""
    import torch_npu

    from ..moe_experts.standard.npu import npu_group_gemm

    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    selected_experts = selected_experts.to(torch.int32)
    permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(hidden_states, selected_experts)
    tokens_per_expert = torch.histc(selected_experts.to(torch.float32), bins=num_experts, min=0, max=num_experts).to(
        torch.int64
    )

    gate_up = npu_group_gemm(permuted_hidden_states, fc1_1_2_weight.transpose(1, 2), tokens_per_expert)
    gate_up = gate_up + gate_up_delta(permuted_hidden_states, tokens_per_expert)
    intermediate = torch_npu.npu_swiglu(gate_up, dim=-1)
    output = npu_group_gemm(intermediate, fc2_weight.transpose(1, 2), tokens_per_expert)
    output = output + down_delta(intermediate, tokens_per_expert)
    return torch_npu.npu_moe_token_unpermute(output, row_ids_map, probs=routing_weights)


def npu_ep_forward(
    num_experts: int,
    routing_weights: Tensor,
    selected_experts: Tensor,
    hidden_states: Tensor,
    fc1_1_2_weight: Tensor,
    fc2_weight: Tensor,
    gate_up_delta: GateUpDelta,
    down_delta: DownDelta,
    ep_group: dist.ProcessGroup | None = None,
) -> Tensor:
    """NPU expert-parallel fused MoE plus caller-supplied LoRA deltas."""
    import torch_npu

    from ..moe_experts.standard.npu import (
        alltoall_combine,
        alltoall_dispatch,
        dispatch_preprocess,
        npu_group_gemm,
    )

    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    selected_experts = selected_experts.to(torch.int32)
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

    group_list = num_global_sum_tokens_per_local_expert.to(torch.int64)
    gate_up = npu_group_gemm(hidden_states, fc1_1_2_weight.transpose(1, 2), group_list)
    gate_up = gate_up + gate_up_delta(hidden_states, group_list)
    intermediate = torch_npu.npu_swiglu(gate_up, dim=-1)
    output = npu_group_gemm(intermediate, fc2_weight.transpose(1, 2), group_list)
    output = output + down_delta(intermediate, group_list)
    return alltoall_combine(
        output,
        routing_weights,
        unpermute_indices,
        input_splits,
        output_splits,
        num_experts,
        num_global_tokens_per_local_expert,
        ep_group,
    )
