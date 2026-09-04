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

"""shared MoE-LoRA NPU implementation."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


def _gate_up_delta(
    x: Tensor,
    lora_a_gate: Tensor,
    lora_b_gate: Tensor,
    lora_a_up: Tensor,
    lora_b_up: Tensor,
    lora_scale_gate: float,
    lora_scale_up: float,
) -> Tensor:
    """Shared 2-D gate/up LoRA delta. ``group_list`` is unused."""
    gate = F.linear(F.linear(x, lora_a_gate), lora_b_gate)
    up = F.linear(F.linear(x, lora_a_up), lora_b_up)
    return torch.cat([gate * lora_scale_gate, up * lora_scale_up], dim=-1)


def _down_delta(mid: Tensor, lora_a_down: Tensor, lora_b_down: Tensor, lora_scale_down: float) -> Tensor:
    """Shared 2-D down LoRA delta."""
    return F.linear(F.linear(mid, lora_a_down), lora_b_down) * lora_scale_down


def _bind_deltas(
    lora_a_gate: Tensor,
    lora_b_gate: Tensor,
    lora_a_up: Tensor,
    lora_b_up: Tensor,
    lora_a_down: Tensor,
    lora_b_down: Tensor,
    lora_scale_gate: float,
    lora_scale_up: float,
    lora_scale_down: float,
):
    """Close shared LoRA tensors over the helper skeleton callbacks."""

    def gate_up(x: Tensor, _group_list: Tensor) -> Tensor:
        return _gate_up_delta(x, lora_a_gate, lora_b_gate, lora_a_up, lora_b_up, lora_scale_gate, lora_scale_up)

    def down(mid: Tensor, _group_list: Tensor) -> Tensor:
        return _down_delta(mid, lora_a_down, lora_b_down, lora_scale_down)

    return gate_up, down


def _npu_fused_lora_moe_forward(
    num_experts: int,
    routing_weights: Tensor,
    selected_experts: Tensor,
    hidden_states: Tensor,
    fc1_1_2_weight: Tensor,
    fc2_weight: Tensor,
    lora_a_gate: Tensor,
    lora_b_gate: Tensor,
    lora_a_up: Tensor,
    lora_b_up: Tensor,
    lora_a_down: Tensor,
    lora_b_down: Tensor,
    lora_scale_gate: float,
    lora_scale_up: float,
    lora_scale_down: float,
) -> Tensor:
    """NPU non-EP fused MoE + shared LoRA."""
    from ..helper import npu_non_ep_forward

    gate_up, down = _bind_deltas(
        lora_a_gate,
        lora_b_gate,
        lora_a_up,
        lora_b_up,
        lora_a_down,
        lora_b_down,
        lora_scale_gate,
        lora_scale_up,
        lora_scale_down,
    )
    return npu_non_ep_forward(
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_2_weight,
        fc2_weight,
        gate_up,
        down,
    )


def _npu_ep_fused_lora_moe_forward(
    num_experts: int,
    routing_weights: Tensor,
    selected_experts: Tensor,
    hidden_states: Tensor,
    fc1_1_2_weight: Tensor,
    fc2_weight: Tensor,
    lora_a_gate: Tensor,
    lora_b_gate: Tensor,
    lora_a_up: Tensor,
    lora_b_up: Tensor,
    lora_a_down: Tensor,
    lora_b_down: Tensor,
    lora_scale_gate: float,
    lora_scale_up: float,
    lora_scale_down: float,
    ep_group: dist.ProcessGroup | None = None,
) -> Tensor:
    """NPU EP fused MoE + shared LoRA."""
    from ..helper import npu_ep_forward

    gate_up, down = _bind_deltas(
        lora_a_gate,
        lora_b_gate,
        lora_a_up,
        lora_b_up,
        lora_a_down,
        lora_b_down,
        lora_scale_gate,
        lora_scale_up,
        lora_scale_down,
    )
    return npu_ep_forward(
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_2_weight,
        fc2_weight,
        gate_up,
        down,
        ep_group=ep_group,
    )


def npu_fused_lora_moe_forward(
    num_experts: int,
    routing_weights: Tensor,
    selected_experts: Tensor,
    hidden_states: Tensor,
    fc1_1_2_weight: Tensor,
    fc2_weight: Tensor,
    lora_a_gate: Tensor,
    lora_b_gate: Tensor,
    lora_a_up: Tensor,
    lora_b_up: Tensor,
    lora_a_down: Tensor,
    lora_b_down: Tensor,
    lora_scale_gate: float,
    lora_scale_up: float,
    lora_scale_down: float,
) -> Tensor:
    """NPU shared-LoRA fused MoE. Branches on EP from parallel state."""
    from .....distributed.parallel_state import get_parallel_state

    kwargs = dict(
        num_experts=num_experts,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        hidden_states=hidden_states,
        fc1_1_2_weight=fc1_1_2_weight,
        fc2_weight=fc2_weight,
        lora_a_gate=lora_a_gate,
        lora_b_gate=lora_b_gate,
        lora_a_up=lora_a_up,
        lora_b_up=lora_b_up,
        lora_a_down=lora_a_down,
        lora_b_down=lora_b_down,
        lora_scale_gate=lora_scale_gate,
        lora_scale_up=lora_scale_up,
        lora_scale_down=lora_scale_down,
    )
    if get_parallel_state().ep_enabled:
        return _npu_ep_fused_lora_moe_forward(ep_group=get_parallel_state().ep_group, **kwargs)
    return _npu_fused_lora_moe_forward(**kwargs)


def wrapper(
    hidden_states: Tensor,
    routing_weights: Tensor,
    selected_experts: Tensor,
    fc1_1_2_weight: Tensor,
    fc2_weight: Tensor,
    lora_a_gate: Tensor,
    lora_b_gate: Tensor,
    lora_a_up: Tensor,
    lora_b_up: Tensor,
    lora_a_down: Tensor,
    lora_b_down: Tensor,
    *,
    num_experts: int,
    lora_scale_gate: float,
    lora_scale_up: float,
    lora_scale_down: float,
) -> Tensor:
    """Shared NPU fused LoRA MoE."""
    return npu_fused_lora_moe_forward(
        num_experts=num_experts,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        hidden_states=hidden_states,
        fc1_1_2_weight=fc1_1_2_weight,
        fc2_weight=fc2_weight,
        lora_a_gate=lora_a_gate,
        lora_b_gate=lora_b_gate,
        lora_a_up=lora_a_up,
        lora_b_up=lora_b_up,
        lora_a_down=lora_a_down,
        lora_b_down=lora_b_down,
        lora_scale_gate=lora_scale_gate,
        lora_scale_up=lora_scale_up,
        lora_scale_down=lora_scale_down,
    )
