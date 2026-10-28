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

"""SwiGLU-OAI expert LoRA with shared or independent adapters.

Tensor layouts match the standard LoRA variants; only the activation semantics
change. NPU implementations reuse the existing dispatch/GEMM/combine pipeline.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .independent import eager as independent_eager
from .independent import npu as independent_npu
from .shared import eager as shared_eager
from .shared import npu as shared_npu


def _activation(alpha: float, limit: float):
    if not math.isfinite(alpha) or alpha <= 0 or not math.isfinite(limit) or limit <= 0:
        raise ValueError("SwiGLU-OAI alpha and limit must be positive finite numbers")

    def activate(gate_up: Tensor) -> Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        return (gate * torch.sigmoid(gate * alpha)) * (up + 1.0)

    return activate


def shared_eager_wrapper(
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
    swiglu_alpha: float,
    swiglu_limit: float,
) -> Tensor:
    """Apply SwiGLU-OAI with shared LoRA through the eager implementation."""
    return shared_eager.wrapper(
        hidden_states,
        routing_weights,
        selected_experts,
        fc1_1_2_weight,
        fc2_weight,
        lora_a_gate,
        lora_b_gate,
        lora_a_up,
        lora_b_up,
        lora_a_down,
        lora_b_down,
        num_experts=num_experts,
        lora_scale_gate=lora_scale_gate,
        lora_scale_up=lora_scale_up,
        lora_scale_down=lora_scale_down,
        activation=_activation(swiglu_alpha, swiglu_limit),
    )


def shared_npu_wrapper(
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
    swiglu_alpha: float,
    swiglu_limit: float,
) -> Tensor:
    """Apply SwiGLU-OAI with shared LoRA through the npu implementation."""
    return shared_npu.wrapper(
        hidden_states,
        routing_weights,
        selected_experts,
        fc1_1_2_weight,
        fc2_weight,
        lora_a_gate,
        lora_b_gate,
        lora_a_up,
        lora_b_up,
        lora_a_down,
        lora_b_down,
        num_experts=num_experts,
        lora_scale_gate=lora_scale_gate,
        lora_scale_up=lora_scale_up,
        lora_scale_down=lora_scale_down,
        activation=_activation(swiglu_alpha, swiglu_limit),
    )


def independent_eager_wrapper(
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
    swiglu_alpha: float,
    swiglu_limit: float,
) -> Tensor:
    """Apply SwiGLU-OAI with independent LoRA through the eager implementation."""
    return independent_eager.wrapper(
        hidden_states,
        routing_weights,
        selected_experts,
        fc1_1_2_weight,
        fc2_weight,
        lora_a_gate,
        lora_b_gate,
        lora_a_up,
        lora_b_up,
        lora_a_down,
        lora_b_down,
        num_experts=num_experts,
        lora_scale_gate=lora_scale_gate,
        lora_scale_up=lora_scale_up,
        lora_scale_down=lora_scale_down,
        activation=_activation(swiglu_alpha, swiglu_limit),
    )


def independent_npu_wrapper(
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
    swiglu_alpha: float,
    swiglu_limit: float,
) -> Tensor:
    """Apply SwiGLU-OAI with independent LoRA through the npu implementation."""
    return independent_npu.wrapper(
        hidden_states,
        routing_weights,
        selected_experts,
        fc1_1_2_weight,
        fc2_weight,
        lora_a_gate,
        lora_b_gate,
        lora_a_up,
        lora_b_up,
        lora_a_down,
        lora_b_down,
        num_experts=num_experts,
        lora_scale_gate=lora_scale_gate,
        lora_scale_up=lora_scale_up,
        lora_scale_down=lora_scale_down,
        activation=_activation(swiglu_alpha, swiglu_limit),
    )
