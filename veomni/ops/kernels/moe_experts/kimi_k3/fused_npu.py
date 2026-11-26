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

"""Kimi K3 situ fused MoE experts (NPU group-GEMM)."""

from __future__ import annotations

from torch import Tensor


def wrapper(
    hidden_states: Tensor,
    routing_weights: Tensor,
    selected_experts: Tensor,
    fc1_1_2_weight: Tensor,
    fc2_weight: Tensor,
    *,
    num_experts: int,
    beta: float = 1.0,
    linear_beta: float | None = None,
) -> Tensor:
    """Delegate to the vendored NPU group-GEMM situ fused-MoE kernel."""
    from .npu_situ_gemm import npu_situ_fused_moe_forward

    return npu_situ_fused_moe_forward(
        num_experts=num_experts,
        routing_weights=routing_weights.to(hidden_states.dtype),
        selected_experts=selected_experts,
        hidden_states=hidden_states,
        fc1_1_weight=None,
        fc1_2_weight=None,
        fc2_weight=fc2_weight,
        fc1_1_2_weight=fc1_1_2_weight,
        beta=beta,
        linear_beta=linear_beta,
    )
