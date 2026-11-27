# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""SwiGLU-OAI MoE experts on the standard Triton grouped-GEMM path."""

from __future__ import annotations

from torch import Tensor

from ..standard.triton import group_gemm_fused_moe_forward
from .eager import _check_merged_layout


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
    swiglu_alpha: float,
    swiglu_limit: float | None = None,
) -> Tensor:
    """Call the standard fused path with the SwiGLU-OAI activation (and EP when enabled)."""
    _check_merged_layout(fc1_1_weight, fc1_2_weight, fc1_1_2_weight)
    return group_gemm_fused_moe_forward(
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        None,
        None,
        fc2_weight,
        fc1_1_2_weight,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
    )
