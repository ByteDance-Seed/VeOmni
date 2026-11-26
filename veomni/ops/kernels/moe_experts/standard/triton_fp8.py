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

"""Opt-in Blackwell MXFP8 standard MoE experts."""

from torch import Tensor

from ..shared.activation import is_silu
from . import triton as standard_triton


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
    assume_distinct_experts: bool = False,
    block_size: int = 128,
    dtype: str = "fp8",
    activation: str = "silu",
) -> Tensor:
    """Quantize both merged expert projections in forward, dgrad and wgrad.

    MXFP8 fixes the scale width to K32; block_size is the generic expert
    contract's blockwise-weight hint, not an MXFP8 recipe option.
    """
    del block_size
    if dtype != "fp8" or not is_silu(activation):
        raise ValueError("fused_triton_fp8 requires FP8 and SiLU/SwiGLU")
    return standard_triton.group_gemm_fused_moe_forward(
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_weight if fc1_1_weight.numel() else None,
        fc1_2_weight if fc1_2_weight.numel() else None,
        fc2_weight,
        fc1_1_2_weight if fc1_1_2_weight.numel() else None,
        swiglu_limit=swiglu_limit,
        assume_distinct_experts=assume_distinct_experts,
        activation=activation,
        use_mxfp8=True,
    )
