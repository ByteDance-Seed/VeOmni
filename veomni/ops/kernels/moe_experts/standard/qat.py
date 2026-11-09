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

"""Standard MoE experts blockwise-FP8/FP4 QAT implementation.

Fake-quantizes the expert weights and the token activation, then runs the same
fused Triton grouped GEMM as the ``fused_triton`` row. ``expert_dtype`` follows
the checkpoint rather than a global default: a V4-Flash checkpoint stores routed
experts in FP4 and its dense projections in FP8, so one recipe name cannot
describe both.
"""

from __future__ import annotations

from torch import Tensor

from ....qat import FP4_BLOCK_SIZE, fp4_fake_quant_weight, fp8_fake_quant_act, fp8_fake_quant_stacked_weight
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
    expert_dtype: str = "fp8",
) -> Tensor:
    """Run fused MoE experts with fake-quantized activations and expert weights.

    Empty weight tensors denote unused layouts and are passed through. Routing
    weights are not quantized: they decide which experts run, so rounding them
    would change the selection, not just the arithmetic.
    """

    if expert_dtype not in ("fp8", "fp4"):
        # The value comes from a free-form checkpoint key, so a typo would
        # otherwise train FP8 while the config claims something else.
        raise ValueError(f"expert_dtype must be 'fp8' or 'fp4', got {expert_dtype!r}")

    def _quant_weight(weight: Tensor) -> Tensor:
        if weight.numel() == 0:
            return weight
        if expert_dtype == "fp4":
            # FP4 group width is fixed by the checkpoint's scale layout, which is
            # one scale per 32 input channels; ``block_size`` is the FP8 tile.
            return fp4_fake_quant_weight(weight, block_size=FP4_BLOCK_SIZE)
        return fp8_fake_quant_stacked_weight(weight, block_size=block_size)

    hidden_q = fp8_fake_quant_act(hidden_states, block_size=block_size) if hidden_states.numel() > 0 else hidden_states
    return standard_triton.wrapper(
        hidden_q,
        routing_weights,
        selected_experts,
        _quant_weight(fc1_1_weight),
        _quant_weight(fc1_2_weight),
        _quant_weight(fc2_weight),
        _quant_weight(fc1_1_2_weight),
        num_experts=num_experts,
        swiglu_limit=swiglu_limit,
        assume_distinct_experts=assume_distinct_experts,
    )
