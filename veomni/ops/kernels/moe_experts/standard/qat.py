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


def quantize_expert_operands(
    hidden_states: Tensor,
    fc1_1_weight: Tensor,
    fc1_2_weight: Tensor,
    fc2_weight: Tensor,
    fc1_1_2_weight: Tensor,
    *,
    expert_dtype: str = "fp8",
    block_size: int = 128,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Apply this row's rounding to the activations and expert weights.

    Kept as a named function rather than inlined so the rounding is readable on its
    own and can be exercised without a fused GEMM. Only this row calls it.

    Empty weight tensors denote unused layouts and are returned unchanged.
    Routing weights are not an input: they decide which experts run, so rounding
    them would change the selection rather than the arithmetic.

    Args:
        hidden_states: Token activations entering the experts.
        fc1_1_weight: First gate/up slab, or an empty tensor when unused.
        fc1_2_weight: Second gate/up slab, or an empty tensor when unused.
        fc2_weight: Down-projection weights.
        fc1_1_2_weight: Fused gate/up slab, or an empty tensor when unused.
        expert_dtype: ``"fp8"`` for stacked FP8 tiles, ``"fp4"`` for 32-wide groups.
        block_size: Side length of the FP8 weight tile; ignored for FP4, whose
            group width is fixed by the checkpoint's scale layout.

    Returns:
        The rounded activation followed by the four rounded weights.

    Raises:
        ValueError: if ``expert_dtype`` is neither ``"fp8"`` nor ``"fp4"``. The
            value comes from a free-form checkpoint key, so a typo would
            otherwise train FP8 while the config claims something else.
    """
    if expert_dtype not in ("fp8", "fp4"):
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
    return (
        hidden_q,
        _quant_weight(fc1_1_weight),
        _quant_weight(fc1_2_weight),
        _quant_weight(fc2_weight),
        _quant_weight(fc1_1_2_weight),
    )


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

    hidden_q, fc1_1_q, fc1_2_q, fc2_q, fc1_1_2_q = quantize_expert_operands(
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
        fc1_1_2_weight,
        expert_dtype=expert_dtype,
        block_size=block_size,
    )
    return standard_triton.wrapper(
        hidden_q,
        routing_weights,
        selected_experts,
        fc1_1_q,
        fc1_2_q,
        fc2_q,
        fc1_1_2_q,
        num_experts=num_experts,
        swiglu_limit=swiglu_limit,
        assume_distinct_experts=assume_distinct_experts,
    )
