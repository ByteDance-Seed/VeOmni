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

"""standard MoE experts eager math (Hugging Face operator order)."""

from __future__ import annotations

from torch import Tensor

from ....activation import resolve
from ..shared.act_fn_experts import merged_experts_act_fn_forward


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
    """Routed expert MLP. Empty ``fc1_*`` means that layout is unused.

    Matches Hugging Face's expert loop bitwise: ``fc2`` first, then the
    routing weights in their own dtype. Non-EP Triton and EP combine use this
    post-``fc2`` routing-weight order as well; Quack still scales the
    intermediate before ``fc2`` and agrees with this row only to rounding.
    Regular autograd, no custom backward. ``assume_distinct_experts`` only
    tightens the Triton grouped-GEMM launch bound, so it is unused here.

    Every activation, and both weight layouts, run the one reference loop in
    :mod:`veomni.ops.kernels.moe_experts.shared.act_fn_experts`. A name outside
    ``veomni.ops.activation`` is resolved through ``ACT2FN`` -- the same table the
    model built its ``act_fn`` from -- so a config that trained before the activation
    became a call argument still trains. That loop owns the expert-parallel dispatch
    too, because the weights it indexes are sharded per rank under EP; a second
    non-EP copy lived here and indexed a shard with global ids.
    """
    del assume_distinct_experts, block_size, dtype
    return merged_experts_act_fn_forward(
        hidden_states,
        selected_experts,
        routing_weights,
        fc1_1_2_weight,
        fc2_weight,
        resolve(activation),
        num_experts,
        swiglu_limit,
        fc1_1_proj=fc1_1_weight,
        fc1_2_proj=fc1_2_weight,
    )
