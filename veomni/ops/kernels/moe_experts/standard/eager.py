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

import torch
import torch.nn.functional as F
from torch import Tensor

from .....distributed.parallel_state import get_parallel_state
from ....activation import apply_activation, resolve, supports
from ..shared.act_fn_experts import merged_experts_act_fn_forward


def _experts_are_sharded() -> bool:
    """Whether the expert weights this row is handed are sharded across ranks.

    Reads ``extra_parallel_sizes`` directly rather than ``ep_enabled``: that property
    indexes the dict unconditionally, and a parallel state built with a different
    ``extra_parallel_names`` (some tests use one) has no ``"ep"`` key. The question
    here is only "is expert parallelism on?", and an absent axis means no.
    """
    sizes = getattr(get_parallel_state(), "extra_parallel_sizes", None)
    return bool(sizes) and sizes.get("ep", 1) > 1


def _expert_activation(gate: Tensor, activation: str) -> Tensor:
    """Evaluate the configured activation, whichever name it is.

    ``veomni.ops.activation`` serves a closed set through ATen spellings that match the
    fused rows bit for bit; anything else resolves through ``ACT2FN``, the same table
    the model built its own ``act_fn`` from. So the row needs no module handed to it.
    """
    from ....activation import resolve, supports

    return apply_activation(gate, activation) if supports(activation) else resolve(activation)(gate)


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
    activation: str = "silu",
) -> Tensor:
    """Routed expert MLP. Empty ``fc1_*`` means that layout is unused.

    Matches Hugging Face's expert loop bitwise: ``fc2`` first, then the
    routing weights in their own dtype. Non-EP Triton and EP combine use this
    post-``fc2`` routing-weight order as well; Quack still scales the
    intermediate before ``fc2`` and agrees with this row only to rounding.
    Regular autograd, no custom backward. ``assume_distinct_experts`` only
    tightens the Triton grouped-GEMM launch bound, so it is unused here.

    An activation a fused row cannot serve reaches this row. A name outside
    ``veomni.ops.activation`` is resolved through ``ACT2FN`` -- the same table the
    model built its ``act_fn`` from -- so a config that trained before the activation
    became a call argument still trains. The expert loop that handles it also owns the
    expert-parallel dispatch, because the weights it indexes are sharded per rank
    under EP.
    """
    del assume_distinct_experts, block_size, expert_dtype
    # Two reasons to take the shared loop rather than the inline one below: an
    # activation no ATen spelling covers, and expert parallelism -- whose weights are
    # sharded per rank, so the inline loop's ``num_experts`` mask indexes past the
    # shard. ``gelu`` is in the served set, so keying on the activation alone would
    # send it down the sharded-weight path.
    if not supports(activation) or _experts_are_sharded():
        # One implementation of the reference path, shared with the fused rows that
        # delegate here. It carries the EP dispatch, which is why this cannot be
        # inlined: indexing a sharded weight by global expert id reads past the shard.
        merged = (
            fc1_1_2_weight
            if fc1_1_2_weight.numel()
            else torch.cat([fc1_1_weight, fc1_2_weight], dim=1)
            if fc1_1_weight.numel()
            else None
        )
        if merged is None:
            raise ValueError("Provide either split fc1 weights or merged fc1_1_2_weight.")
        return merged_experts_act_fn_forward(
            hidden_states,
            selected_experts,
            routing_weights,
            merged,
            fc2_weight,
            resolve(activation),
            num_experts,
            swiglu_limit,
        )
    has_split = fc1_1_weight.numel() > 0
    has_merged = fc1_1_2_weight.numel() > 0
    if has_split == has_merged:
        raise ValueError("Provide either split fc1 weights or merged fc1_1_2_weight, not both or neither.")
    if has_split and fc1_2_weight.numel() == 0:
        raise ValueError("Split fc1 mode requires both fc1_1_weight and fc1_2_weight.")

    output = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        idx = int(expert_idx[0].item())
        top_k_pos, token_idx = torch.where(expert_mask[idx])
        x = hidden_states[token_idx]
        if has_merged:
            gate, up = F.linear(x, fc1_1_2_weight[idx]).chunk(2, dim=-1)
        else:
            gate = F.linear(x, fc1_1_weight[idx])
            up = F.linear(x, fc1_2_weight[idx])
        if swiglu_limit is not None:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        y = F.linear(_expert_activation(gate, activation) * up, fc2_weight[idx])
        y = y * routing_weights[token_idx, top_k_pos, None]
        output = output.index_add(0, token_idx, y.to(output.dtype))
    return output
