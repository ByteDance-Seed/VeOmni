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

"""Cross-variant helpers for ``moe_experts_lora``.

Not a registered kernel family. ``shared/`` here is the LoRA variant name,
not a helper package. NPU permute / EP / SwiGLU / combine lives here so
each variant ``npu.py`` can own its own LoRA deltas.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


GateUpDelta = Callable[[Tensor, Tensor], Tensor]
DownDelta = Callable[[Tensor, Tensor], Tensor]


class _NpuLoRADeltasEP:
    """NPU grouped GEMMs with caller-supplied LoRA deltas, on dispatched tokens.

    This is the per-rank math half of the LoRA NPU path. The all-to-all plumbing is
    ``dispatch_to_ep_class``'s, shared with every other row; only the grouped GEMMs and
    the delta injection are Ascend's and LoRA's.

    ``gate_up_delta`` / ``down_delta`` are closures the variant ``npu.py`` builds from
    its own adapter layout, so shared (2-D, replicated) and independent (3-D, sharded)
    keep their distinct semantics without this class knowing which is which.
    """

    @staticmethod
    def apply(
        permute_tokens: Tensor,
        cumsum: Tensor,
        fc1_1_2_weight: Tensor,
        fc2_weight: Tensor,
        gate_up_delta: GateUpDelta,
        down_delta: DownDelta,
    ) -> Tensor:
        """Run both grouped GEMMs with the LoRA deltas added between them."""
        import torch_npu

        from ....distributed.moe import cumsum_to_counts
        from ..moe_experts.standard.npu import npu_group_gemm

        counts = cumsum_to_counts(cumsum)
        gate_up = npu_group_gemm(permute_tokens, fc1_1_2_weight.transpose(1, 2), counts)
        gate_up = gate_up + gate_up_delta(permute_tokens, counts)
        intermediate = torch_npu.npu_swiglu(gate_up, dim=-1)
        output = npu_group_gemm(intermediate, fc2_weight.transpose(1, 2), counts)
        output = output + down_delta(intermediate, counts)
        # An empty rank must stay on the graph so its all-to-all backward runs.
        return output + permute_tokens * 0


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

    routing_weights = routing_weights.to(dtype=hidden_states.dtype)
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
) -> Tensor:
    """NPU expert-parallel fused MoE plus caller-supplied LoRA deltas.

    Goes through ``dispatch_to_ep_class``, so the all-to-all plumbing is the same one
    every other row uses and a change to it lands in one place.

    Casts the routing weights to the activation dtype first, as every non-LoRA row does
    at its entry point: ``dispatch_to_ep_class`` forwards them untouched and
    ``unpermute`` applies them to the expert outputs, so an fp32 router would upcast
    there.

    With one EP rank there is nothing to dispatch, and the helper's ``preprocess``
    requires a group wider than one (its ``all_gather_into_tensor`` is shaped by
    ``ep_group.size()``). Take the local path instead: it is the same math without the
    all-to-all, and it is what ``ep_size == 1`` means.
    """
    from ....distributed.moe import dispatch_to_ep_class, expert_parallel_size

    routing_weights = routing_weights.to(dtype=hidden_states.dtype)

    if expert_parallel_size() <= 1:
        return npu_non_ep_forward(
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            fc1_1_2_weight,
            fc2_weight,
            gate_up_delta,
            down_delta,
        )

    return dispatch_to_ep_class(
        _NpuLoRADeltasEP,
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_2_weight,
        fc2_weight,
        gate_up_delta,
        down_delta,
    )
