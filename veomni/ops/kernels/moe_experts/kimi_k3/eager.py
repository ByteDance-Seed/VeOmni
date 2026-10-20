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

"""Kimi K3 situ MoE experts eager implementation (HF expert loop)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class _SituExpertsEP:
    """Local SituGLU expert math on tokens already dispatched by EP."""

    @staticmethod
    def apply(
        permute_tokens: Tensor,
        cumsum: Tensor,
        fc1_1_2_weight: Tensor,
        fc2_weight: Tensor,
        beta: float,
        linear_beta: float | None,
    ) -> Tensor:
        """Run ``fc2(situ(fc1(x)))`` on each local expert slice.

        Routing weights are applied later by ``tokens_post_all2all``. The
        SituGLU activation computes in fp32 exactly like the single-device
        expert loop below.
        """
        num_local = fc1_1_2_weight.shape[0]
        ends = cumsum.to(dtype=torch.long)
        starts = permute_tokens.new_zeros(num_local, dtype=torch.long)
        if num_local > 1:
            starts[1:] = ends[:-1]
        pieces: list[Tensor] = []
        for expert_idx in range(num_local):
            start = int(starts[expert_idx])
            end = int(ends[expert_idx])
            x = permute_tokens[start:end]
            gate, up = F.linear(x, fc1_1_2_weight[expert_idx]).chunk(2, dim=-1)
            gate = gate.to(torch.float32)
            up = up.to(torch.float32)
            situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
            if linear_beta is not None:
                up = linear_beta * torch.tanh(up / linear_beta)
            y = F.linear((situ_a * up).to(permute_tokens.dtype), fc2_weight[expert_idx])
            pieces.append(y)
        output = torch.cat(pieces, dim=0)
        # Empty ranks still have to keep ``permute_tokens`` on the graph so the
        # EP all-to-all backward runs on every rank.
        return output + permute_tokens * 0


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
    """Routed SituGLU expert loop over the merged ``gate_up`` layout.

    Regular autograd, no custom backward. The SituGLU activation computes in
    fp32 exactly like the modeling-side ``SituAndMul`` reference. Under expert
    parallelism the weight tensors hold this rank's ``num_experts // ep_size``
    local rows while ``selected_experts`` stays global, so tokens go through
    the shared all-to-all dispatch (``dispatch_to_ep_class``) and only the
    local expert slice runs here.
    """
    num_local = fc1_1_2_weight.shape[0]
    if num_local != num_experts:
        from veomni.distributed.moe import dispatch_to_ep_class
        from veomni.distributed.parallel_state import get_parallel_state

        state = get_parallel_state()
        if not state.ep_enabled or num_local != num_experts // state.ep_size:
            raise ValueError(
                "kimi_k3 eager MoE received expert-sharded weights "
                f"({num_local} local rows for {num_experts} global experts) "
                "without a matching expert-parallel mesh."
            )
        return dispatch_to_ep_class(
            _SituExpertsEP,
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            fc1_1_2_weight,
            fc2_weight,
            beta,
            linear_beta,
        )

    output = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        idx = int(expert_idx[0].item())
        top_k_pos, token_idx = torch.where(expert_mask[idx])
        x = hidden_states[token_idx]
        gate, up = F.linear(x, fc1_1_2_weight[idx]).chunk(2, dim=-1)
        gate = gate.to(torch.float32)
        up = up.to(torch.float32)
        situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
        if linear_beta is not None:
            up = linear_beta * torch.tanh(up / linear_beta)
        y = F.linear((situ_a * up).to(hidden_states.dtype), fc2_weight[idx])
        y = y * routing_weights[token_idx, top_k_pos, None]
        output = output.index_add(0, token_idx, y.to(output.dtype))
    return output
