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

"""gpt_oss MoE experts eager math (HF ``GptOssExperts`` loop)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .....distributed.moe import expert_parallel_size


def _gpt_oss_activation(gate_up: Tensor, alpha: float, limit: float) -> Tensor:
    """Interleaved clamped gate/up activation: ``(up + 1) * gate * sigmoid(alpha * gate)``.

    Unlike the ``standard`` family, GPT-OSS interleaves gate and up in one tensor
    (``[..., ::2]`` / ``[..., 1::2]``) rather than splitting it down the middle.
    """
    gate = gate_up[..., ::2].clamp(max=limit)
    up = gate_up[..., 1::2].clamp(min=-limit, max=limit)
    return (up + 1) * (gate * torch.sigmoid(gate * alpha))


class _GptOssExpertMathEP:
    """GPT-OSS expert math on tokens already dispatched by expert parallelism.

    ``cumsum`` bounds the rows per local expert, so each slice reads the local shard of
    ``gate_up_proj`` / ``down_proj`` with a local index. Routing weights are applied
    afterwards by ``tokens_post_all2all``, matching the non-EP branch's post-``down``
    order.
    """

    @staticmethod
    def apply(
        permute_tokens: Tensor,
        cumsum: Tensor,
        gate_up_proj: Tensor,
        gate_up_proj_bias: Tensor,
        down_proj: Tensor,
        down_proj_bias: Tensor,
        alpha: float,
        limit: float,
    ) -> Tensor:
        """Run the GPT-OSS expert MLP on each local expert's dispatched rows."""
        num_local = gate_up_proj.shape[0]
        ends = cumsum.to(dtype=torch.long)
        starts = permute_tokens.new_zeros(num_local, dtype=torch.long)
        if num_local > 1:
            starts[1:] = ends[:-1]
        pieces: list[Tensor] = []
        for expert_idx in range(num_local):
            current = permute_tokens[int(starts[expert_idx]) : int(ends[expert_idx])]
            gate_up = current @ gate_up_proj[expert_idx] + gate_up_proj_bias[expert_idx]
            gated = _gpt_oss_activation(gate_up, alpha, limit)
            pieces.append((gated @ down_proj[expert_idx] + down_proj_bias[expert_idx]).to(permute_tokens.dtype))
        output = torch.cat(pieces, dim=0)
        # An empty rank must stay on the graph so its all-to-all backward runs.
        return output + permute_tokens * 0


def wrapper(
    hidden_states: Tensor,
    routing_weights: Tensor,
    selected_experts: Tensor,
    gate_up_proj: Tensor,
    gate_up_proj_bias: Tensor,
    down_proj: Tensor,
    down_proj_bias: Tensor,
    *,
    num_experts: int,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> Tensor:
    """Routed GPT-OSS expert MLP. Routing weights are applied after ``down_proj``.

    Weights are right-multiplied (``x @ W + b``), matching HF ``GptOssExperts``.
    Regular autograd, no custom backward.

    Under expert parallelism the weights are sharded along the expert dim, so a global
    expert id would index past the local shard. This dispatches through the same
    all-to-all pipeline the other rows use; the per-rank math stays here because the
    interleaved gate/up layout and the clamped activation are GPT-OSS's own.
    """
    ep_size = expert_parallel_size()
    if ep_size > 1:
        from .....distributed.moe import dispatch_to_ep_class

        if num_experts % ep_size or gate_up_proj.shape[0] * ep_size != num_experts:
            # Multiply back rather than compare against a floored quotient: an uneven
            # split would make `num_experts // ep_size` pass for the wrong shard.
            raise ValueError(
                "GPT-OSS eager experts received expert-parallel weights that are not sharded "
                f"as expected (local {gate_up_proj.shape[0]} vs global {num_experts} over "
                f"ep_size {ep_size}). Enable expert parallelism so tokens are dispatched to "
                "local experts."
            )
        # The non-EP loop applies the routing weights through `.to(output.dtype)`; here
        # `unpermute` does it, so cast up front as every fused row does at its entry.
        return dispatch_to_ep_class(
            _GptOssExpertMathEP,
            num_experts,
            routing_weights.to(dtype=hidden_states.dtype),
            selected_experts,
            hidden_states,
            gate_up_proj,
            gate_up_proj_bias,
            down_proj,
            down_proj_bias,
            alpha,
            limit,
        )

    output = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        idx = int(expert_idx[0].item())
        top_k_pos, token_idx = torch.where(expert_mask[idx])
        current = hidden_states[token_idx]
        gate_up = current @ gate_up_proj[idx] + gate_up_proj_bias[idx]
        gated = _gpt_oss_activation(gate_up, alpha, limit)
        out = gated @ down_proj[idx] + down_proj_bias[idx]
        # HF GptOssExperts applies routing after the down projection.
        weighted = out * routing_weights[token_idx, top_k_pos, None]
        output = output.index_add(0, token_idx, weighted.to(output.dtype))
    return output + hidden_states * 0
