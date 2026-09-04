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

"""standard load-balancing loss eager math (Switch Transformer aux loss)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .....registry import SavedState


@dataclass(frozen=True)
class _Meta:
    """``top_k`` plus whether ``attention_mask`` had any values."""

    top_k: int
    has_mask: bool


def token_mask(gate_logits: Tensor, attention_mask: Tensor) -> Tensor | None:
    """Expand ``[batch, seq]`` across concatenated layers to ``[N]``.

    ``gate_logits`` is ``[N, E]`` after ``torch.cat`` of per-layer
    ``[tokens, E]``. Empty ``attention_mask`` means every token counts.
    ``N`` must be a multiple of ``batch * seq``.
    """
    if attention_mask.numel() == 0:
        return None
    if gate_logits.ndim != 2:
        raise ValueError(f"gate_logits must be [N, E], got {tuple(gate_logits.shape)}")
    token_count, _num_experts = gate_logits.shape
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be [batch, seq], got {tuple(attention_mask.shape)}")
    batch_size, seq_len = attention_mask.shape
    tokens_per_layer = batch_size * seq_len
    if tokens_per_layer == 0 or token_count % tokens_per_layer != 0:
        raise ValueError(
            f"gate_logits rows ({token_count}) must be a multiple of attention_mask {batch_size}*{seq_len}"
        )
    num_layers = token_count // tokens_per_layer
    return attention_mask.reshape(-1).to(device=gate_logits.device, dtype=torch.float32).repeat(num_layers)


def forward(gate_logits: Tensor, attention_mask: Tensor, *, top_k: int) -> tuple[Tensor, SavedState]:
    """Switch Transformer load-balancing loss on concatenated layer logits.

    ``gate_logits`` is ``[N, num_experts]`` after ``torch.cat`` of the
    per-layer tuple. Empty ``attention_mask`` means every token counts.
    Top-k counts are treated as constants in backward.
    """
    if gate_logits.ndim != 2:
        raise ValueError(f"gate_logits must be [N, E], got {tuple(gate_logits.shape)}")
    token_count, num_experts = gate_logits.shape
    device = gate_logits.device
    mask = token_mask(gate_logits, attention_mask)

    probs = torch.softmax(gate_logits.float(), dim=-1)
    _values, selected = torch.topk(probs, top_k, dim=-1)
    if mask is not None:
        weights = mask.unsqueeze(-1)
        router_prob_sum = (probs * weights).sum(dim=0)
        expert_count = torch.zeros(num_experts, device=device, dtype=torch.float32)
        expert_count.scatter_add_(0, selected.reshape(-1), weights.expand_as(selected).reshape(-1))
        total_weight = mask.sum()
    else:
        router_prob_sum = probs.sum(dim=0)
        expert_count = torch.zeros(num_experts, device=device, dtype=torch.float32)
        expert_count.scatter_add_(
            0, selected.reshape(-1), torch.ones(selected.numel(), device=device, dtype=torch.float32)
        )
        total_weight = torch.tensor(float(token_count), device=device)

    if total_weight == 0:
        loss = torch.zeros((), device=device, dtype=torch.float32)
    else:
        loss = torch.dot(expert_count, router_prob_sum) * (num_experts / (total_weight * total_weight))
    return loss, SavedState((gate_logits, attention_mask, expert_count, total_weight), _Meta(top_k, mask is not None))


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, None]:
    """Return ``(grad_gate_logits, None)``. Mask is not differentiated."""
    gate_logits, attention_mask, expert_count, total_weight = saved.tensors
    if total_weight == 0:
        return torch.zeros_like(gate_logits), None

    num_experts = gate_logits.shape[-1]
    scale = grad_output * num_experts / (total_weight * total_weight)
    probs = torch.softmax(gate_logits.float(), dim=-1)
    dot_cs = (probs * expert_count).sum(dim=-1, keepdim=True)
    mask = token_mask(gate_logits, attention_mask)
    weights = mask.unsqueeze(-1) if mask is not None else 1.0
    grad = scale * weights * probs * (expert_count - dot_cs)
    return grad.to(dtype=gate_logits.dtype), None
