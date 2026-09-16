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
# See the License for the specific language governing limitations
# under the License.

"""Helpers for constructing ``VeomniOp`` handles and SwiGLU activation routing.

Read impl names from ``get_ops_config`` at construct time. ``npu`` on
cross-entropy maps to ``chunk_loss``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from veomni.ops import VeomniOp
from veomni.ops.config import get_ops_config


SWIGLU_HIDDEN_ACTS = frozenset({"silu", "swish"})


def uses_swiglu_mlp(hidden_act: str) -> bool:
    """Whether fused SwiGLU / silu MoE kernels match this ``hidden_act``.

    Those kernels are silu-only. ``swish`` is the same function. Any other
    activation stays on ``self.act_fn`` instead of teaching fused SwiGLU a
    generic activation.
    """
    return hidden_act in SWIGLU_HIDDEN_ACTS


def merged_experts_act_fn_forward(
    hidden_states: Tensor,
    top_k_index: Tensor,
    top_k_weights: Tensor,
    gate_up_proj: Tensor,
    down_proj: Tensor,
    act_fn: Callable[[Tensor], Tensor],
    num_experts: int,
) -> Tensor:
    """HF merged-expert loop: ``down(act_fn(gate) * up)`` then routing weights."""
    final_hidden_states = hidden_states.new_zeros(hidden_states.shape)
    with torch.no_grad():
        expert_mask = nn.functional.one_hot(top_k_index, num_classes=num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero()
    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = nn.functional.linear(current_state, gate_up_proj[expert_idx]).chunk(2, dim=-1)
        current_hidden_states = act_fn(gate) * up
        current_hidden_states = nn.functional.linear(current_hidden_states, down_proj[expert_idx])
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
    return final_hidden_states


def resolve_op_impl(field: str, *, npu_as: str | None = None) -> str:
    """Return the impl name on the installed ops config, or ``eager``.

    ``npu_as`` remaps the ``npu`` CE name to ``chunk_loss``. Missing config
    is eager so unit tests can construct a module without ``set_ops_config``.
    """
    cfg = get_ops_config()
    impl = "eager" if cfg is None else getattr(cfg, field, "eager")
    if npu_as is not None and impl == "npu":
        return npu_as
    return impl


def attention_op() -> VeomniOp:
    """Return the interned standard-attention op for the active impl.

    Missing ops config resolves to ``eager``. Construct this only in
    ``__init__`` / ``modify_init`` and store the handle on ``self``.
    Do not call it from ``forward``.
    """
    return VeomniOp("attention", "standard", resolve_op_impl("attn_implementation"))


PACKED_ATTENTION_METADATA_KEYS = frozenset(
    {
        "cu_seqlens",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "cu_seq_lens_q",
        "cu_seq_lens_k",
        "max_length_q",
        "max_length_k",
        "max_seqlen_q",
        "max_seqlen_k",
    }
)


def drop_packed_attention_metadata(kwargs: dict, *, impl: str) -> dict:
    """Strip GDN/varlen metadata that SDPA and eager attention reject.

    Flash and other packed-capable impls keep the keys. Linear-attention
    layers should pass ``cu_seq_lens_q`` explicitly instead of through this
    filter.
    """
    if impl in {"eager", "sdpa"}:
        return {key: value for key, value in kwargs.items() if key not in PACKED_ATTENTION_METADATA_KEYS}
    return kwargs


def resolve_moe_impl() -> str:
    """Return the active ``moe_experts`` impl from the ops config."""
    return resolve_op_impl("moe_implementation")


def resolve_qat_impl() -> str:
    """Return the active model-level quantization recipe, or ``none``."""
    cfg = get_ops_config()
    return "none" if cfg is None else getattr(cfg, "qat_implementation", "none")


def empty_bias(weight: Tensor) -> Tensor:
    """Empty unused-layout bias for a Linear that has ``bias=None``."""
    return weight.new_empty(0)


def linear_bias(linear: nn.Linear) -> Tensor:
    """Return the Linear bias, or the empty unused-layout sentinel."""
    return linear.bias if linear.bias is not None else empty_bias(linear.weight)
