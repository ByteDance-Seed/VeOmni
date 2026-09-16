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


class _MergedExpertsActFnEP:
    """Local non-SiLU expert math on tokens already dispatched by EP."""

    @staticmethod
    def apply(
        permute_tokens: Tensor,
        cumsum: Tensor,
        gate_up_proj: Tensor,
        down_proj: Tensor,
        act_fn: Callable[[Tensor], Tensor],
    ) -> Tensor:
        """Run ``down(act_fn(gate) * up)`` on each local expert slice.

        Routing weights are applied later by ``tokens_post_all2all``.
        """
        num_local = gate_up_proj.shape[0]
        ends = cumsum.to(dtype=torch.long)
        starts = permute_tokens.new_zeros(num_local, dtype=torch.long)
        if num_local > 1:
            starts[1:] = ends[:-1]
        pieces: list[Tensor] = []
        for expert_idx in range(num_local):
            start = int(starts[expert_idx])
            end = int(ends[expert_idx])
            current_state = permute_tokens[start:end]
            gate, up = nn.functional.linear(current_state, gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, down_proj[expert_idx])
            pieces.append(current_hidden_states.to(permute_tokens.dtype))
        output = torch.cat(pieces, dim=0)
        # Empty ranks still have to keep ``permute_tokens`` on the graph so the
        # EP all-to-all backward runs on every rank.
        return output + permute_tokens * 0


def merged_experts_act_fn_forward(
    hidden_states: Tensor,
    top_k_index: Tensor,
    top_k_weights: Tensor,
    gate_up_proj: Tensor,
    down_proj: Tensor,
    act_fn: Callable[[Tensor], Tensor],
    num_experts: int,
) -> Tensor:
    """HF merged-expert loop: ``down(act_fn(gate) * up)`` then routing weights.

    Expert-parallel shards store only local rows on ``gate_up_proj``. Those
    tokens must go through the same EP dispatch as ``veomni_moe``; indexing
    with global expert ids would read past the local shard.
    """
    local_experts = gate_up_proj.shape[0]
    if local_experts != num_experts:
        from veomni.distributed.moe import dispatch_to_ep_class
        from veomni.distributed.parallel_state import get_parallel_state

        state = get_parallel_state()
        if not state.ep_enabled or local_experts != num_experts // state.ep_size:
            raise ValueError(
                "non-SiLU MoE fallback received expert-parallel sharded weights "
                f"(local {local_experts} vs global {num_experts}). "
                "Enable EP so tokens are dispatched to local experts, or use silu."
            )
        return dispatch_to_ep_class(
            _MergedExpertsActFnEP,
            num_experts,
            top_k_weights,
            top_k_index,
            hidden_states,
            gate_up_proj,
            down_proj,
            act_fn,
        )

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
DENSE_ATTENTION_IMPLS = frozenset({"eager", "sdpa"})


def _canonical_attn_impl(impl: str) -> str:
    """Strip the public ``veomni_`` prefix used by ``OpsImplementationConfig``."""
    return impl.removeprefix("veomni_")


def _multi_segment_cu_seqlens(kwargs: dict) -> Tensor | None:
    """Return packed cumulative lengths when they encode more than one sample."""
    for key in ("cu_seq_lens_q", "cu_seqlens_q", "cu_seqlens"):
        value = kwargs.get(key)
        if torch.is_tensor(value) and value.numel() >= 3:
            return value
    return None


def dense_packed_attention_mask(
    *,
    q_len: int,
    kv_len: int,
    cu_seqlens: Tensor,
    attention_mask: Tensor | None,
    batch_size: int,
    impl: str,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tensor:
    """Dense packed causal mask for SDPA/eager, which have no varlen kwargs.

    A 2-D padding mask is composed with packed isolation. A 3-D/4-D mask is
    merged afterwards so padding and custom overlays are not replaced.
    """
    from veomni.ops.kernels.attention.mask.sdpa import _dense_attention_mask_builder
    from veomni.ops.kernels.attention.mask.shape import _to_eager_additive

    padding = attention_mask if attention_mask is not None and attention_mask.ndim == 2 else None
    mask = _dense_attention_mask_builder(
        batch_size,
        q_len,
        kv_len,
        q_offset=kv_len - q_len,
        attention_mask=padding,
        cu_seqlens=cu_seqlens,
        device=device,
        allow_is_causal_skip=False,
    )
    if attention_mask is not None and attention_mask.ndim >= 3:
        mask = _merge_dense_attention_masks(attention_mask, mask)
    if _canonical_attn_impl(impl) == "eager":
        mask = _to_eager_additive(mask, dtype)
    return mask


def _merge_dense_attention_masks(existing: Tensor, packed: Tensor) -> Tensor:
    """Block packed-forbidden positions; keep existing values on the rest.

    Packed isolation is a visibility overlay. Allowed positions retain padding
    and additive bias from ``existing`` instead of being clipped to zero.
    """
    packed_view = packed
    while packed_view.ndim < existing.ndim:
        packed_view = packed_view.unsqueeze(1)
    while packed_view.ndim > existing.ndim:
        if packed_view.shape[1] != 1:
            raise ValueError(
                f"cannot align packed mask {tuple(packed_view.shape)} with existing {tuple(existing.shape)}"
            )
        packed_view = packed_view.squeeze(1)
    if packed_view.shape[-2:] != existing.shape[-2:]:
        raise ValueError(
            f"packed mask q/kv {tuple(packed_view.shape[-2:])} does not match existing {tuple(existing.shape[-2:])}"
        )
    packed_view = packed_view.expand_as(existing)
    packed_keep = packed_view if packed_view.dtype == torch.bool else packed_view >= 0
    packed_keep = packed_keep.to(dtype=torch.bool)
    if existing.dtype == torch.bool:
        return existing & packed_keep
    # A finite sentinel would win over -inf when the query's own sample is
    # fully masked, allowing attention (and gradients) into another sample.
    return existing.masked_fill(~packed_keep, float("-inf"))


def drop_packed_attention_metadata(kwargs: dict, *, impl: str) -> dict:
    """Strip GDN/varlen metadata that SDPA and eager attention reject.

    ``veomni_sdpa`` is the public builder alias of ``sdpa``. Flash and other
    packed-capable impls keep the keys. Linear-attention layers should pass
    ``cu_seq_lens_q`` explicitly instead of through this filter.
    """
    if _canonical_attn_impl(impl) in DENSE_ATTENTION_IMPLS:
        return {key: value for key, value in kwargs.items() if key not in PACKED_ATTENTION_METADATA_KEYS}
    return kwargs


def prepare_dense_attention_inputs(
    kwargs: dict,
    *,
    impl: str,
    attention_mask: Tensor | None,
    hidden_states: Tensor,
) -> tuple[dict, Tensor | None]:
    """Drop packed kwargs for SDPA/eager, after building an isolating dense mask.

    Single-segment or empty ``cu_seq_lens_q`` can be stripped as-is. True
    packed inputs need a dense mask first; dropping lengths alone lets a
    2-D all-ones mask cross-attend across samples. An existing 4-D mask is
    merged with packed isolation so padding and overlays stay in place.
    """
    if _canonical_attn_impl(impl) not in DENSE_ATTENTION_IMPLS:
        return kwargs, attention_mask
    cu_seqlens = _multi_segment_cu_seqlens(kwargs)
    if cu_seqlens is not None:
        q_len = hidden_states.shape[1]
        kv_len = q_len
        if attention_mask is not None and attention_mask.ndim >= 3:
            kv_len = attention_mask.shape[-1]
            if attention_mask.shape[-2] != q_len:
                raise ValueError(
                    "packed SDPA/eager attention requires the existing mask query length "
                    f"to match hidden_states, got {attention_mask.shape[-2]} and {q_len}"
                )
        if kv_len != q_len:
            raise ValueError(
                "packed SDPA/eager attention does not support cached sequences "
                f"(q_len={q_len}, kv_len={kv_len}); use a packed-capable impl or disable cache"
            )
        attention_mask = dense_packed_attention_mask(
            q_len=q_len,
            kv_len=kv_len,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            batch_size=hidden_states.shape[0],
            impl=impl,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    return drop_packed_attention_metadata(kwargs, impl=impl), attention_mask


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
