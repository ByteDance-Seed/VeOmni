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

"""Attention helpers for models."""

from __future__ import annotations

import torch
from torch import Tensor


VARLEN_ATTENTION_TYPES = (
    "flash_attention_2",
    "flash_attention_2_hub",
    "flash_attention_3",
    "flash_attention_3_hub",
    "flash_attention_4",
    "veomni_flash_attention_2",
    "veomni_flash_attention_2_hub",
    "veomni_flash_attention_3",
    "veomni_flash_attention_3_hub",
    "veomni_flash_attention_4",
)
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
    Eager converts bool False to ``finfo.min`` first, then reapplies packed
    isolation as ``-inf`` so a fully-masked sample cannot softmax across
    another sample.
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
        packed_only = _dense_attention_mask_builder(
            batch_size,
            q_len,
            kv_len,
            q_offset=kv_len - q_len,
            attention_mask=None,
            cu_seqlens=cu_seqlens,
            device=device,
            allow_is_causal_skip=False,
        )
        mask = _merge_dense_attention_masks(mask, packed_only)
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
