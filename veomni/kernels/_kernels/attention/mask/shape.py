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
# See the License for the specific language governing limitations
# under the License.

"""Pattern mask APIs dispatched by attention impl."""

from __future__ import annotations

from typing import Any

import torch

from .flash import flash_attention_mask_builder
from .flex import flex_attention_mask_builder
from .magi import MagiAttentionMask, magi_attention_mask_builder
from .sdpa import sdpa_attention_mask_builder


_FLASH = frozenset({"flash_attention_2", "flash_attention_3", "flash_attention_4"})
_SDPA = frozenset({"sdpa", "eager", "native-sparse"})


def causal_mask(
    q_len: int,
    kv_len: int,
    *,
    impl: str,
    device: torch.device | str,
    batch_size: int = 1,
    **kwargs: Any,
):
    """Build a causal mask for ``impl``, or ``None`` when flash uses kwargs."""
    backend = impl.removeprefix("veomni_")
    if backend in _FLASH:
        return flash_attention_mask_builder()
    if backend in _SDPA:
        return sdpa_attention_mask_builder(
            batch_size,
            q_len,
            kv_len,
            q_offset=kv_len - q_len,
            device=device,
            allow_is_causal_skip=False,
        )
    if backend == "flex_attention":
        return flex_attention_mask_builder(
            batch_size=batch_size,
            q_length=q_len,
            kv_length=kv_len,
            **kwargs,
        )
    if backend == "magi_attention":
        return magi_attention_mask_builder(
            batch_size=batch_size,
            q_length=q_len,
            kv_length=kv_len,
            device=device,
            **kwargs,
        )
    raise ValueError(f"unsupported attention impl for causal_mask: {impl!r}")


def sliding_window_mask(
    q_len: int,
    kv_len: int,
    *,
    impl: str,
    device: torch.device | str,
    sliding_window: int,
    batch_size: int = 1,
    **kwargs: Any,
):
    """Sliding-window causal mask. Flash returns ``None``."""
    backend = impl.removeprefix("veomni_")
    if backend in _FLASH:
        return flash_attention_mask_builder()
    if backend in _SDPA:
        return sdpa_attention_mask_builder(
            batch_size,
            q_len,
            kv_len,
            q_offset=kv_len - q_len,
            device=device,
            sliding_window=sliding_window,
            allow_is_causal_skip=False,
        )
    if backend == "flex_attention":
        return flex_attention_mask_builder(
            batch_size=batch_size,
            q_length=q_len,
            kv_length=kv_len,
            **kwargs,
        )
    if backend == "magi_attention":
        raise ValueError("MagiAttention encodes sliding windows in ranges, not sliding_window_mask")
    raise ValueError(f"unsupported attention impl for sliding_window_mask: {impl!r}")


def packed_causal_mask(
    q_len: int,
    kv_len: int,
    *,
    impl: str,
    device: torch.device | str,
    cu_seqlens: torch.Tensor,
    batch_size: int = 1,
    **kwargs: Any,
):
    """Packed causal mask from ``cu_seqlens``. Flash returns ``None``."""
    backend = impl.removeprefix("veomni_")
    if backend in _FLASH:
        return flash_attention_mask_builder()
    if backend in _SDPA:
        return sdpa_attention_mask_builder(
            batch_size,
            q_len,
            kv_len,
            q_offset=kv_len - q_len,
            device=device,
            cu_seqlens=cu_seqlens,
            cu_seqlens_k=kwargs.pop("cu_seq_lens_k", None),
            allow_is_causal_skip=False,
        )
    if backend == "flex_attention":
        return flex_attention_mask_builder(
            batch_size=batch_size,
            q_length=q_len,
            kv_length=kv_len,
            **kwargs,
        )
    if backend == "magi_attention":
        return MagiAttentionMask.from_cu_seqlens(
            cu_seqlens,
            kwargs.pop("cu_seq_lens_k", cu_seqlens),
            device=device,
        )
    raise ValueError(f"unsupported attention impl for packed_causal_mask: {impl!r}")
