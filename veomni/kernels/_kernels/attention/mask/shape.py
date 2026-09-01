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
from transformers.masking_utils import (
    and_masks,
    causal_mask_function,
    or_masks,
    packed_sequence_mask_function,
    sliding_window_overlay,
)

from .flash import flash_attention_mask_builder
from .flex import flex_attention_mask_builder
from .magi import MagiAttentionMask, magi_attention_mask_builder
from .sdpa import _packed_segment_ids, sdpa_attention_mask_builder


_FLASH = frozenset({"flash_attention_2", "flash_attention_3", "flash_attention_4"})
_SDPA = frozenset({"sdpa", "native-sparse"})
_EAGER = frozenset({"eager"})


def _compose_or_and(kwargs: dict[str, Any]) -> dict[str, Any]:
    extra = dict(kwargs)
    mask_function = extra.pop("mask_function", causal_mask_function)
    or_mask_function = extra.pop("or_mask_function", None)
    and_mask_function = extra.pop("and_mask_function", None)
    if or_mask_function is not None:
        mask_function = or_masks(mask_function, or_mask_function)
    if and_mask_function is not None:
        mask_function = and_masks(mask_function, and_mask_function)
    extra["mask_function"] = mask_function
    extra.pop("device", None)
    return extra


def _to_eager_additive(mask: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
    """HF eager adds the mask onto scores, so keep 0 / -inf rather than bool."""
    if mask is None:
        return None
    if mask.dtype.is_floating_point:
        return mask
    min_dtype = torch.finfo(dtype).min
    return torch.where(mask, torch.zeros((), device=mask.device, dtype=dtype), min_dtype)


def _sdpa_or_eager_mask(
    backend: str,
    batch_size: int,
    q_len: int,
    kv_len: int,
    device: torch.device | str,
    extra: dict[str, Any],
    *,
    sliding_window: int | None = None,
    cu_seqlens: torch.Tensor | None = None,
):
    dtype = extra.pop("dtype", torch.float32)
    if sliding_window is not None:
        extra["sliding_window"] = sliding_window
    if cu_seqlens is not None:
        extra["cu_seqlens"] = cu_seqlens
    mask = sdpa_attention_mask_builder(
        batch_size,
        q_len,
        kv_len,
        q_offset=kv_len - q_len,
        device=device,
        allow_is_causal_skip=False,
        **extra,
    )
    if backend in _EAGER:
        return _to_eager_additive(mask, dtype)
    return mask


def _flex_mask(
    batch_size: int,
    q_len: int,
    kv_len: int,
    extra: dict[str, Any],
    *,
    sliding_window: int | None = None,
    cu_seqlens: torch.Tensor | None = None,
    device: torch.device | str,
):
    extra.pop("dtype", None)
    mask_function = extra.get("mask_function", causal_mask_function)
    if sliding_window is not None:
        mask_function = and_masks(mask_function, sliding_window_overlay(sliding_window))
    if cu_seqlens is not None:
        mask_function = and_masks(
            mask_function,
            packed_sequence_mask_function(
                _packed_segment_ids(
                    batch_size=batch_size,
                    q_length=q_len,
                    kv_length=kv_len,
                    cu_seqlens=cu_seqlens,
                    cu_seqlens_k=extra.pop("cu_seq_lens_k", extra.pop("cu_seqlens_k", None)),
                    device=device,
                )
            ),
        )
    extra["mask_function"] = mask_function
    return flex_attention_mask_builder(
        batch_size=batch_size,
        q_length=q_len,
        kv_length=kv_len,
        **extra,
    )


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
    extra = _compose_or_and(kwargs)
    if backend in _FLASH:
        return flash_attention_mask_builder()
    if backend in _SDPA or backend in _EAGER:
        return _sdpa_or_eager_mask(backend, batch_size, q_len, kv_len, device, extra)
    if backend == "flex_attention":
        return _flex_mask(batch_size, q_len, kv_len, extra, device=device)
    if backend == "magi_attention":
        extra.pop("dtype", None)
        extra.pop("mask_function", None)
        return magi_attention_mask_builder(
            batch_size=batch_size,
            q_length=q_len,
            kv_length=kv_len,
            device=device,
            **extra,
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
    extra = _compose_or_and(kwargs)
    cu_seqlens = extra.pop("cu_seqlens", None)
    if backend in _FLASH:
        return flash_attention_mask_builder()
    if backend in _SDPA or backend in _EAGER:
        return _sdpa_or_eager_mask(
            backend,
            batch_size,
            q_len,
            kv_len,
            device,
            extra,
            sliding_window=sliding_window,
            cu_seqlens=cu_seqlens,
        )
    if backend == "flex_attention":
        return _flex_mask(
            batch_size,
            q_len,
            kv_len,
            extra,
            sliding_window=sliding_window,
            cu_seqlens=cu_seqlens,
            device=device,
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
    extra = _compose_or_and(kwargs)
    if backend in _FLASH:
        return flash_attention_mask_builder()
    if backend in _SDPA or backend in _EAGER:
        return _sdpa_or_eager_mask(
            backend,
            batch_size,
            q_len,
            kv_len,
            device,
            extra,
            cu_seqlens=cu_seqlens,
        )
    if backend == "flex_attention":
        return _flex_mask(
            batch_size,
            q_len,
            kv_len,
            extra,
            cu_seqlens=cu_seqlens,
            device=device,
        )
    if backend == "magi_attention":
        extra.pop("dtype", None)
        extra.pop("mask_function", None)
        return MagiAttentionMask.from_cu_seqlens(
            cu_seqlens,
            extra.pop("cu_seq_lens_k", extra.pop("cu_seqlens_k", cu_seqlens)),
            device=device,
        )
    raise ValueError(f"unsupported attention impl for packed_causal_mask: {impl!r}")
