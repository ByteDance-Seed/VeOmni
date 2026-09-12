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

"""HF-signature SDPA mask builder."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor
from transformers.masking_utils import (
    ALL_MASK_ATTENTION_FUNCTIONS,
    and_masks,
    bidirectional_mask_function,
    causal_mask_function,
    sliding_window_overlay,
)

from ..ulysses import effective_sequence_lengths, should_apply_ulysses
from .packed import packed_mask_function


def sdpa_attention_mask_builder(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: torch.Tensor | None = None,
    skip_ulysses: bool = False,
    **kwargs,
) -> Tensor | None:
    """HF-signature SDPA mask for ``sdpa`` / ``veomni_sdpa``.

    Expand Ulysses-local lengths only when the adapter would gather Q/K/V
    itself: sync Ulysses and not ``skip_ulysses``. Then call Transformers'
    ``sdpa`` builder. Cached decode (``q_length != kv_length``) cannot use
    SDPA ``is_causal`` skip. Optional ``sliding_window`` / ``cu_seqlens``
    compose onto ``mask_function``. Canonical causal/bidirectional masks need
    no explicit metadata under Ulysses; custom predicates do.
    """
    sliding_window = kwargs.pop("sliding_window", None)
    cu_seqlens = kwargs.pop("cu_seqlens", None)
    cu_seqlens_k = kwargs.pop("cu_seqlens_k", None)
    if cu_seqlens_k is None:
        cu_seqlens_k = kwargs.pop("cu_seq_lens_k", None)
    if attention_mask is not None and attention_mask.dtype != torch.bool:
        attention_mask = attention_mask != 0
    device = kwargs.get("device", attention_mask.device if attention_mask is not None else "cpu")

    if should_apply_ulysses(skip_ulysses=skip_ulysses):
        if q_offset != 0 or kv_offset != 0:
            raise ValueError("SDPA with Ulysses does not support cached mask offsets.")
        if (
            attention_mask is None
            and cu_seqlens is None
            and mask_function not in (causal_mask_function, bidirectional_mask_function)
        ):
            raise ValueError(
                "SDPA with Ulysses requires full-sequence metadata for a custom mask function; "
                "pass a 2D attention mask or cu_seqlens."
            )
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("SDPA with Ulysses requires a full-sequence 2D attention mask.")
        full_q_length, full_kv_length = effective_sequence_lengths(
            q_length,
            kv_length,
            skip_ulysses=skip_ulysses,
        )
        if attention_mask is not None and attention_mask.shape[-1] != full_kv_length:
            raise ValueError(
                "SDPA with Ulysses requires the full attention-mask sequence length to equal "
                f"the post-Ulysses key length, got attention_mask.shape[-1]={attention_mask.shape[-1]} "
                f"and expected {full_kv_length}."
            )
        q_length, kv_length = full_q_length, full_kv_length
        q_offset = kv_offset = 0

    if sliding_window is not None:
        mask_function = and_masks(mask_function, sliding_window_overlay(sliding_window))
    if cu_seqlens is not None:
        mask_function = packed_mask_function(
            mask_function=mask_function,
            q_length=q_length,
            kv_length=kv_length,
            q_offset=q_offset,
            kv_offset=kv_offset,
            cu_seqlens=cu_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            device=device,
        )

    if q_length != kv_length:
        kwargs["allow_is_causal_skip"] = False

    return ALL_MASK_ATTENTION_FUNCTIONS["sdpa"](
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=mask_function,
        attention_mask=attention_mask,
        **kwargs,
    )
