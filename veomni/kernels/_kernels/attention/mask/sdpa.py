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
    causal_mask_function,
    packed_sequence_mask_function,
    sliding_window_overlay,
)

from .....distributed.parallel_state import get_parallel_state


def sdpa_attention_mask_builder(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
) -> Tensor | None:
    """HF-signature SDPA mask for ``sdpa`` / ``veomni_sdpa``.

    Expands Ulysses-local lengths, then calls Transformers' ``sdpa`` builder.
    Cached decode (``q_length != kv_length``) cannot use SDPA ``is_causal`` skip.
    Optional ``sliding_window`` / ``cu_seqlens`` compose onto ``mask_function``.
    """
    sliding_window = kwargs.pop("sliding_window", None)
    cu_seqlens = kwargs.pop("cu_seqlens", None)
    cu_seqlens_k = kwargs.pop("cu_seqlens_k", None)
    if cu_seqlens_k is None:
        cu_seqlens_k = kwargs.pop("cu_seq_lens_k", None)
    device = kwargs.get("device", attention_mask.device if attention_mask is not None else "cpu")

    if sliding_window is not None:
        mask_function = and_masks(mask_function, sliding_window_overlay(sliding_window))
    if cu_seqlens is not None:
        mask_function = and_masks(
            mask_function,
            packed_sequence_mask_function(
                _packed_segment_ids(
                    batch_size=batch_size,
                    q_length=q_length,
                    kv_length=kv_length,
                    cu_seqlens=cu_seqlens,
                    cu_seqlens_k=cu_seqlens_k,
                    device=device,
                )
            ),
        )

    parallel_state = get_parallel_state()
    if parallel_state.ulysses_enabled:
        if q_offset != 0 or kv_offset != 0:
            raise ValueError("SDPA with Ulysses does not support cached mask offsets.")
        if attention_mask is None or attention_mask.ndim != 2:
            raise ValueError("SDPA with Ulysses requires a full-sequence 2D attention mask.")
        full_sequence_length = q_length * parallel_state.ulysses_size
        if attention_mask.shape[-1] != full_sequence_length:
            raise ValueError(
                "SDPA with Ulysses requires the full attention-mask sequence length to equal "
                f"local q_length * ulysses_size, got attention_mask.shape[-1]={attention_mask.shape[-1]}, "
                f"q_length={q_length}, ulysses_size={parallel_state.ulysses_size}."
            )
        q_length = kv_length = full_sequence_length
        q_offset = kv_offset = 0

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


def _packed_segment_ids(
    *,
    batch_size: int,
    q_length: int,
    kv_length: int,
    cu_seqlens: Tensor,
    cu_seqlens_k: Tensor | None,
    device: torch.device | str,
) -> Tensor:
    """``[batch, kv_length]`` segment ids for HF ``packed_sequence_mask_function``.

    Query indices are global (``q_offset`` already applied by the SDPA builder),
    so packed ids follow the key sequence. ``cu_seqlens_k`` is required when
    ``q_length != kv_length``.
    """
    if cu_seqlens_k is None and q_length != kv_length:
        raise ValueError("packed SDPA with q_length != kv_length requires cu_seqlens_k")
    packed_cu = cu_seqlens if cu_seqlens_k is None else cu_seqlens_k
    return _segment_ids(packed_cu, kv_length, torch.device(device)).unsqueeze(0).expand(batch_size, -1)


def _segment_ids(cu_seqlens: Tensor, length: int, device: torch.device) -> Tensor:
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError(f"cu_seqlens must have shape [n_seg + 1], got {tuple(cu_seqlens.shape)}")
    cu_seqlens = cu_seqlens.to(device=device)
    if int(cu_seqlens[0]) != 0 or int(cu_seqlens[-1]) != length:
        raise ValueError(f"cu_seqlens must run from 0 to {length}, got {cu_seqlens.tolist()}")
    return torch.bucketize(torch.arange(length, device=device), cu_seqlens[1:], right=True)
