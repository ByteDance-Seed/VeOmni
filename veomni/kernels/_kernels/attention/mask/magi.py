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

"""Tensor-native MagiAttention mask contract and Transformers builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from transformers.masking_utils import (
    bidirectional_mask_function,
    causal_mask_function,
)

from .....distributed.parallel_state import get_parallel_state
from ..helper import require_all


@dataclass(frozen=True)
class MagiAttentionMask:
    """Range-based attention mask consumed by MagiAttention's FFA kernel.

    ``q_ranges`` and ``k_ranges`` contain paired half-open token ranges with
    shape ``[num_ranges, 2]`` and dtype ``torch.int32``. ``attn_type_map`` is
    optional; when present, its values are ``0=full``, ``1=causal``,
    ``2=inverse causal``, and ``3=bidirectional causal``.
    """

    q_ranges: torch.Tensor
    k_ranges: torch.Tensor
    attn_type_map: torch.Tensor | None = None

    def __post_init__(self) -> None:
        _validate_ranges(self.q_ranges, self.k_ranges)
        if self.attn_type_map is not None:
            _validate_attn_type_map(self.attn_type_map, num_ranges=self.q_ranges.shape[0], device=self.q_ranges.device)

    @classmethod
    def from_ranges(
        cls,
        q_ranges: torch.Tensor,
        k_ranges: torch.Tensor,
        attn_type_map: torch.Tensor | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> MagiAttentionMask:
        """Build a mask from explicit ranges, casting to the FFA tensor contract."""
        device = q_ranges.device if device is None else torch.device(device)
        q_ranges = q_ranges.to(device=device, dtype=torch.int32).contiguous()
        k_ranges = k_ranges.to(device=device, dtype=torch.int32).contiguous()
        if attn_type_map is not None:
            attn_type_map = attn_type_map.to(device=device, dtype=torch.int32).contiguous()
        return cls(q_ranges, k_ranges, attn_type_map)

    @classmethod
    def from_cu_seqlens(
        cls,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor | None = None,
        *,
        causal: bool = True,
        device: torch.device | str | None = None,
    ) -> MagiAttentionMask:
        """Build a packed range mask from cumulative sequence lengths."""
        if cu_seqlens_k is None:
            cu_seqlens_k = cu_seqlens_q
        device = cu_seqlens_q.device if device is None else torch.device(device)
        q_ranges = _ranges_from_cu_seqlens(cu_seqlens_q, device)
        k_ranges = _ranges_from_cu_seqlens(cu_seqlens_k, device)
        attn_type_map = torch.ones(q_ranges.shape[0], device=device, dtype=torch.int32) if causal else None
        result = cls.from_ranges(q_ranges, k_ranges, attn_type_map, device=device)
        if causal:
            require_all(
                result.q_ranges.diff(dim=1) == result.k_ranges.diff(dim=1),
                "Packed causal MagiAttention requires matching query and key segment lengths.",
            )
        return result


def magi_attention_mask_builder(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
) -> MagiAttentionMask:
    """HF-signature Magi mask for unpacked causal / bidirectional attention."""
    if batch_size != 1:
        raise ValueError(f"MagiAttention mask creation requires physical batch size 1, got {batch_size}.")
    if q_offset != 0 or kv_offset != 0:
        raise ValueError(
            "MagiAttention mask creation does not support KV-cache offsets; "
            f"got q_offset={q_offset} and kv_offset={kv_offset}."
        )
    if mask_function is causal_mask_function:
        causal = True
    elif mask_function is bidirectional_mask_function:
        causal = False
    else:
        raise ValueError(
            "The registered MagiAttention mask builder supports only canonical causal or bidirectional masks. "
            "Packed and model-specific visibility must use MagiAttentionMask.from_ranges or from_cu_seqlens."
        )

    device = kwargs.get("device")
    if device is None and attention_mask is not None:
        device = attention_mask.device
    if device is None:
        raise ValueError("MagiAttention mask creation requires a device or tensor metadata.")
    device = torch.device(device)

    full_q_length, full_kv_length = _full_sequence_lengths(q_length, kv_length)
    if attention_mask is not None and attention_mask.shape[-1] != full_kv_length:
        raise ValueError(
            "MagiAttention attention_mask must describe the full post-Ulysses key sequence, "
            f"got mask length {attention_mask.shape[-1]} and expected {full_kv_length}."
        )

    attn_type_map = torch.ones(1, device=device, dtype=torch.int32) if causal else None
    return MagiAttentionMask.from_ranges(
        torch.tensor([[0, full_q_length]], device=device, dtype=torch.int32),
        torch.tensor([[0, full_kv_length]], device=device, dtype=torch.int32),
        attn_type_map,
        device=device,
    )


def _full_sequence_lengths(q_length: int, kv_length: int) -> tuple[int, int]:
    parallel_state = get_parallel_state()
    scale = parallel_state.ulysses_size if parallel_state.ulysses_enabled else 1
    return q_length * scale, kv_length * scale


def _ranges_from_cu_seqlens(cu_seqlens: torch.Tensor, device: torch.device) -> torch.Tensor:
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError(f"cu_seqlens must have shape [num_sequences + 1], got {tuple(cu_seqlens.shape)}.")
    cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int32)
    if int(cu_seqlens[0]) != 0:
        raise ValueError(f"cu_seqlens must start at 0, got {cu_seqlens.tolist()}.")
    return torch.stack((cu_seqlens[:-1], cu_seqlens[1:]), dim=1).contiguous()


def _validate_ranges(q_ranges: torch.Tensor, k_ranges: torch.Tensor) -> None:
    for name, ranges in (("q_ranges", q_ranges), ("k_ranges", k_ranges)):
        if ranges.dtype != torch.int32 or ranges.ndim != 2 or ranges.shape[1] != 2 or ranges.shape[0] == 0:
            raise ValueError(f"MagiAttentionMask {name} must have shape [num_ranges, 2] and dtype int32.")
    if q_ranges.shape[0] != k_ranges.shape[0] or q_ranges.device != k_ranges.device:
        raise ValueError("MagiAttentionMask q_ranges and k_ranges must share length and device.")
    starts = torch.cat((q_ranges[:, 0], k_ranges[:, 0]))
    ends = torch.cat((q_ranges[:, 1], k_ranges[:, 1]))
    require_all((starts >= 0) & (starts < ends), "MagiAttentionMask ranges must satisfy 0 <= start < end.")


def _validate_attn_type_map(attn_type_map: torch.Tensor, *, num_ranges: int, device: torch.device) -> None:
    if attn_type_map.dtype != torch.int32 or attn_type_map.ndim != 1 or attn_type_map.shape[0] != num_ranges:
        raise ValueError("MagiAttentionMask attn_type_map must have shape [num_ranges] and dtype int32.")
    if attn_type_map.device != device:
        raise ValueError("MagiAttentionMask attn_type_map must be on the same device as q_ranges.")
    require_all(
        (attn_type_map >= 0) & (attn_type_map <= 3), "MagiAttentionMask attn_type_map values must be in [0, 3]."
    )
