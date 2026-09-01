# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing limitations
# under the License.

"""Prepared FA4AttnArg cache keyed by mask tensors and attention shape."""

from dataclasses import dataclass
from threading import Lock

import torch


@dataclass(frozen=True)
class _CacheEntry:
    key: tuple[object, ...]
    attn_arg: object


_CACHE_LOCK = Lock()
_cache_entry: _CacheEntry | None = None


def get_or_prepare_attn_arg(
    query: torch.Tensor,
    key: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor | None,
    metadata_head_dim: int | None = None,
) -> object:
    """Reuse prepared FA4 mask metadata across layers with matching inputs."""
    global _cache_entry

    metadata_head_dim = query.shape[-1] if metadata_head_dim is None else metadata_head_dim
    cache_key = _make_cache_key(query, key, q_ranges, k_ranges, attn_type_map, metadata_head_dim)
    with _CACHE_LOCK:
        if cache_key is not None and _cache_entry is not None and _cache_entry.key == cache_key:
            return _cache_entry.attn_arg

        attn_arg = _prepare_attn_arg(query, key, q_ranges, k_ranges, attn_type_map, metadata_head_dim)
        _cache_entry = _CacheEntry(key=cache_key, attn_arg=attn_arg) if cache_key is not None else None
        return attn_arg


def _prepare_attn_arg(
    query: torch.Tensor,
    key: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor | None,
    metadata_head_dim: int,
) -> object:
    """Build upstream FA4 metadata once for a new mask and attention shape."""
    from ._fa4_cuda import cuda_device_context

    with cuda_device_context(query.device):
        from magi_attention.common.ranges import AttnRanges
        from magi_attention.meta.collection.calc_meta import FA4AttnArg

        q_ranges_list: list[list[int]] = q_ranges.cpu().tolist()
        k_ranges_list: list[list[int]] = k_ranges.cpu().tolist()
        attn_type_map_list: list[int] = (
            [0] * len(q_ranges_list) if attn_type_map is None else attn_type_map.cpu().tolist()
        )
        return FA4AttnArg(
            q_ranges=AttnRanges.from_ranges(q_ranges_list),
            k_ranges=AttnRanges.from_ranges(k_ranges_list),
            attn_type_map=attn_type_map_list,
            seqlen_q=query.shape[0],
            seqlen_k=key.shape[0],
            headdim=metadata_head_dim,
        )


def _make_cache_key(
    query: torch.Tensor,
    key: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor | None,
    metadata_head_dim: int,
) -> tuple[object, ...] | None:
    """Identify unchanged FA4 metadata inputs without reading tensor values."""
    tensor_keys: list[tuple[int, int] | None] = []
    for tensor in (q_ranges, k_ranges, attn_type_map):
        if tensor is None:
            tensor_keys.append(None)
            continue
        try:
            version = tensor._version
        except RuntimeError:
            # Inference tensors do not expose a version counter, so in-place
            # mutation cannot be detected safely.
            return None
        tensor_keys.append((id(tensor), version))

    return (
        query.device,
        query.dtype,
        tuple(query.shape),
        tuple(key.shape),
        metadata_head_dim,
        *tensor_keys,
    )
