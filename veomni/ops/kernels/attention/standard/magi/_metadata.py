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

from ...helper import require_all


@dataclass(frozen=True)
class _CacheEntry:
    """Prepared argument plus strong references anchoring its tensor identities.

    ``attn_arg`` may be ``None`` when only range endpoints have been checked.
    That lets the adapter skip a repeated ``require_all`` reduction before the
    FA4 backend has prepared metadata, without launching the check twice on the
    first layer.
    """

    bounds_key: tuple[object, ...] | None
    attn_arg_key: tuple[object, ...] | None
    attn_arg: object | None
    metadata_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]


_CACHE_LOCK = Lock()
_cache_entry: _CacheEntry | None = None


def ensure_range_bounds(
    query: torch.Tensor,
    key: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
) -> None:
    """Validate range endpoints once while the mask tensors and Q/K shape match.

    ``query`` and ``key`` are the packed three-dimensional tensors consumed by
    FA4. A cache hit skips the ``.all()`` reductions; CUDA still does not sync
    to the host on either path.
    """
    global _cache_entry

    bounds_key = _make_bounds_key(query, key, q_ranges, k_ranges)
    with _CACHE_LOCK:
        if _bounds_hit(bounds_key):
            return
        _validate_range_bounds(query, key, q_ranges, k_ranges)
        if bounds_key is None:
            return
        _cache_entry = _CacheEntry(
            bounds_key=bounds_key,
            attn_arg_key=None,
            attn_arg=None,
            metadata_tensors=(q_ranges, k_ranges, None),
        )


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
    bounds_key = _make_bounds_key(query, key, q_ranges, k_ranges)
    attn_arg_key = _make_cache_key(query, key, q_ranges, k_ranges, attn_type_map, metadata_head_dim)
    with _CACHE_LOCK:
        if _attn_arg_hit(attn_arg_key):
            return _cache_entry.attn_arg
        if not _bounds_hit(bounds_key):
            _validate_range_bounds(query, key, q_ranges, k_ranges)
        attn_arg = _prepare_attn_arg(query, key, q_ranges, k_ranges, attn_type_map, metadata_head_dim)
        if attn_arg_key is None and bounds_key is None:
            _cache_entry = None
            return attn_arg
        _cache_entry = _CacheEntry(
            bounds_key=bounds_key,
            attn_arg_key=attn_arg_key,
            attn_arg=attn_arg if attn_arg_key is not None else None,
            metadata_tensors=(q_ranges, k_ranges, attn_type_map),
        )
        return attn_arg


def _bounds_hit(bounds_key: tuple[object, ...] | None) -> bool:
    return bounds_key is not None and _cache_entry is not None and _cache_entry.bounds_key == bounds_key


def _attn_arg_hit(attn_arg_key: tuple[object, ...] | None) -> bool:
    return (
        attn_arg_key is not None
        and _cache_entry is not None
        and _cache_entry.attn_arg is not None
        and _cache_entry.attn_arg_key == attn_arg_key
    )


def _validate_range_bounds(
    query: torch.Tensor,
    key: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
) -> None:
    """Require range ends to lie within the post-exchange packed sequence lengths."""
    require_all(
        q_ranges[:, 1] <= query.shape[0],
        f"MagiAttention q_ranges must end within the post-exchange query length ({query.shape[0]}).",
    )
    require_all(
        k_ranges[:, 1] <= key.shape[0],
        f"MagiAttention k_ranges must end within the post-exchange key length ({key.shape[0]}).",
    )


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


def _tensor_identities(tensors: tuple[torch.Tensor | None, ...]) -> tuple[object, ...] | None:
    """Identify tensors by id and version, or decline caching when mutation is invisible."""
    keys: list[tuple[int, int] | None] = []
    for tensor in tensors:
        if tensor is None:
            keys.append(None)
            continue
        try:
            version = tensor._version
        except RuntimeError:
            # Inference tensors do not expose a version counter, so in-place
            # mutation cannot be detected safely.
            return None
        keys.append((id(tensor), version))
    return tuple(keys)


def _make_bounds_key(
    query: torch.Tensor,
    key: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
) -> tuple[object, ...] | None:
    """Identify unchanged range-bound checks from mask identity and Q/K shape."""
    identities = _tensor_identities((q_ranges, k_ranges))
    if identities is None:
        return None
    return (tuple(query.shape), tuple(key.shape), *identities)


def _make_cache_key(
    query: torch.Tensor,
    key: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor | None,
    metadata_head_dim: int,
) -> tuple[object, ...] | None:
    """Identify unchanged FA4 metadata inputs without reading tensor values."""
    identities = _tensor_identities((q_ranges, k_ranges, attn_type_map))
    if identities is None:
        return None
    return (
        query.device,
        query.dtype,
        tuple(query.shape),
        tuple(key.shape),
        metadata_head_dim,
        *identities,
    )
