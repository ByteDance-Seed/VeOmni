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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Packed variable-length splitting helpers for SDPA and eager attention."""

from collections.abc import Callable

import torch

from ...helper import (
    _SDPA_CU_SEQLENS_KEYS,
    _SDPA_PACKED_METADATA_KEYS,
    strip_sdpa_packed_metadata,
)


def extract_packed_varlen_cu_seqlens(
    kwargs: dict,
    *,
    batch_size: int,
    attention_mask: torch.Tensor | None,
) -> tuple[dict, tuple[torch.Tensor | None, torch.Tensor | None] | None]:
    """Extract multi-segment ``cu_seqlens`` for unmasked varlen splitting or strip packed metadata.

    When ``attention_mask`` is a 3-D/4-D dense mask or the cumulative lengths
    describe at most one segment per batch row, packed metadata is dropped and
    ``None`` is returned so the caller runs a single attention forward. When
    ``attention_mask is None`` and ``batch_size == 1`` with multiple segments,
    packed metadata is stripped from ``kwargs`` and ``(cu_seqlens_q, cu_seqlens_k)``
    is returned for chunked varlen execution. Multi-segment metadata with a 2-D
    mask or ``batch_size != 1`` is rejected via :func:`strip_sdpa_packed_metadata`.
    """
    packed = [name for name in _SDPA_PACKED_METADATA_KEYS if kwargs.get(name) is not None]
    if not packed:
        return kwargs, None

    multi_segment = [
        name
        for name in _SDPA_CU_SEQLENS_KEYS
        if torch.is_tensor(kwargs.get(name)) and kwargs[name].numel() - 1 > batch_size
    ]
    if not multi_segment or (attention_mask is not None and attention_mask.ndim >= 3):
        return {key: value for key, value in kwargs.items() if key not in packed}, None

    if attention_mask is not None or batch_size != 1:
        return strip_sdpa_packed_metadata(kwargs, batch_size=batch_size, dense_mask=attention_mask), None

    cu_seqlens = kwargs.get("cu_seqlens")
    cu_q = kwargs.get("cu_seq_lens_q")
    if cu_q is None:
        cu_q = kwargs.get("cu_seqlens_q")
    if cu_q is None:
        cu_q = cu_seqlens

    cu_k = kwargs.get("cu_seq_lens_k")
    if cu_k is None:
        cu_k = kwargs.get("cu_seqlens_k")
    if cu_k is None:
        cu_k = cu_seqlens

    for name, tensor in (("cu_seqlens_q", cu_q), ("cu_seqlens_k", cu_k)):
        if tensor is not None and (not torch.is_tensor(tensor) or tensor.ndim != 1 or tensor.numel() < 2):
            raise ValueError(f"{name} must be a 1-D tensor with at least 2 elements.")

    if cu_q is not None and cu_k is not None and cu_q.numel() != cu_k.numel():
        raise ValueError(
            "Packed varlen attention requires the same number of query and key segments, "
            f"got {cu_q.numel() - 1} and {cu_k.numel() - 1}."
        )

    cleaned = {key: value for key, value in kwargs.items() if key not in packed}
    return cleaned, (cu_q, cu_k)


def packed_split_attention_forward(
    forward_fn: Callable,
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Split packed ``[1, H, S, D]`` Q/K/V by ``cu_seqlens`` and run ``forward_fn`` per segment."""
    if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
        raise ValueError(
            "Packed varlen attention requires batch_size == 1 for query, key, and value; "
            f"got query={query.shape[0]}, key={key.shape[0]}, value={value.shape[0]}."
        )
    if key.shape[2] != value.shape[2]:
        raise ValueError(
            "Packed varlen attention requires matching key and value sequence lengths; "
            f"got key={key.shape[2]}, value={value.shape[2]}."
        )
    if cu_seqlens_q is None or cu_seqlens_k is None:
        if query.shape[2] != key.shape[2]:
            raise ValueError(
                "Packed varlen attention with query_length != kv_length requires both query and key cu_seqlens."
            )
        shared = cu_seqlens_q if cu_seqlens_q is not None else cu_seqlens_k
        if shared is None:
            raise ValueError("Packed varlen attention requires cu_seqlens.")
        cu_seqlens_q = shared
        cu_seqlens_k = shared

    lengths_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    lengths_k = lengths_q if cu_seqlens_k is cu_seqlens_q else (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).tolist()
    if any(length <= 0 for length in lengths_q) or any(length <= 0 for length in lengths_k):
        raise ValueError("Packed varlen cu_seqlens segment lengths must be positive.")
    if sum(lengths_q) != query.shape[2]:
        raise ValueError(
            f"Packed varlen cu_seqlens_q total length ({sum(lengths_q)}) must match "
            f"query sequence length ({query.shape[2]})."
        )
    if sum(lengths_k) != key.shape[2]:
        raise ValueError(
            f"Packed varlen cu_seqlens_k total length ({sum(lengths_k)}) must match "
            f"key/value sequence length ({key.shape[2]})."
        )

    q_splits = torch.split(query, lengths_q, dim=2)
    k_splits = torch.split(key, lengths_k, dim=2)
    v_splits = torch.split(value, lengths_k, dim=2)
    attn_outputs = [
        forward_fn(
            module,
            q_chunk,
            k_chunk,
            v_chunk,
            None,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )[0]
        for q_chunk, k_chunk, v_chunk in zip(q_splits, k_splits, v_splits)
    ]
    return torch.cat(attn_outputs, dim=1), None
