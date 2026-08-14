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
# See the License for the specific language governing permissions and
# limitations under the License.
"""Lightweight variable-length metadata helpers for gated delta-rule kernels.

This module intentionally depends only on PyTorch. Model-level metadata
precomputation is shared by the vendored Triton and AscendC backends, so it must
not import the AscendC implementation (and its ``torch_npu`` / ``fla_npu`` /
Triton dependencies) merely to build host-side chunk indices.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


_DEFAULT_VARLEN_CHUNK_SIZES = (16, 32, 64, 128, 608 * 2)


def _ceil_div(a, b):
    return (a + b - 1) // b


def _prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in _ceil_div(_prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_chunk_indices_list(cu_seqlens: list[int] | torch.LongTensor, chunk_size: int) -> list[int]:
    if isinstance(cu_seqlens, torch.Tensor):
        cu_seqlens = [int(x) for x in cu_seqlens.detach().cpu().tolist()]

    indices: list[int] = []
    for seq_idx in range(len(cu_seqlens) - 1):
        length = int(cu_seqlens[seq_idx + 1]) - int(cu_seqlens[seq_idx])
        if length <= 0:
            continue
        for chunk_idx in range((length + chunk_size - 1) // chunk_size):
            indices.extend([seq_idx, chunk_idx])
    return indices


def _next_power_of_2(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def precompute_varlen_metadata(
    cu_seqlens: torch.LongTensor,
    num_heads: int,
    chunk_size: int = 64,
    device: Optional[torch.device | str] = None,
) -> tuple[list[int], Dict[str, Optional[torch.LongTensor]], Dict[str, Optional[list[int]]]]:
    """Precompute variable-length metadata once for all GDN layers.

    Hoists the per-layer ``cu_seqlens.tolist()`` / ``prepare_chunk_indices``
    calls into a single host-side pass so subsequent layers can reuse the
    metadata without importing or invoking a device-specific GDN backend.

    Args:
        cu_seqlens: Cumulative sequence lengths ``[N+1]`` in FlashAttention format.
        num_heads: Number of value heads; use the local head count under Ulysses SP.
        chunk_size: GDN chunk size (must be a power of 2).
        device: If set, place ``chunk_indices`` tensors on this device.

    Returns:
        ``(cu_seqlens_list, chunk_indices, chunk_indices_list)`` suitable for
        the precomputed-metadata arguments accepted by GDN kernels.
    """
    if cu_seqlens.device.type != "cpu":
        cu_seqlens = cu_seqlens.cpu()
    cu_seqlens_list = cu_seqlens.tolist()
    cumsum_block_size = _next_power_of_2((1 << 17) // max(1, num_heads * chunk_size))
    required_sizes = set(_DEFAULT_VARLEN_CHUNK_SIZES) | {chunk_size, cumsum_block_size}
    chunk_indices: Dict[str, Optional[torch.LongTensor]] = {}
    chunk_indices_list: Dict[str, Optional[list[int]]] = {}

    for size in required_sizes:
        key = str(size)
        chunk_indices[key] = prepare_chunk_indices(cu_seqlens, size)
        chunk_indices_list[key] = prepare_chunk_indices_list(cu_seqlens_list, size)
    if device is not None:
        chunk_indices = {k: v.to(device=device) if v is not None else None for k, v in chunk_indices.items()}
    return cu_seqlens_list, chunk_indices, chunk_indices_list


__all__ = ["precompute_varlen_metadata", "prepare_chunk_indices", "prepare_chunk_indices_list"]
