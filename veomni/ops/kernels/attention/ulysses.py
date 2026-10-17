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

"""Shared Ulysses layout exchanges for fused-attention backends."""

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from ....distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
)


def _gather_seq_scatter_heads_bshd(tensor: torch.Tensor, *, group: ProcessGroup) -> torch.Tensor:
    """Gather sequence and scatter heads for one tensor in ``[B, S, H, D]`` layout."""
    if tensor.ndim == 4 and tensor.size(0) == 1:
        tensor = gather_seq_scatter_heads(tensor.squeeze(0), seq_dim=0, head_dim=1, group=group)
        return tensor.unsqueeze(0)

    return gather_seq_scatter_heads(tensor, seq_dim=1, head_dim=2, group=group)


def _repeat_kv_heads_for_ulysses(tensor: torch.Tensor, ulysses_size: int) -> torch.Tensor:
    """Repeat KV heads in ``[B, S, H, D]`` layout until every Ulysses rank owns at least one."""
    head_count = tensor.shape[2]
    if ulysses_size > head_count:
        assert ulysses_size % head_count == 0, (
            f"ulysses_size ({ulysses_size}) must be divisible by num_key_value_heads ({head_count})"
        )
        return torch.repeat_interleave(tensor, dim=2, repeats=ulysses_size // head_count)

    assert head_count % ulysses_size == 0, (
        f"num_key_value_heads ({head_count}) must be divisible by ulysses_size ({ulysses_size})"
    )
    return tensor


def gather_seq_scatter_kv_heads(
    tensor: torch.Tensor,
    *,
    group: ProcessGroup,
    ulysses_size: int,
) -> torch.Tensor:
    """Exchange a KV-head tensor in ``[B, S, H, D]`` layout exactly as ``prepare_ulysses_qkv`` exchanges K/V.

    When ``ulysses_size`` exceeds the head count, heads are repeated first, so
    each rank receives the KV group that serves its contiguous query-head
    slice. Any tensor with one head per KV group (e.g. sparse-attention index
    queries) must use this helper to stay aligned with the exchanged K/V.
    """
    return _gather_seq_scatter_heads_bshd(_repeat_kv_heads_for_ulysses(tensor, ulysses_size), group=group)


def prepare_ulysses_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    group: ProcessGroup,
    ulysses_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Gather sequence and scatter heads for Q/K/V in ``[B, S, H, D]`` layout."""
    query_head_count = query.shape[2]
    assert query_head_count % ulysses_size == 0, (
        f"num_query_heads ({query_head_count}) must be divisible by ulysses_size ({ulysses_size})"
    )
    key = _repeat_kv_heads_for_ulysses(key, ulysses_size)
    value = _repeat_kv_heads_for_ulysses(value, ulysses_size)

    query = _gather_seq_scatter_heads_bshd(query, group=group)
    key = _gather_seq_scatter_heads_bshd(key, group=group)
    value = _gather_seq_scatter_heads_bshd(value, group=group)
    return query, key, value, query_head_count


def slice_ulysses_head_auxiliary(
    auxiliary: torch.Tensor | None,
    *,
    query_head_count: int,
    local_query_head_count: int,
    group: ProcessGroup,
) -> torch.Tensor | None:
    """Select the current Ulysses rank's head slice from a global 1D auxiliary tensor."""
    if auxiliary is None or auxiliary.ndim != 1 or auxiliary.numel() != query_head_count:
        return auxiliary

    head_start = dist.get_rank(group) * local_query_head_count
    return auxiliary.narrow(0, head_start, local_query_head_count).contiguous()


def restore_ulysses_output(output: torch.Tensor, *, group: ProcessGroup) -> torch.Tensor:
    """Gather heads and scatter sequence for attention output in ``[B, S, H, D]`` layout."""
    if output.ndim == 4 and output.size(0) == 1:
        output = output.squeeze(0)
        output = gather_heads_scatter_seq(output, seq_dim=0, head_dim=1, group=group)
        return output.unsqueeze(0)

    return gather_heads_scatter_seq(output, seq_dim=1, head_dim=2, group=group)
