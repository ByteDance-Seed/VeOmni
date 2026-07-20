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
    key_value_head_count = key.shape[2]

    assert query_head_count % ulysses_size == 0, (
        f"num_query_heads ({query_head_count}) must be divisible by ulysses_size ({ulysses_size})"
    )
    if ulysses_size > key_value_head_count:
        assert ulysses_size % key_value_head_count == 0, (
            f"ulysses_size ({ulysses_size}) must be divisible by num_key_value_heads ({key_value_head_count})"
        )
        repeat_count = ulysses_size // key_value_head_count
        key = torch.repeat_interleave(key, dim=2, repeats=repeat_count)
        value = torch.repeat_interleave(value, dim=2, repeats=repeat_count)

    if query.ndim == 4 and query.size(0) == 1:
        query, key, value = query.squeeze(0), key.squeeze(0), value.squeeze(0)
        query = gather_seq_scatter_heads(query, seq_dim=0, head_dim=1, group=group)
        key = gather_seq_scatter_heads(key, seq_dim=0, head_dim=1, group=group)
        value = gather_seq_scatter_heads(value, seq_dim=0, head_dim=1, group=group)
        query, key, value = query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0)
    else:
        query = gather_seq_scatter_heads(query, seq_dim=1, head_dim=2, group=group)
        key = gather_seq_scatter_heads(key, seq_dim=1, head_dim=2, group=group)
        value = gather_seq_scatter_heads(value, seq_dim=1, head_dim=2, group=group)

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
