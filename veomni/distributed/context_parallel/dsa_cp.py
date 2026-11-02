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

"""Collectives for DeepSeek-V4 context parallelism.

Milestone 1 replicates KV rather than minimising traffic, so every exchange here
is an all-gather. They all route through ``gather_outputs``, which wraps
``_Gather.apply`` and therefore carries gradients; nothing here hand-rolls a
differentiable point-to-point.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from ..sequence_parallel.data import gather_outputs


def all_gather_kv(kv: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Replicate the full-resolution MQA KV, ``[B, 1, L, D]`` to ``[B, 1, S, D]``."""
    return gather_outputs(kv, gather_dim=2, group=group)


def all_gather_compressed_rows(
    local_rows: torch.Tensor,
    counts: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Assemble per-rank compressed rows into global window order.

    ``counts`` comes from ``window_owner_counts`` and is identical on every rank,
    so the shapes need no collective to agree on.

    Ranks own different numbers of windows. Padding every rank up to the maximum
    and slicing the valid rows back out afterwards is a performance choice, not a
    correctness one: ``_all_gather`` exchanges shapes first and allocates a tensor
    per rank, so unequal shards would be gathered correctly too. What matched
    shapes buy is the collective underneath — PyTorch's NCCL ``all_gather``
    coalesces equally sized outputs into a single ``ncclAllGather``, and falls
    back to one broadcast per rank when they differ, which is ``cp_size``
    launches per compressor per layer.

    Concatenating each rank's valid rows in rank order is already global window
    order, because windows sort by start token and shards sort by rank.
    """
    # One device-to-host sync for every count at once. Reading ``counts.max()``
    # and then ``counts[rank]`` inside the loop below drains the device queue
    # ``cp_size + 1`` times per call, and there is one call per compressor and
    # per indexer per layer.
    per_rank = counts.tolist()
    n_max = max(per_rank, default=0)
    if n_max == 0:
        return local_rows.new_zeros(local_rows.shape[0], 0, local_rows.shape[-1])

    n_local = local_rows.shape[1]
    if n_local < n_max:
        pad = local_rows.new_zeros(local_rows.shape[0], n_max - n_local, local_rows.shape[-1])
        local_rows = torch.cat([local_rows, pad], dim=1)

    gathered = gather_outputs(local_rows, gather_dim=1, group=group)
    keep = torch.cat(
        [
            torch.arange(rank * n_max, rank * n_max + count, device=gathered.device)
            for rank, count in enumerate(per_rank)
        ]
    )
    return gathered.index_select(1, keep)


def exchange_compressor_halos(
    kv: torch.Tensor,
    gate: torch.Tensor,
    halo: int,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extend ``[B, L, X]`` shards to ``[B, halo + L + halo, X]``.

    The left halo carries the previous window that the first owned window's
    overlap half needs; the right halo carries the tail of an owned window that
    reaches past the shard. Ranks at the ends receive zeros, which are never
    read: rank 0's first owned window is a sample's first window and is masked by
    ``has_previous_window``, and no window extends past the end of the sequence.
    """
    cp_size = dist.get_world_size(group)
    cp_rank = dist.get_rank(group)

    slabs = torch.cat([kv[:, :halo], kv[:, -halo:]], dim=1)
    gate_slabs = torch.cat([gate[:, :halo], gate[:, -halo:]], dim=1)
    all_slabs = gather_outputs(slabs, gather_dim=1, group=group)
    all_gate_slabs = gather_outputs(gate_slabs, gather_dim=1, group=group)

    def _slab(source: torch.Tensor, rank: int, second_half: bool) -> torch.Tensor:
        begin = rank * 2 * halo + (halo if second_half else 0)
        return source[:, begin : begin + halo]

    zeros = kv.new_zeros(kv.shape[0], halo, kv.shape[-1])
    gate_zeros = gate.new_zeros(gate.shape[0], halo, gate.shape[-1])
    left = _slab(all_slabs, cp_rank - 1, True) if cp_rank > 0 else zeros
    left_gate = _slab(all_gate_slabs, cp_rank - 1, True) if cp_rank > 0 else gate_zeros
    right = _slab(all_slabs, cp_rank + 1, False) if cp_rank + 1 < cp_size else zeros
    right_gate = _slab(all_gate_slabs, cp_rank + 1, False) if cp_rank + 1 < cp_size else gate_zeros

    return torch.cat([left, kv, right], dim=1), torch.cat([left_gate, gate, right_gate], dim=1)
