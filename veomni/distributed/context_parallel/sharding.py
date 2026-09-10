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

"""Window ownership arithmetic for DeepSeek-V4 context parallelism.

A compression window belongs to the rank that owns its first token. That rule
lets shard boundaries fall anywhere, which matters because packed sample lengths
are not multiples of the compression rate and the packed metadata deliberately
tolerates the incomplete tail that leaves behind.

Every rank builds the same global ``window_starts``, so ownership is pure local
arithmetic and never needs a collective to agree on.

Nothing in this module communicates. The DeepSeek-V4 HCA compressor, CSA
compressor and Lightning Indexer all call into it *before* their halo exchange,
which is what lets a refusal here reach every rank instead of stranding some of
them in a collective.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


def window_owner_counts(window_starts: torch.Tensor, local_len: int, cp_size: int) -> torch.Tensor:
    """Return how many compression windows each CP rank owns, shape ``[cp_size]``."""
    if local_len <= 0:
        raise ValueError(f"local_len must be positive; got {local_len}")
    if window_starts.numel() == 0:
        return torch.zeros(cp_size, dtype=torch.long, device=window_starts.device)
    owners = torch.div(window_starts, local_len, rounding_mode="floor")
    return torch.bincount(owners, minlength=cp_size)[:cp_size]


def local_window_range(window_starts: torch.Tensor, local_len: int, cp_rank: int) -> tuple[int, int]:
    """Return the ``[begin, end)`` slice of the global window arrays owned by ``cp_rank``.

    ``window_starts`` is ascending, so owners are non-decreasing and each rank's
    windows form one contiguous run.
    """
    if local_len <= 0:
        raise ValueError(f"local_len must be positive; got {local_len}")
    begin = int((window_starts < cp_rank * local_len).sum())
    end = int((window_starts < (cp_rank + 1) * local_len).sum())
    return begin, end


def rebase_window_indices(
    window_indices: torch.Tensor,
    local_len: int,
    cp_rank: int,
    halo: int,
) -> torch.Tensor:
    """Rebase global token indices onto ``[left halo | local shard | right halo]``.

    An owned window can reach up to ``halo`` tokens past the end of the shard,
    and the overlap half of the first owned window can reach ``halo`` tokens
    before its start, so the extended buffer is padded on both sides and every
    index shifts by ``halo``.
    """
    return window_indices - cp_rank * local_len + halo


class CompressorShardPlan(NamedTuple):
    """Which compression windows one CP rank owns.

    ``window_starts`` is the global array every rank builds identically;
    ``begin`` / ``end`` slice this rank's contiguous run out of it, and
    ``counts`` holds every rank's run length, which is what
    ``all_gather_compressed_rows`` needs in order to unpad the gathered rows.
    """

    window_starts: torch.Tensor
    begin: int
    end: int
    counts: torch.Tensor


def plan_compressor_shard(
    *,
    role: str,
    rate: int,
    local_seq_len: int,
    cp_rank: int,
    cp_size: int,
    packed_compression_metadata: dict | None,
    device: torch.device,
) -> CompressorShardPlan:
    """Everything a window-compressing module needs before it touches a collective.

    The DeepSeek-V4 HCA compressor, CSA compressor and Lightning Indexer each
    compress windows over the same token axis at their own head dimension, so
    each runs this identical prologue. One copy on purpose: the narrow-shard
    refusal below is a guard whose regression is a ten-minute NCCL watchdog
    timeout rather than a failing assertion, and a fix for exactly that class of
    bug has already once landed in one copy of this code and missed the others.

    ``role`` names the caller in the refusal, which the traceback would give but
    a log line would not. Because ``rate`` and ``local_seq_len`` are the same on
    every rank, either all of them raise or none does.
    """
    if rate > local_seq_len:
        raise ValueError(
            f"{role} needs shards at least one compression window wide under DeepSeek V4 "
            f"context parallelism; compress rate {rate} exceeds this rank's {local_seq_len} "
            "tokens. An owned window reaches at most one rate past the shard, and that halo "
            "is taken from the single adjacent rank, so a narrower shard would have to reach "
            "across more than one."
        )
    window_starts = (
        packed_compression_metadata[rate]["window_starts"]
        if packed_compression_metadata is not None
        else torch.arange(0, local_seq_len * cp_size - rate + 1, rate, device=device)
    )
    begin, end = local_window_range(window_starts, local_seq_len, cp_rank)
    counts = window_owner_counts(window_starts, local_seq_len, cp_size)
    return CompressorShardPlan(window_starts=window_starts, begin=begin, end=end, counts=counts)


def local_window_token_indices(
    plan: CompressorShardPlan,
    *,
    rate: int,
    local_seq_len: int,
    cp_rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """``[n_owned, rate]`` token indices into ``[left halo | shard | right halo]``.

    Rows come in window order, so reshaping the tokens they gather to
    ``[batch, n_owned, rate, ...]`` sees exactly this rank's own windows and
    nothing else. The second return is the first owned window's global start
    token: the unpacked path turns it into compressed RoPE positions, which is
    sound there because unpacked window starts are consecutive multiples of the
    rate. It is zero when this rank owns no window, where nothing reads it.
    """
    local_starts = plan.window_starts[plan.begin : plan.end]
    window_indices = rebase_window_indices(
        local_starts[:, None] + torch.arange(rate, device=device), local_seq_len, cp_rank, rate
    )
    first_window_position = int(local_starts[0]) if local_starts.numel() > 0 else 0
    return window_indices, first_window_position


def empty_compressed_rows(chunk_kv: torch.Tensor, chunk_gate: torch.Tensor, head_dim: int) -> torch.Tensor:
    """A zero-window compression result that stays attached to *both* inputs.

    A context-parallel rank can own no window at all, and it still has to reach
    the backward of every collective its peers reach -- the compressed-row
    all-gather and both halo all-gathers all-reduce there. Backward
    participation is decided by the autograd graph, so ``new_zeros``, which is
    detached, takes this rank out of it entirely and leaves its peers blocked in
    a collective until the watchdog fires. A result touching only ``chunk_kv``
    does the same to the gate halo's gather alone. Empty slices of both keep the
    result attached to both while materialising nothing.

    All three DeepSeek-V4 window compressors call this, as does the packed
    compression helper they share. They used to carry a copy each, which is how
    the original fix reached one of them and not the other three.
    """
    return chunk_kv[:, :0, :head_dim] + chunk_gate[:, :0, :head_dim]
