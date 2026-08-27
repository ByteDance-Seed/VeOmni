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
"""

from __future__ import annotations

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
