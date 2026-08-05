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

"""Window ownership arithmetic for DeepSeek-V4 context parallelism."""

from __future__ import annotations

import torch

from veomni.distributed.context_parallel.sharding import (
    local_window_range,
    rebase_window_indices,
    window_owner_counts,
)


# Canonical fixture: cp_size=4, seq_len=64, L=16, R=4, samples [(0,38),(38,64)].
# The window at 46 straddles the rank 2/3 boundary and the counts are unequal.
WINDOW_STARTS = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28, 32, 38, 42, 46, 50, 54, 58])
LOCAL_LEN = 16
CP_SIZE = 4


def test_owner_counts_are_unequal_and_sum_to_every_window():
    counts = window_owner_counts(WINDOW_STARTS, LOCAL_LEN, CP_SIZE)
    assert counts.tolist() == [4, 4, 4, 3]
    assert int(counts.sum()) == WINDOW_STARTS.numel()


def test_owner_counts_handle_no_windows():
    empty = torch.zeros(0, dtype=torch.long)
    counts = window_owner_counts(empty, LOCAL_LEN, CP_SIZE)
    assert counts.tolist() == [0, 0, 0, 0]


def test_local_ranges_tile_the_global_window_array():
    ranges = [local_window_range(WINDOW_STARTS, LOCAL_LEN, r) for r in range(CP_SIZE)]
    assert ranges == [(0, 4), (4, 8), (8, 12), (12, 15)]
    # Contiguous and exhaustive: rank r ends exactly where rank r+1 begins.
    assert ranges[0][0] == 0
    assert ranges[-1][1] == WINDOW_STARTS.numel()
    for left, right in zip(ranges, ranges[1:]):
        assert left[1] == right[0]


def test_every_window_is_owned_by_the_rank_holding_its_first_token():
    for rank in range(CP_SIZE):
        begin, end = local_window_range(WINDOW_STARTS, LOCAL_LEN, rank)
        owned = WINDOW_STARTS[begin:end]
        assert torch.all(owned >= rank * LOCAL_LEN)
        assert torch.all(owned < (rank + 1) * LOCAL_LEN)


def test_rebasing_maps_a_straddling_window_inside_the_haloed_shard():
    # Rank 2 owns the window starting at 46; it covers 46..49, so tokens 48 and
    # 49 live on rank 3 and must land in the right halo.
    rate = 4
    window = torch.arange(46, 50).view(1, rate)
    rebased = rebase_window_indices(window, LOCAL_LEN, cp_rank=2, halo=rate)
    # Shard 2 is tokens [32, 48). Extended buffer is [halo | 16 local | halo],
    # so local token 32 sits at index 4 and the window starts at 46 -> 18.
    assert rebased.tolist() == [[18, 19, 20, 21]]
    assert int(rebased.min()) >= 0
    assert int(rebased.max()) < rate + LOCAL_LEN + rate


def test_rebasing_maps_the_previous_window_into_the_left_halo():
    # Rank 1's first owned window starts at 16; its overlap half comes from the
    # window at 12, which lives on rank 0.
    rate = 4
    previous = torch.arange(12, 16).view(1, rate)
    rebased = rebase_window_indices(previous, LOCAL_LEN, cp_rank=1, halo=rate)
    assert rebased.tolist() == [[0, 1, 2, 3]]
