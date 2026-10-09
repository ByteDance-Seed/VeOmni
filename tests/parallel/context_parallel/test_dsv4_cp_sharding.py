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

import pytest
import torch

from veomni.distributed.context_parallel.sharding import (
    empty_compressed_rows,
    local_window_range,
    local_window_token_indices,
    plan_compressor_shard,
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


# ``local_len <= 0`` is unreachable from production: every caller derives it from
# a tensor's sequence dimension, and ``plan_compressor_shard`` refuses
# ``rate > local_seq_len`` with ``rate >= 1`` before it reaches either function.
# The guards are pinned rather than deleted because both functions are exported
# from ``veomni.distributed.context_parallel`` and neither fails usefully without
# them: this one raises ``ZeroDivisionError`` from inside ``torch.div``, and
# ``local_window_range`` below does not fail at all.
@pytest.mark.parametrize("local_len", [0, -4])
def test_owner_counts_refuse_a_non_positive_local_length(local_len):
    with pytest.raises(ValueError, match="local_len must be positive"):
        window_owner_counts(WINDOW_STARTS, local_len, CP_SIZE)


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


# The other half of the guard pair above. This one is the reason they are worth
# keeping: without it a non-positive width answers ``(0, 0)``, an empty shard
# that reads as a rank owning nothing rather than as an error.
@pytest.mark.parametrize("local_len", [0, -4])
def test_local_window_range_refuses_a_non_positive_local_length(local_len):
    with pytest.raises(ValueError, match="local_len must be positive"):
        local_window_range(WINDOW_STARTS, local_len, 0)


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


def test_the_unpacked_plan_builds_the_window_starts_every_rank_agrees_on():
    """Without packed metadata the starts are consecutive multiples of the rate."""
    plan = plan_compressor_shard(
        role="test",
        rate=4,
        local_seq_len=LOCAL_LEN,
        cp_rank=2,
        cp_size=CP_SIZE,
        packed_compression_metadata=None,
        device=torch.device("cpu"),
    )
    assert plan.window_starts.tolist() == list(range(0, LOCAL_LEN * CP_SIZE, 4))
    assert (plan.begin, plan.end) == (8, 12)
    assert plan.counts.tolist() == [4, 4, 4, 4]


def test_the_packed_plan_takes_its_windows_from_the_metadata():
    """Packed starts are not multiples of the rate, so they must come from the caller."""
    metadata = {4: {"window_starts": WINDOW_STARTS}}
    plan = plan_compressor_shard(
        role="test",
        rate=4,
        local_seq_len=LOCAL_LEN,
        cp_rank=3,
        cp_size=CP_SIZE,
        packed_compression_metadata=metadata,
        device=torch.device("cpu"),
    )
    assert plan.window_starts is WINDOW_STARTS
    assert (plan.begin, plan.end) == (12, 15)
    assert plan.counts.tolist() == [4, 4, 4, 3]


def test_the_plan_refuses_a_shard_narrower_than_one_window_and_names_its_caller():
    """One guard for three call sites, so the message has to say which one raised."""
    with pytest.raises(ValueError, match="one compression window wide") as raised:
        plan_compressor_shard(
            role="DeepSeek V4 CSA compressor",
            rate=8,
            local_seq_len=4,
            cp_rank=0,
            cp_size=CP_SIZE,
            packed_compression_metadata=None,
            device=torch.device("cpu"),
        )
    assert "DeepSeek V4 CSA compressor" in str(raised.value), str(raised.value)


def test_token_indices_put_a_straddling_window_inside_the_haloed_shard():
    """Rank 2 owns the window at 46, whose last two tokens live on rank 3."""
    rate = 4
    plan = plan_compressor_shard(
        role="test",
        rate=rate,
        local_seq_len=LOCAL_LEN,
        cp_rank=2,
        cp_size=CP_SIZE,
        packed_compression_metadata={rate: {"window_starts": WINDOW_STARTS}},
        device=torch.device("cpu"),
    )
    indices, first_window_position = local_window_token_indices(
        plan, rate=rate, local_seq_len=LOCAL_LEN, cp_rank=2, device=torch.device("cpu")
    )
    # Rank 2 owns the windows at 32, 38, 42 and 46; the extended buffer is
    # [4 halo | 16 local | 4 halo], so global token 32 sits at index 4.
    assert indices.tolist() == [[4, 5, 6, 7], [10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]]
    assert first_window_position == 32
    assert int(indices.max()) < rate + LOCAL_LEN + rate


def test_the_zero_window_result_keeps_both_inputs_in_the_autograd_graph():
    """The hang-class property: a rank owning no window still reaches every backward.

    ``new_zeros`` here would detach this rank from the graph, so it would skip
    the backward all-reduce of the compressed-row all-gather and of both halo
    all-gathers while its peers block in them. A result built from ``chunk_kv``
    alone would do the same to the gate halo's. Both inputs receiving a gradient
    is what says neither can happen.
    """
    chunk_kv = torch.randn(1, 3, 8, requires_grad=True)
    chunk_gate = torch.randn(1, 3, 8, requires_grad=True)
    empty = empty_compressed_rows(chunk_kv, chunk_gate, head_dim=4)

    assert empty.shape == (1, 0, 4)
    assert empty.requires_grad, "a detached empty result skips the collectives' backward"
    # Reach the loss *through* the empty result, the way the row all-gather does
    # on a rank that owns nothing: the rows are all its peers', and this rank's
    # contribution to them is empty.
    torch.cat([empty, torch.ones(1, 2, 4)], dim=1).sum().backward()
    assert chunk_kv.grad is not None, "the kv halo's backward was never reached"
    assert chunk_gate.grad is not None, "the gate halo's backward was never reached"
