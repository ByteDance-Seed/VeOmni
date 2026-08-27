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

"""How DeepSeek-V4 context parallelism splits a sequence, and how the shards talk.

Two halves of one mechanism, sharing the canonical fixture below so that the
arithmetic and the collectives cannot drift apart: ``window_owner_counts``
computing ``EXPECTED_OWNER_COUNTS`` is exactly why the gather of compressed rows
has to handle uneven per-rank counts.

The arithmetic half is pure host code and runs anywhere. The collectives half
spawns ``CP_SIZE`` ranks and skips without that many devices.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from veomni.distributed.context_parallel.sharding import (
    local_window_range,
    rebase_window_indices,
    window_owner_counts,
)
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


# Canonical fixture: cp_size=4, seq_len=64, L=16, R=4, samples [(0,38),(38,64)].
# The window at 46 straddles the rank 2/3 boundary and the counts are unequal.
CP_SIZE = 4
LOCAL_LEN = 16
SEQ_LEN = CP_SIZE * LOCAL_LEN
WINDOW_STARTS = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28, 32, 38, 42, 46, 50, 54, 58])
EXPECTED_OWNER_COUNTS = [4, 4, 4, 3]
DIM = 8


# ------------------------------ window arithmetic ------------------------------


def test_owner_counts_are_unequal_and_sum_to_every_window():
    counts = window_owner_counts(WINDOW_STARTS, LOCAL_LEN, CP_SIZE)
    assert counts.tolist() == EXPECTED_OWNER_COUNTS
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


# -------------------------------- collectives --------------------------------


def _run(rank: int, world_size: int, init_file: str) -> None:
    """CP collectives reproduce the single-rank tensors and carry gradients."""
    device_type = get_device_type()
    get_torch_device().set_device(rank)
    dist.init_process_group(
        backend=get_dist_comm_backend(),
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )

    from veomni.distributed.context_parallel.dsa_cp import (
        all_gather_compressed_rows,
        all_gather_kv,
        exchange_compressor_halos,
    )

    group = dist.group.WORLD
    torch.manual_seed(0)
    full_kv = torch.randn(1, 1, SEQ_LEN, DIM, device=device_type)
    dist.broadcast(full_kv, src=0)
    local_kv = full_kv[:, :, rank * LOCAL_LEN : (rank + 1) * LOCAL_LEN].clone().requires_grad_(True)

    gathered = all_gather_kv(local_kv, group)
    torch.testing.assert_close(gathered, full_kv)

    # Gradient reaches only this rank's slice, with the value the full-tensor
    # backward would have produced there. Each rank backprops through only its
    # OWN slice of the (identical, replicated) gathered tensor -- summing the
    # WHOLE tensor on every rank would make every rank's contribution identical
    # rather than distinct, and the all-reduce in `_Gather.backward` would then
    # inflate the result by `world_size` instead of reproducing the single-rank
    # baseline.
    gathered[:, :, rank * LOCAL_LEN : (rank + 1) * LOCAL_LEN].sum().backward()
    torch.testing.assert_close(local_kv.grad, torch.ones_like(local_kv))

    # Uneven compressed rows -- the counts the arithmetic above pins -- land in
    # global window order.
    counts = torch.tensor(EXPECTED_OWNER_COUNTS, device=device_type)
    offset = sum(EXPECTED_OWNER_COUNTS[:rank])
    local_rows = (
        torch.arange(offset, offset + EXPECTED_OWNER_COUNTS[rank], device=device_type, dtype=torch.float32)
        .view(1, EXPECTED_OWNER_COUNTS[rank], 1)
        .expand(1, EXPECTED_OWNER_COUNTS[rank], DIM)
        .contiguous()
    )
    total_rows = sum(EXPECTED_OWNER_COUNTS)
    rows = all_gather_compressed_rows(local_rows, counts, group)
    assert rows.shape == (1, total_rows, DIM)
    torch.testing.assert_close(rows[0, :, 0], torch.arange(total_rows, device=device_type, dtype=torch.float32))

    # Halos come from the neighbours, with zeros beyond the ends.
    halo = 4
    local_flat = full_kv[0, 0, rank * LOCAL_LEN : (rank + 1) * LOCAL_LEN].unsqueeze(0)
    kv_ext, gate_ext = exchange_compressor_halos(local_flat, local_flat.clone(), halo, group)
    assert kv_ext.shape == (1, halo + LOCAL_LEN + halo, DIM)
    torch.testing.assert_close(kv_ext[:, halo : halo + LOCAL_LEN], local_flat)
    if rank == 0:
        torch.testing.assert_close(kv_ext[:, :halo], torch.zeros_like(kv_ext[:, :halo]))
    else:
        expected = full_kv[0, 0, rank * LOCAL_LEN - halo : rank * LOCAL_LEN].unsqueeze(0)
        torch.testing.assert_close(kv_ext[:, :halo], expected)
    if rank == world_size - 1:
        torch.testing.assert_close(kv_ext[:, -halo:], torch.zeros_like(kv_ext[:, -halo:]))
    else:
        expected = full_kv[0, 0, (rank + 1) * LOCAL_LEN : (rank + 1) * LOCAL_LEN + halo].unsqueeze(0)
        torch.testing.assert_close(kv_ext[:, -halo:], expected)
    torch.testing.assert_close(gate_ext, kv_ext)

    dist.destroy_process_group()


@pytest.mark.skipif(get_torch_device().device_count() < CP_SIZE, reason="needs 4 devices")
def test_cp_collectives():
    # A bare ``tempfile.NamedTemporaryFile`` races with the ``file://`` init
    # method: PyTorch's file store itself removes the file once every rank has
    # rendezvoused, so the file's own ``__exit__`` unlink then raises
    # ``FileNotFoundError``. Passing a path inside a ``TemporaryDirectory``
    # (as `tests/parallel/ulysses/test_deepseek_v4_ulysses.py` does) avoids the
    # double-delete: only the file disappears, not the directory.
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(_run, args=(CP_SIZE, init_file), nprocs=CP_SIZE, join=True)
