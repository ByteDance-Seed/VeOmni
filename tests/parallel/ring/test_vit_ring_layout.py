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

"""CPU unit tests for the USP layout helpers used by ViT bidirectional ring attention."""

import pytest
import torch

from veomni.distributed.sequence_parallel.ring_attention.layout import (
    _vit_shard_lens,
    usp_zigzag_slice,
    vit_cp_cu_seqlens,
    vit_usp_slice,
    zigzag_reorder,
    zigzag_reorder_packed,
)


LAYOUT_CASES = [
    # (vit_seqlens, cp_size, ulysses_size)
    ([1380, 112], 2, 2),
    ([12, 8], 2, 2),
    ([20, 4, 28, 8], 4, 1),  # 4-row seq: 3 of 4 cp ranks get an empty shard
    ([880] * 4, 4, 2),  # video: one vit seq per temporal step
    ([4, 4, 4], 3, 2),
    ([16], 2, 1),
]


def _local_rows(vit_seqlens, cp_size, cp_rank, ulysses_size, ulysses_rank):
    rows = torch.arange(sum(vit_seqlens))
    return vit_usp_slice(rows, vit_seqlens, 0, cp_size, cp_rank, ulysses_size, ulysses_rank, pad_value=-1)


@pytest.mark.parametrize(("vit_seqlens", "cp_size", "ulysses_size"), LAYOUT_CASES)
def test_vit_usp_slice_covers_every_row_once(vit_seqlens, cp_size, ulysses_size):
    pieces = [
        _local_rows(vit_seqlens, cp_size, c, ulysses_size, u) for c in range(cp_size) for u in range(ulysses_size)
    ]
    assert len({piece.numel() for piece in pieces}) == 1
    merged = torch.cat(pieces)
    real = merged[merged >= 0]
    assert torch.equal(real.sort().values, torch.arange(sum(vit_seqlens)))


@pytest.mark.parametrize(("vit_seqlens", "cp_size", "ulysses_size"), LAYOUT_CASES)
def test_vit_shards_are_merger_aligned_and_balanced(vit_seqlens, cp_size, ulysses_size):
    shard_lens, cp_local_len = _vit_shard_lens(vit_seqlens, cp_size, ulysses_size)
    assert cp_local_len % (4 * ulysses_size) == 0
    for index, seqlen in enumerate(vit_seqlens):
        assert sum(shard_lens[c][index] for c in range(cp_size)) == seqlen
        assert all(shard_lens[c][index] % 4 == 0 for c in range(cp_size))
    totals = [sum(rank_lens) for rank_lens in shard_lens]
    assert max(totals) - min(totals) <= 4  # global round-robin: at most one merger unit apart
    assert max(totals) <= cp_local_len


@pytest.mark.parametrize(("vit_seqlens", "cp_size", "ulysses_size"), LAYOUT_CASES)
def test_vit_ulysses_blocks_start_on_merger_units(vit_seqlens, cp_size, ulysses_size):
    for c in range(cp_size):
        for u in range(ulysses_size):
            rows = _local_rows(vit_seqlens, cp_size, c, ulysses_size, u)
            assert rows.numel() % 4 == 0
            for window in rows.view(-1, 4):
                if window[0] < 0:
                    assert (window < 0).all()
                else:
                    assert window[0] % 4 == 0
                    assert torch.equal(window, window[0] + torch.arange(4))


@pytest.mark.parametrize(("vit_seqlens", "cp_size", "ulysses_size"), LAYOUT_CASES)
def test_vit_cp_cu_seqlens_matches_slice(vit_seqlens, cp_size, ulysses_size):
    global_cu = [0]
    for seqlen in vit_seqlens:
        global_cu.append(global_cu[-1] + seqlen)
    seq_id = torch.repeat_interleave(torch.arange(len(vit_seqlens)), torch.tensor(vit_seqlens))
    cu_per_rank = [vit_cp_cu_seqlens(torch.tensor(global_cu), cp_size, c, ulysses_size) for c in range(cp_size)]
    assert len({cu[-1] for cu in cu_per_rank}) == 1
    assert len({len(cu) for cu in cu_per_rank}) == 1 and len(cu_per_rank[0]) == len(vit_seqlens) + 2
    for c in range(cp_size):
        local = torch.cat(
            [
                vit_usp_slice(seq_id, vit_seqlens, 0, cp_size, c, ulysses_size, u, pad_value=len(vit_seqlens))
                for u in range(ulysses_size)
            ]
        )
        counts = torch.bincount(local, minlength=len(vit_seqlens) + 1).tolist()
        expected = [0]
        for count in counts:
            expected.append(expected[-1] + count)
        assert cu_per_rank[c] == expected


def test_vit_layout_doc_example():
    cu_per_rank = [vit_cp_cu_seqlens([0, 1380, 1492], 2, c, 2) for c in range(2)]
    assert cu_per_rank == [[0, 692, 748, 752], [0, 688, 744, 752]]
    rows = _local_rows([1380, 112], 2, 1, 2, 1)
    assert rows[:312].tolist() == list(range(1068, 1380))
    assert rows[312:368].tolist() == list(range(1436, 1492))
    assert rows[368:].tolist() == [-1] * 8


def test_vit_usp_slice_rejects_unaligned_seq():
    with pytest.raises(ValueError, match="merger_unit"):
        vit_usp_slice(torch.zeros(6), [6], 0, 2, 0, 1, 0)


@pytest.mark.parametrize("doc_cu", [None, [0, 32], [0, 8, 24, 32]])
@pytest.mark.parametrize(("cp_size", "ulysses_size"), [(2, 1), (2, 2)])
def test_usp_zigzag_slice_matches_cp_outer_ulysses_inner(doc_cu, cp_size, ulysses_size):
    x = torch.arange(32).view(1, 32)
    if doc_cu is not None and len(doc_cu) > 2:
        reordered = zigzag_reorder_packed(x, torch.tensor(doc_cu), dim=-1, cp_size=cp_size)
    else:
        reordered = zigzag_reorder(x, dim=-1, cp_size=cp_size)
    cp_chunk = 32 // cp_size
    for c in range(cp_size):
        region = reordered[:, c * cp_chunk : (c + 1) * cp_chunk]
        uly_chunk = cp_chunk // ulysses_size
        for u in range(ulysses_size):
            got = usp_zigzag_slice(x, doc_cu, -1, cp_size, c, ulysses_size, u)
            assert torch.equal(got, region[:, u * uly_chunk : (u + 1) * uly_chunk])
