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

"""CPU unit tests for the USP zig-zag block reordering helpers."""

import pytest
import torch

from veomni.distributed.sequence_parallel.data import (
    zigzag_block_order,
    zigzag_reorder,
    zigzag_undo,
)


def test_block_order_cp2():
    # cp=2 -> blocks [0,3,1,2]: rank0 owns {0,3}, rank1 owns {1,2}
    assert zigzag_block_order(2) == [0, 3, 1, 2]


def test_block_order_cp4():
    assert zigzag_block_order(4) == [0, 7, 1, 6, 2, 5, 3, 4]


@pytest.mark.parametrize("cp_size", [1, 2, 3, 4])
def test_reorder_then_undo_is_identity(cp_size):
    seq = 2 * cp_size * 5
    x = torch.arange(seq).view(1, seq, 1).float()
    reordered = zigzag_reorder(x, dim=1, cp_size=cp_size)
    restored = zigzag_undo(reordered, dim=1, cp_size=cp_size)
    assert torch.equal(restored, x)


def test_reorder_gives_expected_blocks_cp2():
    # 4 blocks of length 2: [b0 b1 b2 b3] -> [b0 b3 b1 b2]
    x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).view(1, 8, 1).float()
    reordered = zigzag_reorder(x, dim=1, cp_size=2).view(-1).tolist()
    assert reordered == [0, 1, 6, 7, 2, 3, 4, 5]


def test_cp1_is_noop():
    x = torch.randn(1, 10, 3)
    assert torch.equal(zigzag_reorder(x, dim=1, cp_size=1), x)
    assert torch.equal(zigzag_undo(x, dim=1, cp_size=1), x)


def test_reorder_requires_divisible_length():
    x = torch.randn(1, 7, 1)
    with pytest.raises(AssertionError):
        zigzag_reorder(x, dim=1, cp_size=2)


# ── varlen (packed) per-document zig-zag ──────────────────────────────────────
from veomni.distributed.sequence_parallel.data import (  # noqa: E402
    local_cu_seqlens,
    zigzag_reorder_varlen,
)


def test_reorder_varlen_shards_each_document_cp2():
    # two documents: doc0 len 12, doc1 len 8; cp=2 -> 4 blocks per doc.
    cp = 2
    doc_lens = [12, 8]
    cu = torch.tensor([0, 12, 20], dtype=torch.int32)
    x = torch.arange(sum(doc_lens)).view(-1, 1).float()
    reordered = zigzag_reorder_varlen(x, cu, dim=0, cp_size=cp)
    chunk = reordered.shape[0] // cp
    rank0 = reordered[:chunk].view(-1).int().tolist()
    rank1 = reordered[chunk:].view(-1).int().tolist()
    # doc0 blocks of len 3: [0,1,2][3,4,5][6,7,8][9,10,11]; rank0 owns {0,3}
    # doc1 blocks of len 2: [12,13][14,15][16,17][18,19]; rank0 owns {0,3}
    assert rank0 == [0, 1, 2, 9, 10, 11, 12, 13, 18, 19]
    assert rank1 == [3, 4, 5, 6, 7, 8, 14, 15, 16, 17]


def test_local_cu_seqlens_cp2():
    cu = torch.tensor([0, 12, 20], dtype=torch.int32)
    local = local_cu_seqlens(cu, cp_size=2)
    # each document length halved for cp=2
    assert local.tolist() == [0, 6, 10]


def test_reorder_varlen_cp1_is_noop():
    cu = torch.tensor([0, 6, 10], dtype=torch.int32)
    x = torch.randn(10, 2)
    assert torch.equal(zigzag_reorder_varlen(x, cu, dim=0, cp_size=1), x)
    assert torch.equal(local_cu_seqlens(cu, cp_size=1), cu)


def test_reorder_varlen_requires_divisible_document():
    # document length 10 is not divisible by 2*cp=4
    cu = torch.tensor([0, 10], dtype=torch.int32)
    x = torch.randn(10, 1)
    with pytest.raises(AssertionError):
        zigzag_reorder_varlen(x, cu, dim=0, cp_size=2)
    with pytest.raises(AssertionError):
        local_cu_seqlens(cu, cp_size=2)
