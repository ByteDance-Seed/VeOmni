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

"""CPU unit tests for USP data plumbing: zig-zag reorder helpers + collator slicing.

Consolidates the pure-CPU (no GPU / no distributed) USP data tests:
  * zig-zag block ordering, dense + varlen reorder/undo round-trips;
  * ``SequenceParallelCollator`` USP varlen (packed) per-document slicing,
    reassembly, cu_seqlens derivation, and alignment error handling.
"""

# ── zig-zag reorder helpers ───────────────────────────────────────────────────
from dataclasses import dataclass

import pytest
import torch

import veomni.data.data_collator as dc
from veomni.data.data_collator import MainCollator, PackingCollator, SequenceParallelCollator
from veomni.distributed.sequence_parallel.data import (
    local_cu_seqlens,
    zigzag_block_order,
    zigzag_reorder,
    zigzag_reorder_varlen,
    zigzag_undo,
)
from veomni.utils.constants import IGNORE_INDEX


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


# ── SequenceParallelCollator USP varlen slicing ───────────────────────────────


@dataclass
class _FakeState:
    cp_size: int
    ulysses_size: int
    cp_rank: int
    ulysses_rank: int

    @property
    def sp_size(self):
        return self.cp_size * self.ulysses_size

    @property
    def sp_rank(self):
        return self.ulysses_rank * self.cp_size + self.cp_rank

    @property
    def sp_enabled(self):
        return self.sp_size > 1


def _make_collator(monkeypatch, state):
    monkeypatch.setattr(dc, "get_parallel_state", lambda: state)
    return SequenceParallelCollator()


def _unaligned_text_features():
    return [
        {
            "input_ids": torch.tensor([10, 11, 12]),
            "labels": torch.tensor([10, 11, 12]),
            "attention_mask": torch.ones(3, dtype=torch.long),
            "position_ids": torch.arange(3),
        },
        {
            "input_ids": torch.tensor([20, 21]),
            "labels": torch.tensor([20, 21]),
            "attention_mask": torch.ones(2, dtype=torch.long),
            "position_ids": torch.arange(2),
        },
    ]


def test_packing_collator_aligns_each_document_for_cp(monkeypatch):
    state = _FakeState(cp_size=2, ulysses_size=1, cp_rank=0, ulysses_rank=0)
    monkeypatch.setattr(dc, "get_parallel_state", lambda: state)

    batch = PackingCollator(pad_to_length=12)(_unaligned_text_features())

    assert batch["input_ids"].tolist() == [[10, 11, 12, 0, 20, 21, 0, 0, 0, 0, 0, 0]]
    assert batch["labels"].tolist() == [
        [10, 11, 12, IGNORE_INDEX, IGNORE_INDEX, 21, IGNORE_INDEX, IGNORE_INDEX, *([IGNORE_INDEX] * 4)]
    ]
    assert batch["attention_mask"].tolist() == [[1] * 12]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3, 0, 1, 2, 3, 0, 0, 0, 0]]
    assert batch[dc._LINEAR_ATTN_TAIL_PADDING_LENGTH] == 4


def test_main_collator_coalesces_fixed_tail_after_cp_document_alignment(monkeypatch):
    state = _FakeState(cp_size=2, ulysses_size=1, cp_rank=0, ulysses_rank=0)
    monkeypatch.setattr(dc, "get_parallel_state", lambda: state)

    batch = MainCollator(pad_to_length=12)(_unaligned_text_features())

    assert batch["cu_seq_lens_q"].tolist() == [0, 4, 8, 12]
    assert batch["cu_seq_lens_k"].tolist() == [0, 4, 8, 12]
    assert int(batch["tail_padding_length"]) == 4
    assert all(length % 4 == 0 for length in torch.diff(batch["cu_seq_lens_q"]).tolist())


def test_compute_cp_cu_seqlens_two_docs(monkeypatch):
    state = _FakeState(cp_size=2, ulysses_size=1, cp_rank=0, ulysses_rank=0)
    collator = _make_collator(monkeypatch, state)
    # two documents of length 8 and 4 -> position_ids restart at each doc
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]])
    cu = collator._compute_cp_cu_seqlens({"position_ids": position_ids})
    assert cu.tolist() == [0, 8, 12]


def test_varlen_slices_reassemble_full_sequence(monkeypatch):
    # cp=2, ulysses=2 -> sp=4. Two docs, each divisible by 2*cp=4.
    cp, uly = 2, 2
    doc_lens = [16, 8]
    seq = sum(doc_lens)
    position_ids = torch.tensor([list(range(doc_lens[0])) + list(range(doc_lens[1]))])
    tokens = torch.arange(seq).view(1, seq)

    # Gather every (cp_rank, ulysses_rank) slice and rebuild the reordered seq.
    gathered = []
    for uly_rank in range(uly):
        for cp_rank in range(cp):
            state = _FakeState(cp_size=cp, ulysses_size=uly, cp_rank=cp_rank, ulysses_rank=uly_rank)
            collator = _make_collator(monkeypatch, state)
            collator._cp_cu_seqlens = collator._compute_cp_cu_seqlens({"position_ids": position_ids})
            local = collator.sp_slice("input_ids", tokens.clone(), dim=1)
            gathered.append((uly_rank, cp_rank, local))

    # Reconstruct per cp-region: for each cp_rank concat ulysses-inner pieces,
    # then the cp regions form the per-document zig-zag reorder of the sequence.
    from veomni.distributed.sequence_parallel.data import zigzag_reorder_varlen

    cu = torch.tensor([0, doc_lens[0], seq], dtype=torch.int32)
    expected_reordered = zigzag_reorder_varlen(tokens, cu, dim=1, cp_size=cp).view(-1).tolist()

    # cp is OUTER, ulysses is INNER: rebuild in that order.
    cp_chunk = seq // cp
    rebuilt = [0] * seq
    for uly_rank, cp_rank, local in gathered:
        uly_chunk = cp_chunk // uly
        base = cp_rank * cp_chunk + uly_rank * uly_chunk
        rebuilt[base : base + uly_chunk] = local.view(-1).tolist()
    assert rebuilt == expected_reordered


def test_single_doc_uses_dense_path(monkeypatch):
    state = _FakeState(cp_size=2, ulysses_size=1, cp_rank=0, ulysses_rank=0)
    collator = _make_collator(monkeypatch, state)
    position_ids = torch.arange(16).view(1, 16)
    cu = collator._compute_cp_cu_seqlens({"position_ids": position_ids})
    # a single monotonically-increasing document -> no packing -> None
    assert cu is None


def test_sp_pad_tail_segment_is_aligned_when_docs_aligned(monkeypatch):
    # cp=2 -> pad multiple is 2*sp_size = 2*(2*1)=4. Two aligned docs (8, 4)
    # totalling 12 (already a multiple of 4) need no padding.
    state = _FakeState(cp_size=2, ulysses_size=1, cp_rank=0, ulysses_rank=0)
    collator = _make_collator(monkeypatch, state)
    position_ids = torch.tensor([[*range(8), *range(4)]])
    cu = collator._compute_cp_cu_seqlens({"position_ids": position_ids})
    assert cu.tolist() == [0, 8, 12]

    # Two aligned docs (8, 4) plus a third aligned doc (8) total 20; still a
    # multiple of 4, still no pad. The SP-pad tail only appears when the total is
    # not already aligned; because every real doc is a multiple of 2*cp, whenever
    # a pad IS added the resulting tail segment is a multiple of 2*cp too (total
    # is a multiple of 2*cp). All resulting segments pass the divisibility check.
    position_ids3 = torch.tensor([[*range(8), *range(4), *range(8)]])
    cu3 = collator._compute_cp_cu_seqlens({"position_ids": position_ids3})
    assert cu3.tolist() == [0, 8, 12, 20]
    seglens = (cu3[1:] - cu3[:-1]).tolist()
    assert all(seg % 4 == 0 for seg in seglens)


def test_unaligned_document_raises_actionable_error(monkeypatch):
    state = _FakeState(cp_size=2, ulysses_size=1, cp_rank=0, ulysses_rank=0)
    collator = _make_collator(monkeypatch, state)
    # two docs: len 8 (ok) and len 6 (NOT divisible by 2*cp=4). The len-6 doc
    # cannot be balanced across the cp group and must raise a clear error.
    position_ids = torch.tensor([[*range(8), *range(6)]])
    with pytest.raises(ValueError, match="divisible by 2.cp_size"):
        collator._compute_cp_cu_seqlens({"position_ids": position_ids})
