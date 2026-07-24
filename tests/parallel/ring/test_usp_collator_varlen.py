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

"""CPU tests for SequenceParallelCollator USP varlen (packed) slicing.

Uses a fake ParallelState so no device mesh / GPUs are needed. Verifies that
per-document zig-zag slicing across the cp group, when gathered back, restores
the full sequence, and that the derived per-document offsets are correct.
"""

from dataclasses import dataclass

import torch

import veomni.data.data_collator as dc
from veomni.data.data_collator import SequenceParallelCollator


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


def _make_collator(monkeypatch, state):
    monkeypatch.setattr(dc, "get_parallel_state", lambda: state)
    return SequenceParallelCollator()


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
    import pytest as _pytest

    state = _FakeState(cp_size=2, ulysses_size=1, cp_rank=0, ulysses_rank=0)
    collator = _make_collator(monkeypatch, state)
    # two docs: len 8 (ok) and len 6 (NOT divisible by 2*cp=4). The len-6 doc
    # cannot be balanced across the cp group and must raise a clear error.
    position_ids = torch.tensor([[*range(8), *range(6)]])
    with _pytest.raises(ValueError, match="divisible by 2.cp_size"):
        collator._compute_cp_cu_seqlens({"position_ids": position_ids})
