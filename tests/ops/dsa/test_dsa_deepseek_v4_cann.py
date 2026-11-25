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

"""``deepseek_v4`` ``cann`` indexer and sparse attention.

CANN call layout is checked on any host, so those cases ask for no accelerator.
The indexer and Sparse FlashMLA are compared with eager on Ascend. The parity cases
use selections the kernel serves -- the whole compressed table and the kernel's
512/1024 widths, the latter over a table both smaller and larger than the
selection. A selection below 512 that does not span its table returned nonfinite
rows on CANN 9.2.1, so the adapter rejects it instead.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.models.transformers.deepseek_v4.packed_utils import (
    build_packed_compression_metadata,
    packed_compressed_causal_ranges,
)
from veomni.ops import resolve_op
from veomni.ops.kernels.dsa.attention.deepseek_v4 import cann as cann_attention
from veomni.ops.kernels.dsa.attention.deepseek_v4 import eager as eager_attention
from veomni.ops.kernels.dsa.indexer.deepseek_v4 import cann as cann_indexer
from veomni.ops.kernels.dsa.sparse_indices import CompressedCandidates
from veomni.ops.kernels.dsa.vendor import npu_sparse_flash_mla as vendor_flash_mla
from veomni.utils.device import IS_NPU_AVAILABLE, get_device_type


_HEADS, _DIM, _RATE = 4, 16, 4

# Only the parity cases below execute kernels; every other case records the
# arguments an adapter hands its kernel, which is host work. Without this marker
# each of them would reserve an accelerator card per runner.
pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class _FakeOps:
    """Records CANN calls and returns ``-1`` in the last selected slot of every row."""

    def __init__(self) -> None:
        self.metadata_calls: list[dict[str, Any]] = []
        self.indexer_calls: list[tuple[tuple[Tensor, ...], dict[str, Any]]] = []

    def lightning_indexer_metadata(self, *args: Any, **kwargs: Any) -> Tensor:
        self.metadata_calls.append({"args": args, **kwargs})
        return torch.empty(1, dtype=torch.int32)

    def lightning_indexer(self, q: Tensor, k: Tensor, weights: Tensor, top_k: int, **kwargs: Any):
        self.indexer_calls.append(((q, k, weights), {"top_k": top_k, **kwargs}))
        shape = (*q.shape[: 2 if kwargs["layout_q"] == "BSND" else 1], 1, top_k)
        indices = torch.arange(top_k, dtype=torch.int32).expand(shape).clone()
        indices[..., -1] = -1
        values = torch.ones(shape, dtype=torch.float32)
        return indices, values


@pytest.fixture
def fake_ops(monkeypatch: pytest.MonkeyPatch) -> _FakeOps:
    ops = _FakeOps()
    monkeypatch.setattr(cann_indexer, "_ops", lambda: ops)
    cann_indexer._causal_plan.cache_clear()
    cann_indexer._packed_plan.cache_clear()
    cann_indexer._context_parallel_range_ends.cache_clear()
    yield ops
    cann_indexer._causal_plan.cache_clear()
    cann_indexer._packed_plan.cache_clear()
    cann_indexer._context_parallel_range_ends.cache_clear()


def _bshd_inputs(batch: int, seq_len: int, compressed_len: int) -> tuple[Tensor, Tensor, Tensor]:
    q = torch.randn(batch, seq_len, _HEADS, _DIM).to(torch.bfloat16)
    k = torch.randn(batch, compressed_len, _DIM).to(torch.bfloat16)
    weights = torch.randn(batch, seq_len, _HEADS)
    return q, k, weights


def _packed_ranges(slices: tuple[tuple[int, int], ...], position_ids: Tensor) -> tuple[Tensor, Tensor]:
    metadata = build_packed_compression_metadata(torch.empty(()), position_ids, slices, (_RATE,))
    return packed_compressed_causal_ranges(metadata[_RATE])


def test_causal_call_reads_model_storage_and_returns_contract_order(fake_ops: _FakeOps):
    q, k, weights = _bshd_inputs(batch=2, seq_len=9, compressed_len=2)

    scores, indices = cann_indexer.wrapper(q.transpose(0, 1), k.transpose(0, 1), weights.transpose(0, 1), _RATE, 4)

    (cann_q, cann_k, cann_weights), kwargs = fake_ops.indexer_calls[0]
    assert cann_q.data_ptr() == q.data_ptr() and cann_q.shape == (2, 9, _HEADS, _DIM)
    assert cann_k.data_ptr() == k.data_ptr() and cann_k.shape == (2, 2, 1, _DIM)
    assert cann_weights.data_ptr() == weights.data_ptr()
    assert (kwargs["layout_q"], kwargs["layout_k"], kwargs["top_k"]) == ("BSND", "BSND", 2)
    assert kwargs["cmp_residual_k"].tolist() == [1, 1]
    assert indices.shape == scores.shape == (2, 9, 2)
    assert indices.dtype == torch.int32 and scores.dtype == torch.float32
    assert (indices[..., 0] == 0).all() and (indices[..., 1] == -1).all()
    assert (scores[..., 0] == 1).all() and torch.isneginf(scores[..., 1]).all()


def test_causal_metadata_is_built_once_per_shape(fake_ops: _FakeOps):
    q, k, weights = _bshd_inputs(batch=1, seq_len=8, compressed_len=2)
    for _ in range(3):
        cann_indexer.wrapper(q.transpose(0, 1), k.transpose(0, 1), weights.transpose(0, 1), _RATE, 2)

    assert len(fake_ops.metadata_calls) == 1
    assert len(fake_ops.indexer_calls) == 3


def test_packed_call_uses_host_boundaries_and_pads_to_contract_width(fake_ops: _FakeOps):
    slices = ((0, 9), (9, 13))
    position_ids = torch.cat([torch.arange(9), torch.arange(4)]).unsqueeze(0)
    range_starts, range_ends = _packed_ranges(slices, position_ids)
    q, k, weights = _bshd_inputs(batch=1, seq_len=13, compressed_len=3)

    scores, indices = cann_indexer.wrapper(
        q.transpose(0, 1),
        k.transpose(0, 1),
        weights.transpose(0, 1),
        _RATE,
        4,
        cu_seqlen_ks=range_starts,
        cu_seqlen_ke=range_ends,
        packed_sequence_slices=slices,
    )

    (cann_q, cann_k, _), kwargs = fake_ops.indexer_calls[0]
    assert cann_q.data_ptr() == q.data_ptr() and cann_q.shape == (13, _HEADS, _DIM)
    assert cann_k.shape == (3, 1, _DIM)
    assert (kwargs["layout_q"], kwargs["layout_k"]) == ("TND", "TND")
    assert kwargs["cu_seqlens_q"].tolist() == [0, 9, 13]
    assert kwargs["cu_seqlens_k"].tolist() == [0, 2, 3]
    assert kwargs["cmp_residual_k"].tolist() == [1, 0]
    # TND caps K at the longest segment (2 entries); the contract width is min(topk, S_kv) = 3.
    assert kwargs["top_k"] == 2
    (metadata,) = fake_ops.metadata_calls
    assert (metadata["batch_size"], metadata["max_seqlen_q"], metadata["max_seqlen_k"]) == (2, 9, 2)
    assert indices.shape == scores.shape == (1, 13, 3)
    assert (indices[..., 1:] == -1).all() and torch.isneginf(scores[..., 1:]).all()
    # TND selects within each segment; the contract addresses the packed buffer, so the
    # second segment's slot 0 is compressed row 2 rather than row 0.
    assert indices[0, :9, 0].tolist() == [0] * 9
    assert indices[0, 9:, 0].tolist() == [2] * 4


def test_single_segment_runs_the_causal_layout(fake_ops: _FakeOps):
    slices = ((0, 8),)
    range_starts, range_ends = _packed_ranges(slices, torch.arange(8).unsqueeze(0))
    q, k, weights = _bshd_inputs(batch=1, seq_len=8, compressed_len=2)

    cann_indexer.wrapper(
        q.transpose(0, 1),
        k.transpose(0, 1),
        weights.transpose(0, 1),
        _RATE,
        2,
        cu_seqlen_ks=range_starts,
        cu_seqlen_ke=range_ends,
        packed_sequence_slices=slices,
    )

    assert fake_ops.indexer_calls[0][1]["layout_q"] == "BSND"


def test_rejects_ranges_the_slices_do_not_imply(fake_ops: _FakeOps):
    slices = ((0, 9), (9, 13))
    # Positions that keep counting across the boundary give the second segment a wider range.
    range_starts, range_ends = _packed_ranges(slices, torch.arange(13).unsqueeze(0))
    q, k, weights = _bshd_inputs(batch=1, seq_len=13, compressed_len=3)

    with pytest.raises(ValueError, match="restarting at 0"):
        cann_indexer.wrapper(
            q.transpose(0, 1),
            k.transpose(0, 1),
            weights.transpose(0, 1),
            _RATE,
            4,
            cu_seqlen_ks=range_starts,
            cu_seqlen_ke=range_ends,
            packed_sequence_slices=slices,
        )
    assert fake_ops.indexer_calls == []


@pytest.mark.parametrize(
    ("slices", "match"),
    [
        (None, "without slices"),
        # A Ulysses or context-parallel shard keeps the global slices for local queries.
        (((0, 8), (8, 16)), "spanning all 8 query rows"),
    ],
)
def test_rejects_explicit_ranges_it_cannot_express(fake_ops: _FakeOps, slices, match: str):
    q, k, weights = _bshd_inputs(batch=1, seq_len=8, compressed_len=2)
    ranges = torch.zeros(8, dtype=torch.int32)

    with pytest.raises(ValueError, match=match):
        cann_indexer.wrapper(
            q.transpose(0, 1),
            k.transpose(0, 1),
            weights.transpose(0, 1),
            _RATE,
            2,
            cu_seqlen_ks=ranges,
            cu_seqlen_ke=ranges,
            packed_sequence_slices=slices,
        )


def test_rejects_supplied_topk_indices(fake_ops: _FakeOps):
    q, k, weights = _bshd_inputs(batch=1, seq_len=8, compressed_len=2)
    with pytest.raises(NotImplementedError, match="topk_indices"):
        cann_indexer.wrapper(
            q.transpose(0, 1),
            k.transpose(0, 1),
            weights.transpose(0, 1),
            _RATE,
            2,
            topk_indices=torch.zeros(1, 8, 2, dtype=torch.int32),
        )


def test_empty_compressed_kv_returns_an_empty_selection(fake_ops: _FakeOps):
    q, k, weights = _bshd_inputs(batch=2, seq_len=3, compressed_len=0)

    scores, indices = cann_indexer.wrapper(q.transpose(0, 1), k.transpose(0, 1), weights.transpose(0, 1), _RATE, 2)

    assert scores.shape == indices.shape == (2, 3, 0)
    assert indices.dtype == torch.int32
    assert fake_ops.indexer_calls == [] and fake_ops.metadata_calls == []


def _causal_ranges(offset: int, seq_len: int) -> tuple[Tensor, Tensor]:
    """The ranges the model builds for a context-parallel shard at ``offset``."""
    starts = torch.zeros(seq_len, dtype=torch.int32)
    ends = ((offset + torch.arange(seq_len) + 1) // _RATE).to(torch.int32)
    return starts, ends


def _install_context_parallel(monkeypatch: pytest.MonkeyPatch, cp_rank: int, cp_size: int) -> None:
    state = SimpleNamespace(cp_enabled=True, ulysses_enabled=False, cp_rank=cp_rank, cp_size=cp_size)
    monkeypatch.setattr(cann_indexer, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(cann_indexer, "get_parallel_state", lambda: state)


@pytest.mark.parametrize(
    ("cp_rank", "cp_size", "seq_len", "offset", "prefix_len", "residual", "global_slots"),
    [
        # 16 rows over cp_size=4: rank 1 owns rows [4, 8) and its last query sees
        # two of the all-gathered table's four compressed rows.
        (1, 4, 4, 4, 2, 0, 4),
        # 10 rows over cp_size=2: rank 0 owns rows [0, 5), narrows hardest, and the
        # non-zero remainder keeps the kernel's column alignment.
        (0, 2, 5, 0, 1, 1, 2),
    ],
    ids=["middle_rank", "first_rank"],
)
def test_context_parallel_narrows_the_table_and_numbers_rows_globally(
    fake_ops: _FakeOps,
    monkeypatch: pytest.MonkeyPatch,
    cp_rank: int,
    cp_size: int,
    seq_len: int,
    offset: int,
    prefix_len: int,
    residual: int,
    global_slots: int,
):
    _install_context_parallel(monkeypatch, cp_rank, cp_size)
    q, k, weights = _bshd_inputs(batch=1, seq_len=seq_len, compressed_len=global_slots)
    starts, ends = _causal_ranges(offset, seq_len)

    scores, indices = cann_indexer.wrapper(
        q.transpose(0, 1),
        k.transpose(0, 1),
        weights.transpose(0, 1),
        _RATE,
        4,
        cu_seqlen_ks=starts,
        cu_seqlen_ke=ends,
    )

    (cann_q, cann_k, _), kwargs = fake_ops.indexer_calls[0]
    assert cann_q.shape == (1, seq_len, _HEADS, _DIM)
    # The table is cut to the prefix the shard's last query can see; the kernel
    # reconstructs ``kv_len = offset + seq_len`` from it and the remainder.
    assert cann_k.shape == (1, prefix_len, 1, _DIM)
    torch.testing.assert_close(cann_k[0, :, 0], k[0, :prefix_len])
    assert kwargs["cmp_residual_k"].tolist() == [residual]
    assert (kwargs["layout_q"], kwargs["layout_k"], kwargs["top_k"]) == ("BSND", "BSND", prefix_len)
    (metadata,) = fake_ops.metadata_calls
    assert (metadata["max_seqlen_q"], metadata["max_seqlen_k"]) == (seq_len, prefix_len)
    # The kernel's selection addresses the narrowed prefix, which starts at the
    # global slot 0, so nothing is rebased here. The contract width is still
    # min(topk, global compressed length), padded with -1 past the prefix.
    assert indices.shape == scores.shape == (1, seq_len, min(4, global_slots))


def test_context_parallel_rejects_ranges_from_another_shard(fake_ops: _FakeOps, monkeypatch: pytest.MonkeyPatch):
    _install_context_parallel(monkeypatch, cp_rank=2, cp_size=4)
    q, k, weights = _bshd_inputs(batch=1, seq_len=4, compressed_len=4)
    # Ranges an offset-0 shard would produce: the kernel would mask from the wrong
    # origin, so the adapter refuses instead of running.
    starts, ends = _causal_ranges(0, 4)

    with pytest.raises(ValueError, match="global row"):
        cann_indexer.wrapper(
            q.transpose(0, 1),
            k.transpose(0, 1),
            weights.transpose(0, 1),
            _RATE,
            4,
            cu_seqlen_ks=starts,
            cu_seqlen_ke=ends,
        )
    assert fake_ops.indexer_calls == []


def test_context_parallel_rejects_range_tables_of_the_wrong_width(fake_ops: _FakeOps, monkeypatch: pytest.MonkeyPatch):
    # A zero-row table compares equal to nothing, and ``.all()`` over an empty
    # comparison reports agreement, so the values alone cannot screen it out.
    _install_context_parallel(monkeypatch, cp_rank=1, cp_size=4)
    q, k, weights = _bshd_inputs(batch=1, seq_len=4, compressed_len=4)
    _, ends = _causal_ranges(4, 4)
    empty = ends.unsqueeze(0)[0:0]
    assert empty.shape == (0, 4)

    with pytest.raises(ValueError, match="one cu_seqlen_ks per local query"):
        cann_indexer.wrapper(
            q.transpose(0, 1),
            k.transpose(0, 1),
            weights.transpose(0, 1),
            _RATE,
            4,
            cu_seqlen_ks=empty,
            cu_seqlen_ke=empty,
        )
    assert fake_ops.indexer_calls == []


def test_ulysses_explicit_ranges_stay_unsupported(fake_ops: _FakeOps, monkeypatch: pytest.MonkeyPatch):
    # Ulysses also passes global ranges without slices, but it partitions the
    # full-sequence queries instead of handing over contiguous ``cp_rank * L``
    # shards, so the CP narrowing would number its rows from the wrong origin.
    state = SimpleNamespace(cp_enabled=False, ulysses_enabled=True, cp_rank=0, cp_size=1)
    monkeypatch.setattr(cann_indexer, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(cann_indexer, "get_parallel_state", lambda: state)
    q, k, weights = _bshd_inputs(batch=1, seq_len=8, compressed_len=2)
    ranges, _ = _causal_ranges(0, 8)

    with pytest.raises(ValueError, match="without slices"):
        cann_indexer.wrapper(
            q.transpose(0, 1),
            k.transpose(0, 1),
            weights.transpose(0, 1),
            _RATE,
            2,
            cu_seqlen_ks=ranges,
            cu_seqlen_ke=ranges,
        )


@pytest.mark.parametrize(
    ("cp_rank", "avail", "residual", "compressed_start", "top_k", "rebased"),
    [
        # Rank 0 runs the whole first sample (2 compressed rows, remainder 1);
        # its 3-row head of the middle sample owns no complete window yet.
        (0, 2, 1, 0, 2, [0, -1, -1, -1]),
        # Rank 1's first 9 rows are the truncated tail of the middle sample: all
        # 3 of its compressed rows with no remainder number the shard's first
        # query at its global row 12 within the sample, and the indices rebase
        # by the sample's compressed start (packed table row 2).
        (1, 3, 0, 2, 3, [2, 3, -1, -1]),
    ],
    ids=["first_rank", "last_rank"],
)
def test_packed_context_parallel_calls_each_intersected_sample(
    fake_ops: _FakeOps,
    monkeypatch: pytest.MonkeyPatch,
    cp_rank: int,
    avail: int,
    residual: int,
    compressed_start: int,
    top_k: int,
    rebased: list[int],
):
    # A global packed sequence of 24 rows over cp_size=2. The middle sample
    # spans the shard boundary and the 3-row third sample (21, 24) has no
    # compressed row of its own, so each rank runs one narrowed BSND call for
    # the intersecting sample that owns compressed rows and leaves the sample
    # without one -1 / -inf, exactly as the eager row would.
    _install_context_parallel(monkeypatch, cp_rank, cp_size=2)
    slices = ((0, 9), (9, 21), (21, 24))
    position_ids = torch.cat([torch.arange(9), torch.arange(12), torch.arange(3)]).unsqueeze(0)
    range_starts, range_ends = _packed_ranges(slices, position_ids)
    q, k, weights = _bshd_inputs(batch=1, seq_len=12, compressed_len=5)

    scores, indices = cann_indexer.wrapper(
        q.transpose(0, 1),
        k.transpose(0, 1),
        weights.transpose(0, 1),
        _RATE,
        4,
        cu_seqlen_ks=range_starts[cp_rank * 12 : cp_rank * 12 + 12],
        cu_seqlen_ke=range_ends[cp_rank * 12 : cp_rank * 12 + 12],
        packed_sequence_slices=slices,
    )

    assert len(fake_ops.indexer_calls) == 1
    (cann_q, cann_k, cann_weights), kwargs = fake_ops.indexer_calls[0]
    assert cann_q.shape == (1, 9, _HEADS, _DIM)
    assert cann_weights.shape == (1, 9, _HEADS)
    # The sample's compressed buffer starts at the sample's first compressed row
    # and is narrowed to the prefix the shard's last query in it can see, with
    # the remainder of that prefix keeping the kernel's column alignment.
    assert cann_k.shape == (1, avail, 1, _DIM)
    torch.testing.assert_close(cann_k[0, :, 0], k[0, compressed_start : compressed_start + avail])
    assert kwargs["cmp_residual_k"].tolist() == [residual]
    assert (kwargs["layout_q"], kwargs["layout_k"], kwargs["top_k"]) == ("BSND", "BSND", top_k)
    assert kwargs["mask_mode"] == cann_indexer._SEGMENT_CAUSAL_MASK_MODE
    # The first 9 rows select the sample's compressed rows rebased into the
    # packed table; the remaining 3 rows own no complete window and stay -1.
    assert indices.shape == scores.shape == (1, 12, 4)
    assert indices[0, :9].equal(torch.tensor([rebased] * 9, dtype=torch.int32))
    assert indices[0, 9:].equal(torch.full((3, 4), -1, dtype=torch.int32))
    assert torch.isinf(scores[0, 9:]).all()


def test_packed_context_parallel_rejects_ranges_from_another_shard(
    fake_ops: _FakeOps, monkeypatch: pytest.MonkeyPatch
):
    _install_context_parallel(monkeypatch, cp_rank=1, cp_size=2)
    slices = ((0, 9), (9, 24))
    position_ids = torch.cat([torch.arange(9), torch.arange(15)]).unsqueeze(0)
    range_starts, range_ends = _packed_ranges(slices, position_ids)
    q, k, weights = _bshd_inputs(batch=1, seq_len=12, compressed_len=5)

    # Ranges an offset-0 shard would produce: the kernel would mask from the
    # wrong origin, so the adapter refuses instead of running.
    with pytest.raises(ValueError, match="global ranges"):
        cann_indexer.wrapper(
            q.transpose(0, 1),
            k.transpose(0, 1),
            weights.transpose(0, 1),
            _RATE,
            4,
            cu_seqlen_ks=range_starts[:12],
            cu_seqlen_ke=range_ends[:12],
            packed_sequence_slices=slices,
        )
    assert fake_ops.indexer_calls == []


def _assert_same_selection(actual: tuple[Tensor, Tensor], expected: tuple[Tensor, Tensor]) -> None:
    """Compare per-row selections independent of the order each backend lists them in."""
    actual_scores, actual_indices = (tensor.cpu() for tensor in actual)
    expected_scores, expected_indices = (tensor.cpu() for tensor in expected)
    assert actual_indices.shape == expected_indices.shape
    torch.testing.assert_close((actual_indices >= 0).sum(-1), (expected_indices >= 0).sum(-1))
    torch.testing.assert_close(
        actual_scores.sort(-1, descending=True).values,
        expected_scores.sort(-1, descending=True).values,
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator"))
@pytest.mark.skipif(
    not IS_NPU_AVAILABLE or importlib.util.find_spec("cann_ops_transformer") is None,
    reason="CANN Lightning Indexer requires Ascend NPU and cann_ops_transformer",
)
@pytest.mark.parametrize("segment_lengths", [(64,), (37, 64, 27)], ids=["causal", "packed"])
def test_cann_matches_eager(segment_lengths: tuple[int, ...]):
    torch.manual_seed(3)
    device = get_device_type()
    heads, dim, topk = 32, 128, 8
    seq_len = sum(segment_lengths)
    compressed_len = sum(length // _RATE for length in segment_lengths)
    q = torch.randn(1, seq_len, heads, dim, device=device).to(torch.bfloat16)
    k = torch.randn(1, compressed_len, dim, device=device).to(torch.bfloat16)
    weights = torch.randn(1, seq_len, heads, device=device) * dim**-0.5
    kwargs: dict[str, Any] = {}
    if len(segment_lengths) > 1:
        bounds = torch.tensor((0, *segment_lengths)).cumsum(0).tolist()
        slices = tuple(zip(bounds[:-1], bounds[1:]))
        position_ids = torch.cat([torch.arange(length) for length in segment_lengths]).unsqueeze(0).to(device)
        metadata = build_packed_compression_metadata(q, position_ids, slices, (_RATE,))
        range_starts, range_ends = packed_compressed_causal_ranges(metadata[_RATE])
        kwargs = {"cu_seqlen_ks": range_starts, "cu_seqlen_ke": range_ends, "packed_sequence_slices": slices}
    operands = (q.transpose(0, 1), k.transpose(0, 1), weights.transpose(0, 1), _RATE, topk)

    expected = resolve_op("dsa_indexer", "deepseek_v4", "eager").wrapper(*operands, **kwargs)
    actual = resolve_op("dsa_indexer", "deepseek_v4", "cann").wrapper(*operands, **kwargs)

    _assert_same_selection(actual, expected)


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator"))
@pytest.mark.skipif(
    not IS_NPU_AVAILABLE or importlib.util.find_spec("cann_ops_transformer") is None,
    reason="CANN Lightning Indexer requires Ascend NPU and cann_ops_transformer",
)
@pytest.mark.parametrize("cp_rank", [0, 1], ids=["first_rank", "last_rank"])
def test_cann_matches_eager_packed_context_parallel(cp_rank: int, monkeypatch: pytest.MonkeyPatch):
    # A packed CP shard: one sample spans the shard boundary, so the per-sample
    # calls must reproduce the truncated tail (rank 1) or the truncated head
    # (rank 0) against the same replicated packed table and global ranges eager
    # sees.
    torch.manual_seed(3)
    _install_context_parallel(monkeypatch, cp_rank, cp_size=2)
    device = get_device_type()
    heads, dim, topk = 32, 128, 8
    segment_lengths = (37, 64, 27)
    seq_len = sum(segment_lengths) // 2
    compressed_len = sum(length // _RATE for length in segment_lengths)
    bounds = torch.tensor((0, *segment_lengths)).cumsum(0).tolist()
    slices = tuple(zip(bounds[:-1], bounds[1:]))
    position_ids = torch.cat([torch.arange(length) for length in segment_lengths]).unsqueeze(0).to(device)
    metadata = build_packed_compression_metadata(torch.empty((), device=device), position_ids, slices, (_RATE,))
    range_starts, range_ends = packed_compressed_causal_ranges(metadata[_RATE])
    q = torch.randn(1, seq_len, heads, dim, device=device).to(torch.bfloat16)
    k = torch.randn(1, compressed_len, dim, device=device).to(torch.bfloat16)
    weights = torch.randn(1, seq_len, heads, device=device) * dim**-0.5
    kwargs = {
        "cu_seqlen_ks": range_starts[cp_rank * seq_len : cp_rank * seq_len + seq_len],
        "cu_seqlen_ke": range_ends[cp_rank * seq_len : cp_rank * seq_len + seq_len],
        "packed_sequence_slices": slices,
    }
    operands = (q.transpose(0, 1), k.transpose(0, 1), weights.transpose(0, 1), _RATE, topk)

    expected = resolve_op("dsa_indexer", "deepseek_v4", "eager").wrapper(*operands, **kwargs)
    actual = resolve_op("dsa_indexer", "deepseek_v4", "cann").wrapper(*operands, **kwargs)

    _assert_same_selection(actual, expected)


_ATTN_HEADS, _ATTN_DIM, _WINDOW, _HCA_RATE = 2, 512, 128, 128


class _FlashMlaRecorder:
    """Records sparse FlashMLA calls and returns zeros shaped like the query."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Tensor | None, ...], dict[str, Any]]] = []

    def __call__(self, q: Tensor, ori_kv: Tensor, cmp_kv: Tensor, indices: Tensor | None, **kwargs: Any) -> Tensor:
        self.calls.append(((q, ori_kv, cmp_kv, indices), kwargs))
        return torch.zeros_like(q)


@pytest.fixture
def flash_mla(monkeypatch: pytest.MonkeyPatch) -> _FlashMlaRecorder:
    recorder = _FlashMlaRecorder()
    monkeypatch.setattr(cann_attention, "npu_sparse_flash_mla", recorder)
    cann_attention._segment_compressed_starts.cache_clear()
    yield recorder
    cann_attention._segment_compressed_starts.cache_clear()


@pytest.fixture
def cp_shard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mark the process as a CP rank: only then may the queries be a local shard."""
    monkeypatch.setattr(cann_attention, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(
        cann_attention,
        "get_parallel_state",
        lambda: SimpleNamespace(cp_enabled=True, ulysses_enabled=False),
    )


def _attention_inputs(
    seq_len: int, compressed_len: int, dtype: torch.dtype = torch.bfloat16, device: str = "cpu"
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    q = torch.randn(1, seq_len, _ATTN_HEADS, _ATTN_DIM, device=device).to(dtype)
    ori_kv = torch.randn(1, seq_len, _ATTN_DIM, device=device).to(dtype)
    cmp_kv = torch.randn(1, compressed_len, _ATTN_DIM, device=device).to(dtype)
    sink = torch.randn(_ATTN_HEADS, device=device).to(dtype)
    return q, ori_kv, cmp_kv, sink


def _segment_positions(segment_lengths: tuple[int, ...], device: str = "cpu") -> Tensor:
    return torch.cat([torch.arange(length) for length in segment_lengths]).unsqueeze(0).to(device)


@pytest.mark.parametrize(
    ("kv_len", "slices", "query_offset"),
    [(8, None, 0), (16, ((0, 16),), 8)],
    ids=["unpacked", "packed_context_parallel"],
)
def test_attention_explicit_mask_runs_the_eager_row(
    flash_mla: _FlashMlaRecorder, kv_len: int, slices, query_offset: int
):
    # Mask modes express causality, not a per-key mask, so a mask the caller supplied
    # must not be dropped: both the unpacked call and the packed per-sample path have
    # to reproduce eager rather than attend to keys the caller excluded.
    torch.manual_seed(11)
    seq_len, compressed_len = 8, kv_len // _RATE
    q = torch.randn(1, seq_len, _ATTN_HEADS, _ATTN_DIM)
    ori_kv = torch.randn(1, kv_len, _ATTN_DIM)
    cmp_kv = torch.randn(1, compressed_len, _ATTN_DIM)
    sink = torch.randn(_ATTN_HEADS)
    positions = (torch.arange(seq_len) + query_offset).unsqueeze(0)
    topk = torch.zeros(1, seq_len, 2, dtype=torch.int32)
    topk[..., 1] = -1
    mask = torch.ones(1, 1, seq_len, kv_len, dtype=torch.bool)
    mask[..., 6] = False
    kwargs = {
        "sliding_window": _WINDOW,
        "compress_ratio": _RATE,
        "candidates": CompressedCandidates(topk_indices=topk),
        "attention_mask": mask,
        "packed_sequence_slices": slices,
        "query_offset": query_offset,
        "sm_scale": 0.5,
    }

    expected = eager_attention.wrapper(q, ori_kv, cmp_kv, sink, positions, **kwargs)
    actual = cann_attention.wrapper(q, ori_kv, cmp_kv, sink, positions, **kwargs)

    assert flash_mla.calls == []
    torch.testing.assert_close(actual, expected)


def test_attention_packed_call_rebases_the_selection_per_segment(flash_mla: _FlashMlaRecorder):
    slices = ((0, 9), (9, 13))
    q, ori_kv, cmp_kv, sink = _attention_inputs(seq_len=13, compressed_len=3)
    topk = torch.full((1, 13, 3), -1, dtype=torch.int32)
    topk[0, :9, 0] = 1
    topk[0, 9:, 0] = 2

    cann_attention.wrapper(
        q,
        ori_kv,
        cmp_kv,
        sink,
        _segment_positions((9, 4)),
        sliding_window=_WINDOW,
        compress_ratio=_RATE,
        candidates=CompressedCandidates(topk_indices=topk),
        packed_sequence_slices=slices,
        sm_scale=0.5,
    )

    ((cann_q, cann_ori_kv, cann_cmp_kv, indices), kwargs) = flash_mla.calls[0]
    assert cann_q.data_ptr() == q.data_ptr()
    assert cann_ori_kv.shape == (1, 13, 1, _ATTN_DIM) and cann_cmp_kv.shape == (1, 3, 1, _ATTN_DIM)
    # The contract addresses the packed compressed buffer, TND each segment's own rows.
    assert indices[0, :9, 0].tolist() == [1] * 9 and indices[0, 9:, 0].tolist() == [0] * 4
    assert (indices[..., 1:] == -1).all()
    assert kwargs["packed_sequence_slices"] == slices
    assert (kwargs["cmp_ratio"], kwargs["softmax_scale"]) == (_RATE, 0.5)
    assert (kwargs["ori_mask_mode"], kwargs["cmp_mask_mode"]) == (4, 3)
    assert (kwargs["ori_win_left"], kwargs["ori_win_right"]) == (_WINDOW - 1, 0)
    assert kwargs["sinks"].dtype == torch.float32


def test_attention_packed_validation_uses_the_widest_segment_table(flash_mla: _FlashMlaRecorder):
    # The producer sizes a TND selection from the widest segment, not from the packed
    # total, and each segment's indices are rebased onto its own table first. Judging
    # the selection against the packed total (4 rows here) would refuse a correct
    # 3-wide selection over two 2-row tables.
    slices = ((0, 8), (8, 16))
    q, ori_kv, cmp_kv, sink = _attention_inputs(seq_len=16, compressed_len=4)
    topk = torch.full((1, 16, 3), -1, dtype=torch.int32)
    topk[0, :8, :2] = torch.tensor([0, 1], dtype=torch.int32)
    topk[0, 8:, :2] = torch.tensor([2, 3], dtype=torch.int32)

    cann_attention.wrapper(
        q,
        ori_kv,
        cmp_kv,
        sink,
        _segment_positions((8, 8)),
        sliding_window=_WINDOW,
        compress_ratio=_RATE,
        candidates=CompressedCandidates(topk_indices=topk),
        packed_sequence_slices=slices,
    )

    ((_, _, _, indices), _) = flash_mla.calls[0]
    # Both samples select their own rows 0 and 1; the third column is padding.
    assert indices.shape == (1, 16, 2)
    assert indices[0, :8].tolist() == [[0, 1]] * 8
    assert indices[0, 8:].tolist() == [[0, 1]] * 8


def test_attention_serves_a_kernel_width_over_a_wider_table(flash_mla: _FlashMlaRecorder):
    # The shipped config selects 512 rows per query while a CP rank holds 1024 or
    # 2048 compressed rows, so the selection is narrower than the table but exactly
    # one of the kernel's widths. That shape is served as-is, not trimmed.
    seq_len, compressed_len = 2052, 513
    q, ori_kv, cmp_kv, sink = _attention_inputs(seq_len=seq_len, compressed_len=compressed_len)
    topk = torch.full((1, seq_len, 512), -1, dtype=torch.int32)
    topk[..., 0] = 0

    cann_attention.wrapper(
        q,
        ori_kv,
        cmp_kv,
        sink,
        torch.arange(seq_len).unsqueeze(0),
        sliding_window=_WINDOW,
        compress_ratio=_RATE,
        candidates=CompressedCandidates(topk_indices=topk),
    )

    ((cann_q, cann_ori_kv, cann_cmp_kv, indices), _) = flash_mla.calls[0]
    assert cann_q.shape == (1, seq_len, _ATTN_HEADS, _ATTN_DIM)
    assert cann_ori_kv.shape == (1, seq_len, 1, _ATTN_DIM)
    assert cann_cmp_kv.shape == (1, compressed_len, 1, _ATTN_DIM)
    assert indices.shape == (1, seq_len, 512)
    assert (indices[..., 0] == 0).all() and (indices[..., 1:] == -1).all()


def test_attention_refuses_a_decode_call_outside_context_parallel(flash_mla: _FlashMlaRecorder):
    # One query ending at the last KV row is the decode layout, which the kernel has
    # no cache input for. Only a CP shard may pass a shorter query block, so outside
    # CP the call must be refused instead of reaching the kernel.
    q, ori_kv, cmp_kv, sink = _attention_inputs(seq_len=8, compressed_len=2)
    topk = torch.zeros(1, 1, 2, dtype=torch.int32)
    topk[..., 1] = -1

    with pytest.raises(ValueError, match="Decode-shaped calls need the eager implementation"):
        cann_attention.wrapper(
            q[:, :1],
            ori_kv,
            cmp_kv,
            sink,
            torch.full((1, 1), 7, dtype=torch.long),
            sliding_window=_WINDOW,
            compress_ratio=_RATE,
            candidates=CompressedCandidates(topk_indices=topk),
            query_offset=7,
        )
    assert flash_mla.calls == []


@pytest.mark.parametrize("slices", [None, ((0, 8),)], ids=["unpacked", "single_segment"])
def test_attention_one_segment_keeps_the_selection(flash_mla: _FlashMlaRecorder, slices):
    q, ori_kv, cmp_kv, sink = _attention_inputs(seq_len=8, compressed_len=2)
    topk = torch.tensor([[[1, -1]]], dtype=torch.int32).expand(1, 8, 2)

    cann_attention.wrapper(
        q,
        ori_kv,
        cmp_kv,
        sink,
        torch.arange(8).unsqueeze(0),
        sliding_window=_WINDOW,
        compress_ratio=_RATE,
        candidates=CompressedCandidates(topk_indices=topk),
        packed_sequence_slices=slices,
    )

    assert flash_mla.calls[0][0][3] is topk


def test_attention_hca_passes_no_selection(flash_mla: _FlashMlaRecorder):
    q, ori_kv, cmp_kv, sink = _attention_inputs(seq_len=256, compressed_len=2)
    candidates = CompressedCandidates(
        range_starts=torch.zeros(256, dtype=torch.int32), range_ends=torch.ones(256, dtype=torch.int32)
    )

    cann_attention.wrapper(
        q,
        ori_kv,
        cmp_kv,
        sink,
        torch.arange(256).unsqueeze(0),
        sliding_window=_WINDOW,
        compress_ratio=_HCA_RATE,
        candidates=candidates,
    )

    ((_, _, _, indices), kwargs) = flash_mla.calls[0]
    assert indices is None and kwargs["cmp_ratio"] == _HCA_RATE


@pytest.mark.parametrize("compressed_len", [None, 0], ids=["sliding_only", "no_compressed_rows_yet"])
def test_attention_without_compressed_rows_runs_eager(flash_mla: _FlashMlaRecorder, compressed_len):
    q, ori_kv, cmp_kv, sink = _attention_inputs(seq_len=6, compressed_len=0, dtype=torch.float32)
    cmp_kv = None if compressed_len is None else cmp_kv
    kwargs = {"sliding_window": 4, "compress_ratio": 0 if cmp_kv is None else _RATE}
    position_ids = torch.arange(6).unsqueeze(0)

    actual = cann_attention.wrapper(q, ori_kv, cmp_kv, sink, position_ids, **kwargs)
    expected = eager_attention.wrapper(q, ori_kv, cmp_kv, sink, position_ids, **kwargs)

    torch.testing.assert_close(actual, expected)
    assert flash_mla.calls == []


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"dropout": 0.1}, "dropout=0"),
        ({"return_attn_weights": True}, "output_attentions"),
        ({"return_lse": True}, "LSE"),
        ({"query_offset": 4}, "end inside the KV buffer"),
    ],
)
def test_attention_rejects_what_the_kernel_cannot_express(flash_mla: _FlashMlaRecorder, overrides, match: str):
    q, ori_kv, cmp_kv, sink = _attention_inputs(seq_len=8, compressed_len=2)
    kwargs = {"sliding_window": _WINDOW, "compress_ratio": _RATE, **overrides}
    topk = torch.zeros(1, 8, 1, dtype=torch.int32)

    with pytest.raises(ValueError, match=match):
        cann_attention.wrapper(
            q,
            ori_kv,
            cmp_kv,
            sink,
            torch.arange(8).unsqueeze(0),
            candidates=CompressedCandidates(topk_indices=topk),
            **kwargs,
        )
    assert flash_mla.calls == []


@pytest.mark.parametrize(
    ("query_offset", "cmp_rows", "residual"), [(0, 1, 2), (6, 3, 0)], ids=["first_rank", "last_rank"]
)
def test_attention_context_parallel_narrows_to_the_query_tail(
    flash_mla: _FlashMlaRecorder, cp_shard: None, query_offset: int, cmp_rows: int, residual: int
):
    # cp_size=2 over 12 rows: rank 0 owns rows [0, 6), rank 1 owns [6, 12), and both
    # receive the all-gathered 12-row KV. The kernel assumes the queries end at the
    # last KV row, so the adapter narrows the buffer to the local prefix.
    q = torch.randn(1, 6, _ATTN_HEADS, _ATTN_DIM).to(torch.bfloat16)
    ori_kv = torch.randn(1, 12, _ATTN_DIM).to(torch.bfloat16)
    cmp_kv = torch.randn(1, 3, _ATTN_DIM).to(torch.bfloat16)
    sink = torch.randn(_ATTN_HEADS).to(torch.bfloat16)
    # The indexer pads its selection to the global width (three compressed rows over
    # twelve tokens), so rank 0 receives three columns over the single row it serves.
    topk = torch.full((1, 6, 3), -1, dtype=torch.int32)
    topk[..., 0] = 0

    cann_attention.wrapper(
        q,
        ori_kv,
        cmp_kv,
        sink,
        torch.arange(query_offset, query_offset + 6).unsqueeze(0),
        sliding_window=_WINDOW,
        compress_ratio=_RATE,
        candidates=CompressedCandidates(topk_indices=topk),
        query_offset=query_offset,
    )

    ((cann_q, cann_ori_kv, cann_cmp_kv, cann_topk), kwargs) = flash_mla.calls[0]
    assert cann_q.shape == q.shape and torch.equal(cann_q, q)
    # The narrowed prefix is materialized into a canonical buffer, so compare values.
    assert cann_ori_kv.shape == (1, query_offset + 6, 1, _ATTN_DIM)
    assert torch.equal(cann_ori_kv.squeeze(2), ori_kv[:, : query_offset + 6])
    assert cann_cmp_kv.shape == (1, cmp_rows, 1, _ATTN_DIM)
    assert torch.equal(cann_cmp_kv.squeeze(2), cmp_kv[:, :cmp_rows])
    # Producer padding past the local table is trimmed instead of rejected.
    assert cann_topk.shape[-1] == cmp_rows
    # The compressed table is aligned to the KV prefix, not to the local query count.
    assert kwargs["cmp_residual_kv"].tolist() == [residual]


def test_attention_packed_context_parallel_calls_each_intersected_sample(flash_mla: _FlashMlaRecorder, cp_shard: None):
    # Global packed sequence of 24 rows over cp_size=2, so this rank owns rows [0, 12).
    # The second sample starts inside the shard, and every local query still needs the
    # rows its own sample contributes before it.
    slices = ((0, 6), (6, 18), (18, 24))
    q = torch.randn(1, 12, _ATTN_HEADS, _ATTN_DIM).to(torch.bfloat16)
    ori_kv = torch.randn(1, 24, _ATTN_DIM).to(torch.bfloat16)
    cmp_kv = torch.randn(1, 5, _ATTN_DIM).to(torch.bfloat16)
    sink = torch.randn(_ATTN_HEADS).to(torch.bfloat16)
    # The selection is padded to the global table width, so each sample carries more
    # columns than the rows it contributes to this shard.
    topk = torch.full((1, 12, 5), -1, dtype=torch.int32)
    topk[0, :6, 0] = 0  # first sample owns compressed row 0
    topk[0, 6:, 0] = 1  # second sample owns rows [1, 4)

    cann_attention.wrapper(
        q,
        ori_kv,
        cmp_kv,
        sink,
        _segment_positions((6, 6)),
        sliding_window=_WINDOW,
        compress_ratio=_RATE,
        candidates=CompressedCandidates(topk_indices=topk),
        packed_sequence_slices=slices,
        query_offset=0,
    )

    assert len(flash_mla.calls) == 2
    ((first_q, first_ori, first_cmp, first_idx), first_kwargs) = flash_mla.calls[0]
    assert first_q.shape[1] == 6 and torch.equal(first_q, q[:, :6])
    assert first_ori.shape == (1, 6, 1, _ATTN_DIM) and first_cmp.shape == (1, 1, 1, _ATTN_DIM)
    assert first_idx[0, :, 0].tolist() == [0] * 6
    # Columns past this sample's own table are dropped before the kernel call.
    assert first_idx.shape[-1] == 1
    # Each sample is its own single-segment call, so no TND slices reach the kernel.
    assert "packed_sequence_slices" not in first_kwargs

    # The second sample's buffer starts at global row 6, not at the shard start.
    ((second_q, second_ori, second_cmp, second_idx), _) = flash_mla.calls[1]
    assert second_q.shape[1] == 6 and torch.equal(second_q, q[:, 6:])
    assert second_ori.shape == (1, 6, 1, _ATTN_DIM)
    assert second_cmp.shape == (1, 1, 1, _ATTN_DIM)
    # Its rows start at compressed row 1, so the selection is rebased by that start.
    assert second_idx[0, :, 0].tolist() == [0] * 6
    assert second_idx.shape[-1] == 1


def test_attention_packed_context_parallel_leaves_short_samples_to_eager(flash_mla: _FlashMlaRecorder, cp_shard: None):
    # A padding tail is encoded as 1-token slices; no compressed row is visible there,
    # so those rows run the eager path while the rest of the shard keeps the kernel.
    slices = ((0, 6), (6, 7), (7, 8))
    q = torch.randn(1, 4, _ATTN_HEADS, _ATTN_DIM)
    ori_kv = torch.randn(1, 8, _ATTN_DIM)
    cmp_kv = torch.randn(1, 1, _ATTN_DIM)
    sink = torch.randn(_ATTN_HEADS)

    out = cann_attention.wrapper(
        q,
        ori_kv,
        cmp_kv,
        sink,
        torch.arange(4, 8).unsqueeze(0),
        sliding_window=_WINDOW,
        compress_ratio=_RATE,
        packed_sequence_slices=slices,
        query_offset=4,
    )

    assert out.shape == q.shape
    assert len(flash_mla.calls) == 1
    ((cann_q, cann_ori, cann_cmp, _), _) = flash_mla.calls[0]
    assert cann_q.shape[1] == 2 and cann_ori.shape == (1, 6, 1, _ATTN_DIM)
    assert cann_cmp.shape == (1, 1, 1, _ATTN_DIM)


def test_attention_packed_context_parallel_rebases_the_eager_fallback(flash_mla: _FlashMlaRecorder, cp_shard: None):
    # A sample can start before the shard and end inside its first compression
    # window, so the truncated prefix holds no compressed row and the sample runs
    # eagerly. Its local queries still start at a non-zero row of that buffer, which
    # the eager row only reaches through ``query_offset``; without it they would
    # read the sample's earlier rows.
    slices = ((0, 7), (7, 10), (10, 26))
    q = torch.randn(1, 8, _ATTN_HEADS, _ATTN_DIM).to(torch.bfloat16)
    ori_kv = torch.randn(1, 26, _ATTN_DIM).to(torch.bfloat16)
    cmp_kv = torch.randn(1, 5, _ATTN_DIM).to(torch.bfloat16)
    sink = torch.randn(_ATTN_HEADS).to(torch.bfloat16)

    out = cann_attention.wrapper(
        q,
        ori_kv,
        cmp_kv,
        sink,
        _segment_positions((2, 6)),
        sliding_window=_WINDOW,
        compress_ratio=_RATE,
        packed_sequence_slices=slices,
        query_offset=8,
    )

    assert out.shape == q.shape and len(flash_mla.calls) == 1
    ((cann_q, _, _, _), _) = flash_mla.calls[0]
    assert cann_q.shape[1] == 6
    # Rows 8 and 9 are rows 1 and 2 of the sample starting at row 7.
    assert torch.equal(
        out[:, :2],
        eager_attention.wrapper(
            q[:, :2],
            ori_kv[:, 7:10],
            cmp_kv[:, 0:0],
            sink,
            torch.arange(1, 3).unsqueeze(0),
            sliding_window=_WINDOW,
            compress_ratio=_RATE,
            query_offset=1,
        ),
    )


def test_attention_packed_context_parallel_keeps_the_gather_in_the_backward_graph(
    flash_mla: _FlashMlaRecorder,
    cp_shard: None,
):
    # The compressed rows reach every rank through a collective all-gather, so a
    # rank that takes the eager fallback must still produce a gradient for
    # ``cmp_kv``. Passing ``None`` instead of a zero-row slice would detach it, and
    # the other ranks would wait in the gather's backward.
    q = torch.randn(1, 4, _ATTN_HEADS, _ATTN_DIM, requires_grad=True)
    ori_kv = torch.randn(1, 8, _ATTN_DIM)
    cmp_kv = torch.randn(1, 1, _ATTN_DIM, requires_grad=True)
    sink = torch.randn(_ATTN_HEADS)

    out = cann_attention.wrapper(
        q,
        ori_kv,
        cmp_kv,
        sink,
        _segment_positions((2, 2)),
        sliding_window=_WINDOW,
        compress_ratio=_RATE,
        packed_sequence_slices=((0, 6), (6, 7), (7, 8)),
        query_offset=4,
    )
    out.sum().backward()

    assert cmp_kv.grad is not None
    assert torch.count_nonzero(cmp_kv.grad) == 0


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator"))
@pytest.mark.skipif(
    not IS_NPU_AVAILABLE or importlib.util.find_spec("cann_ops_transformer") is None,
    reason="CANN Sparse FlashMLA requires Ascend NPU and cann_ops_transformer",
)
@pytest.mark.parametrize("seq_len", [300, 2052], ids=["table_below_width", "table_above_width"])
def test_sparse_flash_mla_matches_eager_on_ascend(seq_len: int):
    """Compare the real kernel's forward and every input gradient with eager.

    Most queries are past their first window here, so the compressed table has
    visible rows and the call takes the index path rather than a fallback. The
    selection is 512 wide, one of the two widths the kernel is built for, and the
    table is narrower in one case and wider in the other, which is the shipped CP
    shape; slots past the table hold ``-1`` and the adapter accepts it.
    """
    torch.manual_seed(5)
    device = get_device_type()
    topk = 512
    compressed_len = seq_len // _RATE
    q = torch.randn(1, seq_len, _ATTN_HEADS, _ATTN_DIM, device=device).to(torch.bfloat16).requires_grad_()
    ori_kv = torch.randn(1, seq_len, _ATTN_DIM, device=device).to(torch.bfloat16).requires_grad_()
    cmp_kv = torch.randn(1, compressed_len, _ATTN_DIM, device=device).to(torch.bfloat16).requires_grad_()
    sink = (torch.randn(_ATTN_HEADS, device=device) * 0.5).requires_grad_()
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # Compressed row ``j`` closes at token ``4j + 3``, so query ``i`` sees the rows
    # below ``(i + 1) // 4``; selecting all of them keeps the selection spanning the
    # table, and the trailing slots stay ``-1``.
    ends = (torch.arange(seq_len, device=device) + 1) // _RATE
    starts = (ends - topk).clamp(min=0)
    columns = torch.arange(topk, device=device)
    selected = torch.where(columns[None, :] < (ends - starts)[:, None], columns[None, :] + starts[:, None], -1)
    candidates = CompressedCandidates(topk_indices=selected.unsqueeze(0).to(torch.int32))
    operands = (q, ori_kv, cmp_kv, sink, position_ids)
    kwargs = {"sliding_window": _WINDOW, "compress_ratio": _RATE, "candidates": candidates}

    expected = resolve_op("dsa_attention", "deepseek_v4", "eager").wrapper(*operands, **kwargs)
    actual = resolve_op("dsa_attention", "deepseek_v4", "cann").wrapper(*operands, **kwargs)
    # bf16 accumulation over a 512-wide head: the worst forward element differs by
    # 2e-2 and the worst gradient element by 4e-2, with a mean absolute error of
    # about 1e-3. The tolerance has to sit above that and far below the row-scale
    # error a wrong selection produced (0.2 on the forward alone).
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)

    rows = torch.randn_like(expected)
    inputs = (q, ori_kv, cmp_kv, sink)
    expected_grads = torch.autograd.grad(expected, inputs, rows, allow_unused=True)
    actual_grads = torch.autograd.grad(actual, inputs, rows, allow_unused=True)
    for name, expected_grad, actual_grad in zip(("q", "ori_kv", "cmp_kv", "sink"), expected_grads, actual_grads):
        assert actual_grad is not None and torch.isfinite(actual_grad).all(), f"{name} gradient is not finite"
        torch.testing.assert_close(actual_grad.float(), expected_grad.float(), atol=5e-2, rtol=5e-2)


def test_attention_rejects_a_selection_narrower_than_the_kernel_serves():
    # The shape the parity case used to run: 16 indices over a 75-row table. On CANN
    # 9.2.1 the kernel keeps the eager direction in the forward (cosine 0.998) but
    # scales some rows down and turns the ``q`` gradient into NaNs. No tolerance
    # makes a nonfinite gradient equal, so the adapter refuses this before the call.
    q = torch.randn(1, 300, _ATTN_HEADS, _ATTN_DIM).to(torch.bfloat16)
    ori_kv = torch.randn(1, 300, _ATTN_DIM).to(torch.bfloat16)
    cmp_kv = torch.randn(1, 75, _ATTN_DIM).to(torch.bfloat16)
    sink = torch.randn(_ATTN_HEADS)
    candidates = CompressedCandidates(topk_indices=torch.zeros(1, 300, 16, dtype=torch.int32))

    with pytest.raises(ValueError, match=r"or be 512/1024 wide; got 16 indices over 75 compressed rows"):
        cann_attention.wrapper(
            q,
            ori_kv,
            cmp_kv,
            sink,
            torch.arange(300).unsqueeze(0),
            sliding_window=_WINDOW,
            compress_ratio=_RATE,
            candidates=candidates,
        )


def test_attention_rejects_padding_that_names_a_row_past_the_local_table():
    # A context-parallel rank receives the producer's global-width selection, padded
    # with ``-1``. Trailing padding is trimmed, but a real index there addresses a
    # compressed row this call does not serve and must stay an error.
    q = torch.randn(1, 6, _ATTN_HEADS, _ATTN_DIM).to(torch.bfloat16)
    ori_kv = torch.randn(1, 6, _ATTN_DIM).to(torch.bfloat16)
    cmp_kv = torch.randn(1, 1, _ATTN_DIM).to(torch.bfloat16)
    sink = torch.randn(_ATTN_HEADS)
    topk = torch.full((1, 6, 3), -1, dtype=torch.int32)
    topk[..., 0] = 0
    topk[..., 2] = 2

    with pytest.raises(ValueError, match=r"selects a row past it"):
        cann_attention.wrapper(
            q,
            ori_kv,
            cmp_kv,
            sink,
            torch.arange(6).unsqueeze(0),
            sliding_window=_WINDOW,
            compress_ratio=_RATE,
            candidates=CompressedCandidates(topk_indices=topk),
        )


@pytest.mark.parametrize(
    ("index_topk", "cmp_ratio", "match"),
    [
        (16, _HCA_RATE, r"supported only when cmp_ratio=4; got cmp_ratio=128"),
        (1025, _RATE, r"index_topk=1025 exceeds sfmla K limit \(1024\)"),
    ],
    ids=["indices_off_csa", "topk_above_limit"],
)
def test_sparse_flash_mla_rejects_selections_the_operator_cannot_take(index_topk: int, cmp_ratio: int, match: str):
    q = ori_kv = cmp_kv = torch.zeros(1, 1, 1, 8, dtype=torch.bfloat16)
    indices = torch.zeros(1, 1, index_topk, dtype=torch.int32)

    with pytest.raises(ValueError, match=match):
        vendor_flash_mla.npu_sparse_flash_mla(q, ori_kv, cmp_kv, indices, cmp_ratio=cmp_ratio)
