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

"""``deepseek_v4`` ``cann`` indexer and sparse attention: CANN call layout on any host, eager parity on Ascend."""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest
import torch
from torch import Tensor

from tests.ops.tol import ATTN_ATOL, ATTN_RTOL
from tests.ops.utils import assert_gradient_direction_and_scale, make_grad_leaves
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
    yield ops
    cann_indexer._causal_plan.cache_clear()
    cann_indexer._packed_plan.cache_clear()


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


_ATTN_HEADS, _ATTN_DIM, _WINDOW, _HCA_RATE = 2, 512, 128, 128
_CANN_AVAILABLE = IS_NPU_AVAILABLE and importlib.util.find_spec("cann_ops_transformer") is not None


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


def test_attention_packed_call_rebases_the_selection_per_segment(flash_mla: _FlashMlaRecorder):
    slices = ((0, 9), (9, 13))
    q, ori_kv, cmp_kv, sink = _attention_inputs(seq_len=13, compressed_len=3)
    topk = torch.full((1, 13, 2), -1, dtype=torch.int32)
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
    assert (indices[..., 1] == -1).all()
    assert kwargs["packed_sequence_slices"] == slices
    assert (kwargs["cmp_ratio"], kwargs["softmax_scale"]) == (_RATE, 0.5)
    assert (kwargs["ori_mask_mode"], kwargs["cmp_mask_mode"]) == (4, 3)
    assert (kwargs["ori_win_left"], kwargs["ori_win_right"]) == (_WINDOW - 1, 0)
    assert kwargs["sinks"].dtype == torch.float32


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
        ({"query_offset": 4}, "one KV row per query row"),
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


def test_attention_ignores_the_mask_and_leaves_kernel_limits_to_the_kernel(flash_mla: _FlashMlaRecorder):
    q, ori_kv, cmp_kv, sink = _attention_inputs(seq_len=8, compressed_len=1, dtype=torch.float32)

    cann_attention.wrapper(
        q,
        ori_kv,
        cmp_kv,
        sink,
        torch.arange(8).unsqueeze(0),
        sliding_window=64,
        compress_ratio=8,
        attention_mask=torch.ones(1, 1, 8, 8, dtype=torch.bool),
    )

    ((cann_q, _, _, _), kwargs) = flash_mla.calls[0]
    assert cann_q.dtype == torch.float32
    assert (kwargs["ori_win_left"], kwargs["cmp_ratio"]) == (63, 8)


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


def _visible_candidates(segment_lengths: tuple[int, ...], rate: int, topk: int, device: str) -> CompressedCandidates:
    """Each query's newest *topk* visible compressed rows, or the whole interval for HCA."""
    bounds = torch.tensor((0, *segment_lengths)).cumsum(0).tolist()
    slices = tuple(zip(bounds[:-1], bounds[1:]))
    metadata = build_packed_compression_metadata(
        torch.empty((), device=device), _segment_positions(segment_lengths, device), slices, (rate,)
    )
    range_starts, range_ends = packed_compressed_causal_ranges(metadata[rate])
    if rate == _HCA_RATE:
        return CompressedCandidates(range_starts=range_starts, range_ends=range_ends)
    slots = range_ends[:, None] - 1 - torch.arange(topk, device=device, dtype=torch.int32)
    indices = torch.where(slots >= range_starts[:, None], slots, -1)
    return CompressedCandidates(topk_indices=indices.unsqueeze(0).to(torch.int32))


@pytest.mark.skipif(not _CANN_AVAILABLE, reason="CANN sparse FlashMLA requires Ascend NPU and cann_ops_transformer")
@pytest.mark.parametrize("rate", [_RATE, _HCA_RATE], ids=["csa", "hca"])
@pytest.mark.parametrize("segment_lengths", [(300,), (200, 260)], ids=["causal", "packed"])
def test_attention_cann_matches_eager(rate: int, segment_lengths: tuple[int, ...]):
    torch.manual_seed(5)
    device = get_device_type()
    seq_len = sum(segment_lengths)
    compressed_len = sum(length // rate for length in segment_lengths)
    q, ori_kv, cmp_kv, sink = _attention_inputs(seq_len, compressed_len, device=device)
    bounds = torch.tensor((0, *segment_lengths)).cumsum(0).tolist()
    kwargs: dict[str, Any] = {
        "sliding_window": _WINDOW,
        "compress_ratio": rate,
        "candidates": _visible_candidates(segment_lengths, rate, topk=16, device=device),
        "packed_sequence_slices": tuple(zip(bounds[:-1], bounds[1:])),
    }
    position_ids = _segment_positions(segment_lengths, device)

    leaves_c = make_grad_leaves(q, ori_kv, cmp_kv)
    leaves_e = make_grad_leaves(q, ori_kv, cmp_kv)
    actual = resolve_op("dsa_attention", "deepseek_v4", "cann").wrapper(
        *leaves_c, sink.float(), position_ids, **kwargs
    )
    expected = resolve_op("dsa_attention", "deepseek_v4", "eager").wrapper(
        *leaves_e, sink.float(), position_ids, **kwargs
    )
    torch.testing.assert_close(actual, expected, atol=ATTN_ATOL, rtol=ATTN_RTOL)

    grad = torch.randn_like(actual)
    cann_grads = torch.autograd.grad((actual * grad).sum(), leaves_c)
    eager_grads = torch.autograd.grad((expected * grad).sum(), leaves_e)
    for cann_grad, eager_grad in zip(cann_grads, eager_grads, strict=True):
        assert_gradient_direction_and_scale(cann_grad, eager_grad, min_cosine=0.98, norm_rtol=0.1)
