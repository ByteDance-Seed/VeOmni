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

"""``qsa_indexer`` ``cann`` selection against the ``eager`` reference.

CANN call layout is checked on any host with a fake ``cann_ops_transformer``.
On Ascend, the kernel's scores are compared with the eager
``relu(q . k).sum(heads) / sqrt(head_dim)`` scores — that comparison is the
gate for the kernel's head-reduction semantics: if the kernel does not apply
ReLU before summing, the sorted-score comparison below fails loudly instead
of silently rerouting blocks.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest
import torch
from torch import Tensor

from veomni.ops import resolve_op
from veomni.ops.kernels.qsa_indexer import cann as cann_indexer
from veomni.ops.kernels.qsa_indexer import eager as eager_indexer
from veomni.utils.device import IS_NPU_AVAILABLE, get_device_type


_HEADS, _DIM, _RATE = 4, 128, 4


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
        batch, seq_len = q.shape[0], q.shape[1]
        shape = (batch, seq_len, 1, top_k)
        indices = torch.arange(top_k, dtype=torch.int32).expand(shape).clone()
        indices[..., -1] = -1
        values = torch.ones(shape, dtype=torch.float32)
        return indices, values


@pytest.fixture
def fake_ops(monkeypatch: pytest.MonkeyPatch) -> _FakeOps:
    ops = _FakeOps()
    monkeypatch.setattr(cann_indexer, "_ops", lambda: ops)
    cann_indexer._plan.cache_clear()
    cann_indexer._head_scale.cache_clear()
    yield ops
    cann_indexer._plan.cache_clear()
    cann_indexer._head_scale.cache_clear()


def _bshd_inputs(batch: int, seq_len: int, compressed_len: int) -> tuple[Tensor, Tensor]:
    q = torch.randn(batch, seq_len, _HEADS, _DIM).to(torch.bfloat16)
    k = torch.randn(batch, compressed_len, _DIM).to(torch.bfloat16)
    return q, k


def test_call_layout_matches_the_lightning_abi(fake_ops: _FakeOps):
    q, k = _bshd_inputs(batch=2, seq_len=9, compressed_len=2)

    indices, scores = cann_indexer.select_blocks(q, k, _RATE, 2)

    (cann_q, cann_k, cann_weights), kwargs = fake_ops.indexer_calls[0]
    assert cann_q.shape == (2, 9, _HEADS, _DIM) and cann_q.dtype == torch.bfloat16 and cann_q.is_contiguous()
    assert cann_k.shape == (2, 2, 1, _DIM) and cann_k.dtype == torch.bfloat16 and cann_k.is_contiguous()
    assert cann_weights.shape == (2, 9, _HEADS) and cann_weights.dtype == torch.float32
    torch.testing.assert_close(cann_weights, torch.full_like(cann_weights, _DIM**-0.5))
    assert (kwargs["layout_q"], kwargs["layout_k"], kwargs["top_k"]) == ("BSND", "BSND", 2)
    assert kwargs["cmp_ratio"] == _RATE
    assert kwargs["mask_mode"] == 3
    assert kwargs["return_value"] == 1
    assert kwargs["cmp_residual_k"].tolist() == [9 % _RATE, 9 % _RATE]
    (metadata,) = fake_ops.metadata_calls
    assert (metadata["batch_size"], metadata["max_seqlen_q"], metadata["max_seqlen_k"]) == (2, 9, 2)
    assert metadata["mask_mode"] == 3 and metadata["cmp_ratio"] == _RATE
    assert indices.shape == scores.shape == (2, 9, 2)
    assert indices.dtype == torch.int32 and scores.dtype == torch.float32
    assert (indices[..., 0] == 0).all() and (indices[..., 1] == -1).all()
    assert (scores[..., 0] == 1).all() and torch.isneginf(scores[..., 1]).all()


def test_plan_is_built_once_per_shape(fake_ops: _FakeOps):
    q, k = _bshd_inputs(batch=1, seq_len=8, compressed_len=2)
    for _ in range(3):
        indices = cann_indexer.wrapper(q, k, _RATE, 2)
        assert indices.shape == (1, 8, 2) and indices.dtype == torch.int32

    assert len(fake_ops.metadata_calls) == 1
    assert len(fake_ops.indexer_calls) == 3


def test_topk_is_capped_at_compressed_blocks_and_pads_the_contract(fake_ops: _FakeOps):
    q, k = _bshd_inputs(batch=2, seq_len=8, compressed_len=2)

    indices, scores = cann_indexer.select_blocks(q, k, _RATE, 6)

    assert fake_ops.indexer_calls[0][1]["top_k"] == 2
    assert indices.shape == scores.shape == (2, 8, 6)
    assert (indices[..., 2:] == -1).all() and torch.isneginf(scores[..., 2:]).all()


def test_selection_carries_no_gradient(fake_ops: _FakeOps):
    q, k = _bshd_inputs(batch=1, seq_len=8, compressed_len=2)
    q.requires_grad_(True)
    k.requires_grad_(True)

    indices = cann_indexer.wrapper(q, k, _RATE, 2)

    assert not indices.requires_grad
    assert indices.grad_fn is None


@pytest.mark.parametrize(
    ("q", "k", "match"),
    [
        (torch.zeros(2, 8), torch.zeros(2, 2, _DIM), r"takes \[B,S,H,D\] queries"),
        (torch.zeros(2, 8, _HEADS, _DIM), torch.zeros(2, 2, _DIM, 1), r"takes \[B,S,H,D\] queries"),
        (torch.zeros(2, 9, _HEADS, _DIM), torch.zeros(2, 3, _DIM), "seq_len // compress_ratio"),
        (torch.zeros(3, 8, _HEADS, _DIM), torch.zeros(2, 2, _DIM), "batch and head dimensions must match"),
    ],
)
def test_rejects_shapes_the_kernel_cannot_take(fake_ops: _FakeOps, q: Tensor, k: Tensor, match: str):
    with pytest.raises(ValueError, match=match):
        cann_indexer.wrapper(q, k, _RATE, 2)
    assert fake_ops.indexer_calls == []


def test_rejects_nonpositive_topk(fake_ops: _FakeOps):
    q, k = _bshd_inputs(batch=1, seq_len=8, compressed_len=2)
    with pytest.raises(ValueError, match="top_k >= 1"):
        cann_indexer.wrapper(q, k, _RATE, 0)
    assert fake_ops.indexer_calls == []


def test_eager_reference_selects_fully_covered_blocks_only():
    torch.manual_seed(7)
    q, k = _bshd_inputs(batch=2, seq_len=13, compressed_len=3)

    indices, scores = eager_indexer.select_blocks(q, k, _RATE, 2)

    cap = ((torch.arange(13) + 1) // _RATE).clamp(max=3)
    assert (indices <= cap[None, :, None]).all()  # -1 padding stays below every cap
    # Rows 0..2 have no fully covered block; row 3 is covered by block 0 alone.
    assert (indices[:3] == -1).all()
    assert (indices[3:4, :, 0] >= 0).all()
    assert torch.isneginf(scores[indices < 0]).all()


def _assert_same_selection(actual: tuple[Tensor, Tensor], expected: tuple[Tensor, Tensor]) -> None:
    """Compare per-row selections independent of the order each backend lists them in."""
    actual_indices, actual_scores = (tensor.cpu() for tensor in actual)
    expected_indices, expected_scores = (tensor.cpu() for tensor in expected)
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
@pytest.mark.parametrize(
    ("batch", "heads", "head_dim", "compress_ratio", "seq_len", "topk"),
    [
        (1, 4, 64, 4, 64, 8),
        (2, 8, 128, 4, 128, 16),
        (1, 8, 128, 1, 96, 24),
        (2, 4, 64, 32, 256, 4),
        (1, 8, 128, 4, 1002, 48),
    ],
    ids=["small", "batched", "every-token-blocks", "wide-blocks", "prefill-scale"],
)
def test_cann_matches_eager(batch: int, heads: int, head_dim: int, compress_ratio: int, seq_len: int, topk: int):
    torch.manual_seed(3)
    device = get_device_type()
    compressed_len = seq_len // compress_ratio
    q = torch.randn(batch, seq_len, heads, head_dim, device=device).to(torch.bfloat16)
    k = torch.randn(batch, compressed_len, head_dim, device=device).to(torch.bfloat16)

    expected = resolve_op("qsa_indexer", "standard", "eager").wrapper(q, k, compress_ratio, topk)
    actual = resolve_op("qsa_indexer", "standard", "cann").wrapper(q, k, compress_ratio, topk)
    assert actual.shape == expected.shape

    # The scores are the ReLU head-reduction gate: sorted per-row score lists
    # must match the eager relu(q.k).sum/sqrt(D) scores within bf16 tolerance.
    _, actual_scores = cann_indexer.select_blocks(q, k, compress_ratio, topk)
    _, expected_scores = eager_indexer.select_blocks(q, k, compress_ratio, topk)
    _assert_same_selection((actual, actual_scores), (expected, expected_scores))


@pytest.mark.skipif(
    not IS_NPU_AVAILABLE or importlib.util.find_spec("cann_ops_transformer") is None,
    reason="CANN Lightning Indexer requires Ascend NPU and cann_ops_transformer",
)
def test_cann_indices_are_visible_blocks_of_the_row():
    torch.manual_seed(11)
    device = get_device_type()
    batch, seq_len, compress_ratio, topk = 2, 128, 4, 16
    compressed_len = seq_len // compress_ratio
    q = torch.randn(batch, seq_len, _HEADS, _DIM, device=device).to(torch.bfloat16)
    k = torch.randn(batch, compressed_len, _DIM, device=device).to(torch.bfloat16)

    indices = resolve_op("qsa_indexer", "standard", "cann").wrapper(q, k, compress_ratio, topk)
    # Every valid id is a fully covered block: below (row + 1) // compress_ratio,
    # capped at the element's own block count.
    cap = ((torch.arange(seq_len, device=indices.device) + 1) // compress_ratio).clamp(max=compressed_len)
    assert (indices <= cap[None, :, None]).all()
    # Well-separated random inputs make near-tie topk flips unlikely, so the
    # valid selections should be exactly the eager set per row: equal valid
    # counts, then equal descending-sorted rows (valid ids first, -1 pads last).
    expected = resolve_op("qsa_indexer", "standard", "eager").wrapper(q, k, compress_ratio, topk)
    torch.testing.assert_close((indices >= 0).sum(-1), (expected >= 0).sum(-1))
    assert torch.equal(
        torch.sort(indices, dim=-1, descending=True).values,
        torch.sort(expected, dim=-1, descending=True).values,
    )
