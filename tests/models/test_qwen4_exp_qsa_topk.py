# Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Finite-mask QSA selection semantics, including the sparse CANN TopKV2 case."""

from types import SimpleNamespace

import pytest
import torch

from veomni.models.transformers.qwen4_exp import packed_utils
from veomni.utils.device import IS_NPU_AVAILABLE


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "npu"])
def test_qsa_mask_sentinel_dispatch(device_type):
    expected = -1.0 if device_type == "npu" else float("-inf")
    assert packed_utils._qsa_masked_fill_value(device_type) == expected


@pytest.mark.parametrize("mask_device", ["cpu", "npu"])
def test_compact_qsa_finite_mask_preserves_packed_selection(monkeypatch, mask_device):
    """Exercise the production selector on CPU with either mask policy."""
    torch.manual_seed(38)
    hidden = torch.randn(1, 25, 3)
    indexer = SimpleNamespace(
        index_kv_heads=1,
        index_n_heads=2,
        index_head_dim=2,
        compress_ratio=4,
        block_topk=2,
        token_budget=8,
        index_qk_proj=torch.nn.Linear(3, 6, bias=False),
        q_layernorm=torch.nn.Identity(),
        k_layernorm=torch.nn.Identity(),
    )
    positions = (torch.ones(1, 25, 2), torch.zeros(1, 25, 2))
    boundaries = torch.tensor([0, 9, 25], dtype=torch.int32)
    kwargs = dict(group=None, rank=0, world_size=1, apply_rotary_pos_emb=lambda q, **kwargs: q)
    # Three query rows per chunk also exercises a short final query chunk.
    monkeypatch.setattr(packed_utils, "_MAX_SCORE_ELEMENTS", 36)
    expected = packed_utils.compact_qsa_select(indexer, hidden, positions, boundaries, **kwargs)
    sentinel = packed_utils._qsa_masked_fill_value(mask_device)
    calls = []

    def mask_value(device_type):
        calls.append(device_type)
        return sentinel

    monkeypatch.setattr(packed_utils, "_qsa_masked_fill_value", mask_value)
    actual = packed_utils.compact_qsa_select(indexer, hidden, positions, boundaries, **kwargs)
    assert calls and set(calls) == {"cpu"}
    # Random continuous scores have no cutoff ties in this fixture. Compare
    # token sets, since TopK does not specify the order of equal masked entries.
    torch.testing.assert_close(actual.sort(dim=-1).values, expected.sort(dim=-1).values)
    for query in range(25):
        selected = actual[0, query]
        selected = selected[selected >= 0]
        start = 0 if query < 9 else 9
        assert ((selected >= start) & (selected <= query)).all()
    assert (actual[0, 0] >= 0).sum() == 1  # No complete block; only the causal tail.
    assert (actual[0, 9] >= 0).sum() == 1  # The next packed example starts fresh.


@pytest.mark.parametrize(
    "device_type",
    ["cpu", pytest.param("npu", marks=pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires Ascend"))],
)
@pytest.mark.parametrize("key_length", [511, 513, 29166])
@pytest.mark.parametrize("quantized", [False, True])
def test_qsa_sparse_topk_boundary(device_type, key_length, quantized):
    """Only launch the repaired finite-mask path on NPU, never the known fault."""
    rows, k = 143, min(512, key_length)
    scores = torch.rand((1, rows, key_length), generator=torch.Generator().manual_seed(38))
    if quantized:
        scores = (scores * 4).floor()  # Exact zeros and ties at the selection cutoff.
    visible = torch.zeros_like(scores, dtype=torch.bool)
    for row in range(rows - 6):
        count = min((1951 + row) // 4, key_length)
        visible[0, row, key_length - count :] = True
    visible[0, -3:, -1] = True
    # The preceding three rows are all masked; the last three have one block.
    reference = scores.masked_fill(~visible, float("-inf")).sort(dim=-1, descending=True).values[..., :k]
    device_scores, device_visible = scores.to(device_type), visible.to(device_type)
    masked = device_scores.masked_fill(~device_visible, packed_utils._qsa_masked_fill_value(device_type))
    picked = masked.topk(k, dim=-1).indices.cpu()
    valid = visible.gather(-1, picked)
    values = scores.gather(-1, picked).masked_fill(~valid, float("-inf"))
    torch.testing.assert_close(values.sort(dim=-1, descending=True).values, reference, rtol=0, atol=0)
    assert torch.equal(valid.sum(dim=-1), visible.sum(dim=-1).clamp(max=k))
    assert not valid[0, -6:-3].any()
    # Equal-score picks may differ from CPU; require exactly the reference
    # score multiset rather than imposing an undocumented TopK tie order.
