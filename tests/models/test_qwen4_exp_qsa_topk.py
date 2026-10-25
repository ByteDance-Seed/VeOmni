# Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Finite-mask QSA selection semantics for Qwen4-Exp NPU vs GPU generated modeling."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from veomni.utils.device import IS_NPU_AVAILABLE


def _load_qwen4_exp_modeling(backend: str):
    return importlib.import_module(
        f"veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_{backend}"
    )


def test_compact_qsa_npu_finite_mask_matches_gpu_packed_selection(monkeypatch):
    """Verify NPU QSA indexer masks invisible blocks with -1.0 (not -inf) and matches GPU selection."""
    gpu_modeling = _load_qwen4_exp_modeling("gpu")
    npu_modeling = _load_qwen4_exp_modeling("npu")

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
        veomni_rope=lambda q, k, cos, sin, **kwargs: (q, k),
    )
    positions = (torch.ones(1, 25, 2), torch.zeros(1, 25, 2))
    boundaries = torch.tensor([0, 9, 25], dtype=torch.int32)

    expected = gpu_modeling.Qwen4ExpTextQSAIndexer.forward(
        indexer,
        hidden,
        positions,
        attention_mask=None,
        past_key_values=None,
        cu_seq_lens_q=boundaries,
    )

    orig_topk = torch.Tensor.topk
    topk_inputs: list[torch.Tensor] = []

    def recording_topk(self: torch.Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True):
        topk_inputs.append(self.detach().clone())
        return orig_topk(self, k, dim=dim, largest=largest, sorted=sorted)

    monkeypatch.setattr(torch.Tensor, "topk", recording_topk)
    actual = npu_modeling.Qwen4ExpTextQSAIndexer.forward(
        indexer,
        hidden,
        positions,
        attention_mask=None,
        past_key_values=None,
        cu_seq_lens_q=boundaries,
    )

    assert topk_inputs, "Expected Qwen4ExpTextQSAIndexer.forward to invoke Tensor.topk"
    for scores in topk_inputs:
        assert torch.isfinite(scores).all(), "NPU QSA indexer must not pass -inf to topk"
        assert (scores == -1.0).any(), "Expected invisible blocks to be masked with -1.0"
        assert (scores[scores != -1.0] >= 0.0).all(), "Unmasked ReLU block scores must be non-negative"

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
def test_qsa_sparse_topk_boundary(device_type: str, key_length: int, quantized: bool):
    """Verify -1.0 finite masking matches -inf selection across TopKV2 boundary sizes."""
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
    masked = device_scores.masked_fill(~device_visible, -1.0)
    picked = masked.topk(k, dim=-1).indices.cpu()
    valid = visible.gather(-1, picked)
    values = scores.gather(-1, picked).masked_fill(~valid, float("-inf"))
    torch.testing.assert_close(values.sort(dim=-1, descending=True).values, reference, rtol=0, atol=0)
    assert torch.equal(valid.sum(dim=-1), visible.sum(dim=-1).clamp(max=k))
    assert not valid[0, -6:-3].any()
