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
# See the License for the specific language governing limitations
# under the License.

"""SageAttention adapter contract and numerical checks vs MATH SDPA."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.kernels.attention_cases import dense_mask, math_sdpa_reference
from tests.kernels.tol import ATTN_ATOL, ATTN_RTOL
from veomni.kernels._kernels.attention.standard import sage as sage_backend
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


_SAGE_AVAILABLE = importlib.util.find_spec("sageattention") is not None


class _FakeAttentionModule(nn.Module):
    def __init__(self, *, is_causal: bool = False):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="veomni_sage_attention")
        self.is_causal = is_causal


def test_sage_attention_hnd_layout_and_is_causal(monkeypatch):
    captured = {}

    def fake_sage(query, key, value, tensor_layout="HND", is_causal=False, sm_scale=None):
        captured.update(query=query, key=key, value=value, is_causal=is_causal, sm_scale=sm_scale)
        return query + 1

    monkeypatch.setattr(sage_backend, "sageattn", fake_sage)
    monkeypatch.setattr(sage_backend, "should_apply_ulysses", lambda: False)
    query = torch.randn(2, 4, 5, 8)
    key = torch.randn(2, 2, 5, 8)
    value = torch.randn(2, 2, 5, 8)

    output, weights = sage_backend.sage_attention_forward(
        _FakeAttentionModule(is_causal=True),
        query,
        key,
        value,
        attention_mask=None,
        scaling=0.125,
        skip_ulysses=True,
        last_loss=0.1,
    )

    torch.testing.assert_close(captured["query"], query)
    torch.testing.assert_close(captured["key"], key)
    torch.testing.assert_close(captured["value"], value)
    assert captured["is_causal"] is True
    assert captured["sm_scale"] == 0.125
    torch.testing.assert_close(output, (query + 1).transpose(1, 2))
    assert output.shape == (2, 5, 4, 8)
    assert weights is None


def test_sage_attention_skip_ulysses_skips_exchange(monkeypatch):
    monkeypatch.setattr(sage_backend, "should_apply_ulysses", lambda: True)
    monkeypatch.setattr(
        sage_backend,
        "prepare_ulysses_qkv",
        lambda *args, **kwargs: pytest.fail("skip_ulysses must not exchange QKV"),
    )
    monkeypatch.setattr(sage_backend, "sageattn", lambda query, key, value, **kwargs: query)
    query = torch.randn(1, 4, 8, 8)
    sage_backend.sage_attention_forward(
        _FakeAttentionModule(),
        query,
        query[:, :2],
        query[:, :2],
        attention_mask=None,
        skip_ulysses=True,
    )


def test_sage_attention_rejects_dense_mask(monkeypatch):
    monkeypatch.setattr(sage_backend, "sageattn", lambda query, key, value, **kwargs: query)
    monkeypatch.setattr(sage_backend, "should_apply_ulysses", lambda: False)
    query = torch.randn(1, 2, 4, 8)
    with pytest.raises(ValueError, match="does not take a dense attention_mask"):
        sage_backend.sage_attention_forward(
            _FakeAttentionModule(),
            query,
            query,
            query,
            attention_mask=torch.ones(1, 1, 4, 4, dtype=torch.bool),
        )


def test_sage_attention_requires_package(monkeypatch):
    monkeypatch.setattr(sage_backend, "sageattn", None)
    monkeypatch.setattr(sage_backend, "should_apply_ulysses", lambda: False)
    query = torch.randn(1, 2, 4, 8)
    with pytest.raises(ImportError, match="sageattention"):
        sage_backend.sage_attention_forward(
            _FakeAttentionModule(),
            query,
            query,
            query,
            attention_mask=None,
        )


def test_sage_attention_rejects_training_graph(monkeypatch):
    monkeypatch.setattr(sage_backend, "sageattn", lambda query, key, value, **kwargs: pytest.fail("sageattn"))
    monkeypatch.setattr(sage_backend, "should_apply_ulysses", lambda: False)
    query = torch.randn(1, 2, 4, 8, requires_grad=True)
    with pytest.raises(RuntimeError, match="inference-only"):
        sage_backend.sage_attention_forward(
            _FakeAttentionModule(),
            query,
            query.detach(),
            query.detach(),
            attention_mask=None,
        )


def test_sage_attention_allows_no_grad_even_if_inputs_require_grad(monkeypatch):
    captured = {}

    def fake_sage(query, key, value, **kwargs):
        captured["called"] = True
        return query

    monkeypatch.setattr(sage_backend, "sageattn", fake_sage)
    monkeypatch.setattr(sage_backend, "should_apply_ulysses", lambda: False)
    query = torch.randn(1, 2, 4, 8, requires_grad=True)
    with torch.no_grad():
        sage_backend.sage_attention_forward(
            _FakeAttentionModule(),
            query,
            query,
            query,
            attention_mask=None,
        )
    assert captured["called"] is True


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="SageAttention numerical comparison requires CUDA")
@pytest.mark.skipif(not _SAGE_AVAILABLE, reason="sageattention is not installed")
@pytest.mark.parametrize("mask_case", ("causal", "full"))
def test_sage_attention_matches_math_sdpa(mask_case):
    device = torch.device(get_device_type())
    dtype = torch.bfloat16
    sequence_length = 128
    query_heads, kv_heads, head_dim = 4, 2, 64
    generator = torch.Generator(device=device).manual_seed(17)
    query = torch.randn(1, query_heads, sequence_length, head_dim, device=device, dtype=dtype, generator=generator)
    key = torch.randn(1, kv_heads, sequence_length, head_dim, device=device, dtype=dtype, generator=generator)
    value = torch.randn(1, kv_heads, sequence_length, head_dim, device=device, dtype=dtype, generator=generator)
    scaling = head_dim**-0.5
    dense = dense_mask(mask_case, sequence_length, device)
    reference_output, _ = math_sdpa_reference(query, key, value, dense, scaling=scaling)
    sage_output, _ = sage_backend.sage_attention_forward(
        _FakeAttentionModule(is_causal=mask_case == "causal"),
        query,
        key,
        value,
        attention_mask=None,
        scaling=scaling,
        is_causal=mask_case == "causal",
    )

    # SageAttention is inference-only; compare the no-grad forward only.
    assert sage_output.grad_fn is None
    torch.testing.assert_close(sage_output, reference_output, rtol=ATTN_RTOL, atol=ATTN_ATOL)
