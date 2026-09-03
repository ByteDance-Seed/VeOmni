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

"""SDPA attention adapter contract and numerical checks vs MATH SDPA."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.kernels.attention_cases import clone_qkv, dense_mask, math_sdpa_reference
from tests.kernels.tol import ATTN_ATOL, ATTN_GRAD_ATOL, ATTN_GRAD_RTOL, ATTN_RTOL, EAGER_ATOL, EAGER_RTOL
from veomni.kernels._kernels.attention.standard import sdpa as sdpa_backend
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


class _FakeAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="veomni_sdpa")
        self.is_causal = True
        self.num_key_value_groups = 1


def test_sdpa_attention_forward_square_causal_layout():
    module = _FakeAttentionModule()
    query = torch.randn(1, 2, 4, 8)
    output, lse = sdpa_backend.sdpa_attention_forward(module, query, query, query, attention_mask=None, dropout=0.0)
    assert output.shape == (1, 4, 2, 8)
    assert lse is None
    assert torch.isfinite(output).all()


def test_sdpa_attention_forward_accepts_dense_mask():
    module = _FakeAttentionModule()
    query = torch.randn(1, 2, 4, 8)
    attention_mask = torch.tril(torch.ones(1, 1, 4, 4, dtype=torch.bool))
    output, lse = sdpa_backend.sdpa_attention_forward(
        module, query, query, query, attention_mask=attention_mask, dropout=0.0
    )
    assert output.shape == (1, 4, 2, 8)
    assert lse is None
    assert torch.isfinite(output).all()


def test_sdpa_attention_rejects_zero_dimensions():
    module = _FakeAttentionModule()
    query = torch.randn(1, 0, 4, 8)
    with pytest.raises(ValueError, match="zero dimensions"):
        sdpa_backend.sdpa_attention_forward(module, query, query, query, attention_mask=None)


def test_sdpa_attention_delegates_active_ulysses_to_shared_helpers(monkeypatch):
    group = object()
    state = SimpleNamespace(ulysses_group=group, ulysses_size=2)
    calls = []

    def fake_prepare(query, key, value, *, group, ulysses_size):
        calls.append(("prepare", query, key, value, group, ulysses_size))
        return query[:, :, :2], key[:, :, :1], value[:, :, :1], 4

    def fake_slice(auxiliary, *, query_head_count, local_query_head_count, group):
        calls.append(("slice", auxiliary, query_head_count, local_query_head_count, group))
        return auxiliary[:local_query_head_count]

    def fake_backend(module, query, key, value, attention_mask, **kwargs):
        calls.append(("backend", query, key, value, attention_mask, kwargs))
        return query.transpose(1, 2), None

    def fake_restore(output, *, group):
        calls.append(("restore", output, group))
        return output

    monkeypatch.setattr(sdpa_backend, "get_parallel_state", lambda: state)
    monkeypatch.setattr(sdpa_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: not skip_ulysses)
    monkeypatch.setattr(sdpa_backend, "prepare_ulysses_qkv", fake_prepare)
    monkeypatch.setattr(sdpa_backend, "slice_ulysses_head_auxiliary", fake_slice)
    monkeypatch.setattr(sdpa_backend, "hf_sdpa_attention_forward", fake_backend)
    monkeypatch.setattr(sdpa_backend, "restore_ulysses_output", fake_restore)
    query = torch.randn(1, 4, 8, 8)
    auxiliary = torch.arange(4)

    output, _ = sdpa_backend.sdpa_attention_forward(
        _FakeAttentionModule(),
        query,
        query[:, :2],
        query[:, :2],
        attention_mask=None,
        s_aux=auxiliary,
    )

    assert [call[0] for call in calls] == ["prepare", "slice", "backend", "restore"]
    assert calls[0][1].shape == (1, 8, 4, 8)
    assert calls[0][4:] == (group, 2)
    torch.testing.assert_close(calls[2][-1]["s_aux"], auxiliary[:2])
    assert output.shape == (1, 8, 2, 8)


def test_sdpa_attention_skip_ulysses_skips_exchange(monkeypatch):
    monkeypatch.setattr(sdpa_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: not skip_ulysses)
    monkeypatch.setattr(
        sdpa_backend,
        "prepare_ulysses_qkv",
        lambda *args, **kwargs: pytest.fail("skip_ulysses must not exchange QKV"),
    )
    captured = {}

    def fake_backend(module, query, key, value, attention_mask, **kwargs):
        captured["kwargs"] = kwargs
        return query.transpose(1, 2), None

    monkeypatch.setattr(sdpa_backend, "hf_sdpa_attention_forward", fake_backend)
    query = torch.randn(1, 4, 8, 8)
    sdpa_backend.sdpa_attention_forward(
        _FakeAttentionModule(),
        query,
        query[:, :2],
        query[:, :2],
        attention_mask=None,
        skip_ulysses=True,
    )
    assert "skip_ulysses" not in captured["kwargs"]


@pytest.mark.parametrize("mask_case", ("causal", "full", "bagel_mixed"))
def test_sdpa_attention_matches_math_reference(mask_case):
    sequence_length = 32
    query_heads, kv_heads, head_dim = 4, 2, 16
    generator = torch.Generator().manual_seed(11)
    query = torch.randn(1, query_heads, sequence_length, head_dim, generator=generator)
    key = torch.randn(1, kv_heads, sequence_length, head_dim, generator=generator)
    value = torch.randn(1, kv_heads, sequence_length, head_dim, generator=generator)
    output_gradient = torch.randn(1, sequence_length, query_heads, head_dim, generator=generator)
    scaling = head_dim**-0.5
    dense = dense_mask(mask_case, sequence_length, "cpu")

    reference_qkv = clone_qkv(query, key, value)
    reference_output, _ = math_sdpa_reference(*reference_qkv, dense, scaling=scaling)
    reference_gradients = torch.autograd.grad(reference_output, reference_qkv, output_gradient)

    module = _FakeAttentionModule()
    module.num_key_value_groups = query_heads // kv_heads
    sdpa_qkv = clone_qkv(query, key, value)
    sdpa_output, _ = sdpa_backend.sdpa_attention_forward(
        module,
        *sdpa_qkv,
        attention_mask=dense,
        dropout=0.0,
        scaling=scaling,
    )
    sdpa_gradients = torch.autograd.grad(sdpa_output, sdpa_qkv, output_gradient)

    torch.testing.assert_close(sdpa_output, reference_output, rtol=EAGER_RTOL, atol=EAGER_ATOL)
    for name, gradient, reference_gradient in zip(
        ("query", "key", "value"),
        sdpa_gradients,
        reference_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            gradient,
            reference_gradient,
            rtol=EAGER_RTOL,
            atol=EAGER_ATOL,
            msg=lambda message, tensor_name=name: f"{tensor_name}: {message}",
        )


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA SDPA numerical comparison")
@pytest.mark.parametrize("mask_case", ("causal", "bagel_mixed"))
def test_sdpa_attention_matches_math_reference_on_cuda(mask_case):
    device = torch.device(get_device_type())
    dtype = torch.bfloat16
    sequence_length = 64
    query_heads, kv_heads, head_dim = 4, 2, 32
    generator = torch.Generator(device=device).manual_seed(12)
    query = torch.randn(1, query_heads, sequence_length, head_dim, device=device, dtype=dtype, generator=generator)
    key = torch.randn(1, kv_heads, sequence_length, head_dim, device=device, dtype=dtype, generator=generator)
    value = torch.randn(1, kv_heads, sequence_length, head_dim, device=device, dtype=dtype, generator=generator)
    output_gradient = torch.randn(
        1, sequence_length, query_heads, head_dim, device=device, dtype=dtype, generator=generator
    )
    scaling = head_dim**-0.5
    dense = dense_mask(mask_case, sequence_length, device)

    reference_qkv = clone_qkv(query, key, value)
    reference_output, _ = math_sdpa_reference(*reference_qkv, dense, scaling=scaling)
    reference_gradients = torch.autograd.grad(reference_output, reference_qkv, output_gradient)

    module = _FakeAttentionModule()
    module.num_key_value_groups = query_heads // kv_heads
    sdpa_qkv = clone_qkv(query, key, value)
    sdpa_output, _ = sdpa_backend.sdpa_attention_forward(
        module,
        *sdpa_qkv,
        attention_mask=dense,
        dropout=0.0,
        scaling=scaling,
    )
    sdpa_gradients = torch.autograd.grad(sdpa_output, sdpa_qkv, output_gradient)

    torch.testing.assert_close(sdpa_output, reference_output, rtol=ATTN_RTOL, atol=ATTN_ATOL)
    for name, gradient, reference_gradient in zip(
        ("query", "key", "value"),
        sdpa_gradients,
        reference_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            gradient,
            reference_gradient,
            rtol=ATTN_GRAD_RTOL,
            atol=ATTN_GRAD_ATOL,
            msg=lambda message, tensor_name=name: f"{tensor_name}: {message}",
        )
