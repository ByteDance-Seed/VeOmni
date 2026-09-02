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

"""Flex attention adapter contract and numerical checks vs MATH SDPA."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask

from tests.kernels.attention_cases import clone_qkv, dense_mask, flex_mask, math_sdpa_reference
from tests.kernels.tol import ATTN_ATOL, ATTN_BF16_GRAD_ATOL, ATTN_GRAD_ATOL, ATTN_GRAD_RTOL, ATTN_RTOL
from veomni.kernels._kernels.attention.standard import flex as flex_backend
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


class _FakeAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="veomni_flex_attention")


class _ToyAttentionLayer(nn.Module):
    def __init__(self, hidden_size: int, query_heads: int, kv_heads: int, head_dim: int):
        super().__init__()
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, query_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(query_heads * head_dim, hidden_size, bias=False)

    def qkv(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, _ = hidden_states.shape
        query = (
            self.q_proj(hidden_states)
            .view(batch_size, sequence_length, self.query_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.k_proj(hidden_states).view(batch_size, sequence_length, self.kv_heads, self.head_dim).transpose(1, 2)
        )
        value = (
            self.v_proj(hidden_states).view(batch_size, sequence_length, self.kv_heads, self.head_dim).transpose(1, 2)
        )
        return query, key, value


def _causal_block_mask(sequence_length: int, device: torch.device):
    return create_block_mask(
        lambda batch_idx, head_idx, query_idx, key_idx: query_idx >= key_idx,
        B=None,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=device,
        BLOCK_SIZE=128,
    )


def test_flex_attention_cpu_forward_uses_block_mask_and_hf_layout():
    sequence_length = 17
    query = torch.randn(2, 4, sequence_length, 8)
    key = torch.randn(2, 2, sequence_length, 8)
    value = torch.randn(2, 2, sequence_length, 8)
    output, auxiliary = flex_backend.flex_attention_forward(
        _FakeAttentionModule(),
        query,
        key,
        value,
        _causal_block_mask(sequence_length, query.device),
        scaling=0.25,
    )
    assert output.shape == (2, sequence_length, 4, 8)
    assert output.dtype == torch.float32
    assert auxiliary is None
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="FlexAttention backward requires CUDA")
def test_flex_attention_short_query_backward_is_finite():
    sequence_length = 65
    head_dim = 16
    device = torch.device(get_device_type())
    query = torch.randn(1, 2, sequence_length, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(1, 1, sequence_length, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn(1, 1, sequence_length, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    output, auxiliary = flex_backend.flex_attention_forward(
        _FakeAttentionModule(),
        query,
        key,
        value,
        _causal_block_mask(sequence_length, device),
    )
    output.float().square().mean().backward()
    assert output.shape == (1, sequence_length, 2, head_dim)
    assert auxiliary is not None
    assert torch.isfinite(output).all()
    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize(
    ("query_heads", "kv_heads", "expected_message"),
    [
        (3, 2, "GQA requires query heads"),
        (4, 0, "does not support query/key/value tensors with zero dimensions"),
    ],
)
def test_flex_attention_rejects_invalid_gqa(query_heads, kv_heads, expected_message):
    query = torch.randn(1, query_heads, 8, 8)
    key = torch.randn(1, kv_heads, 8, 8)
    value = torch.randn(1, kv_heads, 8, 8)
    with pytest.raises(ValueError, match=expected_message):
        flex_backend.flex_attention_forward(
            _FakeAttentionModule(),
            query,
            key,
            value,
            _causal_block_mask(8, query.device),
        )


def test_flex_attention_rejects_unsupported_masks(monkeypatch):
    query = torch.randn(1, 4, 8, 8)
    module = _FakeAttentionModule()
    for unsupported_mask in (None, torch.ones(1, 1, 8, 8, dtype=torch.bool)):
        with pytest.raises(TypeError, match="requires a BlockMask"):
            flex_backend.flex_attention_forward(module, query, query, query, unsupported_mask, sliding_window=4)

    head_specific_mask = create_block_mask(
        lambda batch_idx, head_idx, query_idx, key_idx: query_idx >= key_idx,
        B=None,
        H=query.shape[1],
        Q_LEN=query.shape[2],
        KV_LEN=query.shape[2],
        device=query.device,
        BLOCK_SIZE=128,
    )
    monkeypatch.setattr(flex_backend, "should_apply_ulysses", lambda: True)
    with pytest.raises(ValueError, match="requires a head-broadcast BlockMask"):
        flex_backend.flex_attention_forward(module, query, query, query, head_specific_mask)


def test_flex_attention_accepts_sliding_window_metadata_with_block_mask(monkeypatch):
    captured = {}

    def fake_backend(module, query, key, value, attention_mask, **kwargs):
        captured["attention_mask"] = attention_mask
        captured["kwargs"] = kwargs
        return query.transpose(1, 2), None

    monkeypatch.setattr(flex_backend, "hf_flex_attention_forward", fake_backend)
    monkeypatch.setattr(flex_backend, "should_apply_ulysses", lambda: False)
    query = torch.randn(1, 4, 8, 8)
    block_mask = create_block_mask(
        lambda batch_idx, head_idx, query_idx, key_idx: (query_idx >= key_idx) & (query_idx - key_idx < 4),
        B=None,
        H=None,
        Q_LEN=query.shape[2],
        KV_LEN=query.shape[2],
        device=query.device,
        BLOCK_SIZE=128,
    )
    output, auxiliary = flex_backend.flex_attention_forward(
        _FakeAttentionModule(),
        query,
        query,
        query,
        block_mask,
        sliding_window=4,
    )
    assert captured["attention_mask"] is block_mask
    assert "sliding_window" not in captured["kwargs"]
    assert captured["kwargs"]["kernel_options"] == {"BACKEND": "TRITON"}
    torch.testing.assert_close(output, query.transpose(1, 2))
    assert auxiliary is None


def test_flex_attention_delegates_active_ulysses_to_shared_helpers(monkeypatch):
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
        return query.transpose(1, 2), torch.ones(query.shape[:3])

    def fake_restore(output, *, group):
        calls.append(("restore", output, group))
        return output

    monkeypatch.setattr(flex_backend, "get_parallel_state", lambda: state)
    monkeypatch.setattr(flex_backend, "should_apply_ulysses", lambda: True)
    monkeypatch.setattr(flex_backend, "prepare_ulysses_qkv", fake_prepare)
    monkeypatch.setattr(flex_backend, "slice_ulysses_head_auxiliary", fake_slice)
    monkeypatch.setattr(flex_backend, "hf_flex_attention_forward", fake_backend)
    monkeypatch.setattr(flex_backend, "restore_ulysses_output", fake_restore)
    query = torch.randn(1, 4, 8, 8)
    key = torch.randn(1, 2, 8, 8)
    value = torch.randn(1, 2, 8, 8)
    auxiliary = torch.arange(4)

    output, lse = flex_backend.flex_attention_forward(
        _FakeAttentionModule(),
        query,
        key,
        value,
        _causal_block_mask(8, query.device),
        s_aux=auxiliary,
    )

    assert [call[0] for call in calls] == ["prepare", "slice", "backend", "restore", "restore"]
    assert calls[0][1].shape == (1, 8, 4, 8)
    assert calls[0][4:] == (group, 2)
    torch.testing.assert_close(calls[2][-1]["s_aux"], auxiliary[:2])
    assert output.shape == (1, 8, 2, 8)
    assert lse.shape == (1, 2, 8)


def test_flex_attention_skip_ulysses_skips_exchange(monkeypatch):
    monkeypatch.setattr(flex_backend, "should_apply_ulysses", lambda: True)
    monkeypatch.setattr(
        flex_backend,
        "prepare_ulysses_qkv",
        lambda *args, **kwargs: pytest.fail("skip_ulysses must not exchange QKV"),
    )
    captured = {}

    def fake_backend(module, query, key, value, attention_mask, **kwargs):
        captured["kwargs"] = kwargs
        return query.transpose(1, 2), None

    monkeypatch.setattr(flex_backend, "hf_flex_attention_forward", fake_backend)
    query = torch.randn(1, 4, 8, 8)
    flex_backend.flex_attention_forward(
        _FakeAttentionModule(),
        query,
        query[:, :2],
        query[:, :2],
        _causal_block_mask(8, query.device),
        skip_ulysses=True,
    )
    assert "skip_ulysses" not in captured["kwargs"]


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="FlexAttention numerical comparison requires CUDA")
@pytest.mark.parametrize("mask_case", ("causal", "bagel_mixed"))
def test_flex_attention_matches_math_sdpa(mask_case):
    device = torch.device(get_device_type())
    dtype = torch.bfloat16
    sequence_length = 128
    query_heads, kv_heads, head_dim = 4, 2, 64
    generator = torch.Generator(device=device).manual_seed(9051)
    query, key, value = (
        torch.randn((1, heads, sequence_length, head_dim), device=device, dtype=dtype, generator=generator)
        for heads in (query_heads, kv_heads, kv_heads)
    )
    output_gradient = torch.randn(
        (1, sequence_length, query_heads, head_dim), device=device, dtype=dtype, generator=generator
    )
    scaling = head_dim**-0.5
    dense = dense_mask(mask_case, sequence_length, device)
    block_mask = flex_mask(mask_case, sequence_length, device)

    reference_qkv = clone_qkv(query, key, value)
    reference_output, _ = math_sdpa_reference(*reference_qkv, dense, scaling=scaling)
    reference_gradients = torch.autograd.grad(reference_output, reference_qkv, output_gradient)

    flex_qkv = clone_qkv(query, key, value)
    flex_output, flex_lse = flex_backend.flex_attention_forward(
        _FakeAttentionModule(),
        *flex_qkv,
        block_mask,
        scaling=scaling,
    )
    flex_gradients = torch.autograd.grad(flex_output, flex_qkv, output_gradient)

    torch.testing.assert_close(flex_output, reference_output, rtol=ATTN_RTOL, atol=ATTN_ATOL)
    assert flex_lse is not None
    assert torch.isfinite(flex_lse).all()
    for name, flex_gradient, reference_gradient in zip(
        ("query", "key", "value"),
        flex_gradients,
        reference_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            flex_gradient,
            reference_gradient,
            rtol=ATTN_GRAD_RTOL,
            atol=ATTN_BF16_GRAD_ATOL,
            msg=lambda message, tensor_name=name: f"{tensor_name}: {message}",
        )


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="FlexAttention numerical comparison requires CUDA")
def test_flex_toy_layer_matches_math_sdpa():
    device = torch.device(get_device_type())
    dtype = torch.bfloat16
    hidden_size, query_heads, kv_heads, head_dim, sequence_length = 256, 4, 2, 64, 128
    torch.manual_seed(29)
    math_layer = (
        _ToyAttentionLayer(hidden_size, query_heads, kv_heads, head_dim).to(device=device, dtype=dtype).train()
    )
    flex_layer = copy.deepcopy(math_layer)
    hidden = torch.randn(1, sequence_length, hidden_size, device=device, dtype=dtype)
    math_hidden = hidden.detach().clone().requires_grad_(True)
    flex_hidden = hidden.detach().clone().requires_grad_(True)
    dense = dense_mask("causal", sequence_length, device)
    block_mask = flex_mask("causal", sequence_length, device)
    scaling = head_dim**-0.5

    math_query, math_key, math_value = math_layer.qkv(math_hidden)
    math_output, _ = math_sdpa_reference(math_query, math_key, math_value, dense, scaling=scaling)
    math_logits = math_layer.o_proj(math_output.reshape(1, sequence_length, query_heads * head_dim))

    flex_query, flex_key, flex_value = flex_layer.qkv(flex_hidden)
    flex_output, _ = flex_backend.flex_attention_forward(
        _FakeAttentionModule(),
        flex_query,
        flex_key,
        flex_value,
        block_mask,
        scaling=scaling,
    )
    flex_logits = flex_layer.o_proj(flex_output.reshape(1, sequence_length, query_heads * head_dim))

    torch.testing.assert_close(flex_logits, math_logits, rtol=ATTN_RTOL, atol=ATTN_ATOL)
    output_gradient = torch.randn_like(math_logits)
    math_gradients = torch.autograd.grad(math_logits, (math_hidden, *math_layer.parameters()), output_gradient)
    flex_gradients = torch.autograd.grad(flex_logits, (flex_hidden, *flex_layer.parameters()), output_gradient)
    for math_gradient, flex_gradient in zip(math_gradients, flex_gradients, strict=True):
        torch.testing.assert_close(flex_gradient, math_gradient, rtol=ATTN_GRAD_RTOL, atol=ATTN_GRAD_ATOL)
