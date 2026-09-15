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
from torch.nn import functional as F

from tests.ops.attention.attention_cases import clone_qkv, dense_mask, math_sdpa_reference
from tests.ops.attention.utils import UlyssesHelperRecorder
from tests.ops.tol import (
    ATTN_ATOL,
    ATTN_GRAD_ATOL,
    ATTN_GRAD_RTOL,
    ATTN_RTOL,
    EAGER_ATOL,
    EAGER_GRAD_ATOL,
    EAGER_GRAD_RTOL,
    EAGER_RTOL,
)
from veomni.ops.kernels.attention import ulysses as ulysses_backend
from veomni.ops.kernels.attention.standard import sdpa as sdpa_backend
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


def test_sdpa_attention_rejects_zero_dimensions():
    module = _FakeAttentionModule()
    query = torch.randn(1, 0, 4, 8)
    with pytest.raises(ValueError, match="zero dimensions"):
        sdpa_backend.sdpa_attention_forward(module, query, query, query, attention_mask=None)


@pytest.mark.parametrize("softcap", (0.0, 30.0))
def test_sdpa_attention_rejects_softcap(softcap):
    query = torch.randn(1, 2, 4, 8)
    with pytest.raises(ValueError, match="does not support softcap"):
        sdpa_backend.sdpa_attention_forward(
            _FakeAttentionModule(),
            query,
            query,
            query,
            attention_mask=None,
            softcap=softcap,
        )


def test_sdpa_attention_delegates_active_ulysses_to_shared_helpers(monkeypatch):
    group = object()
    state = SimpleNamespace(ulysses_group=group, ulysses_size=2)
    recorder = UlyssesHelperRecorder()

    def fake_backend(module, query, key, value, attention_mask, **kwargs):
        recorder.calls.append(("backend", query, key, value, attention_mask, kwargs))
        return query.transpose(1, 2), None

    monkeypatch.setattr(sdpa_backend, "get_parallel_state", lambda: state)
    monkeypatch.setattr(sdpa_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: not skip_ulysses)
    monkeypatch.setattr(sdpa_backend, "prepare_ulysses_qkv", recorder.prepare)
    monkeypatch.setattr(sdpa_backend, "slice_ulysses_head_auxiliary", recorder.slice_auxiliary)
    monkeypatch.setattr(sdpa_backend, "hf_sdpa_attention_forward", fake_backend)
    monkeypatch.setattr(sdpa_backend, "restore_ulysses_output", recorder.restore)
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

    assert [call[0] for call in recorder.calls] == ["prepare", "slice", "backend", "restore"]
    assert recorder.calls[0][1].shape == (1, 8, 4, 8)
    assert recorder.calls[0][4:] == (group, 2)
    torch.testing.assert_close(recorder.calls[2][-1]["s_aux"], auxiliary[:2])
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


@pytest.mark.parametrize(
    ("query_heads", "key_value_heads", "ulysses_size", "mask_heads"),
    (
        pytest.param(8, 2, 4, 1, id="local-gqa-ratio"),
        pytest.param(4, 4, 2, 4, id="head-specific-mask"),
    ),
)
def test_sdpa_attention_uses_post_ulysses_head_layout(
    monkeypatch,
    query_heads,
    key_value_heads,
    ulysses_size,
    mask_heads,
):
    rank = 1
    group = object()
    state = SimpleNamespace(ulysses_group=group, ulysses_size=ulysses_size)

    def fake_gather_seq_scatter_heads(tensor, *, seq_dim, head_dim, group):
        del group
        full_sequence = torch.cat([tensor] * ulysses_size, dim=seq_dim)
        local_head_count = tensor.shape[head_dim] // ulysses_size
        return full_sequence.narrow(head_dim, rank * local_head_count, local_head_count).contiguous()

    monkeypatch.setattr(sdpa_backend, "get_parallel_state", lambda: state)
    monkeypatch.setattr(sdpa_backend, "should_apply_ulysses", lambda **kwargs: True)
    monkeypatch.setattr(ulysses_backend, "gather_seq_scatter_heads", fake_gather_seq_scatter_heads)
    monkeypatch.setattr(ulysses_backend, "gather_heads_scatter_seq", lambda tensor, **kwargs: tensor)
    monkeypatch.setattr(ulysses_backend.dist, "get_rank", lambda group: rank)

    local_sequence_length = 3
    full_sequence_length = local_sequence_length * ulysses_size
    batch_size = 2
    head_dim = 8
    generator = torch.Generator().manual_seed(13)
    query = torch.randn(batch_size, query_heads, local_sequence_length, head_dim, generator=generator)
    key = torch.randn(batch_size, key_value_heads, local_sequence_length, head_dim, generator=generator)
    value = torch.randn(batch_size, key_value_heads, local_sequence_length, head_dim, generator=generator)
    causal_mask = torch.tril(torch.ones(full_sequence_length, full_sequence_length, dtype=torch.bool))
    attention_mask = causal_mask[None, None].expand(batch_size, mask_heads, -1, -1).clone()
    if mask_heads > 1:
        for head in range(mask_heads):
            attention_mask[:, head, :, head::mask_heads] = False
            attention_mask[:, head].diagonal().fill_(True)

    reference_qkv = clone_qkv(query, key, value)
    local_query, local_key, local_value, global_query_heads = ulysses_backend.prepare_ulysses_qkv(
        *(tensor.transpose(1, 2) for tensor in reference_qkv),
        group=group,
        ulysses_size=ulysses_size,
    )
    local_query, local_key, local_value = (tensor.transpose(1, 2) for tensor in (local_query, local_key, local_value))
    local_key_value_groups = local_query.shape[1] // local_key.shape[1]
    local_key = torch.repeat_interleave(local_key, local_key_value_groups, dim=1)
    local_value = torch.repeat_interleave(local_value, local_key_value_groups, dim=1)
    if attention_mask.shape[1] == global_query_heads:
        local_attention_mask = attention_mask.narrow(1, rank * local_query.shape[1], local_query.shape[1])
    else:
        local_attention_mask = attention_mask

    reference_output = F.scaled_dot_product_attention(
        local_query,
        local_key,
        local_value,
        attn_mask=local_attention_mask,
    ).transpose(1, 2)
    output_gradient = torch.randn(reference_output.shape, generator=generator)
    reference_gradients = torch.autograd.grad(reference_output, reference_qkv, output_gradient)

    module = _FakeAttentionModule()
    module.num_key_value_groups = query_heads // key_value_heads
    original_key_value_groups = module.num_key_value_groups
    sdpa_qkv = clone_qkv(query, key, value)
    sdpa_output, _ = sdpa_backend.sdpa_attention_forward(
        module,
        *sdpa_qkv,
        attention_mask=attention_mask,
        dropout=0.0,
    )
    sdpa_gradients = torch.autograd.grad(sdpa_output, sdpa_qkv, output_gradient)

    assert module.num_key_value_groups == original_key_value_groups
    torch.testing.assert_close(sdpa_output, reference_output, rtol=EAGER_RTOL, atol=EAGER_ATOL)
    for gradient, reference_gradient in zip(sdpa_gradients, reference_gradients, strict=True):
        torch.testing.assert_close(gradient, reference_gradient, rtol=EAGER_GRAD_RTOL, atol=EAGER_GRAD_ATOL)


@pytest.mark.parametrize("mask_case", ("causal", "full", "2d_mask"))
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
    sdpa_output, lse = sdpa_backend.sdpa_attention_forward(
        module,
        *sdpa_qkv,
        attention_mask=dense,
        dropout=0.0,
        scaling=scaling,
    )
    sdpa_gradients = torch.autograd.grad(sdpa_output, sdpa_qkv, output_gradient)

    assert lse is None
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
@pytest.mark.parametrize("mask_case", ("causal", "2d_mask"))
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
