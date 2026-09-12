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

"""Flash attention adapter contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tests.ops.tol import ATTN_ATOL, ATTN_GRAD_ATOL, ATTN_GRAD_RTOL, ATTN_RTOL
from veomni.ops import resolve_op
from veomni.ops.kernels.attention.standard import flash as flash_backend
from veomni.utils.device import IS_CUDA_AVAILABLE


class _FakeAttentionModule(nn.Module):
    def __init__(self, implementation: str):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation=implementation)
        self.is_causal = True
        self.layer_idx = 7
        self.proj = nn.Linear(4, 4)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="FlashAttention 2 needs a CUDA GPU")
def test_registered_fa2_adapter_matches_sdpa_with_mla_value_dim_and_gradients():
    """Exercise the complete facade/adapter/FA2 path, including MLA padding."""
    try:
        __import__("flash_attn")
    except Exception as exc:
        pytest.skip(f"flash-attn is not available: {exc}")

    torch.manual_seed(17)
    batch, heads, seq_len, head_dim, value_head_dim = 2, 4, 32, 64, 32
    tensors = (
        torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16),
        torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16),
        torch.randn(batch, heads, seq_len, value_head_dim, device="cuda", dtype=torch.float16),
    )
    q_fa, k_fa, v_fa = (tensor.detach().requires_grad_(True) for tensor in tensors)
    q_ref, k_ref, v_ref = (tensor.detach().requires_grad_(True) for tensor in tensors)
    scale = 0.17

    adapter = resolve_op("attention", "standard", "veomni_flash_attention_2").wrapper
    actual, attention_weights = adapter(
        _FakeAttentionModule("veomni_flash_attention_2"),
        q_fa,
        k_fa,
        v_fa,
        None,
        dropout=0.0,
        scaling=scale,
        is_causal=True,
        skip_ulysses=True,
    )
    expected = F.scaled_dot_product_attention(
        q_ref,
        k_ref,
        v_ref,
        dropout_p=0.0,
        is_causal=True,
        scale=scale,
    ).transpose(1, 2)

    assert attention_weights is None
    assert actual.shape == (batch, seq_len, heads, value_head_dim)
    torch.testing.assert_close(actual.float(), expected.float(), atol=ATTN_ATOL, rtol=ATTN_RTOL)

    grad_output = torch.randn_like(actual)
    actual.backward(grad_output)
    expected.backward(grad_output)
    for actual_input, expected_input in zip((q_fa, k_fa, v_fa), (q_ref, k_ref, v_ref), strict=True):
        torch.testing.assert_close(
            actual_input.grad.float(),
            expected_input.grad.float(),
            atol=ATTN_GRAD_ATOL,
            rtol=ATTN_GRAD_RTOL,
        )


@pytest.mark.parametrize(
    ("implementation", "expected_backend"),
    [
        ("veomni_flash_attention_2", "flash_attention_2"),
        ("veomni_flash_attention_3", "flash_attention_3"),
        ("veomni_flash_attention_4", "veomni_flash_attention_4"),
    ],
)
def test_flash_attention_preserves_layout_and_backend_contract(monkeypatch, implementation, expected_backend):
    captured = {}

    def replacement_backend(query, key, value, attention_mask, **kwargs):
        captured.update(query=query, key=key, value=value, attention_mask=attention_mask, kwargs=kwargs)
        return query + 1

    monkeypatch.setattr(flash_backend, "_flash_attention_forward", replacement_backend)
    monkeypatch.setattr(flash_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: False)

    module = _FakeAttentionModule(implementation)
    query = torch.randn(2, 4, 3, 4, dtype=torch.float16)
    key = torch.randn(2, 2, 3, 4, dtype=torch.float16)
    value = torch.randn(2, 2, 3, 4, dtype=torch.float16)
    attention_mask = torch.ones(2, 1, 3, 3, dtype=torch.bool)
    marker = object()

    output, attention_weights = flash_backend.flash_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=0.25,
        scaling=0.5,
        sliding_window=16,
        softcap=30.0,
        is_causal=False,
        contract_marker=marker,
    )

    torch.testing.assert_close(captured["query"], query.transpose(1, 2))
    torch.testing.assert_close(captured["key"], key.transpose(1, 2))
    torch.testing.assert_close(captured["value"], value.transpose(1, 2))
    assert captured["attention_mask"] is attention_mask
    backend_kwargs = captured["kwargs"]
    assert backend_kwargs["query_length"] == query.shape[2]
    assert backend_kwargs["is_causal"] is False
    assert backend_kwargs["dropout"] == 0.25
    assert backend_kwargs["softmax_scale"] == 0.5
    assert backend_kwargs["sliding_window"] == 16
    assert backend_kwargs["softcap"] == 30.0
    assert backend_kwargs["use_top_left_mask"] is False
    assert backend_kwargs["attn_implementation"] == expected_backend
    assert backend_kwargs["layer_idx"] == module.layer_idx
    assert backend_kwargs["contract_marker"] is marker
    assert output.shape == (2, 3, 4, 4)
    torch.testing.assert_close(output, query.transpose(1, 2) + 1)
    assert attention_weights is None


def test_flash_attention_delegates_active_ulysses_to_shared_helpers(monkeypatch):
    group = object()
    state = SimpleNamespace(ulysses_group=group, ulysses_size=2)
    calls = []

    def fake_prepare(query, key, value, *, group, ulysses_size):
        calls.append(("prepare", query, key, value, group, ulysses_size))
        return query[:, :, :2], key[:, :, :1], value[:, :, :1], 4

    def fake_slice(auxiliary, *, query_head_count, local_query_head_count, group):
        calls.append(("slice", auxiliary, query_head_count, local_query_head_count, group))
        return auxiliary[:local_query_head_count]

    def fake_restore(output, *, group):
        calls.append(("restore", output, group))
        return output

    def fake_flash(query, key, value, attention_mask, **kwargs):
        calls.append(("backend", query, key, value, attention_mask, kwargs))
        return query

    monkeypatch.setattr(flash_backend, "get_parallel_state", lambda: state)
    monkeypatch.setattr(flash_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: not skip_ulysses)
    monkeypatch.setattr(flash_backend, "prepare_ulysses_qkv", fake_prepare)
    monkeypatch.setattr(flash_backend, "slice_ulysses_head_auxiliary", fake_slice)
    monkeypatch.setattr(flash_backend, "restore_ulysses_output", fake_restore)
    monkeypatch.setattr(flash_backend, "_flash_attention_forward", fake_flash)
    query = torch.randn(1, 4, 5, 8, dtype=torch.float16)
    auxiliary = torch.arange(4, dtype=torch.float16)

    output, _ = flash_backend.flash_attention_forward(
        _FakeAttentionModule("veomni_flash_attention_2"),
        query,
        query[:, :2],
        query[:, :2],
        attention_mask=None,
        s_aux=auxiliary,
    )

    assert [call[0] for call in calls] == ["prepare", "slice", "backend", "restore"]
    assert calls[0][1].shape == (1, 5, 4, 8)
    assert calls[0][4:] == (group, 2)
    torch.testing.assert_close(calls[2][-1]["s_aux"], auxiliary[:2])
    assert output.shape == (1, 5, 2, 8)


def test_flash_attention_skip_ulysses_skips_exchange_and_is_not_forwarded(monkeypatch):
    captured = {}

    def fake_flash(query, key, value, attention_mask, **kwargs):
        captured["kwargs"] = kwargs
        return query

    monkeypatch.setattr(flash_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: not skip_ulysses)
    monkeypatch.setattr(
        flash_backend,
        "prepare_ulysses_qkv",
        lambda *args, **kwargs: pytest.fail("skip_ulysses must not exchange QKV"),
    )
    monkeypatch.setattr(flash_backend, "_flash_attention_forward", fake_flash)
    query = torch.randn(1, 4, 5, 8, dtype=torch.float16)

    flash_backend.flash_attention_forward(
        _FakeAttentionModule("veomni_flash_attention_2"),
        query,
        query[:, :2],
        query[:, :2],
        attention_mask=None,
        skip_ulysses=True,
        contract_marker=object(),
    )
    assert "skip_ulysses" not in captured["kwargs"]


def test_varlen_flash_attn_padded_input_matches_unpadded():
    """Pin the vendor contract used when packed inputs retain padded tails."""
    if not IS_CUDA_AVAILABLE or torch.version.hip is not None:
        pytest.skip("FlashAttention varlen requires an NVIDIA CUDA GPU")
    try:
        from flash_attn import flash_attn_varlen_func
    except Exception as exc:
        pytest.skip(f"flash-attn is not available: {exc}")

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16
    seqlens = torch.tensor([5, 7], dtype=torch.int32, device=device)
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0), value=0)
    max_seqlen = int(seqlens.max().item())
    total_tokens = int(cu_seqlens[-1].item())
    padded_tokens = total_tokens + 4
    nheads, head_dim = 4, 8
    q = torch.randn(total_tokens, nheads, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    padding = torch.zeros(padded_tokens - total_tokens, nheads, head_dim, device=device, dtype=dtype)

    out_unpadded = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, dropout_p=0.0)
    out_padded = flash_attn_varlen_func(
        torch.cat((q, padding)),
        torch.cat((k, padding)),
        torch.cat((v, padding)),
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        dropout_p=0.0,
    )

    assert out_padded.shape[0] == padded_tokens
    torch.testing.assert_close(out_padded[:total_tokens], out_unpadded, rtol=0.0, atol=0.0)


def test_flash_attention_forwards_fa4_sinks_and_sliding_window(monkeypatch):
    captured = {}

    def fake_flash(query, key, value, attention_mask, **kwargs):
        captured.update(kwargs)
        return torch.zeros_like(query)

    monkeypatch.setattr(flash_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: False)
    monkeypatch.setattr(flash_backend, "_flash_attention_forward", fake_flash)
    module = _FakeAttentionModule("veomni_flash_attention_4")
    query = torch.randn(1, 2, 3, 4)
    sinks = torch.randn(2)

    output, attention_weights = flash_backend.flash_attention_forward(
        module,
        query,
        query[:, :1],
        query[:, :1],
        attention_mask=None,
        scaling=0.5,
        sliding_window=8,
        s_aux=sinks,
    )

    assert output.shape == (1, 3, 2, 4)
    assert attention_weights is None
    assert captured["attn_implementation"] == "veomni_flash_attention_4"
    assert captured["sliding_window"] == 8
    assert captured["s_aux"] is sinks
    assert captured["softmax_scale"] == 0.5
    assert captured["layer_idx"] == 7


def test_flash_attention_rejects_sparse_indices(monkeypatch):
    monkeypatch.setattr(flash_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: False)
    query = torch.randn(1, 2, 3, 4, dtype=torch.float16)

    with pytest.raises(ValueError, match="sparse `indices`"):
        flash_backend.flash_attention_forward(
            _FakeAttentionModule("veomni_flash_attention_2"),
            query,
            query,
            query,
            attention_mask=None,
            indices=torch.zeros(1, 1, dtype=torch.int32),
        )


def test_flash_attention_pads_and_restores_mla_value_head_dim(monkeypatch):
    captured = {}

    def fake_flash(query, key, value, attention_mask, **kwargs):
        captured["value"] = value
        return value + 1

    monkeypatch.setattr(flash_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: False)
    monkeypatch.setattr(flash_backend, "_flash_attention_forward", fake_flash)
    query = torch.randn(1, 2, 3, 8, dtype=torch.float16)
    key = torch.randn(1, 2, 3, 8, dtype=torch.float16)
    value = torch.randn(1, 2, 3, 4, dtype=torch.float16)

    output, _ = flash_backend.flash_attention_forward(
        _FakeAttentionModule("veomni_flash_attention_2"),
        query,
        key,
        value,
        attention_mask=None,
    )

    assert captured["value"].shape == (1, 3, 2, 8)
    assert output.shape == (1, 3, 2, 4)
    torch.testing.assert_close(output, value.transpose(1, 2) + 1)


def test_gpt_oss_attention_passes_learnable_sinks_through_hf_dict():
    from transformers import GptOssConfig
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention

    captured = {}
    backend_name = "veomni_test_gpt_oss_attention"

    def fake_attention_backend(module, query, key, value, attention_mask, **kwargs):
        captured["module"] = module
        captured["s_aux"] = kwargs["s_aux"]
        captured["sliding_window"] = kwargs["sliding_window"]
        return torch.zeros_like(query.transpose(1, 2)), None

    ALL_ATTENTION_FUNCTIONS.register(backend_name, fake_attention_backend)
    try:
        config = GptOssConfig(
            hidden_size=16,
            head_dim=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=1,
            layer_types=["sliding_attention"],
            sliding_window=8,
        )
        config._attn_implementation = backend_name
        attention = GptOssAttention(config, layer_idx=0)
        hidden_states = torch.randn(1, 3, 16)
        cos = torch.ones(1, 3, 2)
        sin = torch.zeros(1, 3, 2)

        output, attention_weights = attention(
            hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=None,
        )
    finally:
        type(ALL_ATTENTION_FUNCTIONS)._global_mapping.pop(backend_name, None)

    assert output.shape == hidden_states.shape
    assert attention_weights is None
    assert captured["module"] is attention
    assert captured["s_aux"] is attention.sinks
    assert captured["sliding_window"] == 8
