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

"""SageAttention adapter contract and numerical checks vs MATH SDPA."""

from __future__ import annotations

import builtins
import importlib.util
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.ops.attention.attention_cases import dense_mask, math_sdpa_reference
from tests.ops.tol import ATTN_ATOL, ATTN_RTOL
from veomni.ops.kernels.attention.standard import sage as sage_backend
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
    monkeypatch.setattr(sage_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: False)
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
        cu_seq_lens_q=None,
        indices=None,
        s_aux=None,
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
    monkeypatch.setattr(sage_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: not skip_ulysses)
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
    monkeypatch.setattr(sage_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: False)
    query = torch.randn(1, 2, 4, 8)
    with pytest.raises(ValueError, match="does not take a dense attention_mask"):
        sage_backend.sage_attention_forward(
            _FakeAttentionModule(),
            query,
            query,
            query,
            attention_mask=torch.ones(1, 1, 4, 4, dtype=torch.bool),
        )


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("indices", torch.tensor([[0, 1]], dtype=torch.int32)),
        ("s_aux", torch.ones(2)),
    ),
)
def test_sage_attention_rejects_unsupported_semantic_metadata(monkeypatch, name, value):
    monkeypatch.setattr(
        sage_backend,
        "sageattn",
        lambda query, key, value, **kwargs: pytest.fail("unsupported metadata must fail before the vendor call"),
    )
    monkeypatch.setattr(
        sage_backend,
        "should_apply_ulysses",
        lambda *, skip_ulysses=False: pytest.fail("unsupported metadata must fail before Ulysses exchange"),
    )
    query = torch.randn(1, 2, 4, 8)

    with pytest.raises(ValueError, match=rf"does not support .*`{name}`"):
        sage_backend.sage_attention_forward(
            _FakeAttentionModule(),
            query,
            query,
            query,
            attention_mask=None,
            **{name: value},
        )


def test_sage_attention_drops_single_segment_and_splits_packed_cu_seqlens(monkeypatch):
    """SageAttention strips single-segment metadata and splits unmasked batch-1 multi-segment calls."""
    calls = []

    def fake_sage(query, key, value, tensor_layout="HND", is_causal=False, sm_scale=None):
        calls.append((query.shape, is_causal, sm_scale))
        scale = (query.shape[-1] ** -0.5) if sm_scale is None else sm_scale
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        if is_causal and query.shape[-2] > 1:
            causal = torch.ones(query.shape[-2], key.shape[-2], dtype=torch.bool, device=query.device).tril()
            scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(weights, value)

    monkeypatch.setattr(sage_backend, "sageattn", fake_sage)
    monkeypatch.setattr(sage_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: False)

    q1 = torch.randn(1, 2, 4, 8)
    # Single-segment metadata ([0, 4]) executes in a single call.
    single_cu = torch.tensor([0, 4], dtype=torch.int32)
    expected_single, _ = sage_backend.sage_attention_forward(_FakeAttentionModule(), q1, q1, q1, attention_mask=None)
    calls.clear()
    actual_single, _ = sage_backend.sage_attention_forward(
        _FakeAttentionModule(),
        q1,
        q1,
        q1,
        attention_mask=None,
        cu_seq_lens_q=single_cu,
        cu_seq_lens_k=single_cu,
        max_length_q=4,
        max_length_k=4,
    )
    assert len(calls) == 1
    torch.testing.assert_close(actual_single, expected_single, rtol=0, atol=0)

    # Multi-segment metadata ([0, 2, 4]) splits per segment and isolates segments.
    calls.clear()
    multi_cu = torch.tensor([0, 2, 4], dtype=torch.int32)
    q2 = q1.clone()
    q2[:, :, 2:, :] += 10.0
    out1, _ = sage_backend.sage_attention_forward(
        _FakeAttentionModule(),
        q1,
        q1,
        q1,
        attention_mask=None,
        cu_seq_lens_q=multi_cu,
        cu_seq_lens_k=multi_cu,
        max_length_q=2,
        max_length_k=2,
        is_causal=False,
    )
    out2, _ = sage_backend.sage_attention_forward(
        _FakeAttentionModule(),
        q2,
        q2,
        q2,
        attention_mask=None,
        cu_seq_lens_q=multi_cu,
        cu_seq_lens_k=multi_cu,
        max_length_q=2,
        max_length_k=2,
        is_causal=False,
    )
    assert [shape for shape, _, _ in calls[:2]] == [torch.Size([1, 2, 2, 8]), torch.Size([1, 2, 2, 8])]
    torch.testing.assert_close(out1[:, :2], out2[:, :2], rtol=0, atol=0)

    # Multi-segment with batch_size > 1 still fails closed before Ulysses or vendor kernel.
    monkeypatch.setattr(
        sage_backend,
        "should_apply_ulysses",
        lambda *, skip_ulysses=False: pytest.fail("unisolated multi-segment metadata must fail before Ulysses"),
    )
    batch2 = torch.randn(2, 2, 4, 8)
    with pytest.raises(ValueError, match="several segments per row"):
        sage_backend.sage_attention_forward(
            _FakeAttentionModule(),
            batch2,
            batch2,
            batch2,
            attention_mask=None,
            cu_seq_lens_q=torch.tensor([0, 2, 4, 8], dtype=torch.int32),
        )


def test_sage_attention_requires_package(monkeypatch):
    real_import = builtins.__import__
    missing = ModuleNotFoundError("No module named 'sageattention'", name="sageattention")
    attempts = []

    def blocked_import(name, *args, **kwargs):
        if name == "sageattention":
            attempts.append(name)
            raise missing
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    monkeypatch.setattr(sage_backend, "sageattn", None)
    monkeypatch.setattr(sage_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: False)
    query = torch.randn(1, 2, 4, 8)
    with pytest.raises(ImportError, match="sageattention") as exc_info:
        sage_backend.sage_attention_forward(
            _FakeAttentionModule(),
            query,
            query,
            query,
            attention_mask=None,
        )
    assert attempts == ["sageattention"]
    assert exc_info.value.__cause__ is missing


def test_sage_attention_preserves_transitive_import_error(monkeypatch):
    real_import = builtins.__import__
    missing_dependency = ModuleNotFoundError("No module named 'sageattention_cuda'", name="sageattention_cuda")

    def broken_import(name, *args, **kwargs):
        if name == "sageattention":
            raise missing_dependency
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    monkeypatch.setattr(sage_backend, "sageattn", None)
    monkeypatch.setattr(sage_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: False)
    query = torch.randn(1, 2, 4, 8)
    with pytest.raises(ModuleNotFoundError) as exc_info:
        sage_backend.sage_attention_forward(
            _FakeAttentionModule(),
            query,
            query,
            query,
            attention_mask=None,
        )
    assert exc_info.value is missing_dependency


def test_sage_attention_rejects_training_graph(monkeypatch):
    monkeypatch.setattr(sage_backend, "sageattn", lambda query, key, value, **kwargs: pytest.fail("sageattn"))
    monkeypatch.setattr(sage_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: False)
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
    monkeypatch.setattr(sage_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: False)
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
