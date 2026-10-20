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

"""Flex attention adapter contract and numerical checks vs MATH SDPA."""

from __future__ import annotations

import contextlib
import copy
import gc
import importlib.util
import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, create_mask

from tests.ops.attention.attention_cases import clone_qkv, dense_mask, flex_mask, math_sdpa_reference
from tests.ops.attention.utils import UlyssesHelperRecorder
from tests.ops.tol import (
    ATTN_ATOL,
    ATTN_BF16_GRAD_ATOL,
    ATTN_GRAD_ATOL,
    ATTN_GRAD_RTOL,
    ATTN_LSE_RTOL,
    ATTN_RTOL,
)
from veomni.ops.kernels.attention.mask import flex as flex_mask_backend
from veomni.ops.kernels.attention.mask.flex import _flex_block_size
from veomni.ops.kernels.attention.standard import flex as flex_backend
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE, get_device_type


_FLEX_COMPILE_AVAILABLE = importlib.util.find_spec("triton") is not None
_FLEX_TENSOR_BOUNDS_AVAILABLE = "seq_lengths" in inspect.signature(BlockMask).parameters


def _mixed_packed_mask(sequence_length, device):
    """Tensor-only predicate and an independently assembled dense reference."""
    span = sequence_length // 5
    document_lengths = ((span, span, span), (span, sequence_length - 4 * span))
    document_modes = (("causal", "noise", "noise"), ("full", "causal"))
    metadata = torch.full((3, sequence_length), -1, device=device, dtype=torch.int32)
    visible = torch.zeros((sequence_length, sequence_length), device=device, dtype=torch.bool)
    cursor = 0
    span_id = 0
    for document_id, (lengths, modes) in enumerate(zip(document_lengths, document_modes, strict=True)):
        clean_spans = []
        for length, mode in zip(lengths, modes, strict=True):
            end = cursor + length
            metadata[0, cursor:end] = document_id
            if mode != "causal":
                metadata[1, cursor:end] = span_id
            if mode == "noise":
                metadata[2, cursor:end] = span_id
            for clean_start, clean_end in clean_spans:
                visible[cursor:end, clean_start:clean_end] = True
            if mode == "causal":
                visible[cursor:end, cursor:end].fill_(True).tril_()
            else:
                visible[cursor:end, cursor:end] = True
            if mode != "noise":
                clean_spans.append((cursor, end))
            cursor = end
            span_id += 1
    document_ids, full_span_ids, noise_span_ids = metadata

    def mask_mod(batch_idx, head_idx, query_idx, key_idx):
        same_document = document_ids[query_idx] == document_ids[key_idx]
        same_full_span = (full_span_ids[query_idx] >= 0) & (full_span_ids[query_idx] == full_span_ids[key_idx])
        foreign_noise_key = (noise_span_ids[key_idx] >= 0) & (noise_span_ids[query_idx] != noise_span_ids[key_idx])
        return same_document & ((query_idx >= key_idx) | same_full_span) & ~foreign_noise_key

    mask = create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=device,
        BLOCK_SIZE=_flex_block_size(device),
        _compile=False,
    )
    return mask, visible[None, None]


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


@pytest.fixture
def cleanup_compiled_cuda_state():
    """Release production-shape Flex state before later GPU tests run."""
    yield
    torch.compiler.reset()
    gc.collect()
    torch.cuda.empty_cache()


def _causal_block_mask(
    sequence_length: int, device: torch.device | str, block_size: int | tuple[int, int] | None = None
):
    return create_block_mask(
        lambda batch_idx, head_idx, query_idx, key_idx: query_idx >= key_idx,
        B=None,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=device,
        BLOCK_SIZE=_flex_block_size(device) if block_size is None else block_size,
    )


@pytest.mark.parametrize(
    (
        "device_type",
        "compute_capability",
        "cute_available",
        "backend",
        "attention_sinks",
        "expected",
    ),
    [
        ("cpu", 0, True, None, False, flex_backend.FLEX_BACKEND_TRITON),
        ("cuda", 80, True, None, False, flex_backend.FLEX_BACKEND_TRITON),
        ("cuda", 90, True, None, False, flex_backend.FLEX_BACKEND_FLASH),
        ("cuda", 90, False, None, False, flex_backend.FLEX_BACKEND_TRITON),
        ("cuda", 90, True, None, True, flex_backend.FLEX_BACKEND_TRITON),
        ("cuda", 100, True, None, False, flex_backend.FLEX_BACKEND_FLASH),
        ("cuda", 90, True, "TRITON", False, flex_backend.FLEX_BACKEND_TRITON),
        ("cuda", 80, True, "FLASH", False, flex_backend.FLEX_BACKEND_FLASH),
    ],
)
def test_resolve_flex_attention(
    monkeypatch,
    device_type,
    compute_capability,
    cute_available,
    backend,
    attention_sinks,
    expected,
):
    monkeypatch.setattr(flex_backend, "get_gpu_compute_capability", lambda device: compute_capability)
    monkeypatch.setattr(flex_backend._fa4, "_flash_attn_cute_available", lambda: cute_available)
    kernel_options = {} if backend is None else {"BACKEND": backend}
    interface = flex_backend.resolve_flex_attention(
        torch.device(device_type),
        kernel_options,
        attention_sinks=attention_sinks,
    )
    if expected == flex_backend.FLEX_BACKEND_FLASH:
        assert interface is flex_backend._flex_attention_fa4
    else:
        assert interface is flex_backend._flex_attention_triton
    assert kernel_options["BACKEND"] == expected


def test_resolve_flex_attention_keeps_fp32_on_triton_on_sm90(monkeypatch):
    monkeypatch.setattr(flex_backend, "get_gpu_compute_capability", lambda device: 90)
    monkeypatch.setattr(flex_backend._fa4, "_flash_attn_cute_available", lambda: True)
    kernel_options = {}
    interface = flex_backend.resolve_flex_attention(
        torch.device("cuda"),
        kernel_options,
        query_dtype=torch.float32,
    )
    assert interface is flex_backend._flex_attention_triton
    assert kernel_options["BACKEND"] == flex_backend.FLEX_BACKEND_TRITON


def test_resolve_flex_attention_keeps_non_multiple_of_32_head_dim_on_triton(monkeypatch):
    monkeypatch.setattr(flex_backend, "get_gpu_compute_capability", lambda device: 90)
    monkeypatch.setattr(flex_backend._fa4, "_flash_attn_cute_available", lambda: True)
    kernel_options = {}
    interface = flex_backend.resolve_flex_attention(
        torch.device("cuda"),
        kernel_options,
        query_dtype=torch.bfloat16,
        head_dim=16,
    )
    assert interface is flex_backend._flex_attention_triton
    assert kernel_options["BACKEND"] == flex_backend.FLEX_BACKEND_TRITON


def test_resolve_flex_attention_keeps_multiple_of_32_head_dim_on_flash(monkeypatch):
    monkeypatch.setattr(flex_backend, "get_gpu_compute_capability", lambda device: 90)
    monkeypatch.setattr(flex_backend._fa4, "_flash_attn_cute_available", lambda: True)
    kernel_options = {}
    interface = flex_backend.resolve_flex_attention(
        torch.device("cuda"),
        kernel_options,
        query_dtype=torch.bfloat16,
        head_dim=64,
    )
    assert interface is flex_backend._flex_attention_fa4
    assert kernel_options["BACKEND"] == flex_backend.FLEX_BACKEND_FLASH


def test_flex_attention_recasts_fp32_qkv_to_module_weight_dtype(monkeypatch):
    captured = {}

    def fake_flash(module, query, key, value, attention_mask, **kwargs):
        del module, key, value, attention_mask, kwargs
        captured["dtype"] = query.dtype
        return query.transpose(1, 2), None

    def fake_resolve(device, kernel_options, **kwargs):
        del device
        captured["query_dtype"] = kwargs.get("query_dtype")
        kernel_options["BACKEND"] = flex_backend.FLEX_BACKEND_FLASH
        return fake_flash

    monkeypatch.setattr(flex_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: False)
    monkeypatch.setattr(flex_backend, "resolve_flex_attention", fake_resolve)

    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(_attn_implementation="veomni_flex_attention")
            self.proj = nn.Linear(8, 8)

    module = Module()
    module.proj.to(dtype=torch.bfloat16)
    query = torch.randn(1, 2, 8, 8)
    flex_backend.flex_attention_forward(
        module,
        query,
        query,
        query,
        _causal_block_mask(8, query.device),
    )
    assert captured["dtype"] == torch.bfloat16
    assert captured["query_dtype"] == torch.bfloat16


def test_resolve_flex_attention_rejects_forced_flash_with_sinks():
    with pytest.raises(ValueError, match="does not support attention sinks"):
        flex_backend.resolve_flex_attention(
            torch.device("cpu"),
            {"BACKEND": flex_backend.FLEX_BACKEND_FLASH},
            attention_sinks=True,
        )


def test_flex_attention_rejects_forced_flash_with_sinks(monkeypatch):
    monkeypatch.setattr(flex_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: False)
    query = torch.randn(1, 4, 8, 8)
    with pytest.raises(ValueError, match="does not support attention sinks"):
        flex_backend.flex_attention_forward(
            _FakeAttentionModule(),
            query,
            query,
            query,
            _causal_block_mask(8, query.device),
            kernel_options={"BACKEND": flex_backend.FLEX_BACKEND_FLASH},
            s_aux=torch.ones(query.shape[1]),
        )


def test_flex_attention_flash_backend_skips_hf_lse_path(monkeypatch):
    captured = {}

    def fake_flash(module, query, key, value, attention_mask, **kwargs):
        captured["kernel_options"] = kwargs["kernel_options"]
        return query.transpose(1, 2), None

    def fail_hf(*args, **kwargs):
        raise AssertionError("FLASH path must not request LSE through the HF adapter")

    def fake_resolve(device, kernel_options, **kwargs):
        del device, kwargs
        kernel_options["BACKEND"] = flex_backend.FLEX_BACKEND_FLASH
        return fake_flash

    monkeypatch.setattr(flex_backend, "resolve_flex_attention", fake_resolve)
    monkeypatch.setattr(flex_backend, "_flex_attention_triton", fail_hf)
    monkeypatch.setattr(flex_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: False)
    query = torch.randn(1, 4, 8, 8)
    output, auxiliary = flex_backend.flex_attention_forward(
        _FakeAttentionModule(),
        query,
        query,
        query,
        _causal_block_mask(8, query.device),
    )
    assert captured["kernel_options"] == {"BACKEND": flex_backend.FLEX_BACKEND_FLASH}
    torch.testing.assert_close(output, query.transpose(1, 2))
    assert auxiliary is None


@pytest.mark.skipif(
    not _FLEX_COMPILE_AVAILABLE,
    reason="HF FlexAttention compiles through inductor, which needs triton",
)
@pytest.mark.skipif(
    IS_NPU_AVAILABLE,
    reason="Ascend inductor cannot lower CPU flex_attention: Subgraph has no dtype",
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
@pytest.mark.parametrize("block_size", (128, (256, 128)))
def test_flex_attention_short_query_backward_is_finite(block_size):
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
        _causal_block_mask(sequence_length, device, block_size),
        kernel_options={"BACKEND": flex_backend.FLEX_BACKEND_TRITON},
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


@pytest.mark.parametrize("backend", [None, "TRITON", "FLASH"])
@pytest.mark.parametrize("token_major", [False, True])
@pytest.mark.parametrize("output_attentions", [None, False, True])
def test_flex_attention_output_attentions_contract(monkeypatch, backend, token_major, output_attentions):
    """Reject attention-weight requests before communication; preserve other calls."""
    query = torch.randn(8, 4, 8) if token_major else torch.randn(1, 4, 8, 8)
    mask = _causal_block_mask(8, query.device)
    kwargs = {} if backend is None else {"kernel_options": {"BACKEND": backend}}
    if output_attentions is not None:
        kwargs["output_attentions"] = output_attentions

    def unexpected_parallel_state():
        """Fail if an unsupported request reaches distributed state lookup."""
        pytest.fail("Attention-weight requests must fail before Ulysses or backend execution")

    if output_attentions:
        monkeypatch.setattr(flex_backend, "get_parallel_state", unexpected_parallel_state)
        with pytest.raises(ValueError, match="does not support `output_attentions=True`"):
            flex_backend.flex_attention_forward(_FakeAttentionModule(), query, query, query, mask, **kwargs)
        return

    def backend_forward(module, query, key, value, attention_mask, **kwargs):
        """Keep backend output observable without compiling an attention kernel."""
        assert attention_mask is mask
        return query.transpose(1, 2), None

    monkeypatch.setattr(flex_backend, "should_apply_ulysses", lambda **kwargs: False)
    monkeypatch.setattr(flex_backend, "_flex_attention_triton", backend_forward)
    monkeypatch.setattr(flex_backend, "_flex_attention_fa4", backend_forward)
    output, auxiliary = flex_backend.flex_attention_forward(
        _FakeAttentionModule(), query, query, query, mask, **kwargs
    )
    torch.testing.assert_close(output, query if token_major else query.transpose(1, 2))
    assert auxiliary is None


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
    monkeypatch.setattr(flex_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: not skip_ulysses)
    with pytest.raises(ValueError, match="requires a head-broadcast BlockMask"):
        flex_backend.flex_attention_forward(module, query, query, query, head_specific_mask)


def test_flex_attention_accepts_sliding_window_metadata_with_block_mask(monkeypatch):
    captured = {}

    def fake_backend(module, query, key, value, attention_mask, **kwargs):
        captured["attention_mask"] = attention_mask
        captured["kwargs"] = kwargs
        return query.transpose(1, 2), None

    monkeypatch.setattr(flex_backend, "_flex_attention_triton", fake_backend)
    monkeypatch.setattr(flex_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: False)
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


@pytest.mark.parametrize("token_major", [False, True])
def test_flex_attention_delegates_active_ulysses_to_shared_helpers(monkeypatch, token_major):
    group = object()
    state = SimpleNamespace(ulysses_group=group, ulysses_size=2)
    recorder = UlyssesHelperRecorder()

    def fake_backend(module, query, key, value, attention_mask, **kwargs):
        recorder.calls.append(("backend", query, key, value, attention_mask, kwargs))
        return query.transpose(1, 2), torch.ones(query.shape[:3])

    monkeypatch.setattr(flex_backend, "get_parallel_state", lambda: state)
    monkeypatch.setattr(flex_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: not skip_ulysses)
    monkeypatch.setattr(flex_backend, "prepare_ulysses_qkv", recorder.prepare)
    monkeypatch.setattr(flex_backend, "slice_ulysses_head_auxiliary", recorder.slice_auxiliary)
    monkeypatch.setattr(flex_backend, "_flex_attention_triton", fake_backend)
    monkeypatch.setattr(flex_backend, "restore_ulysses_output", recorder.restore)
    query = torch.randn(1, 4, 8, 8)
    key = torch.randn(1, 2, 8, 8)
    value = torch.randn(1, 2, 8, 8)
    auxiliary = torch.arange(4)
    if token_major:
        query, key, value = (tensor.squeeze(0).transpose(0, 1) for tensor in (query, key, value))

    output, lse = flex_backend.flex_attention_forward(
        _FakeAttentionModule(),
        query,
        key,
        value,
        _causal_block_mask(8, query.device),
        s_aux=auxiliary,
    )

    assert [call[0] for call in recorder.calls] == ["prepare", "slice", "backend", "restore", "restore"]
    assert recorder.calls[0][1].shape == (1, 8, 4, 8)
    assert recorder.calls[0][4:] == (group, 2)
    torch.testing.assert_close(recorder.calls[2][-1]["s_aux"], auxiliary[:2])
    assert output.shape == ((8, 2, 8) if token_major else (1, 8, 2, 8))
    assert lse.shape == ((2, 8) if token_major else (1, 2, 8))


def test_flex_attention_skip_ulysses_skips_exchange(monkeypatch):
    monkeypatch.setattr(flex_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: not skip_ulysses)
    monkeypatch.setattr(
        flex_backend,
        "prepare_ulysses_qkv",
        lambda *args, **kwargs: pytest.fail("skip_ulysses must not exchange QKV"),
    )
    captured = {}

    def fake_backend(module, query, key, value, attention_mask, **kwargs):
        captured["kwargs"] = kwargs
        return query.transpose(1, 2), None

    monkeypatch.setattr(flex_backend, "_flex_attention_triton", fake_backend)
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
@pytest.mark.parametrize("mask_case", ("causal", "2d_mask"))
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
    reference_output, reference_lse = math_sdpa_reference(*reference_qkv, dense, scaling=scaling)
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
    if flex_backend.resolve_flex_attention(device, {}) is flex_backend._flex_attention_fa4:
        assert flex_lse is None
    else:
        assert flex_lse is not None
        torch.testing.assert_close(flex_lse.float(), reference_lse.float(), rtol=ATTN_LSE_RTOL, atol=ATTN_ATOL)
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
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
def test_flex_toy_layer_matches_math_sdpa(dtype, cleanup_compiled_cuda_state):
    device = torch.device(get_device_type())
    hidden_size, query_heads, kv_heads, head_dim, sequence_length = 3584, 28, 4, 128, 4096
    # Seed-sensitive: other seeds put 1-2 of 1.8M bf16 v_proj.weight grads above 8e-2.
    torch.manual_seed(9051)
    math_layer = (
        _ToyAttentionLayer(hidden_size, query_heads, kv_heads, head_dim).to(device=device, dtype=dtype).train()
    )
    flex_layer = copy.deepcopy(math_layer)
    generator = torch.Generator(device=device).manual_seed(9052)
    hidden = torch.randn(1, sequence_length, hidden_size, device=device, dtype=dtype, generator=generator)
    output_gradient = torch.randn(1, sequence_length, hidden_size, device=device, dtype=dtype, generator=generator)
    math_hidden = hidden.detach().clone().requires_grad_(True)
    flex_hidden = hidden.detach().clone().requires_grad_(True)
    dense = dense_mask("2d_mask", sequence_length, device)
    block_mask = flex_mask("2d_mask", sequence_length, device)
    scaling = head_dim**-0.5

    math_query, math_key, math_value = math_layer.qkv(math_hidden)
    math_output, _ = math_sdpa_reference(math_query, math_key, math_value, dense, scaling=scaling)
    math_logits = math_layer.o_proj(math_output.reshape(1, sequence_length, query_heads * head_dim))

    flex_query, flex_key, flex_value = flex_layer.qkv(flex_hidden)
    flex_output, flex_lse = flex_backend.flex_attention_forward(
        _FakeAttentionModule(),
        flex_query,
        flex_key,
        flex_value,
        block_mask,
        scaling=scaling,
    )
    flex_logits = flex_layer.o_proj(flex_output.reshape(1, sequence_length, query_heads * head_dim))

    torch.testing.assert_close(flex_logits, math_logits, rtol=ATTN_RTOL, atol=ATTN_ATOL)
    if flex_backend.resolve_flex_attention(device, {}) is flex_backend._flex_attention_fa4:
        assert flex_lse is None
    else:
        assert flex_lse is not None
        assert torch.isfinite(flex_lse).all()
    math_gradients = torch.autograd.grad(math_logits, (math_hidden, *math_layer.parameters()), output_gradient)
    flex_gradients = torch.autograd.grad(flex_logits, (flex_hidden, *flex_layer.parameters()), output_gradient)
    gradient_atol = ATTN_BF16_GRAD_ATOL if dtype == torch.bfloat16 else ATTN_GRAD_ATOL
    for math_gradient, flex_gradient in zip(math_gradients, flex_gradients, strict=True):
        assert torch.isfinite(flex_gradient).all()
        torch.testing.assert_close(flex_gradient, math_gradient, rtol=ATTN_GRAD_RTOL, atol=gradient_atol)


@pytest.mark.parametrize("sequence_length", [13, 129, 259])
@pytest.mark.skipif(not _FLEX_TENSOR_BOUNDS_AVAILABLE, reason="BlockMask tensor-derived bounds require seq_lengths")
def test_tensor_length_wrapper_preserves_mixed_visibility_and_sparse_buffers(monkeypatch, sequence_length):
    mask, dense = _mixed_packed_mask(sequence_length, torch.device("cpu"))
    captured = {}

    def fake_raw(query, key, value, *, block_mask):
        captured["mask"] = block_mask
        return query

    monkeypatch.setattr(flex_backend._fa4, "raw_flex_attention", fake_raw)
    query = torch.randn(1, 4, sequence_length, 32)
    flex_backend._fa4._flex_attention_with_tensor_lengths(query, query, query, block_mask=mask)
    rebuilt = captured["mask"]
    assert rebuilt is not mask
    assert rebuilt.seq_lengths == mask.seq_lengths == (sequence_length, sequence_length)
    for name in (
        "kv_num_blocks",
        "kv_indices",
        "full_kv_num_blocks",
        "full_kv_indices",
        "q_num_blocks",
        "q_indices",
        "full_q_num_blocks",
        "full_q_indices",
        "mask_mod",
    ):
        assert getattr(rebuilt, name) is getattr(mask, name)
    assert rebuilt.BLOCK_SIZE == mask.BLOCK_SIZE
    actual = create_mask(rebuilt.mask_mod, B=1, H=1, Q_LEN=sequence_length, KV_LEN=sequence_length, device="cpu")
    torch.testing.assert_close(actual, dense)


@pytest.mark.skipif(not _FLEX_TENSOR_BOUNDS_AVAILABLE, reason="BlockMask tensor-derived bounds require seq_lengths")
@pytest.mark.parametrize("token_major", [False, True])
def test_flex_flash_adapter_uses_dynamic_compile_without_option(monkeypatch, token_major):
    calls = []
    real_compile = torch.compile

    def fake_compile(fn, **options):
        calls.append(options)
        assert fn is flex_backend._fa4._flex_attention_with_tensor_lengths
        return real_compile(fn, backend="eager", **options)

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(
        flex_backend._fa4, "raw_flex_attention", lambda query, key, value, **kwargs: query + key + value
    )
    monkeypatch.setattr(flex_backend, "should_apply_ulysses", lambda **kwargs: False)
    torch.compiler.reset()
    flex_backend._fa4._compiled_flex_attention_fa4.cache_clear()
    initial_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    try:
        with torch._dynamo.config.patch(recompile_limit=4):
            for length in (65, 129, 193, 259, 389, 517, 645, 773, 901, 1027, 1283, 1541, 389, 129, 773, 65):
                mask, _ = _mixed_packed_mask(length, torch.device("cpu"))
                query = torch.randn(length, 4, 32) if token_major else torch.randn(1, 4, length, 32)
                output, auxiliary = flex_backend.flex_attention_forward(
                    _FakeAttentionModule(), query, query, query, mask, kernel_options={"BACKEND": "FLASH"}
                )
                expected = query + query + query
                if not token_major:
                    expected = expected.transpose(1, 2)
                torch.testing.assert_close(output, expected)
                assert auxiliary is None
        assert calls == [{"dynamic": True, "fullgraph": True}]
        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] - initial_graphs <= 4
    finally:
        torch.compiler.reset()
        flex_backend._fa4._compiled_flex_attention_fa4.cache_clear()


@pytest.mark.skipif(not _FLEX_TENSOR_BOUNDS_AVAILABLE, reason="BlockMask tensor-derived bounds require seq_lengths")
def test_dynamic_flex_rejects_mismatched_bounds_before_compile(monkeypatch):
    monkeypatch.setattr(
        flex_backend._fa4,
        "_compiled_flex_attention_fa4",
        lambda: pytest.fail("Mismatched bounds must fail before compiling"),
    )
    query = torch.randn(1, 4, 129, 32)
    with pytest.raises(ValueError, match="sequence lengths to match"):
        flex_backend._flex_attention_fa4(
            _FakeAttentionModule(),
            query,
            query,
            query,
            _causal_block_mask(128, query.device),
            dropout=0.0,
            scaling=None,
            softcap=None,
            kernel_options={"BACKEND": "FLASH"},
        )


@pytest.mark.skipif(not _FLEX_TENSOR_BOUNDS_AVAILABLE, reason="BlockMask tensor-derived bounds require seq_lengths")
def test_flex_cache_exhaustion_raises_without_eager_fallback_on_cpu(monkeypatch):
    """Different dtypes still specialize; exhaustion must not enter eager Flex."""
    from torch._dynamo.exc import FailOnRecompileLimitHit

    real_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: real_compile(fn, backend="eager", **kwargs))
    monkeypatch.setattr(
        flex_backend._fa4, "raw_flex_attention", lambda query, key, value, **kwargs: query + key + value
    )
    torch.compiler.reset()
    flex_backend._fa4._compiled_flex_attention_fa4.cache_clear()
    try:
        with torch._dynamo.config.patch(recompile_limit=1):
            for dtype in (torch.float32, torch.float64):
                query = torch.randn(1, 4, 129, 32, dtype=dtype)
                context = (
                    pytest.raises(FailOnRecompileLimitHit) if dtype == torch.float64 else contextlib.nullcontext()
                )
                with context:
                    flex_backend._flex_attention_fa4(
                        _FakeAttentionModule(),
                        query,
                        query,
                        query,
                        _causal_block_mask(129, query.device),
                        dropout=0.0,
                        scaling=None,
                        softcap=None,
                        kernel_options={"BACKEND": "FLASH"},
                    )
    finally:
        torch.compiler.reset()
        flex_backend._fa4._compiled_flex_attention_fa4.cache_clear()


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Flex-FA4 requires CUDA")
@pytest.mark.parametrize(("query_heads", "kv_heads"), [(28, 4), (7, 1)])
@pytest.mark.parametrize("token_major", [False, True])
def test_flex_fa4_variable_lengths_reuse_graphs_and_match_math_sdpa(
    query_heads, kv_heads, token_major, record_property
):
    device = torch.device(get_device_type())
    if flex_backend.get_gpu_compute_capability(device) < 90 or not flex_backend._fa4._flash_attn_cute_available():
        pytest.skip("Flex-FA4 requires SM90+ and flash-attn-4")
    torch.compiler.reset()
    flex_backend._fa4._compiled_flex_attention_fa4.cache_clear()
    initial_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    try:
        with torch._dynamo.config.patch(recompile_limit=4):
            for length in (65, 129, 193, 259, 389, 517, 645, 773, 901, 1027, 1283, 1541, 389, 129, 773, 65):
                mask, dense = _mixed_packed_mask(length, device)
                tensors = [
                    torch.randn(1, heads, length, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
                    for heads in (query_heads, kv_heads, kv_heads)
                ]
                references = [t.detach().clone().requires_grad_() for t in tensors]
                if token_major:
                    tensors = [t.squeeze(0).transpose(0, 1).contiguous().detach().requires_grad_() for t in tensors]
                expected, _ = math_sdpa_reference(*references, dense, scaling=128**-0.5)
                actual, auxiliary = flex_backend.flex_attention_forward(
                    _FakeAttentionModule(),
                    *tensors,
                    mask,
                    scaling=128**-0.5,
                    kernel_options={"BACKEND": "FLASH"},
                )
                assert auxiliary is None
                if token_major:
                    expected = expected.squeeze(0)
                torch.testing.assert_close(actual, expected, atol=1e-2, rtol=2e-2)
                gradient = torch.randn_like(actual)
                actual_gradients = torch.autograd.grad(actual, tensors, gradient)
                expected_gradients = torch.autograd.grad(expected, references, gradient)
                for actual_grad, expected_grad in zip(actual_gradients, expected_gradients, strict=True):
                    if token_major:
                        expected_grad = expected_grad.squeeze(0).transpose(0, 1)
                    assert torch.isfinite(actual_grad).all()
                    assert (actual_grad.float() - expected_grad.float()).norm() <= 0.01 * expected_grad.float().norm()
            graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"] - initial_graphs
            record_property("unique_graphs", graphs)
            assert graphs <= 4
    finally:
        torch.compiler.reset()
        flex_backend._fa4._compiled_flex_attention_fa4.cache_clear()


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Flex-FA4 requires CUDA")
def test_dynamic_flex_inside_static_outer_compile_matches_standalone_gradients():
    device = torch.device(get_device_type())
    if flex_backend.get_gpu_compute_capability(device) < 90 or not flex_backend._fa4._flash_attn_cute_available():
        pytest.skip("Flex-FA4 requires SM90+ and flash-attn-4")
    module = _FakeAttentionModule()

    def outer(query, key, value, mask):
        return flex_backend._flex_attention_fa4(
            module,
            query,
            key,
            value,
            mask,
            dropout=0.0,
            scaling=128**-0.5,
            softcap=None,
            kernel_options={"BACKEND": "FLASH"},
        )[0]

    torch.compiler.reset()
    compiled = torch.compile(outer, dynamic=False, fullgraph=True)
    try:
        for length in (259, 389, 517, 1027):
            mask, _ = _mixed_packed_mask(length, device)
            tensors = [
                torch.randn(1, heads, length, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
                for heads in (7, 1, 1)
            ]
            for tensor in tensors:
                torch._dynamo.mark_static(tensor, 1)
                torch._dynamo.mark_static(tensor, 3)
            refs = [t.detach().clone().requires_grad_() for t in tensors]
            actual = compiled(*tensors, mask)
            expected, _ = flex_backend._flex_attention_fa4(
                module,
                *refs,
                mask,
                dropout=0.0,
                scaling=128**-0.5,
                softcap=None,
                kernel_options={"BACKEND": "FLASH"},
            )
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=2e-2)
            gradient = torch.randn_like(actual)
            for actual_grad, expected_grad in zip(
                torch.autograd.grad(actual, tensors, gradient),
                torch.autograd.grad(expected, refs, gradient),
                strict=True,
            ):
                assert (actual_grad.float() - expected_grad.float()).norm() <= 0.01 * expected_grad.float().norm()
    finally:
        torch.compiler.reset()
        flex_backend._fa4._compiled_flex_attention_fa4.cache_clear()


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Flex-FA4 requires CUDA")
def test_flex_fa4_cache_exhaustion_raises_instead_of_materializing_dense_scores():
    from torch._dynamo.exc import FailOnRecompileLimitHit

    device = torch.device(get_device_type())
    if flex_backend.get_gpu_compute_capability(device) < 90 or not flex_backend._fa4._flash_attn_cute_available():
        pytest.skip("Flex-FA4 requires SM90+ and flash-attn-4")
    torch.compiler.reset()
    flex_backend._fa4._compiled_flex_attention_fa4.cache_clear()
    try:
        with torch._dynamo.config.patch(recompile_limit=1):
            for dtype in (torch.bfloat16, torch.float16):
                query = torch.randn(1, 7, 129, 128, device=device, dtype=dtype)
                context = (
                    pytest.raises(FailOnRecompileLimitHit) if dtype == torch.float16 else contextlib.nullcontext()
                )
                with context:
                    flex_backend.flex_attention_forward(
                        _FakeAttentionModule(),
                        query,
                        query,
                        query,
                        _causal_block_mask(129, device, block_size=_flex_block_size(device)),
                        kernel_options={"BACKEND": "FLASH"},
                    )
    finally:
        torch.compiler.reset()
        flex_backend._fa4._compiled_flex_attention_fa4.cache_clear()


@pytest.mark.parametrize(
    ("query_shape", "key_shape", "value_shape", "message"),
    [
        ((4, 8), (4, 8), (4, 8), "same ndim"),
        ((1, 1, 4, 8, 8),) * 3 + ("same ndim",),
        ((8, 4, 8), (1, 2, 8, 8), (8, 2, 8), "same ndim"),
        ((0, 4, 8), (8, 2, 8), (8, 2, 8), "zero dimensions"),
        ((1, 4, 8, 8), (2, 2, 8, 8), (2, 2, 8, 8), "batch sizes"),
        ((8, 4, 8), (8, 2, 8), (8, 1, 8), "K/V head counts"),
        ((8, 4, 8), (9, 2, 8), (8, 2, 8), "K/V head counts"),
        ((8, 4, 8), (8, 2, 16), (8, 2, 8), "Q/K head dimensions"),
        ((8, 3, 8), (8, 2, 8), (8, 2, 8), "GQA requires query heads"),
    ],
)
def test_flex_rejects_invalid_qkv_before_exchange(monkeypatch, query_shape, key_shape, value_shape, message):
    monkeypatch.setattr(
        flex_backend, "should_apply_ulysses", lambda **kwargs: pytest.fail("Invalid QKV must fail before exchange")
    )
    with pytest.raises(ValueError, match=message):
        flex_backend.flex_attention_forward(
            _FakeAttentionModule(),
            torch.randn(query_shape),
            torch.randn(key_shape),
            torch.randn(value_shape),
            _causal_block_mask(8, torch.device("cpu")),
        )


@pytest.mark.parametrize("cross_attention", [False, True])
def test_flex_thd_matches_bhsd_outputs_lse_and_gradients_on_cpu(monkeypatch, cross_attention):
    """Use MATH as the kernel to isolate layout, packing, and autograd behavior."""
    if cross_attention:
        q_length, kv_length = 17, 23
        mask = create_block_mask(
            lambda b, h, q, k: q >= k, B=None, H=None, Q_LEN=q_length, KV_LEN=kv_length, device="cpu"
        )
        dense = torch.arange(q_length)[:, None] >= torch.arange(kv_length)[None, :]
    else:
        q_length = kv_length = 65
        mask, dense = _mixed_packed_mask(q_length, torch.device("cpu"))

    def math_backend(module, query, key, value, attention_mask, **kwargs):
        assert query.ndim == key.ndim == value.ndim == 4
        assert attention_mask is mask
        return math_sdpa_reference(query, key, value, dense, scaling=8**-0.5)

    monkeypatch.setattr(flex_backend, "_flex_attention_triton", math_backend)
    monkeypatch.setattr(flex_backend, "should_apply_ulysses", lambda **kwargs: False)
    # Strided token-major inputs also exercise normalization without a copy.
    thd = [
        torch.randn(length, heads * 2, dim)[:, ::2].requires_grad_()
        for length, heads, dim in ((q_length, 4, 8), (kv_length, 2, 8), (kv_length, 2, 16))
    ]
    bhsd = [tensor.detach().transpose(0, 1).unsqueeze(0).requires_grad_() for tensor in thd]
    actual, lse = flex_backend.flex_attention_forward(_FakeAttentionModule(), *thd, mask)
    expected, expected_lse = flex_backend.flex_attention_forward(_FakeAttentionModule(), *bhsd, mask)
    assert actual.shape == (q_length, 4, 16)
    assert lse.shape == (4, q_length)
    torch.testing.assert_close(actual, expected.squeeze(0))
    torch.testing.assert_close(lse, expected_lse.squeeze(0))
    gradient = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, thd, gradient)
    expected_grads = torch.autograd.grad(expected, bhsd, gradient.unsqueeze(0))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad.squeeze(0).transpose(0, 1))


@pytest.mark.skipif(not _FLEX_COMPILE_AVAILABLE, reason="HF FlexAttention needs triton for compilation")
@pytest.mark.skipif(
    IS_NPU_AVAILABLE,
    reason="Ascend inductor cannot lower CPU flex_attention: Subgraph has no dtype",
)
def test_flex_thd_cpu_forward_matches_math_for_different_qk_lengths():
    query, key, value = (torch.randn(length, heads, 8) for length, heads in ((17, 4), (23, 2), (23, 2)))
    mask = create_block_mask(lambda b, h, q, k: q >= k, B=None, H=None, Q_LEN=17, KV_LEN=23, device="cpu")
    dense = torch.arange(17)[:, None] >= torch.arange(23)[None, :]
    expected, _ = math_sdpa_reference(
        *(tensor.transpose(0, 1).unsqueeze(0) for tensor in (query, key, value)), dense, scaling=8**-0.5
    )
    actual, auxiliary = flex_backend.flex_attention_forward(
        _FakeAttentionModule(), query, key, value, mask, scaling=8**-0.5
    )
    torch.testing.assert_close(actual, expected.squeeze(0), atol=1e-5, rtol=1e-5)
    assert auxiliary is None


@pytest.mark.skipif(not _FLEX_TENSOR_BOUNDS_AVAILABLE, reason="BlockMask needs seq_lengths")
@pytest.mark.parametrize("token_major", [False, True])
@pytest.mark.parametrize("skip_ulysses", [False, True])
def test_flex_flash_bounds_follow_qkv_after_ulysses(monkeypatch, token_major, skip_ulysses):
    """Emulate exchange sizes and exercise the real FLASH bounds wrapper."""
    state = SimpleNamespace(ulysses_group=object(), ulysses_size=2)
    monkeypatch.setattr(flex_backend, "get_parallel_state", lambda: state)
    monkeypatch.setattr(flex_backend, "should_apply_ulysses", lambda *, skip_ulysses: not skip_ulysses)
    exchanges = []

    def prepare(query, key, value, **kwargs):
        exchanges.append("prepare")
        return tuple(
            torch.cat((tensor, tensor), dim=1)[:, :, :heads]
            for tensor, heads in zip((query, key, value), (2, 1, 1), strict=True)
        ) + (4,)

    def restore(output, **kwargs):
        exchanges.append("restore")
        return output[:, :5].repeat_interleave(2, dim=2)

    def raw(query, key, value, *, block_mask, **kwargs):
        assert block_mask.seq_lengths == (10, 14)
        assert query.shape[-2] == 10 and key.shape[-2] == value.shape[-2] == 14
        return query

    monkeypatch.setattr(flex_backend, "prepare_ulysses_qkv", prepare)
    monkeypatch.setattr(flex_backend, "restore_ulysses_output", restore)
    monkeypatch.setattr(flex_backend._fa4, "raw_flex_attention", raw)
    monkeypatch.setattr(
        flex_backend._fa4,
        "_compiled_flex_attention_fa4",
        lambda: flex_backend._fa4._flex_attention_with_tensor_lengths,
    )
    q_length, kv_length = (10, 14) if skip_ulysses else (5, 7)
    tensors = [torch.randn(1, heads, length, 32) for heads, length in ((4, q_length), (2, kv_length), (2, kv_length))]
    if token_major:
        tensors = [tensor.squeeze(0).transpose(0, 1) for tensor in tensors]
    mask = create_block_mask(lambda b, h, q, k: q >= k, B=None, H=None, Q_LEN=10, KV_LEN=14, device="cpu")
    output, auxiliary = flex_backend.flex_attention_forward(
        _FakeAttentionModule(), *tensors, mask, skip_ulysses=skip_ulysses, kernel_options={"BACKEND": "FLASH"}
    )
    assert exchanges == ([] if skip_ulysses else ["prepare", "restore"])
    assert output.shape == ((q_length, 4, 32) if token_major else (1, q_length, 4, 32))
    assert auxiliary is None


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Flex-FA4 requires CUDA")
@pytest.mark.parametrize("token_major", [False, True])
def test_flex_fa4_different_qk_lengths_match_math_gradients(token_major):
    device = torch.device(get_device_type())
    if flex_backend.get_gpu_compute_capability(device) < 90 or not flex_backend._fa4._flash_attn_cute_available():
        pytest.skip("Flex-FA4 requires SM90+ and flash-attn-4")
    mask = create_block_mask(
        lambda b, h, q, k: q >= k,
        B=None,
        H=None,
        Q_LEN=65,
        KV_LEN=129,
        device=device,
        BLOCK_SIZE=_flex_block_size(device),
    )
    dense = torch.arange(65, device=device)[:, None] >= torch.arange(129, device=device)[None, :]
    refs = [
        torch.randn(1, heads, length, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        for heads, length in ((7, 65), (1, 129), (1, 129))
    ]
    inputs = [tensor.detach().clone().requires_grad_() for tensor in refs]
    if token_major:
        inputs = [tensor.squeeze(0).transpose(0, 1).contiguous().detach().requires_grad_() for tensor in inputs]
    torch.compiler.reset()
    flex_backend._fa4._compiled_flex_attention_fa4.cache_clear()
    try:
        expected, _ = math_sdpa_reference(*refs, dense, scaling=128**-0.5)
        if token_major:
            expected = expected.squeeze(0)
        actual, auxiliary = flex_backend.flex_attention_forward(
            _FakeAttentionModule(), *inputs, mask, scaling=128**-0.5, kernel_options={"BACKEND": "FLASH"}
        )
        assert auxiliary is None
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=2e-2)
        gradient = torch.randn_like(actual)
        for actual_grad, expected_grad in zip(
            torch.autograd.grad(actual, inputs, gradient), torch.autograd.grad(expected, refs, gradient), strict=True
        ):
            if token_major:
                expected_grad = expected_grad.squeeze(0).transpose(0, 1)
            assert torch.isfinite(actual_grad).all()
            assert (actual_grad.float() - expected_grad.float()).norm() <= 0.01 * expected_grad.float().norm()
    finally:
        torch.compiler.reset()
        flex_backend._fa4._compiled_flex_attention_fa4.cache_clear()


@pytest.mark.parametrize(
    ("capability", "q_dim", "v_dim", "expected_flash"),
    [
        (90, 128, 256, False),
        (90, 288, 128, False),
        (90, 128, 16, False),
        (100, 256, 256, False),
        (100, 128, 128, True),
        (100, 192, 128, False),
    ],
)
def test_flex_fa4_sparse_shape_eligibility(monkeypatch, capability, q_dim, v_dim, expected_flash):
    monkeypatch.setattr(flex_backend, "get_gpu_compute_capability", lambda device: capability)
    monkeypatch.setattr(flex_backend._fa4, "_flash_attn_cute_available", lambda: True)
    selected = flex_backend.resolve_flex_attention(torch.device("cuda"), {}, head_dim=q_dim, value_head_dim=v_dim)
    assert (selected is flex_backend._flex_attention_fa4) is expected_flash


@pytest.mark.parametrize(
    ("capability", "block_size", "expected_flash"),
    [
        (90, 128, True),
        (90, (128, 128), True),
        (90, (256, 128), False),
        (100, 128, False),
        (100, (128, 128), False),
        (100, (256, 128), True),
        (100, (256, 64), False),
        (110, (128, 128), False),
        (110, (256, 128), True),
    ],
)
@pytest.mark.parametrize("backend", [None, "FLASH", "TRITON"])
def test_flex_fa4_block_size_eligibility(monkeypatch, capability, block_size, expected_flash, backend):
    """Automatic routing checks tiles; explicit backend choice retains native validation."""
    monkeypatch.setattr(flex_backend, "get_gpu_compute_capability", lambda device: capability)
    monkeypatch.setattr(flex_mask_backend, "get_gpu_compute_capability", lambda device: capability)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(flex_backend._fa4, "_flash_attn_cute_available", lambda: True)
    mask = _causal_block_mask(389, "cpu", block_size=block_size)
    # Exercise scalar and tuple representations independently of PyTorch normalization.
    mask.BLOCK_SIZE = block_size
    kernel_options = {} if backend is None else {"BACKEND": backend}
    selected = flex_backend.resolve_flex_attention(
        torch.device("cuda"),
        kernel_options,
        query_dtype=torch.bfloat16,
        head_dim=128,
        value_head_dim=128,
        attention_mask=mask,
    )
    expected_flash = backend == "FLASH" if backend is not None else expected_flash
    expected = flex_backend.FLEX_BACKEND_FLASH if expected_flash else flex_backend.FLEX_BACKEND_TRITON
    assert (selected is flex_backend._flex_attention_fa4) is expected_flash
    assert kernel_options["BACKEND"] == expected


def test_flex_fa4_incomplete_block_tables_retain_triton(monkeypatch):
    monkeypatch.setattr(flex_backend, "get_gpu_compute_capability", lambda device: 90)
    monkeypatch.setattr(flex_backend._fa4, "_flash_attn_cute_available", lambda: True)
    mask = _causal_block_mask(129, torch.device("cpu"))
    mask = BlockMask.from_kv_blocks(
        mask.kv_num_blocks, mask.kv_indices, BLOCK_SIZE=mask.BLOCK_SIZE, mask_mod=mask.mask_mod
    )
    selected = flex_backend.resolve_flex_attention(torch.device("cuda"), {}, attention_mask=mask)
    assert selected is flex_backend._flex_attention_triton


def test_flex_fa4_trainable_bias_retains_triton(monkeypatch):
    monkeypatch.setattr(flex_backend, "get_gpu_compute_capability", lambda device: 90)
    monkeypatch.setattr(flex_backend._fa4, "_flash_attn_cute_available", lambda: True)
    selected = flex_backend.resolve_flex_attention(torch.device("cuda"), {}, trainable_position_bias=True)
    assert selected is flex_backend._flex_attention_triton
    with pytest.raises(ValueError, match="trainable position_bias"):
        flex_backend.resolve_flex_attention(torch.device("cuda"), {"BACKEND": "FLASH"}, trainable_position_bias=True)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="requires CUDA")
def test_flex_trainable_position_bias_forward_and_gradients():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.functional import scaled_dot_product_attention

    generator = torch.Generator(device="cuda").manual_seed(9051)
    query, key, value = (
        torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.bfloat16, generator=generator, requires_grad=True)
        for _ in range(3)
    )
    bias = torch.randn(1, 2, 32, 32, device="cuda", dtype=torch.bfloat16, generator=generator, requires_grad=True)
    # Independent leaves keep the MATH oracle separate from compiled backward buffers.
    reference_qkvb = tuple(t.detach().clone().requires_grad_(True) for t in (query, key, value, bias))
    reference_query, reference_key, reference_value, reference_bias = reference_qkvb
    mask = _causal_block_mask(32, query.device)
    output, _ = flex_backend.flex_attention_forward(
        _FakeAttentionModule(), query, key, value, mask, position_bias=bias, scaling=64**-0.5
    )
    with sdpa_kernel(SDPBackend.MATH):
        causal = torch.ones(32, 32, device="cuda", dtype=torch.bool).tril()
        expected = scaled_dot_product_attention(
            reference_query,
            reference_key,
            reference_value,
            attn_mask=reference_bias.masked_fill(~causal, -float("inf")),
        ).transpose(1, 2)
    gradient = torch.randn(output.shape, device=output.device, dtype=output.dtype, generator=generator)
    actual_grads = torch.autograd.grad(output, (query, key, value, bias), gradient)
    expected_grads = torch.autograd.grad(expected, reference_qkvb, gradient)
    torch.testing.assert_close(output, expected, atol=1e-2, rtol=2e-2)
    for actual, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, reference, atol=1e-2, rtol=3e-2)


@pytest.mark.parametrize("head_dim", [192, 256])
def test_flex_fa4_sm90_incompatible_sparse_tiles_retain_triton(monkeypatch, head_dim):
    monkeypatch.setattr(flex_backend, "get_gpu_compute_capability", lambda device: 90)
    monkeypatch.setattr(flex_backend._fa4, "_flash_attn_cute_available", lambda: True)
    selected = flex_backend.resolve_flex_attention(torch.device("cuda"), {}, head_dim=head_dim, value_head_dim=128)
    assert selected is flex_backend._flex_attention_triton


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_flex_fa4_fp8_retains_triton(monkeypatch, dtype):
    monkeypatch.setattr(flex_backend, "get_gpu_compute_capability", lambda device: 100)
    monkeypatch.setattr(flex_backend._fa4, "_flash_attn_cute_available", lambda: True)
    assert (
        flex_backend.resolve_flex_attention(torch.device("cuda"), {}, query_dtype=dtype)
        is flex_backend._flex_attention_triton
    )


def test_flex_fa4_unvalidated_sparse_architecture_retains_triton(monkeypatch):
    monkeypatch.setattr(flex_backend, "get_gpu_compute_capability", lambda device: 120)
    monkeypatch.setattr(flex_backend._fa4, "_flash_attn_cute_available", lambda: True)
    assert flex_backend.resolve_flex_attention(torch.device("cuda"), {}) is flex_backend._flex_attention_triton
