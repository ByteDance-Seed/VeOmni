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

"""Magi attention adapter contract and numerical checks vs MATH SDPA."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.ops.attention.attention_cases import clone_qkv, dense_mask, magi_mask, math_sdpa_reference
from tests.ops.tol import ATTN_ATOL, ATTN_BF16_GRAD_ATOL, ATTN_GRAD_ATOL, ATTN_GRAD_RTOL, ATTN_LSE_RTOL, ATTN_RTOL
from veomni.ops.kernels.attention.standard import magi as magi_backend
from veomni.ops.kernels.attention.standard.magi import _kernel as magi_kernel
from veomni.ops.kernels.attention.standard.magi import _metadata as magi_metadata
from veomni.ops.mask import MagiAttentionMask
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


class _FakeAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="veomni_magi_attention")


def _causal_mask(sequence_length: int, device: torch.device | str = "cpu") -> MagiAttentionMask:
    return MagiAttentionMask.from_ranges(
        torch.tensor([[0, sequence_length]], device=device),
        torch.tensor([[0, sequence_length]], device=device),
        torch.tensor([1], device=device),
    )


def _cp1_state(*, ulysses_size: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        cp_size=1,
        ulysses_group=object() if ulysses_size > 1 else None,
        ulysses_size=ulysses_size,
        async_enabled=False,
    )


def _magi_ffa_available() -> bool:
    if not IS_CUDA_AVAILABLE or importlib.util.find_spec("magi_attention") is None:
        return False
    device = torch.device(get_device_type())
    kernel_mode = magi_kernel.get_kernel_mode(device)
    if kernel_mode == magi_kernel.KERNEL_CUTE_JIT:
        return importlib.util.find_spec("flash_attn_cute") is not None
    if kernel_mode != magi_kernel.KERNEL_CUTLASS:
        return False
    try:
        from flash_attn_cute.ffa_fa3 import flash_attn_interface
    except (ImportError, OSError, RuntimeError):
        return False
    return all(
        callable(getattr(flash_attn_interface, name, None)) for name in ("_flash_attn_forward", "_flash_attn_backward")
    )


_MAGI_FFA_AVAILABLE = _magi_ffa_available()
_MAGI_FFA_REASON = (
    "MagiAttention numerical tests require a supported NVIDIA GPU with its CUTLASS overlay or CUTE DSL/JIT backend"
)


@pytest.fixture(autouse=True)
def _isolate_magi_metadata_cache(monkeypatch):
    monkeypatch.setattr(magi_metadata, "_cache_entry", None)


def test_magi_attention_preserves_ffa_layout_and_scale(monkeypatch):
    captured = {}

    def fake_backend(query, key, value, q_ranges, k_ranges, attn_type_map, **kwargs):
        captured.update(
            query=query,
            key=key,
            value=value,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            kwargs=kwargs,
        )
        output = query + 1
        return output, SimpleNamespace(lse=torch.ones(query.shape[:2]))

    monkeypatch.setattr(magi_backend, "get_parallel_state", lambda: _cp1_state())
    monkeypatch.setattr(magi_backend, "_magi_attention_forward", fake_backend)
    query = torch.randn(1, 4, 8, 16)
    key = torch.randn(1, 2, 8, 16)
    value = torch.randn(1, 2, 8, 16)
    attn_type_map = torch.tensor([1], dtype=torch.int32)
    attention_mask = MagiAttentionMask.from_ranges(
        torch.tensor([[0, 8]]),
        torch.tensor([[0, 8]]),
        attn_type_map,
    )

    output, lse = magi_backend.magi_attention_forward(
        _FakeAttentionModule(),
        query,
        key,
        value,
        attention_mask,
        scaling=0.25,
        softcap=30.0,
    )

    assert captured["query"].shape == (8, 4, 16)
    assert captured["key"].shape == (8, 2, 16)
    assert captured["value"].shape == (8, 2, 16)
    assert captured["q_ranges"] is attention_mask.q_ranges
    assert captured["k_ranges"] is attention_mask.k_ranges
    assert captured["attn_type_map"] is attn_type_map
    assert captured["kwargs"] == {"softmax_scale": 0.25, "softcap": 30.0}
    torch.testing.assert_close(output, query.transpose(1, 2) + 1)
    assert lse.shape == (1, 4, 8)


def test_magi_attention_rejects_s_aux_before_ulysses(monkeypatch):
    """Sink softmax is not implemented; fail closed before any collective."""

    def unexpected_call(*args, **kwargs):
        pytest.fail("s_aux reached Magi Ulysses or FA4 handling")

    monkeypatch.setattr(magi_backend, "get_parallel_state", unexpected_call)
    monkeypatch.setattr(magi_backend, "should_apply_ulysses", unexpected_call)
    monkeypatch.setattr(magi_backend, "prepare_ulysses_qkv", unexpected_call)
    monkeypatch.setattr(magi_backend, "_magi_attention_forward", unexpected_call)
    query = torch.randn(1, 4, 8, 16)
    with pytest.raises(ValueError, match="does not implement attention sinks"):
        magi_backend.magi_attention_forward(
            _FakeAttentionModule(),
            query,
            query,
            query,
            _causal_mask(8),
            s_aux=torch.arange(4),
        )


def test_magi_attention_rejects_unsupported_features(monkeypatch):
    monkeypatch.setattr(magi_backend, "get_parallel_state", lambda: _cp1_state())
    query = torch.randn(1, 4, 8, 16)
    with pytest.raises(TypeError, match="requires a MagiAttentionMask"):
        magi_backend.magi_attention_forward(_FakeAttentionModule(), query, query, query, None)
    with pytest.raises(ValueError, match="does not support attention dropout"):
        magi_backend.magi_attention_forward(_FakeAttentionModule(), query, query, query, _causal_mask(8), dropout=0.1)
    with pytest.raises(ValueError, match="encode visibility in MagiAttentionMask"):
        magi_backend.magi_attention_forward(
            _FakeAttentionModule(), query, query, query, _causal_mask(8), sliding_window=4
        )


def test_magi_attention_rejects_batch_and_cp_greater_than_one(monkeypatch):
    query = torch.randn(2, 4, 8, 16)
    monkeypatch.setattr(magi_backend, "get_parallel_state", lambda: _cp1_state())
    with pytest.raises(ValueError, match="requires batch size 1"):
        magi_backend.magi_attention_forward(_FakeAttentionModule(), query, query, query, _causal_mask(8))

    query = query[:1]
    state = _cp1_state()
    state.cp_size = 2
    monkeypatch.setattr(magi_backend, "get_parallel_state", lambda: state)
    with pytest.raises(ValueError, match="supports cp_size == 1"):
        magi_backend.magi_attention_forward(_FakeAttentionModule(), query, query, query, _causal_mask(8))


def test_magi_attention_rejects_invalid_gqa_before_ulysses(monkeypatch):
    monkeypatch.setattr(magi_backend, "get_parallel_state", lambda: _cp1_state(ulysses_size=2))
    monkeypatch.setattr(magi_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: not skip_ulysses)
    monkeypatch.setattr(
        magi_backend,
        "prepare_ulysses_qkv",
        lambda *args, **kwargs: pytest.fail("invalid GQA must fail before Ulysses collectives"),
    )
    query = torch.randn(1, 6, 8, 16)
    key = torch.randn(1, 4, 8, 16)
    value = torch.randn(1, 4, 8, 16)
    with pytest.raises(ValueError, match="GQA requires query heads"):
        magi_backend.magi_attention_forward(_FakeAttentionModule(), query, key, value, _causal_mask(8))


def test_magi_attention_delegates_active_ulysses_to_shared_helpers(monkeypatch):
    state = _cp1_state(ulysses_size=2)
    calls = []

    def fake_prepare(query, key, value, *, group, ulysses_size):
        calls.append(("prepare", query, key, value, group, ulysses_size))
        return (
            torch.randn(1, 16, 2, 16),
            torch.randn(1, 16, 1, 16),
            torch.randn(1, 16, 1, 16),
            4,
        )

    def fake_backend(query, key, value, q_ranges, k_ranges, attn_type_map, **kwargs):
        calls.append(("backend", query, key, value, q_ranges, k_ranges, attn_type_map, kwargs))
        return query, SimpleNamespace(lse=torch.ones(query.shape[:2]))

    def fake_restore(output, *, group):
        calls.append(("restore", output, group))
        return output[:, :8].repeat_interleave(2, dim=2)

    monkeypatch.setattr(magi_backend, "get_parallel_state", lambda: state)
    monkeypatch.setattr(magi_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: not skip_ulysses)
    monkeypatch.setattr(magi_backend, "prepare_ulysses_qkv", fake_prepare)
    monkeypatch.setattr(magi_backend, "_magi_attention_forward", fake_backend)
    monkeypatch.setattr(magi_backend, "restore_ulysses_output", fake_restore)
    query = torch.randn(1, 4, 8, 16)
    key = torch.randn(1, 2, 8, 16)
    value = torch.randn(1, 2, 8, 16)

    output, lse = magi_backend.magi_attention_forward(
        _FakeAttentionModule(),
        query,
        key,
        value,
        _causal_mask(16),
    )

    assert [call[0] for call in calls] == ["prepare", "backend", "restore", "restore"]
    assert calls[0][1].shape == (1, 8, 4, 16)
    assert calls[0][4:] == (state.ulysses_group, 2)
    assert calls[1][1].shape == (16, 2, 16)
    assert output.shape == (1, 8, 4, 16)
    assert lse.shape == (1, 4, 8)


def test_magi_attention_skip_ulysses_uses_local_sequence(monkeypatch):
    monkeypatch.setattr(magi_backend, "get_parallel_state", lambda: _cp1_state(ulysses_size=2))
    monkeypatch.setattr(magi_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: not skip_ulysses)
    monkeypatch.setattr(
        magi_backend,
        "prepare_ulysses_qkv",
        lambda *args, **kwargs: pytest.fail("skip_ulysses must not exchange QKV"),
    )
    query = torch.randn(1, 4, 8, 16)
    captured = {}

    def fake_backend(query, key, value, q_ranges, k_ranges, attn_type_map, **kwargs):
        captured["query_shape"] = tuple(query.shape)
        return query, SimpleNamespace(lse=None)

    monkeypatch.setattr(magi_backend, "_magi_attention_forward", fake_backend)
    magi_backend.magi_attention_forward(
        _FakeAttentionModule(),
        query,
        query[:, :2],
        query[:, :2],
        _causal_mask(8),
        skip_ulysses=True,
    )
    assert captured["query_shape"] == (8, 4, 16)


def test_magi_attention_rejects_global_ranges_when_ulysses_is_off(monkeypatch):
    monkeypatch.setattr(magi_backend, "get_parallel_state", lambda: _cp1_state())
    monkeypatch.setattr(magi_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: False)
    query = torch.randn(1, 4, 4, 16)
    with pytest.raises(ValueError, match="post-exchange query length \\(4\\)"):
        magi_backend.magi_attention_forward(
            _FakeAttentionModule(),
            query,
            query,
            query,
            _causal_mask(8),
        )


def _count_range_bound_reductions(monkeypatch) -> dict[str, int]:
    """Count ``require_all`` launches used by cached Magi range-endpoint checks."""
    counts = {"require_all": 0}
    real_require_all = magi_metadata.require_all

    def counting_require_all(condition, message):
        counts["require_all"] += 1
        return real_require_all(condition, message)

    monkeypatch.setattr(magi_metadata, "require_all", counting_require_all)
    return counts


def _run_mocked_magi_layer(monkeypatch, query: torch.Tensor, attention_mask: MagiAttentionMask) -> None:
    monkeypatch.setattr(magi_backend, "get_parallel_state", lambda: _cp1_state())
    monkeypatch.setattr(
        magi_backend,
        "_magi_attention_forward",
        lambda query, key, value, *args, **kwargs: (query, SimpleNamespace(lse=None)),
    )
    magi_backend.magi_attention_forward(
        _FakeAttentionModule(),
        query,
        query[:, :2],
        query[:, :2],
        attention_mask,
    )


def test_magi_attention_range_bound_cache_hit_miss_and_mutation(monkeypatch):
    """Same mask across layers checks once; in-place range or seq-len changes recheck.

    CUDA saves later ``.all()`` launches, not host synchronizations.
    """
    counts = _count_range_bound_reductions(monkeypatch)
    attention_mask = _causal_mask(8)
    for _ in range(3):
        _run_mocked_magi_layer(monkeypatch, torch.randn(1, 4, 8, 16), attention_mask)
    assert counts["require_all"] == 2

    attention_mask.q_ranges[0, 1] = 7
    _run_mocked_magi_layer(monkeypatch, torch.randn(1, 4, 8, 16), attention_mask)
    _run_mocked_magi_layer(monkeypatch, torch.randn(1, 4, 8, 16), attention_mask)
    assert counts["require_all"] == 4

    _run_mocked_magi_layer(monkeypatch, torch.randn(1, 4, 16, 16), attention_mask)
    assert counts["require_all"] == 6


@pytest.mark.skipif(not _MAGI_FFA_AVAILABLE, reason=_MAGI_FFA_REASON)
@pytest.mark.parametrize(
    ("mask_case", "dtype", "sequence_length", "query_heads", "kv_heads", "head_dim"),
    (
        *(
            pytest.param(mask_case, dtype, 128, 4, 2, 64, id=f"{mask_case}-{dtype_name}")
            for mask_case in ("causal", "full", "2d_mask")
            for dtype, dtype_name in ((torch.bfloat16, "bf16"), (torch.float16, "fp16"))
        ),
        pytest.param("2d_mask", torch.bfloat16, 4096, 28, 4, 128, id="2d-mask-production-bf16"),
    ),
)
def test_magi_attention_matches_math_sdpa(
    monkeypatch,
    mask_case,
    dtype,
    sequence_length,
    query_heads,
    kv_heads,
    head_dim,
):
    device = torch.device(get_device_type())
    monkeypatch.setattr(magi_backend, "get_parallel_state", lambda: _cp1_state())
    kernel_mode, build_flags = magi_kernel.prepare_kernel(device)
    if (
        dtype == torch.float16
        and kernel_mode == magi_kernel.KERNEL_CUTLASS
        and build_flags is not None
        and build_flags.get("FLASHATTENTION_DISABLE_FP16", False)
    ):
        pytest.skip("The installed CUTLASS overlay does not include FP16 kernels.")

    generator = torch.Generator(device=device).manual_seed(9300)
    query, key, value = (
        torch.randn((1, heads, sequence_length, head_dim), device=device, dtype=dtype, generator=generator)
        for heads in (query_heads, kv_heads, kv_heads)
    )
    output_gradient = torch.randn(
        (1, sequence_length, query_heads, head_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    scaling = head_dim**-0.5
    dense = dense_mask(mask_case, sequence_length, device)
    ranges = magi_mask(mask_case, sequence_length, device)

    reference_qkv = clone_qkv(query, key, value)
    reference_output, reference_lse = math_sdpa_reference(*reference_qkv, dense, scaling=scaling)
    reference_gradients = torch.autograd.grad(reference_output, reference_qkv, output_gradient)

    magi_qkv = clone_qkv(query, key, value)
    magi_output, magi_lse = magi_backend.magi_attention_forward(
        _FakeAttentionModule(),
        *magi_qkv,
        ranges,
        scaling=scaling,
    )
    magi_gradients = torch.autograd.grad(magi_output, magi_qkv, output_gradient)

    torch.testing.assert_close(magi_output, reference_output, rtol=ATTN_RTOL, atol=ATTN_ATOL)
    if magi_lse is not None:
        torch.testing.assert_close(magi_lse.float(), reference_lse.float(), rtol=ATTN_LSE_RTOL, atol=ATTN_ATOL)
    gradient_atol = ATTN_BF16_GRAD_ATOL if dtype == torch.bfloat16 else ATTN_GRAD_ATOL
    for name, gradient, reference_gradient in zip(
        ("query", "key", "value"),
        magi_gradients,
        reference_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            gradient,
            reference_gradient,
            rtol=ATTN_GRAD_RTOL,
            atol=gradient_atol,
            msg=lambda message, tensor_name=name: f"{tensor_name}: {message}",
        )
