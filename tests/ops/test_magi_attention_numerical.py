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

import copy
import gc
import importlib.util
import json
import os
import statistics
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask

from veomni.ops.kernels.attention import flash as flash_backend
from veomni.ops.kernels.attention import flex as flex_backend
from veomni.ops.kernels.attention import magi as magi_backend
from veomni.utils.device import (
    IS_CUDA_AVAILABLE,
    empty_cache,
    get_device_type,
    get_gpu_compute_capability,
    get_torch_device,
    synchronize,
)


_MASK_CASES = ("causal", "full", "bagel_mixed")


def _is_magi_ffa_available() -> bool:
    if not IS_CUDA_AVAILABLE or importlib.util.find_spec("magi_attention") is None:
        return False

    return get_gpu_compute_capability() >= 100 and importlib.util.find_spec("flash_attn_cute") is not None


_MAGI_FFA_AVAILABLE = _is_magi_ffa_available()
_MAGI_FFA_REASON = "MagiAttention requires an NVIDIA SM100+ GPU and the CUTE DSL/JIT backend from the gpu extra"

_QUERY_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 64
_SEQUENCE_LENGTH = 128

_BAGEL_HIDDEN_SIZE = 3584
_BAGEL_QUERY_HEADS = 28
_BAGEL_KV_HEADS = 4
_BAGEL_HEAD_DIM = 128
_BAGEL_SEQUENCE_LENGTH = 4096
_RUN_PROFILE = os.environ.get("RUN_MAGI_ATTENTION_PROFILE") == "1"
_PROFILE_SEQUENCE_LENGTHS = (4096, 8192, 20000)
_PROFILE_ITERATIONS = 5


class _AttentionModule(nn.Module):
    def __init__(self, implementation: str, *, is_causal: bool = False):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation=implementation)
        self.is_causal = is_causal
        self.layer_idx = 0
        self.proj = nn.Linear(1, 1, bias=False)


def _cp1_state(*, ulysses_enabled: bool = False):
    return SimpleNamespace(
        cp_size=1,
        ulysses_enabled=ulysses_enabled,
        ulysses_group=None,
        ulysses_size=1,
    )


def _build_span_splits(sequence_length: int) -> list[int]:
    quarter = sequence_length // 4
    return [quarter, quarter, quarter, sequence_length - 3 * quarter]


def _build_bagel_dense_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
    modes = ("causal", "noise", "full", "causal")
    visible = torch.zeros((sequence_length, sequence_length), device=device, dtype=torch.bool)
    clean_spans: list[tuple[int, int]] = []
    span_start = 0
    for length, mode in zip(_build_span_splits(sequence_length), modes, strict=True):
        span_end = span_start + length
        for clean_start, clean_end in clean_spans:
            visible[span_start:span_end, clean_start:clean_end] = True
        if mode == "causal":
            visible[span_start:span_end, span_start:span_end].fill_(True).tril_()
        else:
            visible[span_start:span_end, span_start:span_end] = True
        if mode != "noise":
            clean_spans.append((span_start, span_end))
        span_start = span_end
    return visible.unsqueeze(0).unsqueeze(0).contiguous()


def _build_dense_mask(mask_case: str, sequence_length: int, device: torch.device) -> torch.Tensor:
    if mask_case == "causal":
        return torch.ones(
            (1, 1, sequence_length, sequence_length),
            device=device,
            dtype=torch.bool,
        ).tril_()
    if mask_case == "full":
        return torch.ones(
            (1, 1, sequence_length, sequence_length),
            device=device,
            dtype=torch.bool,
        )
    if mask_case == "bagel_mixed":
        return _build_bagel_dense_mask(sequence_length, device)
    raise ValueError(f"Unsupported mask case: {mask_case}")


def _build_magi_mask(mask_case: str, sequence_length: int, device: torch.device) -> magi_backend.MagiAttentionMask:
    if mask_case in {"causal", "full"}:
        ranges = torch.tensor([[0, sequence_length]], device=device, dtype=torch.int32)
        attn_type_map = torch.tensor([1], device=device, dtype=torch.int32) if mask_case == "causal" else None
        return magi_backend.MagiAttentionMask(ranges, ranges.clone(), attn_type_map)

    if mask_case != "bagel_mixed":
        raise ValueError(f"Unsupported mask case: {mask_case}")

    modes = ("causal", "noise", "full", "causal")
    q_ranges: list[list[int]] = []
    k_ranges: list[list[int]] = []
    attn_types: list[int] = []
    clean_spans: list[tuple[int, int]] = []
    span_start = 0
    for length, mode in zip(_build_span_splits(sequence_length), modes, strict=True):
        span_end = span_start + length
        for clean_start, clean_end in clean_spans:
            q_ranges.append([span_start, span_end])
            k_ranges.append([clean_start, clean_end])
            attn_types.append(0)

        q_ranges.append([span_start, span_end])
        k_ranges.append([span_start, span_end])
        attn_types.append(1 if mode == "causal" else 0)

        if mode != "noise":
            clean_spans.append((span_start, span_end))
        span_start = span_end

    return magi_backend.MagiAttentionMask(
        q_ranges=torch.tensor(q_ranges, device=device, dtype=torch.int32),
        k_ranges=torch.tensor(k_ranges, device=device, dtype=torch.int32),
        attn_type_map=torch.tensor(attn_types, device=device, dtype=torch.int32),
    )


def _build_flex_mask(dense_mask: torch.Tensor):
    sequence_length = dense_mask.shape[-1]
    return create_block_mask(
        lambda batch_idx, head_idx, query_idx, key_idx: dense_mask[0, 0, query_idx, key_idx],
        B=None,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=dense_mask.device,
        BLOCK_SIZE=128,
    )


def _materialize_magi_mask(
    attention_mask: magi_backend.MagiAttentionMask,
    sequence_length: int,
) -> torch.Tensor:
    visible = torch.zeros((sequence_length, sequence_length), dtype=torch.bool)
    attn_types = (
        torch.zeros(attention_mask.q_ranges.shape[0], dtype=torch.int32)
        if attention_mask.attn_type_map is None
        else attention_mask.attn_type_map.cpu()
    )
    for q_range, k_range, attn_type in zip(
        attention_mask.q_ranges.cpu(),
        attention_mask.k_ranges.cpu(),
        attn_types,
        strict=True,
    ):
        q_start, q_end = q_range.tolist()
        k_start, k_end = k_range.tolist()
        slice_mask = torch.ones((q_end - q_start, k_end - k_start), dtype=torch.bool)
        if attn_type.item() == 1:
            slice_mask.tril_()
        visible[q_start:q_end, k_start:k_end] |= slice_mask
    return visible.unsqueeze(0).unsqueeze(0)


def _clone_qkv(qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    return tuple(tensor.detach().clone().requires_grad_(True) for tensor in qkv)


def _math_sdpa_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dense_mask: torch.Tensor,
    *,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=dense_mask,
            dropout_p=0.0,
            scale=scaling,
            enable_gqa=True,
        ).transpose(1, 2)

    repeat_count = query.shape[1] // key.shape[1]
    expanded_key = key.repeat_interleave(repeat_count, dim=1)
    logits = torch.einsum("bhqd,bhkd->bhqk", query.float(), expanded_key.float()) * scaling
    lse = torch.logsumexp(logits.masked_fill(~dense_mask, -torch.inf), dim=-1)
    return output, lse


def _assert_backend_matches_reference(
    backend_name: str,
    output: torch.Tensor,
    lse: torch.Tensor | None,
    gradients: tuple[torch.Tensor, ...],
    reference_output: torch.Tensor,
    reference_lse: torch.Tensor,
    reference_gradients: tuple[torch.Tensor, ...],
    *,
    dtype: torch.dtype,
) -> None:
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output,
        reference_output,
        rtol=3e-2,
        atol=3e-2,
        msg=lambda message: f"{backend_name} output: {message}",
    )

    if lse is not None:
        assert torch.isfinite(lse).all()
        torch.testing.assert_close(
            lse.float(),
            reference_lse.float(),
            rtol=5e-3,
            atol=3e-2,
            msg=lambda message: f"{backend_name} LSE: {message}",
        )

    gradient_atol = 8e-2 if dtype == torch.bfloat16 else 5e-2
    for name, gradient, reference_gradient in zip(
        ("query", "key", "value"),
        gradients,
        reference_gradients,
        strict=True,
    ):
        assert torch.isfinite(gradient).all()
        assert torch.isfinite(reference_gradient).all()
        torch.testing.assert_close(
            gradient,
            reference_gradient,
            rtol=8e-2,
            atol=gradient_atol,
            msg=lambda message, tensor_name=name: f"{backend_name} {tensor_name} gradient: {message}",
        )


@pytest.mark.parametrize("mask_case", _MASK_CASES)
def test_magi_mask_cases_match_dense_visibility(mask_case):
    device = torch.device("cpu")
    dense_mask = _build_dense_mask(mask_case, _SEQUENCE_LENGTH, device)
    magi_mask = _build_magi_mask(mask_case, _SEQUENCE_LENGTH, device)

    assert torch.equal(_materialize_magi_mask(magi_mask, _SEQUENCE_LENGTH), dense_mask)


@pytest.mark.skipif(not _MAGI_FFA_AVAILABLE, reason=_MAGI_FFA_REASON)
@pytest.mark.parametrize("mask_case", _MASK_CASES)
def test_magi_attention_matches_dense_reference_and_peer_backends(monkeypatch, mask_case):
    device = torch.device(get_device_type())
    dtype = torch.bfloat16
    monkeypatch.setattr(magi_backend, "get_parallel_state", lambda: _cp1_state())
    monkeypatch.setattr(flex_backend, "get_parallel_state", lambda: _cp1_state())
    monkeypatch.setattr(flash_backend, "get_parallel_state", lambda: _cp1_state())

    seed = 9300 + _MASK_CASES.index(mask_case) * 10 + int(dtype == torch.bfloat16)
    generator = torch.Generator(device=device).manual_seed(seed)
    qkv = (
        torch.randn(
            (1, _QUERY_HEADS, _SEQUENCE_LENGTH, _HEAD_DIM),
            device=device,
            dtype=dtype,
            generator=generator,
        ),
        torch.randn(
            (1, _KV_HEADS, _SEQUENCE_LENGTH, _HEAD_DIM),
            device=device,
            dtype=dtype,
            generator=generator,
        ),
        torch.randn(
            (1, _KV_HEADS, _SEQUENCE_LENGTH, _HEAD_DIM),
            device=device,
            dtype=dtype,
            generator=generator,
        ),
    )
    output_gradient = torch.randn(
        (1, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    scaling = _HEAD_DIM**-0.5
    dense_mask = _build_dense_mask(mask_case, _SEQUENCE_LENGTH, device)
    magi_mask = _build_magi_mask(mask_case, _SEQUENCE_LENGTH, device)
    flex_mask = _build_flex_mask(dense_mask)

    reference_qkv = _clone_qkv(qkv)
    reference_output, reference_lse = _math_sdpa_reference(
        *reference_qkv,
        dense_mask,
        scaling=scaling,
    )
    reference_gradients = torch.autograd.grad(reference_output, reference_qkv, output_gradient)

    magi_qkv = _clone_qkv(qkv)
    magi_output, magi_lse = magi_backend.magi_attention_forward(
        _AttentionModule("veomni_magi_attention_with_sp").to(device),
        *magi_qkv,
        magi_mask,
        scaling=scaling,
    )
    magi_gradients = torch.autograd.grad(magi_output, magi_qkv, output_gradient)
    _assert_backend_matches_reference(
        "MagiAttention",
        magi_output,
        magi_lse,
        magi_gradients,
        reference_output,
        reference_lse,
        reference_gradients,
        dtype=dtype,
    )

    flex_qkv = _clone_qkv(qkv)
    flex_output, flex_lse = flex_backend.flex_attention_forward(
        _AttentionModule("veomni_flex_attention_with_sp").to(device),
        *flex_qkv,
        flex_mask,
        scaling=scaling,
        kernel_options={"BACKEND": "TRITON"},
    )
    flex_gradients = torch.autograd.grad(flex_output, flex_qkv, output_gradient)
    _assert_backend_matches_reference(
        "FlexAttention",
        flex_output,
        flex_lse,
        flex_gradients,
        reference_output,
        reference_lse,
        reference_gradients,
        dtype=dtype,
    )

    if mask_case in {"causal", "full"}:
        flash_qkv = _clone_qkv(qkv)
        flash_output, flash_lse = flash_backend.flash_attention_forward(
            _AttentionModule(
                "veomni_flash_attention_2_with_sp",
                is_causal=mask_case == "causal",
            ).to(device=device, dtype=dtype),
            *flash_qkv,
            attention_mask=None,
            scaling=scaling,
        )
        flash_gradients = torch.autograd.grad(flash_output, flash_qkv, output_gradient)
        _assert_backend_matches_reference(
            "FlashAttention",
            flash_output,
            flash_lse,
            flash_gradients,
            reference_output,
            reference_lse,
            reference_gradients,
            dtype=dtype,
        )


class _BagelLikeAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="veomni_magi_attention_with_sp")
        self.q_proj = nn.Linear(_BAGEL_HIDDEN_SIZE, _BAGEL_QUERY_HEADS * _BAGEL_HEAD_DIM, bias=True)
        self.k_proj = nn.Linear(_BAGEL_HIDDEN_SIZE, _BAGEL_KV_HEADS * _BAGEL_HEAD_DIM, bias=True)
        self.v_proj = nn.Linear(_BAGEL_HIDDEN_SIZE, _BAGEL_KV_HEADS * _BAGEL_HEAD_DIM, bias=True)
        self.o_proj = nn.Linear(_BAGEL_QUERY_HEADS * _BAGEL_HEAD_DIM, _BAGEL_HIDDEN_SIZE, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask,
        *,
        backend: str,
        sdpa_backend: SDPBackend = SDPBackend.MATH,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        query = (
            self.q_proj(hidden_states)
            .view(
                batch_size,
                sequence_length,
                _BAGEL_QUERY_HEADS,
                _BAGEL_HEAD_DIM,
            )
            .transpose(1, 2)
        )
        key = (
            self.k_proj(hidden_states)
            .view(
                batch_size,
                sequence_length,
                _BAGEL_KV_HEADS,
                _BAGEL_HEAD_DIM,
            )
            .transpose(1, 2)
        )
        value = (
            self.v_proj(hidden_states)
            .view(
                batch_size,
                sequence_length,
                _BAGEL_KV_HEADS,
                _BAGEL_HEAD_DIM,
            )
            .transpose(1, 2)
        )
        scaling = _BAGEL_HEAD_DIM**-0.5

        if backend == "sdpa":
            enable_gqa = sdpa_backend == SDPBackend.MATH
            if not enable_gqa:
                repeat_count = query.shape[1] // key.shape[1]
                key = key.repeat_interleave(repeat_count, dim=1)
                value = value.repeat_interleave(repeat_count, dim=1)
            with sdpa_kernel(backends=[sdpa_backend]):
                output = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    scale=scaling,
                    enable_gqa=enable_gqa,
                ).transpose(1, 2)
            lse = None
        elif backend == "flex":
            output, lse = flex_backend.flex_attention_forward(
                self,
                query,
                key,
                value,
                attention_mask,
                scaling=scaling,
                kernel_options={"BACKEND": "TRITON"},
            )
        elif backend == "magi":
            output, lse = magi_backend.magi_attention_forward(
                self,
                query,
                key,
                value,
                attention_mask,
                scaling=scaling,
            )
        else:
            raise ValueError(f"Unsupported attention backend: {backend}")

        output = output.reshape(batch_size, sequence_length, _BAGEL_QUERY_HEADS * _BAGEL_HEAD_DIM)
        return self.o_proj(output), lse


@pytest.mark.skipif(not _MAGI_FFA_AVAILABLE, reason=_MAGI_FFA_REASON)
def test_bagel_like_magi_layer_matches_math_sdpa(monkeypatch):
    device = torch.device(get_device_type())
    dtype = torch.bfloat16
    monkeypatch.setattr(magi_backend, "get_parallel_state", lambda: _cp1_state())
    torch.manual_seed(9051)
    reference_layer = _BagelLikeAttentionLayer().to(device=device, dtype=dtype).train()
    magi_layer = copy.deepcopy(reference_layer)
    generator = torch.Generator(device=device).manual_seed(9052)
    hidden_states = torch.randn(
        (1, _BAGEL_SEQUENCE_LENGTH, _BAGEL_HIDDEN_SIZE),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    reference_input = hidden_states.detach().clone().requires_grad_(True)
    magi_input = hidden_states.detach().clone().requires_grad_(True)
    dense_mask = _build_dense_mask("bagel_mixed", _BAGEL_SEQUENCE_LENGTH, device)
    magi_mask = _build_magi_mask("bagel_mixed", _BAGEL_SEQUENCE_LENGTH, device)
    output_gradient = torch.randn(
        (1, _BAGEL_SEQUENCE_LENGTH, _BAGEL_HIDDEN_SIZE),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    parameter_names = tuple(name for name, _ in reference_layer.named_parameters())

    reference_output, _ = reference_layer(reference_input, dense_mask, backend="sdpa")
    reference_gradients = torch.autograd.grad(
        reference_output,
        (reference_input, *reference_layer.parameters()),
        output_gradient,
    )

    magi_output, magi_lse = magi_layer(magi_input, magi_mask, backend="magi")
    torch.testing.assert_close(magi_output, reference_output, rtol=3e-2, atol=3e-2)
    assert magi_lse is not None
    assert torch.isfinite(magi_lse).all()
    magi_gradients = torch.autograd.grad(
        magi_output,
        (magi_input, *magi_layer.parameters()),
        output_gradient,
    )

    gradient_atol = 8e-2 if dtype == torch.bfloat16 else 5e-2
    for name, magi_gradient, reference_gradient in zip(
        ("hidden_states", *parameter_names),
        magi_gradients,
        reference_gradients,
        strict=True,
    ):
        assert torch.isfinite(magi_gradient).all()
        assert torch.isfinite(reference_gradient).all()
        torch.testing.assert_close(
            magi_gradient,
            reference_gradient,
            rtol=8e-2,
            atol=gradient_atol,
            msg=lambda message, gradient_name=name: f"{gradient_name}: {message}",
        )


def _profile_bagel_like_iteration(
    layer: _BagelLikeAttentionLayer,
    hidden_states: torch.Tensor,
    attention_mask,
    output_gradient: torch.Tensor,
    *,
    backend: str,
    sdpa_backend: SDPBackend = SDPBackend.MATH,
) -> tuple[float, bool]:
    device_api = get_torch_device()
    synchronize()
    start = device_api.Event(enable_timing=True)
    end = device_api.Event(enable_timing=True)
    start.record()
    output, lse = layer(
        hidden_states,
        attention_mask,
        backend=backend,
        sdpa_backend=sdpa_backend,
    )
    gradients = torch.autograd.grad(output, (hidden_states, *layer.parameters()), output_gradient)
    end.record()
    synchronize()

    finite = bool(torch.isfinite(output).all().item()) and all(
        bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    )
    if lse is not None:
        finite = finite and bool(torch.isfinite(lse).all().item())
    elapsed_ms = start.elapsed_time(end)
    del output, lse, gradients
    return elapsed_ms, finite


def _profile_bagel_like_backend(sequence_length: int, backend: str) -> dict[str, object]:
    device_api = get_torch_device()
    gc.collect()
    empty_cache()

    device = torch.device(get_device_type())
    dtype = torch.bfloat16
    torch.manual_seed(12001 + sequence_length)
    layer = _BagelLikeAttentionLayer().to(device=device, dtype=dtype).train()
    generator = torch.Generator(device=device).manual_seed(12002 + sequence_length)
    hidden_states = torch.randn(
        (1, sequence_length, _BAGEL_HIDDEN_SIZE),
        device=device,
        dtype=dtype,
        generator=generator,
        requires_grad=True,
    )
    output_gradient = torch.randn(
        hidden_states.shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    if backend == "efficient_attention":
        attention_mask = _build_dense_mask("bagel_mixed", sequence_length, device)
        layer_backend = "sdpa"
        sdpa_backend = SDPBackend.EFFICIENT_ATTENTION
        mask_kind = "dense_bool"
    elif backend == "flex_attention":
        dense_mask = _build_dense_mask("bagel_mixed", sequence_length, device)
        attention_mask = _build_flex_mask(dense_mask)
        layer_backend = "flex"
        sdpa_backend = SDPBackend.MATH
        mask_kind = "native_BlockMask"
        torch.compiler.reset()
    elif backend == "magi_attention":
        attention_mask = _build_magi_mask("bagel_mixed", sequence_length, device)
        layer_backend = "magi"
        sdpa_backend = SDPBackend.MATH
        mask_kind = "MagiAttentionMask"
    else:
        raise ValueError(f"Unsupported profiling backend: {backend}")

    device_api.reset_peak_memory_stats()
    first_iteration_ms, first_finite = _profile_bagel_like_iteration(
        layer,
        hidden_states,
        attention_mask,
        output_gradient,
        backend=layer_backend,
        sdpa_backend=sdpa_backend,
    )
    first_iteration_peak_allocated_gib = device_api.max_memory_allocated() / 1024**3
    post_first_warmup_ms, warmup_finite = _profile_bagel_like_iteration(
        layer,
        hidden_states,
        attention_mask,
        output_gradient,
        backend=layer_backend,
        sdpa_backend=sdpa_backend,
    )

    gc.collect()
    empty_cache()
    device_api.reset_peak_memory_stats()
    steady_state_times_ms = []
    all_finite = first_finite and warmup_finite
    for _ in range(_PROFILE_ITERATIONS):
        elapsed_ms, iteration_finite = _profile_bagel_like_iteration(
            layer,
            hidden_states,
            attention_mask,
            output_gradient,
            backend=layer_backend,
            sdpa_backend=sdpa_backend,
        )
        steady_state_times_ms.append(elapsed_ms)
        all_finite = all_finite and iteration_finite

    return {
        "backend": backend,
        "mask": mask_kind,
        "first_iteration_ms": first_iteration_ms,
        "first_iteration_peak_allocated_gib": first_iteration_peak_allocated_gib,
        "post_first_warmup_ms": post_first_warmup_ms,
        "steady_state_iterations": _PROFILE_ITERATIONS,
        "steady_state_times_ms": steady_state_times_ms,
        "steady_state_median_ms": statistics.median(steady_state_times_ms),
        "peak_allocated_gib": device_api.max_memory_allocated() / 1024**3,
        "all_outputs_and_gradients_finite": all_finite,
    }


@pytest.mark.skipif(not _MAGI_FFA_AVAILABLE, reason=_MAGI_FFA_REASON)
@pytest.mark.skipif(
    not _RUN_PROFILE,
    reason="Set RUN_MAGI_ATTENTION_PROFILE=1 to run the BAGEL-like CUDA profile",
)
@pytest.mark.benchmark
@pytest.mark.parametrize("sequence_length", _PROFILE_SEQUENCE_LENGTHS)
def test_bagel_like_layer_profiles_efficient_sdpa_flex_and_magi(sequence_length):
    device_api = get_torch_device()
    try:
        efficient_result = _profile_bagel_like_backend(sequence_length, "efficient_attention")
        flex_result = _profile_bagel_like_backend(sequence_length, "flex_attention")
        magi_result = _profile_bagel_like_backend(sequence_length, "magi_attention")
    except device_api.OutOfMemoryError as error:
        free_bytes, total_bytes = device_api.mem_get_info()
        pytest.fail(
            json.dumps(
                {
                    "sequence_length": sequence_length,
                    "error": str(error),
                    "allocated_gib": device_api.memory_allocated() / 1024**3,
                    "reserved_gib": device_api.memory_reserved() / 1024**3,
                    "free_gib": free_bytes / 1024**3,
                    "total_gib": total_bytes / 1024**3,
                },
                indent=2,
            )
        )

    result = {
        "sequence_length": sequence_length,
        "dtype": str(torch.bfloat16),
        "batch_size": 1,
        "hidden_size": _BAGEL_HIDDEN_SIZE,
        "query_heads": _BAGEL_QUERY_HEADS,
        "kv_heads": _BAGEL_KV_HEADS,
        "head_dim": _BAGEL_HEAD_DIM,
        "efficient_attention": efficient_result,
        "flex_attention": flex_result,
        "magi_attention": magi_result,
        "flex_speedup_vs_efficient": (
            efficient_result["steady_state_median_ms"] / flex_result["steady_state_median_ms"]
        ),
        "magi_speedup_vs_efficient": (
            efficient_result["steady_state_median_ms"] / magi_result["steady_state_median_ms"]
        ),
        "magi_speedup_vs_flex": flex_result["steady_state_median_ms"] / magi_result["steady_state_median_ms"],
    }
    print(f"BAGEL-like mixed-visibility profile:\n{json.dumps(result, indent=2)}")

    assert efficient_result["all_outputs_and_gradients_finite"]
    assert flex_result["all_outputs_and_gradients_finite"]
    assert magi_result["all_outputs_and_gradients_finite"]
