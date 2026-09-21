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
import json
import os
import statistics
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.functional import scaled_dot_product_attention

from veomni.ops import build_ALL_OPS
from veomni.ops.kernels import attention as veomni_attention
from veomni.ops.kernels.attention import flex as flex_backend
from veomni.utils.device import IS_CUDA_AVAILABLE, empty_cache, get_device_type, get_torch_device, synchronize


class _FakeAttentionModule(nn.Module):
    def __init__(self, *, training: bool = False):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="flex_attention")
        self.train(training)


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


def test_flex_module_compute_slot_preserves_the_hf_adapter_contract_and_diagnostics(monkeypatch):
    captured = {}

    def replacement_backend(module, query, key, value, attention_mask, **kwargs):
        captured.update(
            module=module,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            kwargs=kwargs,
        )
        return query.transpose(1, 2) + 1, torch.ones(query.shape[:-1])

    monkeypatch.setattr(flex_backend, "_flex_attention_forward", replacement_backend)
    module = _FakeAttentionModule()
    query = torch.randn(2, 4, 8, 8)
    key = torch.randn(2, 2, 8, 8)
    value = torch.randn(2, 2, 8, 12)
    block_mask = _causal_block_mask(8, query.device)
    kernel_options = {"BACKEND": "TRITON", "BLOCKS_ARE_CONTIGUOUS": True}

    output, auxiliary = veomni_attention.flex_attention_forward(
        module,
        query,
        key,
        value,
        block_mask,
        scaling=0.19,
        softcap=30.0,
        kernel_options=kernel_options,
        output_attentions=False,
    )

    assert captured["module"] is module
    assert captured["query"] is query
    assert captured["key"] is key
    assert captured["value"] is value
    assert captured["attention_mask"] is block_mask
    assert captured["kwargs"] == {
        "dropout": 0.0,
        "scaling": 0.19,
        "softcap": 30.0,
        "kernel_options": kernel_options,
        "output_attentions": False,
    }
    torch.testing.assert_close(output, query.transpose(1, 2) + 1)
    assert auxiliary is not None
    assert dict(build_ALL_OPS())["_flex_attention_forward"] is replacement_backend


def test_flex_attention_cpu_forward_uses_native_block_mask_and_hf_layout():
    sequence_length = 17
    query = torch.randn(2, 4, sequence_length, 8)
    key = torch.randn(2, 2, sequence_length, 8)
    value = torch.randn(2, 2, sequence_length, 8)
    block_mask = _causal_block_mask(sequence_length, query.device)

    output, auxiliary = veomni_attention.flex_attention_forward(
        _FakeAttentionModule(),
        query,
        key,
        value,
        block_mask,
        scaling=0.25,
    )

    assert output.shape == (2, sequence_length, 4, 8)
    assert output.dtype == torch.float32
    assert auxiliary is None
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="FlexAttention backward requires CUDA")
def test_flex_attention_short_query_uses_default_triton_backend():
    sequence_length = 65
    head_dim = 16
    device = torch.device(get_device_type())
    query = torch.randn(1, 2, sequence_length, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(1, 1, sequence_length, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn(1, 1, sequence_length, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)

    output, auxiliary = veomni_attention.flex_attention_forward(
        _FakeAttentionModule(),
        query,
        key,
        value,
        _causal_block_mask(sequence_length, device),
    )
    output.float().square().mean().backward()

    assert output.shape == (1, sequence_length, 2, head_dim)
    assert auxiliary is not None
    assert torch.isfinite(auxiliary).all()
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
        veomni_attention.flex_attention_forward(
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
            veomni_attention.flex_attention_forward(
                module,
                query,
                query,
                query,
                unsupported_mask,
                sliding_window=4,
            )

    head_specific_mask = create_block_mask(
        lambda batch_idx, head_idx, query_idx, key_idx: query_idx >= key_idx,
        B=None,
        H=query.shape[1],
        Q_LEN=query.shape[2],
        KV_LEN=query.shape[2],
        device=query.device,
        BLOCK_SIZE=128,
    )
    monkeypatch.setattr(
        flex_backend,
        "get_parallel_state",
        lambda: SimpleNamespace(ulysses_enabled=True),
    )
    with pytest.raises(ValueError, match="requires a head-broadcast BlockMask"):
        veomni_attention.flex_attention_forward(module, query, query, query, head_specific_mask)


def test_flex_attention_accepts_sliding_window_metadata_with_block_mask(monkeypatch):
    captured = {}

    def fake_backend(module, query, key, value, attention_mask, **kwargs):
        captured["attention_mask"] = attention_mask
        captured["kwargs"] = kwargs
        return query.transpose(1, 2), None

    monkeypatch.setattr(flex_backend, "_flex_attention_forward", fake_backend)
    monkeypatch.setattr(flex_backend, "get_parallel_state", lambda: SimpleNamespace(ulysses_enabled=False))
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

    output, auxiliary = veomni_attention.flex_attention_forward(
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
    state = SimpleNamespace(ulysses_enabled=True, ulysses_group=group, ulysses_size=2)
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
    monkeypatch.setattr(flex_backend, "prepare_ulysses_qkv", fake_prepare)
    monkeypatch.setattr(flex_backend, "slice_ulysses_head_auxiliary", fake_slice)
    monkeypatch.setattr(flex_backend, "_flex_attention_forward", fake_backend)
    monkeypatch.setattr(flex_backend, "restore_ulysses_output", fake_restore)
    query = torch.randn(1, 4, 8, 8)
    key = torch.randn(1, 2, 8, 8)
    value = torch.randn(1, 2, 8, 8)
    block_mask = _causal_block_mask(8, query.device)
    auxiliary = torch.arange(4)

    output, lse = veomni_attention.flex_attention_forward(
        _FakeAttentionModule(),
        query,
        key,
        value,
        block_mask,
        s_aux=auxiliary,
    )

    assert [call[0] for call in calls] == ["prepare", "slice", "backend", "restore", "restore"]
    assert calls[0][1].shape == (1, 8, 4, 8)
    assert calls[0][2].shape == (1, 8, 2, 8)
    assert calls[0][4:] == (group, 2)
    assert calls[1][1] is auxiliary
    assert calls[1][2:] == (4, 2, group)
    assert calls[2][1].shape == (1, 2, 8, 8)
    torch.testing.assert_close(calls[2][-1]["s_aux"], auxiliary[:2])
    assert calls[3][1].shape == (1, 8, 2, 8)
    assert calls[4][1].shape == (1, 8, 2, 1)
    assert output.shape == (1, 8, 2, 8)
    assert lse.shape == (1, 2, 8)


_HIDDEN_SIZE = 3584
_QUERY_HEADS = 28
_KV_HEADS = 4
_HEAD_DIM = 128
_CORRECTNESS_SEQUENCE_LENGTH = 4096
_PROFILE_SEQUENCE_LENGTHS = [4096, 8192, 20000]
_SAMPLE_MODES = ["causal", "noise", "full", "causal"]
_PROFILE_ITERATIONS = 5
_RUN_PROFILE = os.getenv("RUN_FLEX_ATTENTION_PROFILE") == "1"


def _build_sample_splits(sequence_length: int) -> list[int]:
    quarter = sequence_length // len(_SAMPLE_MODES)
    return [quarter, quarter, quarter, sequence_length - 3 * quarter]


def _build_dense_visibility_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
    visible = torch.zeros((sequence_length, sequence_length), device=device, dtype=torch.bool)
    clean_spans = []
    span_start = 0
    for length, mode in zip(_build_sample_splits(sequence_length), _SAMPLE_MODES, strict=True):
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


def _build_visibility_metadata(sequence_length: int, device: torch.device) -> torch.Tensor:
    metadata = torch.full((3, sequence_length), -1, device=device, dtype=torch.int32)
    cursor = 0
    for span_id, (length, mode) in enumerate(zip(_build_sample_splits(sequence_length), _SAMPLE_MODES, strict=True)):
        span_end = cursor + length
        metadata[0, cursor:span_end] = 0
        if mode != "causal":
            metadata[1, cursor:span_end] = span_id
        if mode == "noise":
            metadata[2, cursor:span_end] = span_id
        cursor = span_end
    return metadata


def _build_mixed_visibility_block_mask(metadata: torch.Tensor):
    document_ids, full_span_ids, noise_span_ids = metadata

    def mask_mod(batch_idx, head_idx, query_idx, key_idx):
        same_document = document_ids[query_idx] == document_ids[key_idx]
        causal = query_idx >= key_idx
        same_full_or_noise_span = (full_span_ids[query_idx] >= 0) & (
            full_span_ids[query_idx] == full_span_ids[key_idx]
        )
        foreign_noise_key = (noise_span_ids[key_idx] >= 0) & (noise_span_ids[query_idx] != noise_span_ids[key_idx])
        return same_document & (causal | same_full_or_noise_span) & ~foreign_noise_key

    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=metadata.shape[1],
        KV_LEN=metadata.shape[1],
        device=metadata.device,
        BLOCK_SIZE=128,
    )


class _MixedVisibilityAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="flex_attention")
        self.q_proj = nn.Linear(_HIDDEN_SIZE, _QUERY_HEADS * _HEAD_DIM, bias=True)
        self.k_proj = nn.Linear(_HIDDEN_SIZE, _KV_HEADS * _HEAD_DIM, bias=True)
        self.v_proj = nn.Linear(_HIDDEN_SIZE, _KV_HEADS * _HEAD_DIM, bias=True)
        self.o_proj = nn.Linear(_QUERY_HEADS * _HEAD_DIM, _HIDDEN_SIZE, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask,
        *,
        backend: str,
        sdpa_backend: SDPBackend | None = None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch_size, sequence_length, _QUERY_HEADS, _HEAD_DIM).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch_size, sequence_length, _KV_HEADS, _HEAD_DIM).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch_size, sequence_length, _KV_HEADS, _HEAD_DIM).transpose(1, 2)
        scale = _HEAD_DIM**-0.5

        if backend == "sdpa":
            if sdpa_backend is None:
                raise ValueError("SDPA comparisons must select an explicit SDPBackend.")

            sdpa_key = key
            sdpa_value = value
            enable_gqa = True
            if sdpa_backend == SDPBackend.EFFICIENT_ATTENTION:
                # PyTorch's efficient-attention backend does not accept enable_gqa,
                # so materialize the same grouped K/V heads before kernel dispatch.
                repeat_count = query.shape[1] // key.shape[1]
                sdpa_key = key.repeat_interleave(repeat_count, dim=1)
                sdpa_value = value.repeat_interleave(repeat_count, dim=1)
                enable_gqa = False

            with sdpa_kernel(backends=[sdpa_backend]):
                output = scaled_dot_product_attention(
                    query,
                    sdpa_key,
                    sdpa_value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    scale=scale,
                    enable_gqa=enable_gqa,
                ).transpose(1, 2)
            auxiliary = None
        elif backend == "flex":
            output, auxiliary = veomni_attention.flex_attention_forward(
                self,
                query,
                key,
                value,
                attention_mask,
                scaling=scale,
                kernel_options={"BACKEND": "TRITON"},
            )
        else:
            raise ValueError(f"Unsupported attention backend: {backend}")

        output = output.reshape(batch_size, sequence_length, _QUERY_HEADS * _HEAD_DIM)
        return self.o_proj(output), auxiliary


def test_mixed_visibility_block_mask_matches_dense_mask():
    sequence_length = 129
    device = torch.device("cpu")
    dense_mask = _build_dense_visibility_mask(sequence_length, device)
    block_mask = _build_mixed_visibility_block_mask(_build_visibility_metadata(sequence_length, device))
    query_idx = torch.arange(sequence_length)[:, None]
    key_idx = torch.arange(sequence_length)[None, :]

    reconstructed = block_mask.mask_mod(0, 0, query_idx, key_idx)

    assert torch.equal(reconstructed, dense_mask[0, 0])


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="FlexAttention backward requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bagel_like_layer_matches_math_sdpa(dtype):
    device = torch.device(get_device_type())
    torch.manual_seed(9051)
    sdpa_layer = _MixedVisibilityAttentionLayer().to(device=device, dtype=dtype).train()
    flex_layer = copy.deepcopy(sdpa_layer)
    generator = torch.Generator(device=device).manual_seed(9052)
    hidden_states = torch.randn(
        (1, _CORRECTNESS_SEQUENCE_LENGTH, _HIDDEN_SIZE),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    sdpa_input = hidden_states.detach().clone().requires_grad_(True)
    flex_input = hidden_states.detach().clone().requires_grad_(True)
    dense_mask = _build_dense_visibility_mask(_CORRECTNESS_SEQUENCE_LENGTH, device)
    block_mask = _build_mixed_visibility_block_mask(_build_visibility_metadata(_CORRECTNESS_SEQUENCE_LENGTH, device))

    output_gradient = torch.randn(
        (1, _CORRECTNESS_SEQUENCE_LENGTH, _HIDDEN_SIZE),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    parameter_names = tuple(name for name, _ in sdpa_layer.named_parameters())

    sdpa_output, _ = sdpa_layer(
        sdpa_input,
        dense_mask,
        backend="sdpa",
        sdpa_backend=SDPBackend.MATH,
    )
    assert torch.isfinite(sdpa_output).all()
    sdpa_gradients = torch.autograd.grad(sdpa_output, (sdpa_input, *sdpa_layer.parameters()), output_gradient)

    flex_output, flex_lse = flex_layer(flex_input, block_mask, backend="flex")

    torch.testing.assert_close(flex_output, sdpa_output, rtol=3e-2, atol=3e-2)
    assert flex_lse is not None
    assert torch.isfinite(flex_lse).all()

    flex_gradients = torch.autograd.grad(flex_output, (flex_input, *flex_layer.parameters()), output_gradient)

    gradient_names = ("hidden_states", *parameter_names)
    gradient_atol = 8e-2 if dtype == torch.bfloat16 else 5e-2
    for name, flex_gradient, sdpa_gradient in zip(gradient_names, flex_gradients, sdpa_gradients, strict=True):
        assert torch.isfinite(flex_gradient).all()
        assert torch.isfinite(sdpa_gradient).all()
        torch.testing.assert_close(
            flex_gradient,
            sdpa_gradient,
            rtol=8e-2,
            atol=gradient_atol,
            msg=lambda message, gradient_name=name: f"{gradient_name}: {message}",
        )


def _profile_forward_backward(
    layer: _MixedVisibilityAttentionLayer,
    hidden_states: torch.Tensor,
    attention_mask,
    output_gradient: torch.Tensor,
    *,
    backend: str,
    sdpa_backend: SDPBackend | None = None,
) -> tuple[float, bool]:
    device_api = get_torch_device()
    synchronize()
    start = device_api.Event(enable_timing=True)
    end = device_api.Event(enable_timing=True)
    start.record()
    output, auxiliary = layer(
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
    if auxiliary is not None:
        finite = finite and bool(torch.isfinite(auxiliary).all().item())
    elapsed_ms = start.elapsed_time(end)
    del output, auxiliary, gradients
    return elapsed_ms, finite


def _profile_mixed_visibility_backend(sequence_length: int, *, backend: str) -> dict[str, object]:
    device_api = get_torch_device()
    gc.collect()
    empty_cache()
    device_api.reset_peak_memory_stats()

    device = torch.device(get_device_type())
    dtype = torch.bfloat16
    torch.manual_seed(12001 + sequence_length)
    layer = _MixedVisibilityAttentionLayer().to(device=device, dtype=dtype).train()
    generator = torch.Generator(device=device).manual_seed(12002 + sequence_length)
    hidden_states = torch.randn(
        (1, sequence_length, _HIDDEN_SIZE),
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
        attention_mask = _build_dense_visibility_mask(sequence_length, device)
        layer_backend = "sdpa"
        sdpa_backend = SDPBackend.EFFICIENT_ATTENTION
        mask_kind = "dense_bool"
    elif backend == "flex_attention":
        attention_mask = _build_mixed_visibility_block_mask(_build_visibility_metadata(sequence_length, device))
        layer_backend = "flex"
        sdpa_backend = None
        mask_kind = "native_BlockMask"
        # Reset the in-process compile state so the first iteration captures graph
        # setup. Persistent compiler caches may still shorten this iteration.
        torch.compiler.reset()
    else:
        raise ValueError(f"Unsupported profiling backend: {backend}")

    device_api.reset_peak_memory_stats()
    first_iteration_ms, first_finite = _profile_forward_backward(
        layer,
        hidden_states,
        attention_mask,
        output_gradient,
        backend=layer_backend,
        sdpa_backend=sdpa_backend,
    )
    first_iteration_peak_allocated_gib = device_api.max_memory_allocated() / 1024**3
    post_first_warmup_ms, warmup_finite = _profile_forward_backward(
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
        elapsed_ms, iteration_finite = _profile_forward_backward(
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
        "first_iteration_after_compiler_reset": backend == "flex_attention",
        "first_iteration_peak_allocated_gib": first_iteration_peak_allocated_gib,
        "post_first_warmup_ms": post_first_warmup_ms,
        "steady_state_iterations": _PROFILE_ITERATIONS,
        "steady_state_times_ms": steady_state_times_ms,
        "steady_state_median_ms": statistics.median(steady_state_times_ms),
        "peak_allocated_gib": device_api.max_memory_allocated() / 1024**3,
        "all_outputs_and_gradients_finite": all_finite,
    }


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Attention profiling requires CUDA")
@pytest.mark.skipif(
    not _RUN_PROFILE,
    reason="Set RUN_FLEX_ATTENTION_PROFILE=1 to run the BAGEL-like CUDA profile",
)
@pytest.mark.benchmark
@pytest.mark.parametrize("sequence_length", _PROFILE_SEQUENCE_LENGTHS)
def test_bagel_like_layer_profiles_efficient_sdpa_against_flex(sequence_length):
    device_api = get_torch_device()
    try:
        efficient_result = _profile_mixed_visibility_backend(sequence_length, backend="efficient_attention")
        flex_result = _profile_mixed_visibility_backend(sequence_length, backend="flex_attention")
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
        "hidden_size": _HIDDEN_SIZE,
        "query_heads": _QUERY_HEADS,
        "kv_heads": _KV_HEADS,
        "expanded_sdpa_kv_heads": _QUERY_HEADS,
        "head_dim": _HEAD_DIM,
        "efficient_attention": efficient_result,
        "flex_attention": flex_result,
        "flex_speedup": efficient_result["steady_state_median_ms"] / flex_result["steady_state_median_ms"],
    }
    print(f"BAGEL-like mixed-visibility profile:\n{json.dumps(result, indent=2)}")

    assert efficient_result["all_outputs_and_gradients_finite"]
    assert flex_result["all_outputs_and_gradients_finite"]
