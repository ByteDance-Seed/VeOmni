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
    block_mask = _causal_block_mask(8, query.device)
    module = _FakeAttentionModule()

    with pytest.raises(TypeError, match="requires a BlockMask"):
        veomni_attention.flex_attention_forward(
            module,
            query,
            query,
            query,
            torch.ones(1, 1, 8, 8, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="must be encoded in the supplied BlockMask"):
        veomni_attention.flex_attention_forward(module, query, query, query, block_mask, sliding_window=4)

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
_SEQUENCE_LENGTH = 129
_SAMPLE_SPLITS = [[32, 32, 32, 33]]
_SAMPLE_MODES = [["causal", "noise", "full", "causal"]]


def _build_dense_visibility_mask(device: torch.device) -> torch.Tensor:
    visible = torch.zeros((_SEQUENCE_LENGTH, _SEQUENCE_LENGTH), device=device, dtype=torch.bool)
    sample_start = 0
    for split_lengths, modes in zip(_SAMPLE_SPLITS, _SAMPLE_MODES, strict=True):
        clean_spans = []
        span_start = sample_start
        for length, mode in zip(split_lengths, modes, strict=True):
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
        sample_start = span_start
    return visible.unsqueeze(0).unsqueeze(0).contiguous()


def _build_visibility_metadata(device: torch.device) -> torch.Tensor:
    metadata = torch.full((3, _SEQUENCE_LENGTH), -1, device=device, dtype=torch.int32)
    cursor = 0
    span_id = 0
    for document_id, (split_lengths, modes) in enumerate(zip(_SAMPLE_SPLITS, _SAMPLE_MODES, strict=True)):
        for length, mode in zip(split_lengths, modes, strict=True):
            span_end = cursor + length
            metadata[0, cursor:span_end] = document_id
            if mode != "causal":
                metadata[1, cursor:span_end] = span_id
            if mode == "noise":
                metadata[2, cursor:span_end] = span_id
            cursor = span_end
            span_id += 1
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
        Q_LEN=_SEQUENCE_LENGTH,
        KV_LEN=_SEQUENCE_LENGTH,
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

    def forward(self, hidden_states: torch.Tensor, attention_mask, *, backend: str):
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch_size, sequence_length, _QUERY_HEADS, _HEAD_DIM).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch_size, sequence_length, _KV_HEADS, _HEAD_DIM).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch_size, sequence_length, _KV_HEADS, _HEAD_DIM).transpose(1, 2)
        scale = _HEAD_DIM**-0.5

        if backend == "sdpa":
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                output = scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    scale=scale,
                    enable_gqa=True,
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
    dense_mask = _build_dense_visibility_mask(torch.device("cpu"))
    block_mask = _build_mixed_visibility_block_mask(_build_visibility_metadata(torch.device("cpu")))
    query_idx = torch.arange(_SEQUENCE_LENGTH)[:, None]
    key_idx = torch.arange(_SEQUENCE_LENGTH)[None, :]

    reconstructed = block_mask.mask_mod(0, 0, query_idx, key_idx)

    assert torch.equal(reconstructed, dense_mask[0, 0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention backward requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bagel_like_layer_matches_dense_sdpa(dtype):
    device = torch.device("cuda")
    torch.manual_seed(9051)
    sdpa_layer = _MixedVisibilityAttentionLayer().to(device=device, dtype=dtype).train()
    flex_layer = copy.deepcopy(sdpa_layer)
    generator = torch.Generator(device=device).manual_seed(9052)
    hidden_states = torch.randn(
        (1, _SEQUENCE_LENGTH, _HIDDEN_SIZE),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    sdpa_input = hidden_states.detach().clone().requires_grad_(True)
    flex_input = hidden_states.detach().clone().requires_grad_(True)
    dense_mask = _build_dense_visibility_mask(device)
    block_mask = _build_mixed_visibility_block_mask(_build_visibility_metadata(device))

    sdpa_output, _ = sdpa_layer(sdpa_input, dense_mask, backend="sdpa")
    flex_output, flex_lse = flex_layer(flex_input, block_mask, backend="flex")

    torch.testing.assert_close(flex_output, sdpa_output, rtol=3e-2, atol=3e-2)
    assert flex_lse is not None
    assert torch.isfinite(flex_lse).all()

    output_gradient = torch.randn(flex_output.shape, device=device, dtype=dtype, generator=generator)
    parameter_names = tuple(name for name, _ in sdpa_layer.named_parameters())
    sdpa_gradients = torch.autograd.grad(sdpa_output, (sdpa_input, *sdpa_layer.parameters()), output_gradient)
    flex_gradients = torch.autograd.grad(flex_output, (flex_input, *flex_layer.parameters()), output_gradient)

    gradient_names = ("hidden_states", *parameter_names)
    gradient_atol = 8e-2 if dtype == torch.bfloat16 else 5e-2
    for name, flex_gradient, sdpa_gradient in zip(gradient_names, flex_gradients, sdpa_gradients, strict=True):
        assert torch.isfinite(flex_gradient).all()
        torch.testing.assert_close(
            flex_gradient,
            sdpa_gradient,
            rtol=8e-2,
            atol=gradient_atol,
            msg=lambda message, gradient_name=name: f"{gradient_name}: {message}",
        )
