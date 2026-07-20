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

from veomni.ops.kernels.attention import flex_attention_forward


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


def _build_block_mask(metadata: torch.Tensor):
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
            output, auxiliary = flex_attention_forward(
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


def test_compact_metadata_block_mask_matches_dense_2d_visibility():
    dense_mask = _build_dense_visibility_mask(torch.device("cpu"))
    block_mask = _build_block_mask(_build_visibility_metadata(torch.device("cpu")))
    query_idx = torch.arange(_SEQUENCE_LENGTH)[:, None]
    key_idx = torch.arange(_SEQUENCE_LENGTH)[None, :]

    reconstructed = block_mask.mask_mod(0, 0, query_idx, key_idx)

    assert dense_mask.shape == (1, 1, _SEQUENCE_LENGTH, _SEQUENCE_LENGTH)
    assert dense_mask.dtype == torch.bool
    assert torch.equal(reconstructed, dense_mask[0, 0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention backward requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_single_layer_mixed_visibility_flex_matches_dense_sdpa(dtype):
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
    block_mask = _build_block_mask(_build_visibility_metadata(device))

    sdpa_output, _ = sdpa_layer(sdpa_input, dense_mask, backend="sdpa")
    flex_output, flex_lse = flex_layer(flex_input, block_mask, backend="flex")

    torch.testing.assert_close(flex_output, sdpa_output, rtol=3e-2, atol=3e-2)
    assert flex_lse is not None
    assert torch.isfinite(flex_lse).all()

    output_gradient = torch.randn(flex_output.shape, device=device, dtype=dtype, generator=generator)
    parameter_names = tuple(name for name, _ in sdpa_layer.named_parameters())
    sdpa_parameters = tuple(sdpa_layer.parameters())
    flex_parameters = tuple(flex_layer.parameters())
    sdpa_gradients = torch.autograd.grad(sdpa_output, (sdpa_input, *sdpa_parameters), output_gradient)
    flex_gradients = torch.autograd.grad(flex_output, (flex_input, *flex_parameters), output_gradient)

    gradient_names = ("hidden_states", *parameter_names)
    gradient_atol = 8e-2 if dtype == torch.bfloat16 else 5e-2
    for name, flex_gradient, sdpa_gradient in zip(
        gradient_names,
        flex_gradients,
        sdpa_gradients,
        strict=True,
    ):
        assert torch.isfinite(flex_gradient).all()
        torch.testing.assert_close(
            flex_gradient,
            sdpa_gradient,
            rtol=8e-2,
            atol=gradient_atol,
            msg=lambda message, gradient_name=name: f"{gradient_name}: {message}",
        )
