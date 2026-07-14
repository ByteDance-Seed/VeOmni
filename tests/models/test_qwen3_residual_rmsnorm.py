# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Unit tests for Qwen3's post-attention residual-add + RMSNorm fast path."""

from __future__ import annotations

from contextlib import contextmanager

import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from veomni.models.transformers.qwen3.generated import patched_modeling_qwen3_gpu as qwen3_gpu


@contextmanager
def _temporary_opslot_binding(slot, impl_name: str):
    prev_kernel = slot._kernel
    prev_impl_name = slot._impl_name
    slot.bind(impl_name)
    try:
        yield
    finally:
        slot._kernel = prev_kernel
        slot._impl_name = prev_impl_name


def _make_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=1,
    )


def _eager_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    x_f = x.to(torch.float32)
    variance = x_f.pow(2).mean(-1, keepdim=True)
    x_f = x_f * torch.rsqrt(variance + eps)
    return weight * x_f.to(input_dtype)


def test_qwen3_decoder_layer_residual_add_slot_matches_eager_semantics():
    torch.manual_seed(0)
    config = _make_config()
    config._attn_implementation = "eager"
    layer = qwen3_gpu.Qwen3DecoderLayer(config, layer_idx=0).eval()
    rotary_emb = qwen3_gpu.Qwen3RotaryEmbedding(config)

    hidden_states = torch.randn(2, 8, config.hidden_size, dtype=torch.float32)
    position_ids = torch.arange(hidden_states.shape[1], dtype=torch.long).unsqueeze(0)
    position_embeddings = rotary_emb(hidden_states, position_ids)

    with _temporary_opslot_binding(qwen3_gpu.veomni_rms_norm_residual_add, "eager"):
        eager_out = layer(
            hidden_states.clone(),
            attention_mask=None,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )

    calls = {"count": 0}

    def _fake_residual_add(hidden_states_arg, residual_arg, weight_arg, eps_arg):
        calls["count"] += 1
        updated_residual = residual_arg + hidden_states_arg
        normed = _eager_rms_norm(updated_residual, weight_arg, eps_arg)
        return normed, updated_residual

    prev_kernel = qwen3_gpu.veomni_rms_norm_residual_add._kernel
    prev_impl = qwen3_gpu.veomni_rms_norm_residual_add._impl_name

    try:
        qwen3_gpu.veomni_rms_norm_residual_add._kernel = _fake_residual_add
        qwen3_gpu.veomni_rms_norm_residual_add._impl_name = "test_fake"

        fused_out = layer(
            hidden_states.clone(),
            attention_mask=None,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
    finally:
        qwen3_gpu.veomni_rms_norm_residual_add._kernel = prev_kernel
        qwen3_gpu.veomni_rms_norm_residual_add._impl_name = prev_impl

    assert calls["count"] == 1
    assert torch.allclose(fused_out, eager_out, atol=1e-6, rtol=1e-6)
