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

"""Wan T2V models_kernel consume tests.

Direct-import the staged wrapper. Do not register or use
``build_foundation_model``. Compare a tiny transformer against official
``diffusers.WanTransformer3DModel``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from diffusers import WanTransformer3DModel as OfficialWanTransformer3DModel

from tests.models_kernel.compare import (
    assert_outputs_and_grads_match,
    eager_kernels_config,
)
from veomni.kernels import VeomniKernel
from veomni.kernels.config import get_kernels_config, set_kernels_config
from veomni.models_kernel.diffusers.wan_t2v.wan_transformer.configuration_wan_transformer import (
    WanTransformer3DModelConfig,
)


def _tiny_kwargs() -> dict:
    return {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 4,
        "attention_head_dim": 16,
        "in_channels": 4,
        "out_channels": 4,
        "text_dim": 32,
        "freq_dim": 16,
        "ffn_dim": 64,
        "num_layers": 1,
        "cross_attn_norm": True,
        "qk_norm": "rms_norm_across_heads",
        "eps": 1e-6,
        "rope_max_seq_len": 64,
    }


def _tiny_ours_config() -> WanTransformer3DModelConfig:
    return WanTransformer3DModelConfig(**_tiny_kwargs(), attn_implementation="eager")


def _build_ours(config: WanTransformer3DModelConfig, kernels: SimpleNamespace | None = None):
    from veomni.models_kernel.diffusers.wan_t2v.wan_transformer.modeling_wan_transformer import (
        WanTransformer3DModel,
    )

    previous = get_kernels_config()
    set_kernels_config(kernels if kernels is not None else eager_kernels_config())
    try:
        return WanTransformer3DModel(config)
    finally:
        set_kernels_config(previous)


def _wan_inputs() -> dict[str, torch.Tensor]:
    return {
        "hidden_states": torch.randn(1, 4, 2, 8, 8),
        "timestep": torch.tensor([500], dtype=torch.long),
        "encoder_hidden_states": torch.randn(1, 8, 32),
    }


def test_wan_t2v_constructs_local_kernels():
    model = _build_ours(_tiny_ours_config())
    processor = model.blocks[0].attn1.processor
    assert isinstance(processor.veomni_attn, VeomniKernel)
    assert processor.veomni_attn.kernel == "attention"
    assert processor.veomni_attn.impl == "eager"


def test_wan_t2v_instances_keep_distinct_impls():
    eager = _build_ours(_tiny_ours_config(), eager_kernels_config())
    other_cfg = eager_kernels_config()
    other_cfg.attn_implementation = "sdpa"
    other = _build_ours(_tiny_ours_config(), other_cfg)

    assert eager.blocks[0].attn1.processor.veomni_attn.impl == "eager"
    assert other.blocks[0].attn1.processor.veomni_attn.impl == "sdpa"

    set_kernels_config(other_cfg)
    assert eager.blocks[0].attn1.processor.veomni_attn.impl == "eager"


def test_wan_t2v_eager_matches_official():
    torch.manual_seed(0)
    official = OfficialWanTransformer3DModel(**_tiny_kwargs())
    ours = _build_ours(_tiny_ours_config())
    ours.load_state_dict(official.state_dict())
    inputs = _wan_inputs()

    def call(model):
        output = OfficialWanTransformer3DModel.forward(model, **inputs, return_dict=False)
        return output[0] if isinstance(output, tuple) else output

    assert_outputs_and_grads_match(official, ours, call)


def test_wan_t2v_flash2_kernel_passes_full_sequence_varlen_kwargs():
    kernels = eager_kernels_config()
    kernels.attn_implementation = "veomni_flash_attention_2"
    model = _build_ours(_tiny_ours_config(), kernels).to(dtype=torch.bfloat16)
    attn = model.blocks[0].attn1
    processor = attn.processor
    assert processor.veomni_attn.impl == "veomni_flash_attention_2"
    assert processor._use_flash2

    captured: dict = {}

    def record(_module, query, _key, _value, attention_mask=None, **kwargs):
        captured.update(kwargs)
        return query.transpose(1, 2), None

    processor.veomni_attn = record
    hidden = torch.randn(2, 8, attn.to_q.in_features, dtype=torch.bfloat16)
    processor(attn, hidden, encoder_hidden_states=None, attention_mask=None, rotary_emb=None)

    assert captured["max_length_q"] == 8
    assert captured["max_length_k"] == 8
    assert captured["cu_seq_lens_q"].tolist() == [0, 8, 16]
    assert captured["cu_seq_lens_k"].tolist() == [0, 8, 16]


def test_wan_t2v_eager_skips_full_sequence_varlen_kwargs():
    model = _build_ours(_tiny_ours_config()).to(dtype=torch.bfloat16)
    attn = model.blocks[0].attn1
    captured: dict = {}

    def record(_module, query, _key, _value, attention_mask=None, **kwargs):
        captured.update(kwargs)
        return query.transpose(1, 2), None

    assert not attn.processor._use_flash2
    attn.processor.veomni_attn = record
    hidden = torch.randn(2, 8, attn.to_q.in_features, dtype=torch.bfloat16)
    attn.processor(attn, hidden, encoder_hidden_states=None, attention_mask=None, rotary_emb=None)

    assert "cu_seq_lens_q" not in captured
    assert "max_length_q" not in captured
