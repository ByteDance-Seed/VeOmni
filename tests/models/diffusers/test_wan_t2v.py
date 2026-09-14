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

"""Wan T2V models registry, op-selection, and parity tests.

Compare a tiny transformer against official ``diffusers.WanTransformer3DModel``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from diffusers import WanTransformer3DModel as OfficialWanTransformer3DModel

from tests.models.compare import (
    assert_outputs_and_grads_match,
    eager_ops_config,
)
from tests.models.tiny_configs import tiny_wan_t2v_condition_config as _tiny_condition_config
from tests.models.tiny_configs import tiny_wan_t2v_config as _tiny_config
from veomni.models.diffusers.wan_t2v.wan_transformer.configuration_wan_transformer import (
    WanTransformer3DModelConfig,
)
from veomni.ops import VeomniOp
from veomni.ops.config import get_ops_config, set_ops_config


_OFFICIAL_FORWARD = OfficialWanTransformer3DModel.forward


def _build_ours(config: WanTransformer3DModelConfig, ops: SimpleNamespace | None = None):
    from veomni.models.diffusers.wan_t2v.wan_transformer.modeling_wan_transformer import (
        WanTransformer3DModel,
    )

    previous = get_ops_config()
    set_ops_config(ops if ops is not None else eager_ops_config())
    try:
        return WanTransformer3DModel(config)
    finally:
        set_ops_config(previous)


def _wan_inputs() -> dict[str, torch.Tensor]:
    return {
        "hidden_states": torch.randn(1, 4, 2, 8, 8),
        "timestep": torch.tensor([500], dtype=torch.long),
        "encoder_hidden_states": torch.randn(1, 8, 32),
    }


def test_wan_t2v_configs_roundtrip_through_registry(tmp_path):
    from veomni.models import build_config, get_model_class
    from veomni.models.diffusers.wan_t2v.wan_condition.configuration_wan_condition import (
        WanTransformer3DConditionModelConfig,
    )

    transformer_path = tmp_path / "transformer"
    condition_path = tmp_path / "condition"
    _tiny_config().save_pretrained(transformer_path)
    _tiny_condition_config().save_pretrained(condition_path)

    transformer = build_config(str(transformer_path))
    condition = build_config(str(condition_path))

    assert type(transformer) is WanTransformer3DModelConfig
    assert type(condition) is WanTransformer3DConditionModelConfig
    assert get_model_class(transformer).__name__ == "WanTransformer3DModel"
    assert get_model_class(condition).__name__ == "WanTransformer3DConditionModel"


def test_wan_t2v_condition_builds_from_registered_class_without_assets(monkeypatch):
    from veomni.models import MODELING_REGISTRY
    from veomni.models.diffusers.wan_t2v.wan_condition import modeling_wan_condition

    monkeypatch.setattr(modeling_wan_condition, "get_device_type", lambda: "cpu")
    monkeypatch.setattr(modeling_wan_condition, "get_parallel_state", lambda: SimpleNamespace(dp_rank=0))
    monkeypatch.setattr(modeling_wan_condition.WanTransformer3DConditionModel, "_load_components", lambda self: None)

    model_class = MODELING_REGISTRY["WanTransformer3DConditionModel"]()
    model = model_class._from_config(_tiny_condition_config())

    assert type(model) is modeling_wan_condition.WanTransformer3DConditionModel
    assert model.config.model_type == "WanTransformer3DConditionModel"


def test_wan_t2v_constructs_local_kernels():
    model = _build_ours(_tiny_config())
    processor = model.blocks[0].attn1.processor
    assert isinstance(processor.veomni_attn, VeomniOp)
    assert processor.veomni_attn.op == "attention"
    assert processor.veomni_attn.impl == "eager"


def test_wan_t2v_instances_keep_distinct_impls():
    eager = _build_ours(_tiny_config(), eager_ops_config())
    other_cfg = eager_ops_config()
    other_cfg.attn_implementation = "sdpa"
    other = _build_ours(_tiny_config(), other_cfg)

    assert eager.blocks[0].attn1.processor.veomni_attn.impl == "eager"
    assert other.blocks[0].attn1.processor.veomni_attn.impl == "sdpa"

    set_ops_config(other_cfg)
    assert eager.blocks[0].attn1.processor.veomni_attn.impl == "eager"


def test_wan_t2v_eager_matches_official():
    torch.manual_seed(0)
    config = _tiny_config()
    official = OfficialWanTransformer3DModel(**config.to_diffuser_dict())
    ours = _build_ours(config)
    ours.load_state_dict(official.state_dict())
    inputs = _wan_inputs()

    def call(model):
        output = _OFFICIAL_FORWARD(model, **inputs, return_dict=False)
        return output[0] if isinstance(output, tuple) else output

    assert_outputs_and_grads_match(official, ours, call)


def test_wan_t2v_flash2_kernel_passes_full_sequence_varlen_kwargs(available_nvidia_ops):
    ops = eager_ops_config()
    ops.attn_implementation = "veomni_flash_attention_2"
    model = _build_ours(_tiny_config(), ops).to(dtype=torch.bfloat16)
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
    model = _build_ours(_tiny_config()).to(dtype=torch.bfloat16)
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
