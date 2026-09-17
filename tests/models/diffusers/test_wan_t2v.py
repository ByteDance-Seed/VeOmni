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
import torch.nn.functional as F
from diffusers import WanTransformer3DModel as OfficialWanTransformer3DModel

from tests.models.compare import (
    assert_outputs_and_grads_match,
    eager_ops_config,
    ops_config_scope,
)
from tests.models.tiny_configs import tiny_wan_t2v_condition_config as _tiny_condition_config
from tests.models.tiny_configs import tiny_wan_t2v_config as _tiny_config
from veomni.models.diffusers.wan_t2v.wan_transformer.configuration_wan_transformer import (
    WanTransformer3DModelConfig,
)


_OFFICIAL_FORWARD = OfficialWanTransformer3DModel.forward


def _build_ours(config: WanTransformer3DModelConfig, ops: SimpleNamespace | None = None):
    from veomni.models.diffusers.wan_t2v.wan_transformer.modeling_wan_transformer import (
        WanTransformer3DModel,
    )

    with ops_config_scope(ops if ops is not None else eager_ops_config()):
        return WanTransformer3DModel(config)


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


def test_wan_t2v_public_training_forward_matches_official(monkeypatch):
    """Compare ragged-sample predictions and mean sample loss through the public entry."""
    from veomni.models import get_model_class

    torch.manual_seed(0)
    config = _tiny_config()
    official = OfficialWanTransformer3DModel(**config.to_diffuser_dict())
    # Registry resolution installs the production backbone forward. Restore that
    # class-level patch after this test, including when an assertion fails.
    monkeypatch.setattr(OfficialWanTransformer3DModel, "forward", OfficialWanTransformer3DModel.forward)
    with ops_config_scope(eager_ops_config()):
        ours = get_model_class(config)(config)
    ours.load_state_dict(official.state_dict())
    samples = [
        {
            "hidden_states": torch.randn(1, 4, frames, height, width, requires_grad=True),
            "timestep": torch.tensor([timestep]),
            "encoder_hidden_states": torch.randn(1, text_len, 32, requires_grad=True),
        }
        for frames, height, width, text_len, timestep in [(2, 4, 4, 3, 250), (1, 4, 6, 5, 750)]
    ]
    ours_samples = [
        {key: value.detach().clone().requires_grad_(value.requires_grad) for key, value in sample.items()}
        for sample in samples
    ]
    targets = [torch.randn_like(sample["hidden_states"]) + 2 * i for i, sample in enumerate(samples)]
    expected_predictions = []

    def call(model):
        if model is official:
            expected_predictions.extend(_OFFICIAL_FORWARD(model, **sample, return_dict=False)[0] for sample in samples)
            return torch.stack(
                [
                    F.mse_loss(prediction, target)
                    for prediction, target in zip(expected_predictions, targets, strict=True)
                ]
            ).mean()
        output = model(
            latents=None,
            **{key: [sample[key] for sample in ours_samples] for key in samples[0]},
            training_target=targets,
        )
        assert set(output.loss) == {"mse_loss"}
        assert len(output.predictions) == len(samples)
        for actual, expected in zip(output.predictions, expected_predictions, strict=True):
            torch.testing.assert_close(actual, expected)
        return output.loss["mse_loss"]

    assert_outputs_and_grads_match(official, ours, call)
    for actual, expected in zip(ours_samples, samples, strict=True):
        for key in ("hidden_states", "encoder_hidden_states"):
            assert expected[key].grad is not None
            torch.testing.assert_close(actual[key].grad, expected[key].grad)


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
