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

"""Qwen-Image models registry, condition, and Diffusers parity tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from diffusers import QwenImageTransformer2DModel as OfficialQwenImageTransformer2DModel

from tests.models.compare import assert_outputs_and_grads_match, eager_ops_config
from tests.models.tiny_configs import tiny_qwen_image_condition_config as _tiny_condition_config
from tests.models.tiny_configs import tiny_qwen_image_config as _tiny_config
from veomni.models.diffusers.qwen_image.qwen_image_transformer.configuration_qwen_image_transformer import (
    QwenImageTransformer2DModelConfig,
)
from veomni.ops.config import get_ops_config, set_ops_config


_OFFICIAL_FORWARD = OfficialQwenImageTransformer2DModel.forward


def _build_ours(config: QwenImageTransformer2DModelConfig):
    from veomni.models.diffusers.qwen_image.qwen_image_transformer.modeling_qwen_image_transformer import (
        QwenImageTransformer2DModel,
    )

    previous = get_ops_config()
    set_ops_config(eager_ops_config())
    try:
        return QwenImageTransformer2DModel(config)
    finally:
        set_ops_config(previous)


def _qwen_image_inputs() -> dict:
    return {
        "hidden_states": torch.randn(1, 4, 16),
        "encoder_hidden_states": torch.randn(1, 3, 32),
        "encoder_hidden_states_mask": torch.ones(1, 3, dtype=torch.bool),
        "timestep": torch.tensor([0.5]),
        "img_shapes": [[(1, 2, 2)]],
    }


def test_qwen_image_configs_roundtrip_through_registry(tmp_path):
    from veomni.models import build_config, get_model_class
    from veomni.models.diffusers.qwen_image.qwen_image_condition.configuration_qwen_image_condition import (
        QwenImageConditionModelConfig,
    )

    transformer_path = tmp_path / "transformer"
    condition_path = tmp_path / "condition"
    _tiny_config().save_pretrained(transformer_path)
    _tiny_condition_config().save_pretrained(condition_path)

    transformer = build_config(str(transformer_path))
    condition = build_config(str(condition_path))

    assert type(transformer) is QwenImageTransformer2DModelConfig
    assert type(condition) is QwenImageConditionModelConfig
    assert get_model_class(transformer).__name__ == "QwenImageTransformer2DModel"
    assert get_model_class(condition).__name__ == "QwenImageConditionModel"


def test_qwen_image_condition_builds_from_registered_class_without_assets(monkeypatch):
    from veomni.models import MODELING_REGISTRY
    from veomni.models.diffusers.qwen_image.qwen_image_condition import modeling_qwen_image_condition

    monkeypatch.setattr(modeling_qwen_image_condition, "get_device_type", lambda: "cpu")
    monkeypatch.setattr(modeling_qwen_image_condition, "get_parallel_state", lambda: SimpleNamespace(dp_rank=0))
    monkeypatch.setattr(modeling_qwen_image_condition.QwenImageConditionModel, "_load_components", lambda self: None)

    model_class = MODELING_REGISTRY["QwenImageConditionModel"]()
    model = model_class._from_config(_tiny_condition_config())

    assert type(model) is modeling_qwen_image_condition.QwenImageConditionModel
    assert model.config.model_type == "QwenImageConditionModel"


def test_qwen_image_matches_official_without_sp():
    from veomni.models.diffusers.qwen_image.qwen_image_transformer.modeling_qwen_image_transformer import (
        QwenImageTransformer2DModel_forward,
    )

    torch.manual_seed(0)
    config = _tiny_config()
    official = OfficialQwenImageTransformer2DModel(**config.to_diffuser_dict())
    ours = _build_ours(config)
    ours.load_state_dict(official.state_dict())
    inputs = _qwen_image_inputs()

    def call(model):
        if model is official:
            output = _OFFICIAL_FORWARD(model, **inputs, return_dict=False)
        else:
            output = QwenImageTransformer2DModel_forward(model, **inputs, return_dict=False)
        return output[0]

    assert_outputs_and_grads_match(official, ours, call)
