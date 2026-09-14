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

import pytest
import torch
import torch.nn.functional as F
from diffusers import QwenImageTransformer2DModel as OfficialQwenImageTransformer2DModel

from tests.models.compare import assert_outputs_and_grads_match, eager_ops_config
from tests.models.tiny_configs import tiny_qwen_image_condition_config as _tiny_condition_config
from tests.models.tiny_configs import tiny_qwen_image_config as _tiny_config
from veomni.models.diffusers.qwen_image.qwen_image_transformer.configuration_qwen_image_transformer import (
    QwenImageTransformer2DModelConfig,
)
from veomni.ops.config import get_ops_config, set_ops_config


_OFFICIAL_FORWARD = OfficialQwenImageTransformer2DModel.forward


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


@pytest.mark.parametrize("training", [False, True], ids=["inference", "training"])
def test_qwen_image_public_forward_matches_official_without_sp(monkeypatch, training):
    """Compare public inference and ragged supervised batches with upstream math."""
    from veomni.models import get_model_class

    torch.manual_seed(0)
    config = _tiny_config()
    official = OfficialQwenImageTransformer2DModel(**config.to_diffuser_dict())
    monkeypatch.setattr(OfficialQwenImageTransformer2DModel, "forward", OfficialQwenImageTransformer2DModel.forward)
    previous = get_ops_config()
    set_ops_config(eager_ops_config())
    try:
        ours = get_model_class(config)(config)
    finally:
        set_ops_config(previous)
    ours.load_state_dict(official.state_dict())
    samples = [
        {
            "hidden_states": torch.randn(1, height * width, 16, requires_grad=True),
            "encoder_hidden_states": torch.randn(1, text_len, 32, requires_grad=True),
            "encoder_hidden_states_mask": torch.tensor([[True] * (text_len - 1) + [False]]),
            "timestep": torch.tensor([timestep]),
            "img_shapes": [[(1, height, width)]],
        }
        for height, width, text_len, timestep in [(2, 2, 3, 0.25), (2, 3, 5, 0.75)]
    ]
    ours_samples = [
        {
            key: value.detach().clone().requires_grad_(value.requires_grad) if torch.is_tensor(value) else value
            for key, value in sample.items()
        }
        for sample in samples
    ]
    targets = [torch.randn_like(sample["hidden_states"]) + 2 * i for i, sample in enumerate(samples)]
    expected_predictions = []

    def call(model):
        if model is official:
            expected_predictions.extend(_OFFICIAL_FORWARD(model, **sample, return_dict=False)[0] for sample in samples)
            if not training:
                return tuple(expected_predictions)
            return torch.stack(
                [
                    F.mse_loss(prediction, target)
                    for prediction, target in zip(expected_predictions, targets, strict=True)
                ]
            ).mean()
        if training:
            output = model(
                **{key: [sample[key] for sample in ours_samples] for key in samples[0]},
                training_target=targets,
            )
            assert set(output.loss) == {"mse_loss"}
            predictions = output.predictions
        else:
            predictions = [model(**sample, return_dict=False)[0] for sample in ours_samples]
        assert len(predictions) == len(samples)
        for actual, expected in zip(predictions, expected_predictions, strict=True):
            torch.testing.assert_close(actual, expected)
        return output.loss["mse_loss"] if training else tuple(predictions)

    assert_outputs_and_grads_match(official, ours, call)
    for actual, expected in zip(ours_samples, samples, strict=True):
        for key in ("hidden_states", "encoder_hidden_states"):
            assert expected[key].grad is not None
            torch.testing.assert_close(actual[key].grad, expected[key].grad)
