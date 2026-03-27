from types import SimpleNamespace

import pytest

from veomni.models import build_foundation_model
from veomni.trainer.vlm_trainer import (
    VeOmniVLMArguments,
    VLMMDataArguments,
    VLMMModelArguments,
    VLMTrainer,
    _get_vlm_visual_module,
)
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to


_FREEZE_VIT_VLM_CASES_TRANSFORMERS_V4 = [
    pytest.param("./tests/toy_config/qwen2vl_toy", id="qwen2_vl"),
    pytest.param("./tests/toy_config/qwen25vl_toy", id="qwen2_5_vl"),
    pytest.param("./tests/toy_config/qwen3vl_toy", id="qwen3_vl"),
]

_FREEZE_VIT_VLM_CASES_TRANSFORMERS_V5 = [
    pytest.param("./tests/toy_config/qwen3_5_toy/config.json", id="qwen3_5"),
    pytest.param("./tests/toy_config/qwen3_5_moe_toy/config.json", id="qwen3_5_moe"),
]

_FREEZE_VIT_VLM_CASES = (
    _FREEZE_VIT_VLM_CASES_TRANSFORMERS_V5
    if is_transformers_version_greater_or_equal_to("5.0.0")
    else _FREEZE_VIT_VLM_CASES_TRANSFORMERS_V4
)


@pytest.mark.parametrize(
    "freeze_vit",
    [
        pytest.param(False, id="freeze_vit_disabled"),
        pytest.param(True, id="freeze_vit_enabled"),
    ],
)
@pytest.mark.parametrize("config_path", _FREEZE_VIT_VLM_CASES)
def test_freeze_vit_on_vlm_model(config_path, freeze_vit):
    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="float32",
        attn_implementation="eager",
        init_device="meta",
    )
    visual = _get_vlm_visual_module(model)
    assert visual is not None

    args = VeOmniVLMArguments(
        model=VLMMModelArguments(config_path=config_path),
        data=VLMMDataArguments(train_path="dummy"),
    )
    args.train.freeze_vit = freeze_vit

    trainer = VLMTrainer.__new__(VLMTrainer)
    trainer.base = SimpleNamespace(
        args=args,
        model=model,
        model_config=model.config,
    )

    trainer._freeze_model_module()

    if freeze_vit:
        assert all(not param.requires_grad for param in visual.parameters())
    else:
        assert all(param.requires_grad for param in visual.parameters())
