from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from veomni.lora import LoraLinear, is_veomni_lora_model
from veomni.models import build_foundation_model
from veomni.trainer.base import BaseTrainer
from veomni.trainer.vlm_trainer import (
    VeOmniVLMArguments,
    VLMMDataArguments,
    VLMMModelArguments,
    VLMTrainer,
    _get_vlm_visual_module,
)

from ..tools.training_utils import make_eager_ops_config


_PRODUCTION_CONFIGS = [
    pytest.param(
        "configs/multimodal/qwen3_5_moe/qwen3_5_moe_vl_lora.yaml",
        "tests/toy_config/qwen3_5_moe_toy/config.json",
        id="qwen3_5_moe",
    ),
    pytest.param(
        "configs/multimodal/qwen3_vl/qwen3_vl_moe_lora.yaml",
        "tests/toy_config/qwen3vlmoe_toy/config.json",
        id="qwen3_vl_moe",
    ),
    pytest.param(
        "configs/multimodal/qwen3_omni/qwen3_omni_lora.yaml",
        "tests/toy_config/qwen3omni_toy/config.json",
        id="qwen3_omni",
    ),
]


def _make_args(config_path, lora_config, *, freeze_vit=False, freeze_audio_tower=False):
    args = VeOmniVLMArguments(
        model=VLMMModelArguments(
            config_path=config_path,
            ops_implementation=make_eager_ops_config(),
            lora_config=lora_config,
        ),
        data=VLMMDataArguments(train_path="dummy"),
    )
    args.train.freeze_vit = freeze_vit
    args.train.freeze_audio_tower = freeze_audio_tower
    return args


def _make_trainer(model, args, model_config=None):
    trainer = VLMTrainer.__new__(VLMTrainer)
    trainer.base = BaseTrainer.__new__(BaseTrainer)
    trainer.base.args = args
    trainer.base.model = model
    trainer.base.model_config = model_config or model.config
    return trainer


def _build_meta_trainer(config_path, lora_config, **freeze_kwargs):
    args = _make_args(config_path, lora_config, **freeze_kwargs)
    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="float32",
        init_device="meta",
        ops_implementation=args.model.ops_implementation,
    )
    return _make_trainer(model, args)


def _trainable_lora_names(model):
    return [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and (".lora_A." in name or ".lora_B." in name)
    ]


def test_vlm_lora_preserves_vision_adapters_when_vit_is_frozen():
    trainer = _build_meta_trainer(
        "tests/toy_config/qwen3vl_toy/config.json",
        {"rank": 4, "alpha": 8, "lora_modules": ["q_proj", "qkv"]},
        freeze_vit=True,
    )

    trainer._freeze_model_module()

    model = trainer.base.model
    visual = _get_vlm_visual_module(model)
    visual_lora = [module for module in visual.modules() if isinstance(module, LoraLinear)]
    language_lora = [
        module for name, module in model.named_modules() if "language_model" in name and isinstance(module, LoraLinear)
    ]
    assert is_veomni_lora_model(model) and visual_lora and language_lora
    assert all(not module.base_layer.weight.requires_grad for module in visual_lora + language_lora)
    assert all(param.requires_grad for module in visual_lora for param in module.lora_A.parameters())
    assert all(param.requires_grad for module in language_lora for param in module.lora_A.parameters())
    assert trainer.base.vision_lora_enabled is True
    assert trainer.base.vision_requires_grad is True


def test_llm_only_lora_freezes_entire_vlm_visual_tower():
    trainer = _build_meta_trainer(
        "tests/toy_config/qwen3vl_toy/config.json",
        {"rank": 4, "alpha": 8, "lora_modules": ["q_proj"]},
        freeze_vit=False,
    )

    trainer._freeze_model_module()

    visual = _get_vlm_visual_module(trainer.base.model)
    assert all(not param.requires_grad for param in visual.parameters())
    assert trainer.base.vision_lora_enabled is False
    assert trainer.base.vision_requires_grad is False


@pytest.mark.parametrize("yaml_path,config_path", _PRODUCTION_CONFIGS)
def test_production_multimodal_lora_configs_have_trainable_adapters(yaml_path, config_path):
    lora_config = yaml.safe_load(Path(yaml_path).read_text())["model"]["lora_config"]
    trainer = _build_meta_trainer(config_path, lora_config)

    trainer._freeze_model_module()

    assert _trainable_lora_names(trainer.base.model)
    assert trainer.base.model.base_model.wrapped_dense
    assert trainer.base.model.base_model.wrapped_moe


class _FakeOmniModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.thinker = torch.nn.Module()
        self.thinker.visual = torch.nn.Module()
        self.thinker.visual.proj = torch.nn.Linear(4, 4)
        self.thinker.visual.merger = torch.nn.Linear(4, 4)
        self.thinker.audio_tower = torch.nn.Module()
        self.thinker.audio_tower.proj1 = torch.nn.Linear(4, 4)
        self.text_proj = torch.nn.Linear(4, 4)

    def disable_talker(self):
        pass


def test_omni_lora_ignores_tower_freeze_flags():
    model = _FakeOmniModel()
    args = _make_args(
        "tests/toy_config/qwen3vl_toy/config.json",
        {
            "rank": 2,
            "alpha": 4,
            "lora_modules": ["thinker.visual.proj", "thinker.audio_tower.proj1", "text_proj"],
        },
        freeze_vit=True,
        freeze_audio_tower=True,
    )
    trainer = _make_trainer(model, args, SimpleNamespace(model_type="qwen3_omni_moe"))

    trainer._freeze_model_module()

    wrapped = trainer.base.model
    assert any(param.requires_grad for param in wrapped.text_proj.parameters())
    assert any(param.requires_grad for param in wrapped.thinker.visual.proj.parameters())
    assert any(param.requires_grad for param in wrapped.thinker.audio_tower.proj1.parameters())
    assert all(not param.requires_grad for param in wrapped.thinker.visual.merger.parameters())
    assert trainer.base.vision_lora_enabled is True
    assert trainer.base.vision_requires_grad is True


@pytest.mark.parametrize(
    "lora_config,merger_trainable",
    [
        pytest.param({"rank": 2, "alpha": 4, "lora_modules": ["text_proj"]}, False, id="llm_lora"),
        pytest.param(None, True, id="full_tuning"),
    ],
)
def test_omni_freeze_vit_without_vision_lora(lora_config, merger_trainable):
    trainer = _make_trainer(
        _FakeOmniModel(),
        _make_args(
            "tests/toy_config/qwen3vl_toy/config.json",
            lora_config,
            freeze_vit=True,
            freeze_audio_tower=True,
        ),
        SimpleNamespace(model_type="qwen3_omni_moe"),
    )

    trainer._freeze_model_module()

    visual = trainer.base.model.thinker.visual
    assert all(not param.requires_grad for param in visual.proj.parameters())
    assert all(param.requires_grad is merger_trainable for param in visual.merger.parameters())
    assert all(
        param.requires_grad is merger_trainable for param in trainer.base.model.thinker.audio_tower.proj1.parameters()
    )
    assert trainer.base.vision_lora_enabled is False
    assert trainer.base.vision_requires_grad is merger_trainable
    if lora_config:
        inputs = torch.randn(1, 4)
        assert not visual.proj(inputs).requires_grad
        assert not trainer.base.model.thinker.audio_tower.proj1(inputs).requires_grad
