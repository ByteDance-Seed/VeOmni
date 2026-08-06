from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from veomni.lora import LoraLinear, is_veomni_lora_model
from veomni.lora.state_dict import get_lora_state_dict
from veomni.lora.weight_loading import load_lora_weights
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
        False,
        False,
        id="qwen3_5_moe",
    ),
    pytest.param(
        "configs/multimodal/qwen3_vl/qwen3_vl_moe_lora.yaml",
        "tests/toy_config/qwen3vlmoe_toy/config.json",
        True,
        False,
        id="qwen3_vl_moe",
    ),
    pytest.param(
        "configs/multimodal/qwen3_omni/qwen3_omni_lora.yaml",
        "tests/toy_config/qwen3omni_toy/config.json",
        False,
        False,
        id="qwen3_omni",
    ),
]
_PRODUCTION_CONFIG_PATHS = {case.values[0] for case in _PRODUCTION_CONFIGS}


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


@pytest.mark.parametrize("freeze_vit", [False, True])
def test_vlm_lora_wraps_language_and_preserves_vision_adapters(freeze_vit):
    trainer = _build_meta_trainer(
        "tests/toy_config/qwen3vl_toy/config.json",
        {"rank": 4, "alpha": 8, "lora_modules": ["q_proj", "qkv"]},
        freeze_vit=freeze_vit,
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


@pytest.mark.parametrize("bias", ["none", "all"])
def test_vlm_lora_vision_only_targets_override_freeze_vit(bias):
    trainer = _build_meta_trainer(
        "tests/toy_config/qwen3vl_toy/config.json",
        {"rank": 4, "alpha": 8, "lora_modules": ["qkv"], "bias": bias},
        freeze_vit=True,
    )
    trainer._freeze_model_module()

    visual = _get_vlm_visual_module(trainer.base.model)
    trainable_visual_names = {name for name, param in visual.named_parameters() if param.requires_grad}
    assert _trainable_lora_names(trainer.base.model)
    assert trainer.base.vision_lora_enabled is True
    if bias == "none":
        assert all({"lora_A", "lora_B"} & set(name.split(".")) for name in trainable_visual_names)
    else:
        assert any(name.endswith("bias") for name in trainable_visual_names)


def test_llm_only_lora_freezes_entire_vlm_visual_tower():
    trainer = _build_meta_trainer(
        "tests/toy_config/qwen3vl_toy/config.json",
        {"rank": 4, "alpha": 8, "lora_modules": ["q_proj"]},
        freeze_vit=True,
    )

    trainer._freeze_model_module()

    visual = _get_vlm_visual_module(trainer.base.model)
    assert all(not param.requires_grad for param in visual.parameters())
    assert trainer.base.vision_lora_enabled is False


def test_multimodal_lora_config_inventory_is_audited():
    found = set()
    for path in Path("configs/multimodal").rglob("*.yaml"):
        config = yaml.safe_load(path.read_text())
        if config and config.get("model", {}).get("lora_config"):
            found.add(path.as_posix())
    assert found == _PRODUCTION_CONFIG_PATHS


@pytest.mark.parametrize("yaml_path,config_path,freeze_vit,freeze_audio_tower", _PRODUCTION_CONFIGS)
def test_production_multimodal_lora_configs_have_trainable_adapters(
    yaml_path,
    config_path,
    freeze_vit,
    freeze_audio_tower,
):
    lora_config = yaml.safe_load(Path(yaml_path).read_text())["model"]["lora_config"]
    trainer = _build_meta_trainer(
        config_path,
        lora_config,
        freeze_vit=freeze_vit,
        freeze_audio_tower=freeze_audio_tower,
    )

    trainer._freeze_model_module()

    assert _trainable_lora_names(trainer.base.model)
    assert trainer.base.model.base_model.wrapped_dense
    assert trainer.base.model.base_model.wrapped_moe


def test_vlm_lora_optimizer_and_parallelization_boundary(monkeypatch):
    trainer = _build_meta_trainer(
        "tests/toy_config/qwen3vl_toy/config.json",
        {"rank": 4, "alpha": 8, "lora_modules": ["q_proj", "qkv"]},
    )
    trainer._freeze_model_module()
    captured_optimizer = {}
    captured_parallel = {}

    def fake_build_optimizer(model, **kwargs):
        captured_optimizer.update(kwargs)
        return object()

    def fake_build_parallelize_model(model, **kwargs):
        captured_parallel["model"] = model
        captured_parallel.update(kwargs)
        return model

    monkeypatch.setattr("veomni.trainer.vlm_trainer.build_optimizer", fake_build_optimizer)
    monkeypatch.setattr("veomni.trainer.base.build_parallelize_model", fake_build_parallelize_model)
    trainer._build_optimizer()
    trainer.base._build_parallelized_model()

    assert [group["lr"] for group in captured_optimizer["param_groups"]] == [
        trainer.base.args.train.vit_lr,
        trainer.base.args.train.optimizer.lr,
    ]
    assert is_veomni_lora_model(captured_parallel["model"])
    assert captured_parallel["is_peft_model"] is True


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


@pytest.mark.parametrize("freeze_towers", [False, True])
def test_omni_lora_freeze_flags_gate_tower_adapters(freeze_towers):
    model = _FakeOmniModel()
    args = _make_args(
        "tests/toy_config/qwen3vl_toy/config.json",
        {
            "rank": 2,
            "alpha": 4,
            "lora_modules": ["thinker.visual.proj", "thinker.audio_tower.proj1", "text_proj"],
        },
        freeze_vit=freeze_towers,
        freeze_audio_tower=freeze_towers,
    )
    trainer = _make_trainer(model, args, SimpleNamespace(model_type="qwen3_omni_moe"))

    trainer._freeze_model_module()

    wrapped = trainer.base.model
    assert any(param.requires_grad for param in wrapped.text_proj.parameters())
    assert any(param.requires_grad for param in wrapped.thinker.visual.proj.parameters())
    assert any(param.requires_grad for param in wrapped.thinker.audio_tower.proj1.parameters()) is not freeze_towers
    assert all(not param.requires_grad for param in wrapped.thinker.visual.merger.parameters())
    assert trainer.base.vision_lora_enabled is True


def test_llm_only_lora_freezes_entire_omni_visual_tower():
    model = _FakeOmniModel()
    args = _make_args(
        "tests/toy_config/qwen3vl_toy/config.json",
        {"rank": 2, "alpha": 4, "lora_modules": ["text_proj"]},
        freeze_vit=True,
    )
    trainer = _make_trainer(model, args, SimpleNamespace(model_type="qwen3_omni_moe"))

    trainer._freeze_model_module()

    assert all(not param.requires_grad for param in trainer.base.model.thinker.visual.parameters())
    assert trainer.base.vision_lora_enabled is False


def test_full_tuning_freezes_omni_visual_backbone_but_keeps_merger_trainable():
    model = _FakeOmniModel()
    args = _make_args("tests/toy_config/qwen3vl_toy/config.json", None, freeze_vit=True)
    trainer = _make_trainer(model, args, SimpleNamespace(model_type="qwen3_omni_moe"))

    trainer._freeze_model_module()

    visual = trainer.base.model.thinker.visual
    assert all(not param.requires_grad for param in visual.proj.parameters())
    assert all(param.requires_grad for param in visual.merger.parameters())
    assert trainer.base.vision_lora_enabled is False


class _TinyVLMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = torch.nn.Module()
        self.visual.qkv = torch.nn.Linear(4, 4)
        self.language_model = torch.nn.Module()
        self.language_model.q_proj = torch.nn.Linear(4, 4)
        self.config = SimpleNamespace(model_type="qwen3_vl")


def _build_tiny_trainer(lora_config):
    model = _TinyVLMModel()
    args = _make_args("tests/toy_config/qwen3vl_toy/config.json", lora_config)
    return _make_trainer(model, args)


def test_vlm_lora_adapter_save_reload_preserves_language_and_vision_keys(tmp_path):
    trainer = _build_tiny_trainer({"rank": 2, "alpha": 4, "lora_modules": ["q_proj", "qkv"]})
    trainer._freeze_model_module()
    with torch.no_grad():
        for name, param in trainer.base.model.named_parameters():
            if ".lora_B." in name:
                param.fill_(0.25)

    trainer.base.model.save_pretrained(str(tmp_path))
    expected = get_lora_state_dict(trainer.base.model, config=trainer.base.model.get_lora_config())
    assert any("visual.qkv" in name for name in expected)
    assert any("language_model.q_proj" in name for name in expected)

    reloaded = _build_tiny_trainer({"lora_adapter": str(tmp_path), "is_trainable": True})
    reloaded._freeze_model_module()
    load_lora_weights(reloaded.base.model, str(tmp_path), init_device="cpu")
    actual = get_lora_state_dict(reloaded.base.model, config=reloaded.base.model.get_lora_config())

    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])
