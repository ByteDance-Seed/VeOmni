from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from veomni.lora import LoraLinear, is_veomni_lora_model
from veomni.models import build_foundation_model
from veomni.trainer.vlm_trainer import (
    VeOmniVLMArguments,
    VLMMDataArguments,
    VLMMModelArguments,
    VLMModelRuntime,
    _get_vlm_visual_module,
)

from ..tools.training_utils import make_eager_ops_config, unbuilt_runtime


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


def _make_runtime(model, args, model_config=None):
    runtime = unbuilt_runtime(args.model, cls=VLMModelRuntime, train=args.train)
    runtime.model = model
    runtime.model_config = model_config or model.config
    return runtime


def _build_meta_runtime(config_path, lora_config, **freeze_kwargs):
    args = _make_args(config_path, lora_config, **freeze_kwargs)
    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="float32",
        init_device="meta",
        ops_implementation=args.model.ops_implementation,
    )
    return _make_runtime(model, args)


def _trainable_lora_names(model):
    return [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and (".lora_A." in name or ".lora_B." in name)
    ]


def test_vlm_lora_preserves_vision_adapters_when_vit_is_frozen():
    runtime = _build_meta_runtime(
        "tests/toy_config/qwen3vl_toy/config.json",
        {"rank": 4, "alpha": 8, "lora_modules": ["q_proj", "qkv"]},
        freeze_vit=True,
    )

    runtime.freeze_model()

    # ``is_veomni_lora_model`` is an isinstance check, so unwrap the runtime handle.
    model = runtime.model
    visual = _get_vlm_visual_module(model)
    visual_lora = [module for module in visual.modules() if isinstance(module, LoraLinear)]
    language_lora = [
        module for name, module in model.named_modules() if "language_model" in name and isinstance(module, LoraLinear)
    ]
    assert is_veomni_lora_model(model) and visual_lora and language_lora
    assert all(not module.base_layer.weight.requires_grad for module in visual_lora + language_lora)
    assert all(param.requires_grad for module in visual_lora for param in module.lora_A.parameters())
    assert all(param.requires_grad for module in language_lora for param in module.lora_A.parameters())


def test_llm_only_lora_freezes_entire_vlm_visual_tower():
    runtime = _build_meta_runtime(
        "tests/toy_config/qwen3vl_toy/config.json",
        {"rank": 4, "alpha": 8, "lora_modules": ["q_proj"]},
        freeze_vit=False,
    )

    runtime.freeze_model()

    visual = _get_vlm_visual_module(runtime.model)
    assert all(not param.requires_grad for param in visual.parameters())


@pytest.mark.parametrize("yaml_path,config_path", _PRODUCTION_CONFIGS)
def test_production_multimodal_lora_configs_have_trainable_adapters(yaml_path, config_path):
    lora_config = yaml.safe_load(Path(yaml_path).read_text())["model"]["lora_config"]
    runtime = _build_meta_runtime(config_path, lora_config)

    runtime.freeze_model()

    assert _trainable_lora_names(runtime.model)
    assert runtime.model.base_model.wrapped_dense
    assert runtime.model.base_model.wrapped_moe


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
    runtime = _make_runtime(model, args, SimpleNamespace(model_type="qwen3_omni_moe"))

    runtime.freeze_model()

    wrapped = runtime.model
    assert any(param.requires_grad for param in wrapped.text_proj.parameters())
    assert any(param.requires_grad for param in wrapped.thinker.visual.proj.parameters())
    assert any(param.requires_grad for param in wrapped.thinker.audio_tower.proj1.parameters())
    assert all(not param.requires_grad for param in wrapped.thinker.visual.merger.parameters())


@pytest.mark.parametrize(
    "lora_config,merger_trainable",
    [
        pytest.param({"rank": 2, "alpha": 4, "lora_modules": ["text_proj"]}, False, id="llm_lora"),
        pytest.param(None, True, id="full_tuning"),
    ],
)
def test_omni_freeze_vit_without_vision_lora(lora_config, merger_trainable):
    runtime = _make_runtime(
        _FakeOmniModel(),
        _make_args(
            "tests/toy_config/qwen3vl_toy/config.json",
            lora_config,
            freeze_vit=True,
            freeze_audio_tower=True,
        ),
        SimpleNamespace(model_type="qwen3_omni_moe"),
    )

    runtime.freeze_model()

    visual = runtime.model.thinker.visual
    assert all(not param.requires_grad for param in visual.proj.parameters())
    assert all(param.requires_grad is merger_trainable for param in visual.merger.parameters())
    assert all(
        param.requires_grad is merger_trainable for param in runtime.model.thinker.audio_tower.proj1.parameters()
    )
    if lora_config:
        inputs = torch.randn(1, 4)
        assert not visual.proj(inputs).requires_grad
        assert not runtime.model.thinker.audio_tower.proj1(inputs).requires_grad
