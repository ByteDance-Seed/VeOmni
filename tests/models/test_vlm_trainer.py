from types import SimpleNamespace

import pytest
import torch

from veomni.models import build_foundation_model
from veomni.models.model_runtime import VeOmniModelRuntime
from veomni.trainer import vlm_trainer
from veomni.trainer.vlm_trainer import (
    VeOmniVLMArguments,
    VLMMDataArguments,
    VLMMModelArguments,
    VLMModelRuntime,
    _get_vlm_visual_module,
)

from ..tools.training_utils import make_eager_ops_config, unbuilt_runtime


_FREEZE_VIT_VLM_CASES = [
    pytest.param("./tests/toy_config/qwen2vl_toy/config.json", id="qwen2_vl"),
    pytest.param("./tests/toy_config/qwen3_5_toy/config.json", id="qwen3_5"),
    pytest.param("./tests/toy_config/qwen3_5_moe_toy/config.json", id="qwen3_5_moe"),
    pytest.param("./tests/toy_config/qwen25vl_toy/config.json", id="qwen2_5_vl"),
    pytest.param("./tests/toy_config/qwen3vl_toy/config.json", id="qwen3_vl"),
    pytest.param("./tests/toy_config/qwen3vlmoe_toy/config.json", id="qwen3_vl_moe"),
    pytest.param("./tests/toy_config/qwen4_exp_toy/config.json", id="qwen4_exp"),
]


class _Gemma3VLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.vision_tower = torch.nn.Linear(4, 4)
        self.model.language_model = torch.nn.Linear(4, 4)


def test_get_vlm_visual_module_finds_gemma3_vision_tower():
    model = _Gemma3VLM()

    assert _get_vlm_visual_module(model) is model.model.vision_tower


def test_freeze_vit_freezes_gemma3_vision_tower(monkeypatch):
    model = _Gemma3VLM()
    runtime = unbuilt_runtime(
        SimpleNamespace(lora_config=None),
        cls=VLMModelRuntime,
        train=SimpleNamespace(freeze_vit=True, freeze_audio_tower=False),
    )
    monkeypatch.setattr(vlm_trainer, "pretty_print_trainable_parameters", lambda _model: None)
    monkeypatch.setattr(vlm_trainer.helper, "print_device_mem_info", lambda _message: None)
    runtime.model = model
    runtime.model_config = SimpleNamespace(model_type="gemma3")

    runtime._freeze_model_module()

    assert all(not param.requires_grad for param in model.model.vision_tower.parameters())
    assert all(param.requires_grad for param in model.model.language_model.parameters())


def test_optimizer_uses_vit_lr_for_gemma3_vision_tower(monkeypatch):
    model = _Gemma3VLM()
    runtime = unbuilt_runtime(
        SimpleNamespace(optimizer=SimpleNamespace(lr=2e-4)),
        cls=VLMModelRuntime,
        train=SimpleNamespace(vit_lr=3e-5),
    )
    captured = {}

    def fake_build_optimizer(_runtime, param_groups=None):
        captured["param_groups"] = param_groups

    monkeypatch.setattr(VeOmniModelRuntime, "_build_optimizer", fake_build_optimizer)
    runtime.model = model
    runtime.model_config = SimpleNamespace(model_type="gemma3")

    runtime._build_optimizer()

    vision_param_ids = {id(param) for param in model.model.vision_tower.parameters()}
    vit_group, other_group = captured["param_groups"]
    assert vit_group["lr"] == runtime.train_args.vit_lr
    assert {id(param) for param in vit_group["params"]} == vision_param_ids
    assert not vision_param_ids.intersection(id(param) for param in other_group["params"])


@pytest.mark.parametrize("freeze_vit", [False, True])
@pytest.mark.parametrize("config_path", _FREEZE_VIT_VLM_CASES)
def test_freeze_vit_on_vlm_model(config_path, freeze_vit):
    # This test only constructs the model on `meta` and verifies freeze
    # behaviour — it never runs forward. Use an all-eager ops config so the
    # build works everywhere: it pins every per-op field (including the
    # Qwen3.5 GatedDeltaNet trio that has no FLA backend on NPU and the
    # GPU-only liger/triton defaults that fail NPU validation). Eager paths
    # that raise only at forward time are fine because this test never
    # forwards.
    ops_implementation = make_eager_ops_config()
    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="float32",
        init_device="meta",
        ops_implementation=ops_implementation,
    )
    visual = _get_vlm_visual_module(model)
    assert visual is not None

    args = VeOmniVLMArguments(
        model=VLMMModelArguments(config_path=config_path, ops_implementation=ops_implementation),
        data=VLMMDataArguments(train_path="dummy"),
    )
    args.train.freeze_vit = freeze_vit
    runtime = unbuilt_runtime(args.model, cls=VLMModelRuntime, train=args.train)
    runtime.model = model
    runtime.model_config = model.config

    runtime._freeze_model_module()

    assert all(param.requires_grad is not freeze_vit for param in visual.parameters())
