import pytest
import torch
import torch.nn as nn

from veomni.utils.lora_utils import build_lora_key_overrides, build_peft_lora_targets, convert_fused_moe_lora_targets


class _TargetParameterPeftModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = nn.Module()
        self.base_model.model.model = nn.Module()
        self.base_model.model.model.layers = nn.ModuleList([nn.Module()])
        experts = nn.Module()
        experts.base_layer = nn.Module()
        experts.base_layer.gate_up_proj = nn.Parameter(torch.empty(2, 4, 3))
        experts.base_layer.register_buffer("down_proj", torch.empty(2, 3, 4))
        experts.lora_A = nn.ParameterDict({"default": nn.Parameter(torch.empty(2, 1))})
        self.base_model.model.model.layers[0].mlp = nn.Module()
        self.base_model.model.model.layers[0].mlp.experts = experts


class _DenseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.gate_proj = nn.Linear(4, 8, bias=False)
        self.model.layers[0].mlp.up_proj = nn.Linear(4, 8, bias=False)
        self.model.layers[0].mlp.down_proj = nn.Linear(8, 4, bias=False)
        self.model.layers[0].custom = nn.Parameter(torch.empty(2, 2))


class _FusedMoeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].self_attn = nn.Module()
        self.model.layers[0].self_attn.q_proj = nn.Linear(4, 4, bias=False)
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.experts = nn.Module()
        self.model.layers[0].mlp.experts.gate_up_proj = nn.Parameter(torch.empty(2, 8, 4))
        self.model.layers[0].mlp.experts.down_proj = nn.Parameter(torch.empty(2, 4, 8))


def _convert_test_fused_moe_lora_targets(_model, lora_modules, target_parameter_patterns):
    return convert_fused_moe_lora_targets(
        lora_modules,
        target_parameter_patterns,
        "model.layers.*.mlp.experts.gate_up_proj",
        "model.layers.*.mlp.experts.down_proj",
    )


def test_build_lora_key_overrides_handles_target_parameters_base_layer():
    model = _TargetParameterPeftModel()

    overrides = build_lora_key_overrides(model)

    assert overrides["model.layers.0.mlp.experts.gate_up_proj"] == (
        "base_model.model.model.layers.0.mlp.experts.base_layer.gate_up_proj"
    )
    assert overrides["model.layers.0.mlp.experts.down_proj"] == (
        "base_model.model.model.layers.0.mlp.experts.base_layer.down_proj"
    )
    assert "model.layers.0.mlp.experts.lora_A.default" not in overrides


def test_build_peft_lora_targets_keeps_dense_modules_with_explicit_parameters():
    model = _DenseModel()
    lora_config = {
        "lora_modules": ["gate_proj", "up_proj", "down_proj"],
        "target_parameters": ["model.layers.0.custom"],
    }

    target_modules, target_parameters = build_peft_lora_targets(model, lora_config)

    assert target_modules == ["gate_proj", "up_proj", "down_proj"]
    assert target_parameters == ["model.layers.0.custom"]


def test_build_peft_lora_targets_uses_model_owned_fused_mapping():
    model = _FusedMoeModel()
    model._convert_lora_targets_to_parameters = _convert_test_fused_moe_lora_targets
    lora_config = {
        "lora_modules": ["q_proj", "gate_proj", "up_proj", "down_proj"],
    }

    target_modules, target_parameters = build_peft_lora_targets(model, lora_config)

    assert target_modules == ["q_proj"]
    assert target_parameters == [
        "model.layers.0.mlp.experts.down_proj",
        "model.layers.0.mlp.experts.gate_up_proj",
    ]


def test_build_peft_lora_targets_merges_explicit_parameters_with_fused_mapping():
    model = _FusedMoeModel()
    model.extra = nn.Parameter(torch.empty(1))
    model._convert_lora_targets_to_parameters = _convert_test_fused_moe_lora_targets
    lora_config = {
        "lora_modules": ["q_proj", "gate_proj"],
        "target_parameters": ["extra"],
    }

    target_modules, target_parameters = build_peft_lora_targets(model, lora_config)

    assert target_modules == ["q_proj"]
    assert target_parameters == ["extra", "model.layers.0.mlp.experts.gate_up_proj"]


def test_build_peft_lora_targets_rejects_unmatched_parameter_pattern():
    model = _DenseModel()
    lora_config = {
        "lora_modules": ["gate_proj"],
        "target_parameters": ["missing.parameter"],
    }

    with pytest.raises(ValueError, match="missing.parameter"):
        build_peft_lora_targets(model, lora_config)
