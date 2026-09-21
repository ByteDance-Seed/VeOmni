# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""CPU unit tests for the native LoRA target mapping / key-override helpers."""

import pytest
import torch
import torch.nn as nn

from veomni.lora.target_mapping import convert_fused_moe_lora_targets, resolve_fused_moe_lora_targets
from veomni.lora.weight_loading import build_lora_key_overrides


class _TargetParameterLoraModel(nn.Module):
    """Mimics the wrapped layout after MoE-LoRA injection (fused param under base_layer)."""

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


def test_convert_fused_moe_lora_targets_rewrites_semantic_names():
    target_modules, target_parameters = convert_fused_moe_lora_targets(
        ["q_proj", "gate_proj", "up_proj", "down_proj"],
        [],
        "model.layers.*.mlp.experts.gate_up_proj",
        "model.layers.*.mlp.experts.down_proj",
    )
    assert target_modules == ["q_proj"]
    assert target_parameters == [
        "model.layers.*.mlp.experts.gate_up_proj",
        "model.layers.*.mlp.experts.down_proj",
    ]


def test_build_lora_key_overrides_handles_fused_expert_base_layer():
    overrides = build_lora_key_overrides(_TargetParameterLoraModel())

    assert overrides["model.layers.0.mlp.experts.gate_up_proj"] == (
        "base_model.model.model.layers.0.mlp.experts.base_layer.gate_up_proj"
    )
    assert overrides["model.layers.0.mlp.experts.down_proj"] == (
        "base_model.model.model.layers.0.mlp.experts.base_layer.down_proj"
    )
    assert "model.layers.0.mlp.experts.lora_A.default" not in overrides


def test_resolve_keeps_dense_modules_without_hook():
    model = _DenseModel()
    lora_config = {"lora_modules": ["gate_proj", "up_proj", "down_proj"]}

    resolved = resolve_fused_moe_lora_targets(model, lora_config)

    assert resolved["lora_modules"] == ["gate_proj", "up_proj", "down_proj"]
    assert resolved.get("target_parameters") is None


def test_resolve_uses_model_owned_fused_mapping():
    model = _FusedMoeModel()
    model._convert_lora_targets_to_parameters = _convert_test_fused_moe_lora_targets
    lora_config = {"lora_modules": ["q_proj", "gate_proj", "up_proj", "down_proj"]}

    resolved = resolve_fused_moe_lora_targets(model, lora_config)

    assert resolved["lora_modules"] == ["q_proj"]
    assert resolved["target_parameters"] == [
        "model.layers.*.mlp.experts.gate_up_proj",
        "model.layers.*.mlp.experts.down_proj",
    ]


def test_resolve_merges_explicit_parameters_with_fused_mapping():
    model = _FusedMoeModel()
    model.extra = nn.Parameter(torch.empty(1))
    model._convert_lora_targets_to_parameters = _convert_test_fused_moe_lora_targets
    lora_config = {"lora_modules": ["q_proj", "gate_proj"], "target_parameters": ["extra"]}

    resolved = resolve_fused_moe_lora_targets(model, lora_config)

    assert resolved["lora_modules"] == ["q_proj"]
    assert resolved["target_parameters"] == ["extra", "model.layers.*.mlp.experts.gate_up_proj"]


def test_resolve_rejects_unmatched_parameter_pattern():
    model = _FusedMoeModel()
    model._convert_lora_targets_to_parameters = _convert_test_fused_moe_lora_targets
    lora_config = {"lora_modules": ["gate_proj"], "target_parameters": ["missing.parameter"]}

    with pytest.raises(ValueError, match="missing.parameter"):
        resolve_fused_moe_lora_targets(model, lora_config)
