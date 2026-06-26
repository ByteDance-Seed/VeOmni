import torch
import torch.nn as nn

from veomni.utils.lora_utils import build_lora_key_overrides


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
