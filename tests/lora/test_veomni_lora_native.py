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
"""CPU single-process tests for the VeOmni-native dense LoRA stack (Phase 1).

No distributed / GPU / HF-model build required — a tiny ``nn.Module`` exercises:

* ``VeOmniLoraConfig`` YAML <-> config <-> ``adapter_config.json`` round-trips.
* Dense injection: PEFT-identical FQNs, trainable-only-LoRA, no-op-at-init.
* ``get_lora_state_dict`` PEFT on-disk key format.
* ``save_pretrained`` -> ``from_pretrained`` + native weight load round-trip.
* ``merge_and_unload`` numerical equivalence.
* rank/alpha patterns, exclude_modules, rslora scaling.

``test_peft_bidirectional_interop`` additionally asserts cross-compatibility
with stock ``peft`` when it is installed (dev/test-only dependency).
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from veomni.lora import VeOmniLoraConfig, VeOmniLoraModel
from veomni.lora.layers import LoraLinear
from veomni.lora.state_dict import get_lora_state_dict, load_adapter_state_dict
from veomni.lora.weight_loading import load_lora_weights


torch.manual_seed(0)


class Toy(nn.Module):
    """Tiny 3-linear MLP; ``lin1`` / ``lin2`` are LoRA targets, ``lin3`` is not."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.lin1 = nn.Linear(dim, dim, bias=False)
        self.lin2 = nn.Linear(dim, dim, bias=False)
        self.lin3 = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin3(torch.relu(self.lin2(torch.relu(self.lin1(x)))))


def _base_config(**overrides) -> VeOmniLoraConfig:
    kwargs = dict(r=8, lora_alpha=16, target_modules=["lin1", "lin2"])
    kwargs.update(overrides)
    return VeOmniLoraConfig(**kwargs)


def _randomize_lora_b(model: VeOmniLoraModel) -> None:
    """Simulate a trained adapter: make B non-zero so the delta is observable."""
    with torch.no_grad():
        for _, module in model.named_modules():
            if isinstance(module, LoraLinear):
                for lin in module.lora_B.values():
                    lin.weight.normal_(0.0, 0.02)


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


def test_config_yaml_roundtrip():
    cfg = VeOmniLoraConfig.from_yaml(
        {"rank": 8, "alpha": 16, "lora_modules": ["q_proj", "v_proj"], "lora_dropout": 0.05, "use_rslora": True}
    )
    assert cfg.r == 8 and cfg.lora_alpha == 16
    assert cfg.target_modules == ["q_proj", "v_proj"]
    assert cfg.lora_dropout == 0.05 and cfg.use_rslora is True
    assert cfg.moe_mode is None and cfg.has_moe is False


def test_config_yaml_moe_mode():
    indep = VeOmniLoraConfig.from_yaml({"rank": 4, "alpha": 8, "target_parameters": ["m.experts.gate_up_proj"]})
    assert indep.moe_mode == "independent" and indep.has_moe
    shared = VeOmniLoraConfig.from_yaml(
        {"rank": 4, "alpha": 8, "target_parameters": ["m.experts.gate_up_proj"], "share_expert_lora": True}
    )
    assert shared.moe_mode == "shared"


def test_adapter_config_json_roundtrip(tmp_path):
    cfg = _base_config(lora_dropout=0.1, use_rslora=True, rank_pattern={r".*lin2": 16})
    cfg.save_pretrained(str(tmp_path))
    loaded = VeOmniLoraConfig.from_pretrained(str(tmp_path))
    assert loaded.r == cfg.r
    assert loaded.lora_alpha == cfg.lora_alpha
    assert loaded.target_modules == cfg.target_modules
    assert loaded.lora_dropout == cfg.lora_dropout
    assert loaded.use_rslora == cfg.use_rslora
    assert loaded.rank_pattern == cfg.rank_pattern

    import json

    with open(tmp_path / "adapter_config.json") as f:
        raw = json.load(f)
    assert raw["peft_type"] == "LORA"
    assert "veomni_lora" in raw


# ----------------------------------------------------------------------------
# Injection
# ----------------------------------------------------------------------------


def test_dense_injection_structure():
    model = VeOmniLoraModel(Toy(), _base_config())
    names = [n for n, _ in model.named_parameters()]

    assert all(n.startswith("base_model.model.") for n in names)
    assert "base_model.model.lin1.base_layer.weight" in names
    assert "base_model.model.lin1.lora_A.default.weight" in names
    assert "base_model.model.lin1.lora_B.default.weight" in names
    # lin3 is not a target -> plain weight, no base_layer/lora.
    assert "base_model.model.lin3.weight" in names
    assert not any("lin3.lora_A" in n for n in names)


def test_only_lora_trainable():
    model = VeOmniLoraModel(Toy(), _base_config())
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable, "expected some trainable params"
    assert all(".lora_A." in n or ".lora_B." in n for n in trainable)


def test_init_is_noop():
    torch.manual_seed(1)
    base = Toy()
    x = torch.randn(4, 32)
    expected = base(x)
    model = VeOmniLoraModel(copy.deepcopy(base), _base_config())
    got = model(x)
    torch.testing.assert_close(got, expected)


def test_exclude_modules():
    cfg = _base_config(target_modules=["lin1", "lin2"], exclude_modules=["lin2"])
    model = VeOmniLoraModel(Toy(), cfg)
    names = [n for n, _ in model.named_parameters()]
    assert any("lin1.lora_A" in n for n in names)
    assert not any("lin2.lora_A" in n for n in names)


def test_rank_alpha_pattern():
    cfg = _base_config(rank_pattern={r".*lin2$": 16}, alpha_pattern={r".*lin2$": 64})
    model = VeOmniLoraModel(Toy(), cfg)
    base = model.base_model.model
    lin1: LoraLinear = base.get_submodule("lin1")
    lin2: LoraLinear = base.get_submodule("lin2")
    assert lin1.r["default"] == 8 and lin2.r["default"] == 16
    assert lin2.lora_alpha["default"] == 64


def test_rslora_scaling():
    import math

    plain = LoraLinear(nn.Linear(8, 8, bias=False), "default", r=16, lora_alpha=32, use_rslora=False)
    rs = LoraLinear(nn.Linear(8, 8, bias=False), "default", r=16, lora_alpha=32, use_rslora=True)
    assert plain.scaling["default"] == 32 / 16
    assert rs.scaling["default"] == 32 / math.sqrt(16)


# ----------------------------------------------------------------------------
# State dict / save / load
# ----------------------------------------------------------------------------


def test_state_dict_peft_key_format():
    model = VeOmniLoraModel(Toy(), _base_config())
    sd = get_lora_state_dict(model, config=model.get_lora_config())
    assert sd, "empty adapter state dict"
    for k in sd:
        assert k.startswith("base_model.model.")
        assert ".default." not in k
        assert ".lora_A." in k or ".lora_B." in k
    assert "base_model.model.lin1.lora_A.weight" in sd


def test_save_and_reload_roundtrip(tmp_path):
    torch.manual_seed(2)
    base = Toy()
    model = VeOmniLoraModel(copy.deepcopy(base), _base_config())
    _randomize_lora_b(model)

    x = torch.randn(4, 32)
    trained_out = model(x)

    model.save_pretrained(str(tmp_path), safe_serialization=True)

    # Fresh base + rebuild wrappers from the saved config, then load weights.
    reloaded = VeOmniLoraModel.from_pretrained(copy.deepcopy(base), str(tmp_path))
    load_lora_weights(reloaded, str(tmp_path), init_device="cpu")

    torch.testing.assert_close(reloaded(x), trained_out)


def test_saved_state_matches_live(tmp_path):
    model = VeOmniLoraModel(Toy(), _base_config())
    _randomize_lora_b(model)
    model.save_pretrained(str(tmp_path))

    on_disk = load_adapter_state_dict(str(tmp_path), device="cpu")
    live = get_lora_state_dict(model, config=model.get_lora_config())
    assert set(on_disk) == set(live)
    for k in live:
        torch.testing.assert_close(on_disk[k], live[k].detach())


def test_merge_and_unload():
    torch.manual_seed(3)
    model = VeOmniLoraModel(Toy(), _base_config())
    _randomize_lora_b(model)

    x = torch.randn(4, 32)
    lora_out = model(x)

    merged = model.merge_and_unload()
    # No LoRA layers remain; plain Linears only.
    assert not any(isinstance(m, LoraLinear) for m in merged.modules())
    torch.testing.assert_close(merged(x), lora_out)


# ----------------------------------------------------------------------------
# PEFT bidirectional interop (dev/test-only dependency)
# ----------------------------------------------------------------------------


def test_peft_bidirectional_interop(tmp_path):
    peft = pytest.importorskip("peft")

    torch.manual_seed(4)
    base = Toy()
    x = torch.randn(4, 32)

    # (1) PEFT trains + saves; VeOmni loads and matches.
    peft_cfg = peft.LoraConfig(r=8, lora_alpha=16, target_modules=["lin1", "lin2"])
    peft_model = peft.get_peft_model(copy.deepcopy(base), peft_cfg)
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if "lora_B" in n:
                p.normal_(0.0, 0.02)
    peft_out = peft_model(x)

    peft_dir = tmp_path / "from_peft"
    peft_model.save_pretrained(str(peft_dir))

    veomni_model = VeOmniLoraModel.from_pretrained(copy.deepcopy(base), str(peft_dir))
    load_lora_weights(veomni_model, str(peft_dir), init_device="cpu")
    torch.testing.assert_close(veomni_model(x), peft_out, rtol=1e-4, atol=1e-4)

    # (2) VeOmni trains + saves; PEFT loads and matches.
    veomni_dir = tmp_path / "from_veomni"
    veomni_train = VeOmniLoraModel(copy.deepcopy(base), _base_config(r=8, lora_alpha=16))
    _randomize_lora_b(veomni_train)
    veomni_out = veomni_train(x)
    veomni_train.save_pretrained(str(veomni_dir), safe_serialization=True)

    peft_reload = peft.PeftModel.from_pretrained(copy.deepcopy(base), str(veomni_dir))
    torch.testing.assert_close(peft_reload(x), veomni_out, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
