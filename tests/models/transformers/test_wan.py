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
# See the License for the specific language governing limitations
# under the License.

"""Wan models registry, op-selection, and parity tests.

Compare a toy DiT against ``tests/models/refs/wan.py``.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.models.compare import (
    assert_outputs_and_grads_match,
    eager_ops_config,
    ops_config_scope,
)
from tests.models.refs.wan import WanConfig as RefWanConfig
from tests.models.refs.wan import WanModel as RefWanModel
from tests.models.tiny_configs import tiny_wan_config as _tiny_config
from veomni.models.transformers.wan import fa3_fp8
from veomni.models.transformers.wan import modeling_wan as wan_modeling
from veomni.models.transformers.wan.config_wan import WanConfig
from veomni.models.transformers.wan.fa3_fp8 import should_use_fa3_fp8


def _tiny_ref_config() -> RefWanConfig:
    config = _tiny_config()
    return RefWanConfig(
        patch_size=config.patch_size,
        dim=config.dim,
        eps=config.eps,
        ffn_dim=config.ffn_dim,
        freq_dim=config.freq_dim,
        in_dim=config.in_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        out_dim=config.out_dim,
        text_dim=config.text_dim,
        text_len=config.text_len,
        has_image_input=config.has_image_input,
    )


def _build_ours(config: WanConfig, ops: SimpleNamespace | None = None):
    from veomni.models.transformers.wan.modeling_wan import WanModel

    with ops_config_scope(ops if ops is not None else eager_ops_config()):
        return WanModel(config)


def _wan_inputs(in_dim: int, text_len: int, text_dim: int) -> dict[str, torch.Tensor]:
    return {
        "x": torch.randn(2, in_dim, 2, 8, 8),
        "timestep": torch.rand(2),
        "context": torch.randn(2, text_len, text_dim),
    }


@pytest.mark.parametrize(
    ("filename", "has_image_input"),
    (
        ("wani2v_14b.json", "true"),
        ("want2v_1.3b.json", "false"),
        ("want2v_14b.json", "false"),
    ),
)
def test_wan_repository_configs_load_through_registry(filename, has_image_input):
    from veomni.models import build_config, get_model_class

    config_path = Path(__file__).parents[3] / "configs/model_configs/wan" / filename
    config = build_config(str(config_path))

    assert type(config) is WanConfig
    assert config.architectures == ["WanModel"]
    assert config.has_image_input == has_image_input
    assert get_model_class(config).__name__ == "WanModel"


def test_wan_sage_constructs_veomni_sage_attention(available_nvidia_ops):
    sage_cfg = eager_ops_config()
    sage_cfg.attn_implementation = "sageattention"
    model = _build_ours(_tiny_config(), sage_cfg)
    assert model.blocks[0].self_attn.attn.veomni_attn.op == "attention"
    assert model.blocks[0].self_attn.attn.veomni_attn.impl == "veomni_sage_attention"


def test_should_use_fa3_fp8_policy():
    assert should_use_fa3_fp8("flash_attention_3", is_self_attn=True, last_loss=0.1)
    assert should_use_fa3_fp8("veomni_flash_attention_3", is_self_attn=True, last_loss=1.0)
    assert not should_use_fa3_fp8("flash_attention_3", is_self_attn=True, last_loss=None)
    assert not should_use_fa3_fp8("flash_attention_3", is_self_attn=True, last_loss=math.nan)
    assert not should_use_fa3_fp8("flash_attention_3", is_self_attn=False, last_loss=0.1)
    assert not should_use_fa3_fp8("eager", is_self_attn=True, last_loss=0.1)
    assert not should_use_fa3_fp8("veomni_sage_attention", is_self_attn=True, last_loss=0.1)


def test_wan_fa3_constructs_generic_flash_attention_3(available_nvidia_ops):
    fa3_cfg = eager_ops_config()
    fa3_cfg.attn_implementation = "flash_attention_3"
    model = _build_ours(_tiny_config(), fa3_cfg)
    handle = model.blocks[0].self_attn.attn.veomni_attn
    assert handle.op == "attention"
    assert handle.impl == "flash_attention_3"


def test_wan_fa3_finite_last_loss_quantizes(monkeypatch, available_nvidia_ops):
    fa3_cfg = eager_ops_config()
    fa3_cfg.attn_implementation = "flash_attention_3"
    model = _build_ours(_tiny_config(), fa3_cfg)
    attn = model.blocks[0].self_attn.attn
    captured = {}

    def fake_flash_attn_func(query, key, value, **kwargs):
        captured.update(q=query, k=key, v=value, kwargs=kwargs)
        return torch.zeros(query.shape[0], query.shape[1], query.shape[2], query.shape[3], dtype=torch.bfloat16)

    monkeypatch.setattr(fa3_fp8, "flash_attn_interface", SimpleNamespace(flash_attn_func=fake_flash_attn_func))
    monkeypatch.setattr(attn, "veomni_attn", lambda *args, **kwargs: pytest.fail("generic attention"))
    attn.veomni_attn.impl = "flash_attention_3"

    hidden = torch.randn(1, 8, 32)
    output = attn(hidden, hidden, hidden, last_loss=0.1, isSelfAttn=True)

    assert captured["q"].dtype == torch.float8_e4m3fn
    assert "q_descale" in captured["kwargs"]
    assert "k_descale" in captured["kwargs"]
    assert "v_descale" in captured["kwargs"]
    assert captured["kwargs"]["original_q"].shape == (1, 8, 4, 8)
    assert output.shape == (1, 8, 32)


@pytest.mark.parametrize(
    ("kwargs",),
    (
        ({"last_loss": None, "isSelfAttn": True},),
        ({"last_loss": math.nan, "isSelfAttn": True},),
        ({"last_loss": 0.1, "isSelfAttn": False},),
    ),
)
def test_wan_fa3_skips_fp8_outside_policy(monkeypatch, available_nvidia_ops, kwargs):
    fa3_cfg = eager_ops_config()
    fa3_cfg.attn_implementation = "flash_attention_3"
    model = _build_ours(_tiny_config(), fa3_cfg)
    attn = model.blocks[0].self_attn.attn
    monkeypatch.setattr(
        wan_modeling,
        "flash_attention_3_fp8",
        lambda *args, **kw: pytest.fail("fp8 path"),
    )

    called = {}

    def fake_generic(module, query, key, value, attention_mask, **kw):
        called["generic"] = True
        return torch.zeros(query.shape[0], query.shape[2], query.shape[1], query.shape[3])

    fake_generic.impl = "flash_attention_3"
    monkeypatch.setattr(attn, "veomni_attn", fake_generic)

    hidden = torch.randn(1, 8, 32)
    output = attn(hidden, hidden, hidden, **kwargs)
    assert called["generic"] is True
    assert output.shape == (1, 8, 32)


def test_wan_eager_matches_official():
    torch.manual_seed(0)
    official = RefWanModel(_tiny_ref_config())
    ours = _build_ours(_tiny_config())
    ours.load_state_dict(official.state_dict())
    inputs = _wan_inputs(in_dim=4, text_len=8, text_dim=16)

    def call(model):
        return model(**inputs)

    assert_outputs_and_grads_match(official, ours, call)
