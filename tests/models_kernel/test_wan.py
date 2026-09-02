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

"""Wan models_kernel consume tests.

Direct-import the staged class. Do not register or use
``build_foundation_model``. Compare a toy DiT against
``tests/models_kernel/refs/wan.py``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from tests.models_kernel.compare import (
    assert_outputs_and_grads_match,
    eager_kernels_config,
)
from tests.models_kernel.refs.wan import WanConfig as RefWanConfig
from tests.models_kernel.refs.wan import WanModel as RefWanModel
from veomni.kernels import VeomniKernel
from veomni.kernels.config import get_kernels_config, set_kernels_config
from veomni.models_kernel.transformers.wan import fa3_fp8
from veomni.models_kernel.transformers.wan import modeling_wan as wan_modeling
from veomni.models_kernel.transformers.wan.config_wan import WanConfig
from veomni.models_kernel.transformers.wan.fa3_fp8 import should_use_fa3_fp8


def _tiny_kwargs() -> dict:
    return {
        "patch_size": [1, 2, 2],
        "dim": 32,
        "eps": 1e-6,
        "ffn_dim": 64,
        "freq_dim": 16,
        "in_dim": 4,
        "num_heads": 4,
        "num_layers": 2,
        "out_dim": 4,
        "text_dim": 16,
        "text_len": 8,
    }


def _tiny_ours_config() -> WanConfig:
    config = WanConfig(**_tiny_kwargs(), attn_implementation="eager")
    config.has_image_input = "false"
    return config


def _tiny_ref_config() -> RefWanConfig:
    return RefWanConfig(**_tiny_kwargs(), has_image_input="false")


def _build_ours(config: WanConfig, kernels: SimpleNamespace | None = None):
    from veomni.models_kernel.transformers.wan.modeling_wan import WanModel

    previous = get_kernels_config()
    set_kernels_config(kernels if kernels is not None else eager_kernels_config())
    try:
        return WanModel(config)
    finally:
        set_kernels_config(previous)


def _wan_inputs(in_dim: int, text_len: int, text_dim: int) -> dict[str, torch.Tensor]:
    return {
        "x": torch.randn(2, in_dim, 2, 8, 8),
        "timestep": torch.rand(2),
        "context": torch.randn(2, text_len, text_dim),
    }


def test_wan_sage_constructs_veomni_sage_attention():
    sage_cfg = eager_kernels_config()
    sage_cfg.attn_implementation = "sageattention"
    model = _build_ours(_tiny_ours_config(), sage_cfg)
    assert model.blocks[0].self_attn.attn.veomni_attn.kernel == "attention"
    assert model.blocks[0].self_attn.attn.veomni_attn.impl == "veomni_sage_attention"


def test_should_use_fa3_fp8_policy():
    assert should_use_fa3_fp8("flash_attention_3", is_self_attn=True, last_loss=0.1)
    assert should_use_fa3_fp8("veomni_flash_attention_3", is_self_attn=True, last_loss=1.0)
    assert not should_use_fa3_fp8("flash_attention_3", is_self_attn=True, last_loss=None)
    assert not should_use_fa3_fp8("flash_attention_3", is_self_attn=True, last_loss=math.nan)
    assert not should_use_fa3_fp8("flash_attention_3", is_self_attn=False, last_loss=0.1)
    assert not should_use_fa3_fp8("eager", is_self_attn=True, last_loss=0.1)
    assert not should_use_fa3_fp8("veomni_sage_attention", is_self_attn=True, last_loss=0.1)


def test_wan_fa3_constructs_generic_flash_attention_3():
    fa3_cfg = eager_kernels_config()
    fa3_cfg.attn_implementation = "flash_attention_3"
    model = _build_ours(_tiny_ours_config(), fa3_cfg)
    handle = model.blocks[0].self_attn.attn.veomni_attn
    assert handle.kernel == "attention"
    assert handle.impl == "flash_attention_3"


def test_wan_fa3_finite_last_loss_quantizes(monkeypatch):
    fa3_cfg = eager_kernels_config()
    fa3_cfg.attn_implementation = "flash_attention_3"
    model = _build_ours(_tiny_ours_config(), fa3_cfg)
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
def test_wan_fa3_skips_fp8_outside_policy(monkeypatch, kwargs):
    fa3_cfg = eager_kernels_config()
    fa3_cfg.attn_implementation = "flash_attention_3"
    model = _build_ours(_tiny_ours_config(), fa3_cfg)
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


def test_wan_constructs_local_kernels():
    model = _build_ours(_tiny_ours_config())
    block = model.blocks[0]
    assert isinstance(block.self_attn.norm_q.veomni_rms_norm, VeomniKernel)
    assert block.self_attn.norm_q.veomni_rms_norm.impl == "eager"
    assert block.self_attn.attn.veomni_attn.kernel == "attention"
    assert block.self_attn.attn.veomni_attn.impl == "eager"


def test_wan_instances_keep_distinct_impls():
    eager = _build_ours(_tiny_ours_config(), eager_kernels_config())
    other_cfg = eager_kernels_config()
    other_cfg.rms_norm_implementation = "liger_kernel"
    other = _build_ours(_tiny_ours_config(), other_cfg)

    assert eager.blocks[0].self_attn.norm_q.veomni_rms_norm.impl == "eager"
    assert other.blocks[0].self_attn.norm_q.veomni_rms_norm.impl == "liger_kernel"

    set_kernels_config(other_cfg)
    assert eager.blocks[0].self_attn.norm_q.veomni_rms_norm.impl == "eager"


def test_wan_eager_matches_official():
    torch.manual_seed(0)
    official = RefWanModel(_tiny_ref_config())
    ours = _build_ours(_tiny_ours_config())
    ours.load_state_dict(official.state_dict())
    inputs = _wan_inputs(in_dim=4, text_len=8, text_dim=16)

    def call(model):
        return model(**inputs)

    assert_outputs_and_grads_match(official, ours, call)
