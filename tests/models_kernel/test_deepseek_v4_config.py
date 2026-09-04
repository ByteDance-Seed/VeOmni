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

"""DeepSeek-V4 config fields and ``validate_build_prerequisites``."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config as UpstreamConfig

from tests.models_kernel.compare import eager_kernels_config
from veomni.kernels.config import get_kernels_config, set_kernels_config
from veomni.models_kernel.transformers.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config


def _config_asking_for_the_loss(coef: float = 1.0, **overrides) -> DeepseekV4Config:
    return DeepseekV4Config(
        num_hidden_layers=2,
        layer_types=["compressed_sparse_attention"] * 2,
        dsa_indexer_loss=True,
        dsa_indexer_loss_coef=coef,
        **overrides,
    )


@contextmanager
def _kernels_config_installed(**overrides):
    previous = get_kernels_config()
    installed = eager_kernels_config()
    for key, value in overrides.items():
        setattr(installed, key, value)
    set_kernels_config(installed)
    try:
        yield installed
    finally:
        set_kernels_config(previous)


def test_the_two_fields_are_declared_on_the_model_config():
    base = UpstreamConfig(num_hidden_layers=2, layer_types=["compressed_sparse_attention"] * 2).to_dict()

    overridden = DeepseekV4Config.from_dict(dict(base), dsa_indexer_loss=True, dsa_indexer_loss_coef=0.25)
    assert overridden.dsa_indexer_loss is True
    assert overridden.dsa_indexer_loss_coef == 0.25

    dropped = UpstreamConfig.from_dict(dict(base), dsa_indexer_loss=True)
    assert not hasattr(dropped, "dsa_indexer_loss")


def test_the_defaults_leave_the_objective_off():
    config = DeepseekV4Config(num_hidden_layers=2, layer_types=["compressed_sparse_attention"] * 2)
    assert config.dsa_indexer_loss is False
    assert config.dsa_indexer_loss_coef == 1.0


@pytest.mark.parametrize("coef", [-1.0, -1e-8, float("nan"), float("inf"), float("-inf")])
def test_indexer_loss_coef_rejects_negative_and_non_finite(coef):
    with pytest.raises(ValueError, match="dsa_indexer_loss_coef"):
        _config_asking_for_the_loss(coef=coef).validate_build_prerequisites()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dsa_indexer_loss", "false"),
        ("dsa_indexer_loss", "true"),
        ("dsa_indexer_loss", 1),
        ("dsa_indexer_loss_coef", "0.5"),
        ("dsa_indexer_loss_coef", None),
    ],
)
def test_the_two_fields_are_refused_when_they_arrive_with_the_wrong_type(field, value):
    config = _config_asking_for_the_loss()
    setattr(config, field, value)
    with pytest.raises(TypeError, match=field):
        config.validate_build_prerequisites()


@pytest.mark.parametrize("coef", [0.5, 1.0, 100.0])
def test_indexer_loss_coef_accepts_finite_positive_weights(coef):
    with _kernels_config_installed(dsa_indexer_implementation="tilelang", dsa_attention_implementation="tilelang"):
        _config_asking_for_the_loss(coef=coef).validate_build_prerequisites()


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, "dsa_indexer_implementation"),
        ({"dsa_indexer_implementation": "tilelang"}, "dsa_attention_implementation"),
        (
            {"dsa_indexer_implementation": "cudnn", "dsa_attention_implementation": "tilelang"},
            "dsa_indexer_implementation",
        ),
    ],
)
def test_the_two_kernel_prerequisites_are_refused_at_model_build(overrides, expected):
    with _kernels_config_installed(**overrides):
        with pytest.raises(ValueError, match=expected):
            _config_asking_for_the_loss().validate_build_prerequisites()


def test_a_zero_coefficient_is_not_held_to_the_prerequisites():
    with _kernels_config_installed():
        _config_asking_for_the_loss(coef=0.0).validate_build_prerequisites()


def test_undeclared_is_not_the_same_as_absent(tmp_path: Path):
    _config_asking_for_the_loss(coef=1.0).save_pretrained(tmp_path)
    assert UpstreamConfig.from_pretrained(tmp_path).dsa_indexer_loss is True
    assert UpstreamConfig.from_pretrained(tmp_path, dsa_indexer_loss=False).dsa_indexer_loss is False

    config_json = tmp_path / "config.json"
    saved = json.loads(config_json.read_text())
    saved.pop("dsa_indexer_loss", None)
    saved.pop("dsa_indexer_loss_coef", None)
    config_json.write_text(json.dumps(saved))

    assert not hasattr(UpstreamConfig.from_pretrained(tmp_path, dsa_indexer_loss=True), "dsa_indexer_loss")
    assert DeepseekV4Config.from_pretrained(tmp_path, dsa_indexer_loss=True).dsa_indexer_loss is True
