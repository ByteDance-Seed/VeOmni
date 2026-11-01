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
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSeek-V4 config fields and ``validate_build_prerequisites``."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config as UpstreamConfig

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import eager_ops_config
from veomni.models.transformers.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _config_asking_for_the_loss(
    coef: float = 1.0,
    *,
    ops_overrides: dict | None = None,
    **overrides,
) -> DeepseekV4Config:
    ops = eager_ops_config()
    for key, value in (ops_overrides or {}).items():
        setattr(ops, key, value)
    return DeepseekV4Config(
        num_hidden_layers=2,
        layer_types=["compressed_sparse_attention"] * 2,
        dsa_indexer_loss=True,
        dsa_indexer_loss_coef=coef,
        ops_implementation=ops,
        **overrides,
    )


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
    _config_asking_for_the_loss(
        coef=coef,
        ops_overrides={"dsa_indexer_implementation": "tilelang", "dsa_attention_implementation": "tilelang"},
    ).validate_build_prerequisites()


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
    with pytest.raises(ValueError, match=expected):
        _config_asking_for_the_loss(ops_overrides=overrides).validate_build_prerequisites()


def test_a_zero_coefficient_is_not_held_to_the_prerequisites():
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


def test_dspark_mtp_config_fields_and_defaults():
    config = DeepseekV4Config(num_hidden_layers=4, layer_types=["compressed_sparse_attention"] * 4)
    assert config.dspark_block_size == 0
    assert config.dspark_target_layer_ids == [1, 2, 3]
    assert config.mtp_loss_coef == 1.0
    assert config.freeze_mtp is False
    assert config.moe_config_for_layer(0) == (config.n_routed_experts, config.num_experts_per_tok)
    assert config.moe_config_for_layer(4) == (config.n_routed_experts, config.num_experts_per_tok)

    custom = DeepseekV4Config(
        num_hidden_layers=4,
        compress_ratios=[0, 0, 4, 128, 0, 0, 0],
        dspark_block_size=5,
        dspark_n_routed_experts=2,
        dspark_num_experts_per_tok=1,
    )
    assert custom.num_nextn_predict_layers == 3
    assert custom.moe_config_for_layer(4) == (2, 1)

    explicit_zero = DeepseekV4Config(
        num_hidden_layers=4,
        compress_ratios=[0, 0, 4, 128, 0, 0, 0],
        num_nextn_predict_layers=0,
    )
    assert explicit_zero.num_nextn_predict_layers == 0

    explicit_one = DeepseekV4Config(
        num_hidden_layers=4,
        compress_ratios=[0, 0, 4, 128, 0, 0, 0],
        num_nextn_predict_layers=1,
    )
    assert explicit_one.num_nextn_predict_layers == 1

    # Verify from_dict override where dspark_block_size is passed as a kwarg override
    # while compress_ratios is in the base dict (simulating YAML model_config override).
    from_dict_cfg = DeepseekV4Config.from_dict(
        {"num_hidden_layers": 4, "compress_ratios": [0, 0, 4, 128, 0, 0, 0]},
        dspark_block_size=5,
    )
    assert from_dict_cfg.num_nextn_predict_layers == 3
    assert from_dict_cfg.dspark_block_size == 5
    assert "_explicit_num_nextn_predict_layers" not in copy.deepcopy(from_dict_cfg).to_dict()

    # On-disk config.json with placeholder "num_nextn_predict_layers": 1 + 43+3 compress_ratios
    # infers 3 stages unless the caller explicitly overrides num_nextn_predict_layers in kwargs.
    on_disk_dict = {
        "num_hidden_layers": 4,
        "num_nextn_predict_layers": 1,
        "compress_ratios": [0, 0, 4, 128, 0, 0, 0],
        "dspark_target_layer_ids": [1, 2, 3],
    }
    inferred_from_disk = DeepseekV4Config.from_dict(on_disk_dict, dspark_block_size=5)
    assert inferred_from_disk.num_nextn_predict_layers == 3

    overridden_to_one = DeepseekV4Config.from_dict(
        on_disk_dict,
        dspark_block_size=5,
        num_nextn_predict_layers=1,
    )
    assert overridden_to_one.num_nextn_predict_layers == 1

    # Overriding num_hidden_layers via from_dict kwargs recomputes default dspark_target_layer_ids.
    shrunk_cfg = DeepseekV4Config.from_dict(
        on_disk_dict,
        num_hidden_layers=2,
        dspark_block_size=5,
    )
    assert shrunk_cfg.num_hidden_layers == 2
    assert shrunk_cfg.dspark_target_layer_ids == [0, 1]
    shrunk_cfg.validate_build_prerequisites()

    # Non-default explicit dspark_target_layer_ids in config_dict is preserved so validation catches it.
    custom_targets_dict = dict(on_disk_dict, dspark_target_layer_ids=[0, 2, 3])
    shrunk_custom_targets = DeepseekV4Config.from_dict(
        custom_targets_dict,
        num_hidden_layers=2,
        dspark_block_size=5,
    )
    assert shrunk_custom_targets.dspark_target_layer_ids == [0, 2, 3]
    with pytest.raises(ValueError, match="dspark_target_layer_ids"):
        shrunk_custom_targets.validate_build_prerequisites()

    # Increasing num_hidden_layers via from_dict without explicit layer_types raises ValueError.
    with pytest.raises(ValueError, match="Pass layer_types explicitly"):
        DeepseekV4Config.from_dict(on_disk_dict, num_hidden_layers=6)


@pytest.mark.parametrize("coef", [-1.0, float("nan"), float("inf")])
def test_mtp_loss_coef_rejects_negative_and_non_finite(coef):
    config = DeepseekV4Config(
        num_hidden_layers=4,
        layer_types=["compressed_sparse_attention"] * 4,
        dspark_block_size=5,
        mtp_loss_coef=coef,
    )
    with pytest.raises(ValueError, match="mtp_loss_coef"):
        config.validate_build_prerequisites()


def test_dspark_target_layer_ids_rejects_out_of_range_when_mtp_enabled():
    config = DeepseekV4Config(
        num_hidden_layers=4,
        layer_types=["compressed_sparse_attention"] * 4,
        dspark_block_size=5,
        dspark_target_layer_ids=[2, 4],
    )
    with pytest.raises(ValueError, match="dspark_target_layer_ids"):
        config.validate_build_prerequisites()
