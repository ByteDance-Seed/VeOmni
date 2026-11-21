# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Both original HF releases and checkpoints exported with the current model layout can be loaded."""

from pathlib import Path

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models.auto import build_foundation_model
from veomni.models.checkpoint.weights import _convert_weight_key


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("fixture", ["qwen2vl_toy", "qwen25vl_toy"])
def test_public_checkpoint_keys_map_to_both_towers(fixture):
    config = Path(__file__).resolve().parents[1] / "toy_config" / fixture
    model = build_foundation_model(
        config_path=str(config),
        init_device="meta",
        ops_implementation=OpsImplementationConfig(attn_implementation="eager"),
    )
    current = set(model.state_dict())
    legacy = {
        key.removeprefix("model.")
        if key.startswith("model.visual.")
        else key.replace("model.language_model.", "model.")
        for key in current
    }
    assert "visual.patch_embed.proj.weight" in legacy
    assert "model.layers.0.self_attn.q_proj.weight" in legacy
    assert {_convert_weight_key(key, model) for key in legacy} == current
    assert all(_convert_weight_key(key, model) == key for key in current)
