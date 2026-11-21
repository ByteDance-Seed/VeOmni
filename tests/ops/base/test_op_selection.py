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

"""Per-site op selection tests: OpsSelection, SiteOverride, VeomniOp interning, and config round-trips."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import pytest
import torch
import yaml

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments import OpsImplementationConfig
from veomni.ops import OP_REGISTRY, OpsSelection, SiteOverride, VeomniOp, register_op


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _round_forward(x: torch.Tensor, *, block_size: int = 128) -> tuple[torch.Tensor, object]:
    from veomni.ops.registry import SavedState

    del block_size
    return x, SavedState(())


def _round_backward(grad_output: torch.Tensor, saved: object) -> tuple[torch.Tensor]:
    del saved
    return (grad_output,)


@pytest.fixture
def quantizing_row():
    """Register a CPU-resolvable test row on ``linear/standard`` for one test."""
    register_op(
        "linear",
        "standard",
        "test_round",
        _round_forward,
        _round_backward,
        description="test-only rounding row",
    )
    try:
        yield "test_round"
    finally:
        for key in [key for key in OP_REGISTRY._entries if key[2] == "test_round"]:
            del OP_REGISTRY._entries[key]


def test_bare_string_and_unset_selection():
    selection = OpsSelection.from_config_value("liger_kernel")
    assert selection.resolve("layers.0.mlp") == ("liger_kernel", {})
    assert selection.resolve(None) == ("liger_kernel", {})
    assert OpsSelection().resolve("layers.0.q_a_proj") == (None, {})


def test_site_overrides_anchored_and_ordered():
    selection = OpsSelection.from_config_value(
        {
            "name": "eager",
            "overrides": [
                {"site": r"layers\.\d+\.q_a_proj", "name": "qat"},
                {"site": r"layers\.0\.q_a_proj", "name": "eager"},
            ],
        }
    )
    assert selection.resolve("layers.3.q_a_proj") == ("qat", {})
    assert selection.resolve("layers.0.q_a_proj") == ("eager", {})
    # Anchored fullmatch: layers.0.q_a_proj does not match layers.10.q_a_proj.
    assert selection.resolve("layers.10.q_a_proj") == ("qat", {})


def test_override_merges_kwargs_on_same_row_and_resets_on_row_switch():
    selection = OpsSelection.from_config_value(
        {
            "name": "qat",
            "kwargs": {"block_size": 128},
            "overrides": [
                {"site": r"layers\.\d+\.o_a_proj", "kwargs": {"block_size": 64}},
                {"site": r"layers\.0\.q_a_proj", "name": "eager"},
            ],
        }
    )
    assert selection.resolve("layers.2.q_a_proj") == ("qat", {"block_size": 128})
    assert selection.resolve("layers.2.o_a_proj") == ("qat", {"block_size": 64})
    # Switching to a different row clears the previous row's kwargs.
    assert selection.resolve("layers.0.q_a_proj") == ("eager", {})


def test_handle_kwargs_override_call_site_and_intern_by_kwargs(quantizing_row):
    selection = OpsSelection.from_config_value(
        {
            "name": quantizing_row,
            "kwargs": {"block_size": 128},
            "overrides": [{"site": r"layers\.7\.q_a_proj", "kwargs": {"block_size": 64}}],
        }
    )
    first = VeomniOp("linear", "standard", selection, site="layers.3.q_a_proj")
    second = VeomniOp("linear", "standard", selection, site="layers.4.q_a_proj")
    special = VeomniOp("linear", "standard", selection, site="layers.7.q_a_proj")
    assert first is second
    assert first is not special
    assert (first.kwargs, special.kwargs) == ({"block_size": 128}, {"block_size": 64})

    # Pickle round-trip preserves the interned handle and its kwargs.
    restored = pickle.loads(pickle.dumps(special))
    assert restored is special
    assert restored.kwargs == {"block_size": 64}

    # Handle kwargs take precedence over call-site kwargs.
    seen = {}

    def record(*args, **kwargs):
        seen.update(kwargs)
        return torch.zeros(1)

    special_copy = object.__new__(VeomniOp)
    special_copy.op, special_copy.variant, special_copy.impl, special_copy.kwargs = (
        "linear",
        "standard",
        quantizing_row,
        {"block_size": 64},
    )
    special_copy._entry = type(special._entry)(
        op="linear",
        variant="standard",
        impl=quantizing_row,
        description="probe",
        wrapper=record,
    )
    special_copy(torch.zeros(1), block_size=32)
    assert seen == {"block_size": 64}


def test_config_round_trip_and_kv_round_overrides():
    plain = OpsImplementationConfig(linear_implementation="eager")
    assert plain.to_dict()["linear_implementation"] == "eager"

    configured = OpsImplementationConfig(
        linear_implementation={
            "name": "qat",
            "kwargs": {"block_size": 64},
            "overrides": [{"site": r"layers\.0\..*", "name": "eager"}],
        },
        kv_round_implementation={
            "name": "none",
            "overrides": [{"site": r"layers\.0\.window_kv", "name": "eager"}],
        },
    )
    assert isinstance(configured.linear_implementation, OpsSelection)
    assert isinstance(configured.kv_round_implementation, OpsSelection)
    assert configured.kv_round_implementation.resolve("layers.0.window_kv") == ("eager", {})
    assert configured.kv_round_implementation.resolve("layers.1.window_kv") == ("none", {})

    reloaded = OpsImplementationConfig(**configured.to_dict())
    assert reloaded.linear_implementation.all_impls() == configured.linear_implementation.all_impls()
    assert reloaded.linear_implementation.kwargs == {"block_size": 64}


def test_hardware_validation_covers_override_rows():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        pytest.skip("this case asserts the non-NPU refusal")

    with pytest.raises(ValueError, match="requires Ascend NPU"):
        OpsImplementationConfig(
            moe_implementation={"name": "eager", "overrides": [{"site": r"layers\.0\.experts", "name": "fused_npu"}]}
        )


def test_shipped_deepseek_v4_mixed_precision_yaml_parses_cleanly():
    yaml_path = Path(__file__).resolve().parents[3] / "configs" / "text" / "deepseek_v4_mixed_precision.yaml"
    raw = yaml.safe_load(yaml_path.read_text())
    ops = OpsImplementationConfig(**raw["model"]["ops_implementation"])
    assert ops.linear_implementation.resolve("layers.0.q_a_proj") == ("eager", {})
    assert ops.linear_implementation.resolve("layers.1.q_a_proj") == ("qat", {})
    assert ops.gated_mlp_implementation.resolve("layers.0.mlp") == ("liger_kernel", {})
    assert ops.gated_mlp_implementation.resolve("layers.1.mlp") == ("qat", {})
    assert ops.moe_implementation == "deepgemm"
    assert ops.kv_round_implementation == "triton"


def test_quantizing_moe_rows_refuse_unsupported_activations():
    from veomni.ops.kernels.moe_experts.standard import deepgemm, qat

    zero = torch.empty(0)
    with pytest.raises(ValueError, match="moe_implementation='deepgemm' computes SiLU only"):
        deepgemm.wrapper(zero, zero, zero, zero, zero, zero, zero, num_experts=1, activation="gelu")
    with pytest.raises(ValueError, match="moe_implementation='qat' cannot run on hidden_act='relu'"):
        qat.wrapper(zero, zero, zero, zero, zero, zero, zero, num_experts=1, activation="relu")


def test_site_override_invalid_regex_raises():
    with pytest.raises(re.error):
        SiteOverride(site="layers.(0.mlp")
