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

"""Per-site op selection and override tests."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import pytest
import torch
import yaml

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments import OpsImplementationConfig, VeOmniArguments
from veomni.arguments.parser import _instantiate_recursive
from veomni.ops import OP_REGISTRY, OpsSelection, SiteOverride, VeomniOp, register_op
from veomni.utils.device import IS_NPU_AVAILABLE


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_per_site_op_selection_and_overrides():
    """End-to-end check for OpsSelection overrides, handle interning/pickle, config round-trip, and shipped YAML."""
    from veomni.ops.registry import SavedState

    # 1. Bare string / unset / ordered anchored regex matching / kwargs merge vs row-switch reset.
    assert OpsSelection.from_config_value("liger_kernel").resolve("layers.0.mlp") == ("liger_kernel", {})
    assert OpsSelection().resolve("layers.0.q_a_proj") == (None, {})

    selection = OpsSelection.from_config_value(
        {
            "name": "qat",
            "kwargs": {"block_size": 128},
            "overrides": [
                {"site": r"layers\.\d+\.o_a_proj", "kwargs": {"block_size": 64}},
                {"site": r"layers\.0\..*", "name": "eager"},
            ],
        }
    )
    assert selection.resolve("layers.2.q_a_proj") == ("qat", {"block_size": 128})
    assert selection.resolve("layers.2.o_a_proj") == ("qat", {"block_size": 64})
    assert selection.resolve("layers.0.q_a_proj") == ("eager", {})
    # Anchored fullmatch: layers.0.* does not match layers.10.*.
    assert selection.resolve("layers.10.q_a_proj") == ("qat", {"block_size": 128})

    with pytest.raises(re.error):
        SiteOverride(site="layers.(0.mlp")

    # 2. VeomniOp handle interning by (op, variant, impl, kwargs), pickle round-trip, and kwargs precedence.
    def _fwd(x: torch.Tensor, *, block_size: int = 128):
        del block_size
        return x, SavedState(())

    def _bwd(grad: torch.Tensor, saved: object):
        del saved
        return (grad,)

    register_op("linear", "standard", "test_round", _fwd, _bwd, description="test-only rounding row")
    try:
        row_sel = OpsSelection.from_config_value(
            {
                "name": "test_round",
                "kwargs": {"block_size": 128},
                "overrides": [{"site": r"layers\.7\.q_a_proj", "kwargs": {"block_size": 64}}],
            }
        )
        h1 = VeomniOp("linear", "standard", row_sel, site="layers.3.q_a_proj")
        h2 = VeomniOp("linear", "standard", row_sel, site="layers.4.q_a_proj")
        h3 = VeomniOp("linear", "standard", row_sel, site="layers.7.q_a_proj")
        assert h1 is h2
        assert h1 is not h3
        assert (h1.kwargs, h3.kwargs) == ({"block_size": 128}, {"block_size": 64})
        assert pickle.loads(pickle.dumps(h3)) is h3

        seen = {}
        probe = object.__new__(VeomniOp)
        probe.op, probe.variant, probe.impl, probe.kwargs = "linear", "standard", "test_round", {"block_size": 64}
        probe._entry = type(h3._entry)(
            op="linear",
            variant="standard",
            impl="test_round",
            description="probe",
            wrapper=lambda *a, **kw: seen.update(kw) or torch.zeros(1),
        )
        probe(torch.zeros(1), block_size=32)
        assert seen == {"block_size": 64}
    finally:
        for key in [k for k in OP_REGISTRY._entries if k[2] == "test_round"]:
            del OP_REGISTRY._entries[key]

    # 3. Config dict round-trip, kv_round overrides, host validation across overrides, and shipped YAML.
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
    assert configured.kv_round_implementation.resolve("layers.0.window_kv") == ("eager", {})
    assert configured.kv_round_implementation.resolve("layers.1.window_kv") == ("none", {})
    reloaded = OpsImplementationConfig(**configured.to_dict())
    assert reloaded.linear_implementation.all_impls() == configured.linear_implementation.all_impls()
    assert reloaded.linear_implementation.kwargs == {"block_size": 64}

    if not IS_NPU_AVAILABLE:
        with pytest.raises(ValueError, match="requires Ascend NPU"):
            OpsImplementationConfig(
                moe_implementation={
                    "name": "eager",
                    "overrides": [{"site": r"layers\.0\.experts", "name": "fused_npu"}],
                }
            )

    yaml_path = Path(__file__).resolve().parents[3] / "configs" / "text" / "deepseek_v4_mixed_precision.yaml"
    raw = yaml.safe_load(yaml_path.read_text())
    parsed = _instantiate_recursive(VeOmniArguments, raw)
    ops = parsed.model.ops_implementation
    assert ops.linear_implementation.resolve("layers.0.q_a_proj") == ("eager", {})
    assert ops.linear_implementation.resolve("layers.1.q_a_proj") == ("qat", {})
    assert ops.gated_mlp_implementation.resolve("layers.0.mlp") == ("liger_kernel", {})
    assert ops.gated_mlp_implementation.resolve("layers.1.mlp") == ("qat", {})
    assert ops.moe_implementation == "deepgemm"
    assert ops.kv_round_implementation == "triton"
