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

"""Per-site op selection: one family field, different rows at different sites."""

from __future__ import annotations

import pickle

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments import OpsImplementationConfig
from veomni.ops import VeomniOp
from veomni.ops.selection import OpsSelection, SiteOverride


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_a_bare_string_is_one_row_for_every_site():
    """The pre-override shape stays valid: a row name applies everywhere."""
    selection = OpsSelection.from_config_value("liger_kernel")
    assert selection.resolve_impl("layers.0.mlp") == "liger_kernel"
    assert selection.resolve_impl("layers.40.mlp") == "liger_kernel"
    assert selection.resolve_impl(None) == "liger_kernel"


def test_an_unset_field_asks_the_host_for_its_default():
    """``None`` is not "no row": it is "let the registry pick", as an unset field is today."""
    assert OpsSelection().resolve_impl("layers.0.q_a_proj") is None


def test_an_override_can_switch_rows_for_some_sites():
    """The common shape: one row by default, a different one where the kernel differs."""
    selection = OpsSelection.from_config_value(
        {
            "name": "eager",
            "overrides": [{"site": r"layers\.(1[0-9]|2[0-9])\.q_a_proj", "name": "qat"}],
        }
    )
    assert selection.resolve_impl("layers.15.q_a_proj") == "qat"
    assert selection.resolve_impl("layers.5.q_a_proj") == "eager"


def test_an_override_can_turn_quantization_off_by_naming_a_reference_row():
    """There is no separate "disable" flag: the row name is the whole decision."""
    selection = OpsSelection.from_config_value(
        {
            "name": "qat",
            "overrides": [{"site": r"layers\.0\.mlp", "name": "liger_kernel"}],
        }
    )
    assert selection.resolve_impl("layers.7.mlp") == "qat"
    assert selection.resolve_impl("layers.0.mlp") == "liger_kernel"


def test_a_later_override_wins_over_an_earlier_one():
    """Ordering is the only conflict rule, so a narrow rule goes after a broad one."""
    selection = OpsSelection.from_config_value(
        {
            "name": "eager",
            "overrides": [
                {"site": r"layers\.\d+\.q_a_proj", "name": "qat"},
                {"site": r"layers\.7\.q_a_proj", "name": "eager"},
            ],
        }
    )
    assert selection.resolve_impl("layers.3.q_a_proj") == "qat"
    assert selection.resolve_impl("layers.7.q_a_proj") == "eager"


def test_matching_is_anchored_so_a_suffix_does_not_leak():
    """``fullmatch``: ``layers.0.mlp`` must not match ``layers.10.mlp``."""
    selection = OpsSelection.from_config_value(
        {"name": "eager", "overrides": [{"site": r"layers\.0\.mlp", "name": "liger_kernel"}]}
    )
    assert selection.resolve_impl("layers.0.mlp") == "liger_kernel"
    assert selection.resolve_impl("layers.10.mlp") == "eager"


# ---------------------------------------------------------------------------
# Row arguments
# ---------------------------------------------------------------------------


def test_kwargs_travel_with_the_selection():
    """A row's arguments are part of the selection, not of the row's name."""
    selection = OpsSelection.from_config_value({"name": "qat", "kwargs": {"dtype": "fp8", "block_size": 64}})
    assert selection.resolve_kwargs("layers.3.q_a_proj") == {"dtype": "fp8", "block_size": 64}


def test_an_override_merges_kwargs_by_default():
    """A rule that adjusts one argument keeps the rest."""
    selection = OpsSelection.from_config_value(
        {
            "name": "qat",
            "kwargs": {"dtype": "fp8", "block_size": 128},
            "overrides": [{"site": r"layers\.\d+\.o_a_proj", "kwargs": {"block_size": 64}}],
        }
    )
    assert selection.resolve_kwargs("layers.2.o_a_proj") == {"dtype": "fp8", "block_size": 64}
    assert selection.resolve_kwargs("layers.2.q_a_proj") == {"dtype": "fp8", "block_size": 128}


def test_an_override_can_replace_the_whole_kwargs_set():
    """Switching to a row with a different signature needs the inherited set dropped.

    ``liger_kernel`` takes no ``dtype``, so inheriting one from the enclosing
    quantizing block would be refused when the handle is built.
    """
    selection = OpsSelection.from_config_value(
        {
            "name": "qat",
            "kwargs": {"dtype": "fp8"},
            "overrides": [{"site": r"layers\.0\.mlp", "name": "liger_kernel", "kwargs": {}, "replace_kwargs": True}],
        }
    )
    assert selection.resolve_kwargs("layers.0.mlp") == {}
    assert selection.resolve_kwargs("layers.7.mlp") == {"dtype": "fp8"}


def test_replace_kwargs_without_kwargs_is_refused():
    """The combination has no meaning, so it fails rather than being ignored."""
    with pytest.raises(ValueError, match="replace_kwargs requires kwargs"):
        SiteOverride(site="x", replace_kwargs=True)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_an_unknown_key_is_named_rather_than_ignored():
    """A typo in YAML would otherwise silently drop the setting it meant to write."""
    with pytest.raises(ValueError, match=r"Unknown ops selection keys: \['kwarg'\]"):
        OpsSelection.from_config_value({"name": "eager", "kwarg": {"dtype": "fp8"}})


def test_an_invalid_site_regex_is_refused():
    with pytest.raises(ValueError, match="is not a valid regex"):
        SiteOverride(site="layers.(0.mlp")


def test_an_override_without_a_site_is_refused():
    with pytest.raises(ValueError, match="requires a site pattern"):
        SiteOverride.from_dict({"name": "eager"})


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_a_row_argument_the_row_does_not_take_is_refused():
    """The row's own signature is the check, so a typo cannot reach a forward."""
    selection = OpsSelection.from_config_value({"name": "eager", "kwargs": {"dtype": "fp8"}})
    with pytest.raises(ValueError, match=r"does not accept \['dtype'\]"):
        VeomniOp("swiglu_mlp", "standard", selection, site="layers.0.mlp")


def test_a_row_argument_the_row_takes_is_kept_on_the_handle():
    selection = OpsSelection.from_config_value({"name": "eager", "kwargs": {"swiglu_limit": 10.0}})
    handle = VeomniOp("swiglu_mlp", "standard", selection, site="layers.0.mlp")
    assert handle.kwargs == {"swiglu_limit": 10.0}


def test_a_call_can_still_override_a_handle_argument():
    """The checkpoint's storage dtype is more specific than the config's."""
    selection = OpsSelection.from_config_value({"name": "eager", "kwargs": {"swiglu_limit": 10.0}})
    handle = VeomniOp("swiglu_mlp", "standard", selection, site="layers.0.mlp")
    seen = {}

    def record(*args, **kwargs):
        seen.update(kwargs)
        return torch.zeros(1)

    handle._entry = type(handle._entry)(
        op="swiglu_mlp",
        variant="standard",
        impl="eager",
        description="test",
        wrapper=record,
    )
    handle(torch.zeros(1), swiglu_limit=5.0)
    assert seen == {"swiglu_limit": 5.0}


def test_sites_with_the_same_row_and_arguments_share_a_handle():
    """Intern keys on the resolved row *and* its arguments, so nothing leaks across."""
    selection = OpsSelection.from_config_value({"name": "eager", "kwargs": {"swiglu_limit": 10.0}})
    first = VeomniOp("swiglu_mlp", "standard", selection, site="layers.3.mlp")
    second = VeomniOp("swiglu_mlp", "standard", selection, site="layers.4.mlp")
    assert first is second


def test_sites_with_different_arguments_do_not_share_a_handle():
    selection = OpsSelection.from_config_value(
        {
            "name": "eager",
            "kwargs": {"swiglu_limit": 10.0},
            "overrides": [{"site": r"layers\.7\.mlp", "kwargs": {"swiglu_limit": 5.0}}],
        }
    )
    normal = VeomniOp("swiglu_mlp", "standard", selection, site="layers.3.mlp")
    special = VeomniOp("swiglu_mlp", "standard", selection, site="layers.7.mlp")
    assert normal is not special
    assert (normal.kwargs, special.kwargs) == ({"swiglu_limit": 10.0}, {"swiglu_limit": 5.0})


def test_an_interned_handle_survives_pickle_with_its_arguments():
    selection = OpsSelection.from_config_value({"name": "eager", "kwargs": {"swiglu_limit": 10.0}})
    handle = VeomniOp("swiglu_mlp", "standard", selection, site="layers.0.mlp")
    restored = pickle.loads(pickle.dumps(handle))
    assert restored is handle
    assert restored.kwargs == {"swiglu_limit": 10.0}


def test_the_config_accepts_both_shapes_and_round_trips_the_mapping_one():
    """A bare string stays valid; the mapping form survives ``to_dict``."""
    plain = OpsImplementationConfig(linear_implementation="eager")
    assert plain.to_dict()["linear_implementation"] == "eager"

    mapping = {
        "name": "qat",
        "kwargs": {"dtype": "fp8"},
        "overrides": [{"site": r"layers\.0\..*", "name": "eager"}],
    }
    configured = OpsImplementationConfig(linear_implementation=mapping)
    assert isinstance(configured.linear_implementation, OpsSelection)

    reloaded = OpsImplementationConfig(**configured.to_dict())
    assert reloaded.linear_implementation.all_impls() == configured.linear_implementation.all_impls()
    assert reloaded.linear_implementation.kwargs == {"dtype": "fp8"}


def test_a_deprecated_name_is_rewritten_in_every_form():
    """The rename tables must not apply to bare strings only.

    A value that works as ``moe_implementation: fused_mlu_triton`` and fails as
    ``moe_implementation: {name: fused_mlu_triton}`` is the kind of asymmetry
    nobody finds until they copy a config.
    """
    plain = OpsImplementationConfig(moe_implementation="fused_mlu_triton")
    assert plain.moe_implementation == "fused_triton"

    named = OpsImplementationConfig(moe_implementation={"name": "fused_mlu_triton"})
    assert named.moe_implementation.name == "fused_triton"

    overridden = OpsImplementationConfig(
        moe_implementation={
            "name": "eager",
            "overrides": [{"site": r"layers\.\d+\.experts", "name": "fused_mlu_triton"}],
        }
    )
    assert overridden.moe_implementation.overrides[0].name == "fused_triton"


def test_hardware_validation_covers_every_row_an_override_can_produce():
    """A rule that never runs on this host is still checked.

    Config-time validation has no site, so it has to clear the default *and* every
    override. Checking only the default would let an NPU-only row sit in a rule
    until the layer that matches it builds.
    """
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        pytest.skip("this case asserts the non-NPU refusal")

    with pytest.raises(ValueError, match="requires Ascend NPU"):
        OpsImplementationConfig(
            moe_implementation={"name": "eager", "overrides": [{"site": r"layers\.0\.experts", "name": "fused_npu"}]}
        )
