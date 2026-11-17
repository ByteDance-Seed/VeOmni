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
from veomni.ops import OP_REGISTRY, RowPrecision, VeomniOp
from veomni.ops.selection import OpsSelection, SiteOverride


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


#: A quantizing row the CPU can resolve, so the argument plumbing is testable
#: without an SM90 host. It rounds nothing; what it proves is that ``kwargs`` reach
#: a row that declares itself quantizing, and that a non-quantizing row refuses them.
def _round_forward(x: torch.Tensor, *, block_size: int = 128) -> tuple[torch.Tensor, object]:
    from veomni.ops.registry import SavedState

    del block_size
    return x, SavedState(())


def _round_backward(grad_output: torch.Tensor, saved: object) -> tuple[torch.Tensor]:
    del saved
    return (grad_output,)


@pytest.fixture
def quantizing_row():
    """Register ``test_round`` on the ``linear/standard`` family for one test."""
    from veomni.ops import OP_REGISTRY, register_op

    register_op(
        "linear",
        "standard",
        "test_round",
        _round_forward,
        _round_backward,
        description="test-only emulated row",
        precision=RowPrecision.EMULATED,
    )
    try:
        yield "test_round"
    finally:
        for key in [key for key in OP_REGISTRY._entries if key[2] == "test_round"]:
            del OP_REGISTRY._entries[key]


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
    selection = OpsSelection.from_config_value({"name": "qat", "kwargs": {"block_size": 64}})
    assert selection.resolve_kwargs("layers.3.q_a_proj") == {"block_size": 64}


def test_an_override_merges_kwargs_by_default():
    """A rule that adjusts one argument keeps the rest."""
    selection = OpsSelection.from_config_value(
        {
            "name": "qat",
            "kwargs": {"block_size": 128},
            "overrides": [{"site": r"layers\.\d+\.o_a_proj", "kwargs": {"block_size": 64}}],
        }
    )
    assert selection.resolve_kwargs("layers.2.o_a_proj") == {"block_size": 64}
    assert selection.resolve_kwargs("layers.2.q_a_proj") == {"block_size": 128}


def test_an_override_can_replace_the_whole_kwargs_set():
    """Switching to a reference row needs the inherited set dropped.

    ``liger_kernel`` is not a rounding row, so it takes no row arguments;
    inheriting the enclosing block's ``block_size`` would be refused.
    """
    selection = OpsSelection.from_config_value(
        {
            "name": "qat",
            "kwargs": {"block_size": 64},
            "overrides": [{"site": r"layers\.0\.mlp", "name": "liger_kernel", "kwargs": {}, "replace_kwargs": True}],
        }
    )
    assert selection.resolve_kwargs("layers.0.mlp") == {}
    assert selection.resolve_kwargs("layers.7.mlp") == {"block_size": 64}


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


def test_a_non_quantizing_row_refuses_row_arguments(quantizing_row):
    """Row arguments describe how to round, so a reference row must not take any.

    Accepting them would leave the config saying ``dtype: fp8`` while the row ran
    no rounding at all.
    """
    selection = OpsSelection.from_config_value({"name": "eager", "kwargs": {"block_size": 64}})
    with pytest.raises(ValueError, match="is not a rounding row"):
        VeomniOp("linear", "standard", selection, site="layers.0.q_a_proj")


def test_a_quantizing_row_still_has_to_declare_the_key(quantizing_row):
    """A typo must not be dropped: the row would run its own default instead."""
    selection = OpsSelection.from_config_value({"name": quantizing_row, "kwargs": {"block_siz": 64}})
    with pytest.raises(ValueError, match=r"does not accept \['block_siz'\]"):
        VeomniOp("linear", "standard", selection, site="layers.0.q_a_proj")


def test_a_row_argument_the_row_takes_is_kept_on_the_handle(quantizing_row):
    selection = OpsSelection.from_config_value({"name": quantizing_row, "kwargs": {"block_size": 64}})
    handle = VeomniOp("linear", "standard", selection, site="layers.0.q_a_proj")
    assert handle.kwargs == {"block_size": 64}


def test_the_config_beats_the_value_the_modeling_passes(quantizing_row):
    """Three sources, in order: row default, then modeling, then config.

    A checkpoint records what its weights were stored as; the config records what
    this run is asked to train at. The config has to be the one that survives, or a
    YAML saying ``block_size: 64`` would be ignored wherever the modeling passes its
    own value -- which is every DeepSeek-V4 expert call.
    """
    selection = OpsSelection.from_config_value({"name": quantizing_row, "kwargs": {"block_size": 64}})
    handle = VeomniOp("linear", "standard", selection, site="layers.0.q_a_proj")
    seen = {}

    def record(*args, **kwargs):
        seen.update(kwargs)
        return torch.zeros(1)

    handle._entry = type(handle._entry)(
        op="linear",
        variant="standard",
        impl=quantizing_row,
        description="test",
        wrapper=record,
    )
    handle(torch.zeros(1), block_size=32)
    assert seen == {"block_size": 64}


def test_a_key_the_config_does_not_set_keeps_the_modeling_value(quantizing_row):
    """Silence in the config is not an override: the call site's value passes through."""
    selection = OpsSelection.from_config_value({"name": quantizing_row, "kwargs": {}})
    handle = VeomniOp("linear", "standard", selection, site="layers.0.q_a_proj")
    seen = {}

    def record(*args, **kwargs):
        seen.update(kwargs)
        return torch.zeros(1)

    handle._entry = type(handle._entry)(
        op="linear",
        variant="standard",
        impl=quantizing_row,
        description="test",
        wrapper=record,
    )
    handle(torch.zeros(1), block_size=32)
    assert seen == {"block_size": 32}


def test_a_parameter_the_row_requires_is_not_configurable():
    """``num_experts`` is the module's shape, not a preference.

    It has no default because the row cannot run without it, so the modeling always
    supplies it. Letting a config pin it would let a typo launch the kernel over the
    wrong expert count -- refused at config parse, like any other key the row does
    not offer.
    """
    with pytest.raises(ValueError, match=r"'qat' does not accept \['num_experts'\]"):
        OpsImplementationConfig(moe_implementation={"name": "qat", "kwargs": {"num_experts": 256}})

    # The keys that *are* configurable still are, including the checkpoint's
    # storage dtype, which is the one the modeling also passes at the call site.
    configured = OpsImplementationConfig(
        moe_implementation={"name": "qat", "kwargs": {"expert_dtype": "fp4", "block_size": 64}}
    )
    assert configured.moe_implementation.kwargs == {"expert_dtype": "fp4", "block_size": 64}


def test_sites_with_the_same_row_and_arguments_share_a_handle(quantizing_row):
    """Intern keys on the resolved row *and* its arguments, so nothing leaks across."""
    selection = OpsSelection.from_config_value({"name": quantizing_row, "kwargs": {"block_size": 64}})
    first = VeomniOp("linear", "standard", selection, site="layers.3.q_a_proj")
    second = VeomniOp("linear", "standard", selection, site="layers.4.q_a_proj")
    assert first is second


def test_sites_with_different_arguments_do_not_share_a_handle(quantizing_row):
    selection = OpsSelection.from_config_value(
        {
            "name": quantizing_row,
            "kwargs": {"block_size": 128},
            "overrides": [{"site": r"layers\.7\.q_a_proj", "kwargs": {"block_size": 64}}],
        }
    )
    normal = VeomniOp("linear", "standard", selection, site="layers.3.q_a_proj")
    special = VeomniOp("linear", "standard", selection, site="layers.7.q_a_proj")
    assert normal is not special
    assert (normal.kwargs, special.kwargs) == ({"block_size": 128}, {"block_size": 64})


def test_an_interned_handle_survives_pickle_with_its_arguments(quantizing_row):
    selection = OpsSelection.from_config_value({"name": quantizing_row, "kwargs": {"block_size": 64}})
    handle = VeomniOp("linear", "standard", selection, site="layers.0.q_a_proj")
    restored = pickle.loads(pickle.dumps(handle))
    assert restored is handle
    assert restored.kwargs == {"block_size": 64}


def test_the_config_accepts_both_shapes_and_round_trips_the_mapping_one():
    """A bare string stays valid; the mapping form survives ``to_dict``."""
    plain = OpsImplementationConfig(linear_implementation="eager")
    assert plain.to_dict()["linear_implementation"] == "eager"

    mapping = {
        "name": "qat",
        "kwargs": {"block_size": 64},
        "overrides": [{"site": r"layers\.0\..*", "name": "eager"}],
    }
    configured = OpsImplementationConfig(linear_implementation=mapping)
    assert isinstance(configured.linear_implementation, OpsSelection)

    reloaded = OpsImplementationConfig(**configured.to_dict())
    assert reloaded.linear_implementation.all_impls() == configured.linear_implementation.all_impls()
    assert reloaded.linear_implementation.kwargs == {"block_size": 64}


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


def test_config_time_refuses_row_arguments_on_a_reference_row():
    """The mistake should not wait for a GPU: `qat` does not resolve on every host.

    A rule that inherits a reference row from the enclosing block and adds only
    `dtype` is the same error as writing it on `name:`, and is judged the same way.
    """
    with pytest.raises(ValueError, match="'fused_triton' is not a rounding row"):
        OpsImplementationConfig(moe_implementation={"name": "fused_triton", "kwargs": {"block_size": 64}})

    with pytest.raises(ValueError, match="'eager' is not a rounding row"):
        OpsImplementationConfig(
            moe_implementation={
                "name": "qat",
                "kwargs": {"block_size": 128},
                "overrides": [{"site": r"layers\.0\.experts", "name": "eager", "kwargs": {"block_size": 64}}],
            }
        )


def test_a_quantizing_row_may_carry_arguments():
    configured = OpsImplementationConfig(moe_implementation={"name": "qat", "kwargs": {"block_size": 64}})
    assert configured.moe_implementation.kwargs == {"block_size": 64}


# ---------------------------------------------------------------------------
# The precision axis: emulated vs native
# ---------------------------------------------------------------------------


def test_every_row_reports_a_precision_and_the_three_values_are_distinct():
    """The axis has to separate the three cases a bool could not.

    ``deepgemm_fp8`` rounds *and* runs the GEMM in FP8, so ``quantizes=True`` made it
    indistinguishable from ``qat`` -- which rounds but multiplies in the model
    dtype. Both answers were wrong: treating it as emulated claimed the parameters
    stay high precision, and treating it as exact claimed nothing is rounded.
    """
    expected = {
        ("moe_experts", "fused_triton"): RowPrecision.EXACT,
        ("moe_experts", "qat"): RowPrecision.EMULATED,
        ("moe_experts", "deepgemm_fp8"): RowPrecision.NATIVE,
    }
    for (op, impl), want in expected.items():
        assert OP_REGISTRY.precision(op, impl) is want

    # ``exact`` is the only value that does not round.
    assert [m.rounds_operands for m in RowPrecision] == [False, True, True]
    # ... and only emulation is parameterised, which is why a native row takes no
    # config arguments even though it rounds.
    assert [m.configurable for m in RowPrecision] == [False, True, False]


def test_an_unregistered_row_has_no_precision():
    """``None`` means "not a row", which callers must not confuse with ``EXACT``."""
    assert OP_REGISTRY.precision("moe_experts", "no_such_row") is None
    assert OP_REGISTRY.quantizes("moe_experts", "no_such_row") is None


def test_emulated_impls_is_derived_from_the_registry_not_a_hand_written_list():
    """The guard sets are computed, so a new emulated row is complete on registration.

    ``deepgemm_fp8`` is the case that motivated it: listed by hand, it was missing
    from ``moe_experts.QUANTIZING_IMPLS`` and so slipped past the non-SwiGLU guard.
    """
    assert OP_REGISTRY.emulated_impls("moe_experts") == frozenset({"qat"})
    # A native row rounds but is not emulated -- the distinction the sets now carry.
    assert "deepgemm_fp8" in OP_REGISTRY.rounding_impls("moe_experts")
    assert "deepgemm_fp8" not in OP_REGISTRY.emulated_impls("moe_experts")


def test_a_native_row_refuses_row_arguments():
    """Its format is its name, so a config cannot choose a width or dtype for it.

    Accepting ``expert_dtype: fp4`` here would be the same silent disagreement as
    accepting ``block_size`` on ``eager``: the config would say FP4 while the row
    cast to FP8.
    """
    with pytest.raises(ValueError, match="'deepgemm_fp8' is a native row of 'moe_experts'"):
        OpsImplementationConfig(moe_implementation={"name": "deepgemm_fp8", "kwargs": {"expert_dtype": "fp4"}})

    # No row arguments at all, so the empty mapping is not a loophole either.
    assert OpsImplementationConfig(moe_implementation={"name": "deepgemm_fp8"}).moe_implementation.name == (
        "deepgemm_fp8"
    )


def test_the_native_row_still_takes_the_family_signature_at_the_call_site():
    """``_check_kwargs`` constrains a config, not a call.

    ``DeepseekV4Experts`` passes ``expert_dtype`` at every call whatever row it
    resolved, so a native row must still accept it positionally-equivalent: the
    check would otherwise break every DeepSeek-V4 expert forward.
    """
    from veomni.ops.registry import OpEntry

    seen = {}

    def record(*args, **kwargs):
        seen.update(kwargs)
        return torch.zeros(1)

    handle = object.__new__(VeomniOp)
    handle.op, handle.variant, handle.impl, handle.kwargs = "moe_experts", "standard", "deepgemm_fp8", {}
    handle._entry = OpEntry(
        op="moe_experts", variant="standard", impl="deepgemm_fp8", description="probe", wrapper=record
    )
    handle(torch.zeros(1), num_experts=8, expert_dtype="fp4")
    assert seen == {"num_experts": 8, "expert_dtype": "fp4"}


def test_a_native_row_is_refused_where_a_deployment_kernel_is_not_reached():
    """The non-SwiGLU guard asks "does this row round?", so it must catch both kinds.

    ``deepgemm_fp8`` bakes SwiGLU into its grouped GEMM exactly as ``qat`` does, so
    falling back to the Python act_fn loop would run it at full precision. It was
    the hand-written ``frozenset({"qat"})`` that let it through.
    """
    from torch import nn

    from veomni.ops.kernels.moe_experts import reject_quantizing_act_fn

    class _Experts(nn.Module):
        def __init__(self, impl):
            super().__init__()
            self.use_swiglu_mlp = False
            self.veomni_moe = type("_Handle", (), {"impl": impl})()

    for impl in ("qat", "deepgemm_fp8"):
        with pytest.raises(ValueError, match="cannot run with a non-SwiGLU activation"):
            reject_quantizing_act_fn(_Experts(impl))

    # A row that rounds nothing keeps the fallback: it has no fused kernel to miss.
    reject_quantizing_act_fn(_Experts("fused_triton"))
