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
from veomni.ops import OP_REGISTRY, VeomniOp
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
        description="test-only rounding row",
        dtypes=("fp8",),
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


def test_an_override_inherits_kwargs_it_does_not_use(quantizing_row):
    """Switching to a row with a different signature needs nothing cleared.

    ``eager`` takes no ``dtype``, so the inherited one is simply not passed to it. The
    selection still carries it -- resolving is a name question, and what a row accepts
    is answered when the handle is built.
    """
    selection = OpsSelection.from_config_value(
        {
            "name": quantizing_row,
            "kwargs": {"block_size": 64},
            "overrides": [{"site": r"layers\.0\.q_a_proj", "name": "eager"}],
        }
    )
    dropped = VeomniOp("linear", "standard", selection, site="layers.0.q_a_proj")
    assert dropped.impl == "eager"
    # ``eager`` declares ``block_size`` for the family-wide signature even though it
    # ignores it, so nothing is dropped here; the dropping is exercised by
    # ``test_a_reference_row_simply_ignores_the_arguments_it_does_not_take``.
    assert dropped.kwargs == {"block_size": 64}
    # The selection itself still carries the inherited argument at both sites: the
    # filtering is a property of the row, not of the override.
    assert selection.resolve_kwargs("layers.0.q_a_proj") == {"block_size": 64}
    # The site that keeps the quantizing row still gets it.
    kept = VeomniOp("linear", "standard", selection, site="layers.7.q_a_proj")
    assert kept.kwargs == {"block_size": 64}


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


def test_a_reference_row_simply_ignores_the_arguments_it_does_not_take(quantizing_row):
    """The enclosing block describes the family; the row decides what it can honour.

    A site may name a reference row under a block that sets ``dtype`` for the
    quantized sites. That row rounds nothing, so the key is not passed to it -- not
    an error, because the config is not claiming that row rounds.
    """
    selection = OpsSelection.from_config_value({"name": "eager", "kwargs": {"dtype": "fp8"}})
    handle = VeomniOp("linear", "standard", selection, site="layers.0.q_a_proj")
    assert handle.impl == "eager"
    assert handle.kwargs == {}


def test_a_key_no_row_of_the_family_takes_is_refused_at_config_time(quantizing_row):
    """Dropping inherited keys is safe only because a real typo is caught earlier.

    ``block_siz`` is not an argument of any ``linear`` row, so it fails when the
    config is read rather than being filtered away at each site.
    """
    with pytest.raises(ValueError, match=r"\['block_siz'\] is not a row argument of any 'linear' row"):
        OpsImplementationConfig(linear_implementation={"name": "qat", "kwargs": {"block_siz": 64}})


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
    with pytest.raises(ValueError, match=r"\['num_experts'\] is not a row argument of any 'moe_experts' row"):
        OpsImplementationConfig(moe_implementation={"name": "qat", "kwargs": {"num_experts": 256}})

    # The keys that *are* configurable still are, including the checkpoint's
    # storage dtype, which is the one the modeling also passes at the call site.
    configured = OpsImplementationConfig(
        moe_implementation={"name": "qat", "kwargs": {"dtype": "fp4", "block_size": 64}}
    )
    assert configured.moe_implementation.kwargs == {"dtype": "fp4", "block_size": 64}


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


def test_config_time_refuses_a_key_no_row_of_the_family_takes():
    """The mistake should not wait for a GPU: `qat` does not resolve on every host.

    ``expert_dtypo`` is not an argument of any ``moe_experts`` row, so it fails when
    the config is read. A key that *some* row takes is allowed even on a site that
    resolves elsewhere -- that is the inherited-argument case.
    """
    with pytest.raises(ValueError, match=r"\['expert_dtypo'\] is not a row argument of any 'moe_experts' row"):
        OpsImplementationConfig(moe_implementation={"name": "qat", "kwargs": {"expert_dtypo": "fp4"}})

    # An override naming a reference row while the enclosing block sets arguments is
    # a family description, not a mistake: the handle drops what that row cannot take.
    configured = OpsImplementationConfig(
        moe_implementation={
            "name": "qat",
            "kwargs": {"block_size": 128},
            "overrides": [{"site": r"layers\.0\.experts", "name": "eager"}],
        }
    )
    assert configured.moe_implementation.kwargs == {"block_size": 128}


def test_a_quantizing_row_may_carry_arguments():
    configured = OpsImplementationConfig(moe_implementation={"name": "qat", "kwargs": {"block_size": 64}})
    assert configured.moe_implementation.kwargs == {"block_size": 64}


# ---------------------------------------------------------------------------
# The precision axis: emulated vs native
# ---------------------------------------------------------------------------


def test_dtypes_declares_which_formats_a_row_rounds_to():
    """A row that rounds names its formats; a reference row names none.

    ``dtypes`` is what a config's ``dtype`` is validated against, and its emptiness
    is also how the registry distinguishes a rounding row from a reference one
    without a hand-written list per family.
    """
    assert OP_REGISTRY.row_dtypes("moe_experts", "qat") == ("fp8", "fp4")
    assert OP_REGISTRY.row_dtypes("moe_experts", "deepgemm") == ("fp8",)
    assert OP_REGISTRY.row_dtypes("linear", "torchao") == ("fp8", "nvfp4", "mxfp4")
    assert OP_REGISTRY.row_dtypes("gated_mlp", "torchao") == ("fp8", "nvfp4", "mxfp4")
    # A reference row rounds nothing, so it declares no format.
    assert OP_REGISTRY.row_dtypes("moe_experts", "fused_triton") == ()
    assert OP_REGISTRY.row_dtypes("linear", "eager") == ()
    assert OP_REGISTRY.row_dtypes("moe_experts", "no_such_row") is None


def test_the_handle_reports_whether_it_rounds():
    """``VeomniOp.rounds`` is derived from ``dtypes``, not a hand-kept name list.

    Built from rows that resolve on CPU: ``qat`` needs SM90 and ``deepgemm`` needs
    SM90/SM100, so their declaration is asserted through ``row_dtypes`` above.
    """
    from veomni.ops.registry import OpEntry

    def handle(impl, dtypes):
        op = object.__new__(VeomniOp)
        op.op, op.variant, op.impl, op.kwargs = "linear", "standard", impl, {}
        op._entry = OpEntry(
            op="linear",
            variant="standard",
            impl=impl,
            description="probe",
            wrapper=lambda *a, **k: None,
            dtypes=dtypes,
        )
        return op

    assert handle("qat", ("fp8",)).rounds
    assert handle("deepgemm", ("fp8",)).rounds
    assert not handle("fused_triton", ()).rounds


def test_the_per_family_quantizing_sets_name_the_rows_that_round():
    """The families that own one list the rows a guard must refuse a fallback for.

    ``deepgemm`` is the case that motivated listing it: absent, a non-SwiGLU
    activation reached the plain-``F.linear`` fallback and trained the routed
    experts at full precision while the config said ``deepgemm``.
    """
    from veomni.ops.kernels.gated_mlp._module import QUANTIZING_IMPLS as gated
    from veomni.ops.kernels.moe_experts import QUANTIZING_IMPLS as moe

    assert moe == frozenset({"qat", "deepgemm"})
    assert gated == frozenset({"qat", "torchao"})


def test_the_per_family_sets_agree_with_the_registry():
    """The hand-written sets must say what each row already declares.

    A row states the formats it rounds to (``dtypes``) at registration, and the
    guard sets list the rows a fallback would train unquantized. Those are the same
    fact recorded twice, so this pins them together: adding a rounding row without
    listing it -- the mistake that let ``deepgemm`` slip past the activation guard --
    fails here instead of at a user's first non-SwiGLU run.
    """
    from veomni.ops.kernels.gated_mlp._module import QUANTIZING_IMPLS as gated
    from veomni.ops.kernels.linear._module import QUANTIZING_IMPLS as linear
    from veomni.ops.kernels.moe_experts import QUANTIZING_IMPLS as moe

    for op, declared in (("linear", linear), ("gated_mlp", gated), ("moe_experts", moe)):
        rounds = {impl for impl in OP_REGISTRY.all_impls(op) if OP_REGISTRY.row_dtypes(op, impl)}
        assert declared == rounds, f"{op}: QUANTIZING_IMPLS disagrees with the registry's dtypes"


def test_a_native_row_ignores_the_arguments_it_does_not_take():
    """Its format is fixed by the row, so ``dtype`` is not passed to it.

    ``deepgemm_fp8`` casts to FP8 at the width DeepGEMM fixes. An enclosing
    ``kwargs`` that sets ``dtype: fp4`` is filtered away rather than raising --
    the block describes the family, and ``qat`` at another site does take it.
    """
    configured = OpsImplementationConfig(moe_implementation={"name": "deepgemm", "kwargs": {"dtype": "fp4"}})
    assert configured.moe_implementation.kwargs == {"dtype": "fp4"}
    # The handle is where the filtering happens, and it needs a device the row
    # resolves on. What matters is that the key never reaches the call.
    from veomni.ops.registry import OpEntry

    seen = {}

    def record(*args, **kwargs):
        seen.update(kwargs)
        return torch.zeros(1)

    handle = object.__new__(VeomniOp)
    handle.op, handle.variant, handle.impl, handle.kwargs = "moe_experts", "standard", "deepgemm", {}
    handle._entry = OpEntry(op="moe_experts", variant="standard", impl="deepgemm", description="probe", wrapper=record)
    handle(torch.zeros(1), num_experts=8)
    assert seen == {"num_experts": 8}


def test_the_native_row_still_takes_the_family_signature_at_the_call_site():
    """``_check_kwargs`` constrains a config, not a call.

    ``DeepseekV4Experts`` passes ``dtype`` at every call whatever row it
    resolved, so a native row must still accept it positionally-equivalent: the
    check would otherwise break every DeepSeek-V4 expert forward.
    """
    from veomni.ops.registry import OpEntry

    seen = {}

    def record(*args, **kwargs):
        seen.update(kwargs)
        return torch.zeros(1)

    handle = object.__new__(VeomniOp)
    handle.op, handle.variant, handle.impl, handle.kwargs = "moe_experts", "standard", "deepgemm", {}
    handle._entry = OpEntry(op="moe_experts", variant="standard", impl="deepgemm", description="probe", wrapper=record)
    handle(torch.zeros(1), num_experts=8, dtype="fp4")
    assert seen == {"num_experts": 8, "dtype": "fp4"}


def test_a_native_row_is_refused_where_a_deployment_kernel_is_not_reached():
    """Quantizing moe_experts rows refuse activations their fused kernel cannot evaluate.

    ``deepgemm`` bakes SiLU into its grouped GEMM, so it refuses non-SiLU activations
    directly inside the wrapper rather than falling back to BF16 eager.
    """
    from veomni.ops.kernels.moe_experts.standard import deepgemm, qat

    zero = torch.empty(0)
    with pytest.raises(ValueError, match="moe_implementation='deepgemm' computes SiLU only"):
        deepgemm.wrapper(zero, zero, zero, zero, zero, zero, zero, num_experts=1, activation="gelu")
    with pytest.raises(ValueError, match="moe_implementation='qat' cannot run on hidden_act='relu'"):
        qat.wrapper(zero, zero, zero, zero, zero, zero, zero, num_experts=1, activation="relu")
