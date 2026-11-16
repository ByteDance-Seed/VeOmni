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

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments import OpsImplementationConfig
from veomni.ops import VeomniOp
from veomni.ops.selection import (
    LowPrecisionConfig,
    OpsSelection,
    SiteOverride,
    low_precision_impl,
)


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


def test_a_low_precision_block_resolves_through_the_backend_table():
    selection = OpsSelection.from_config_value(
        {"low_precision": {"enable": True, "dtype": "fp8", "qat": True, "backend": "tilelang"}}
    )
    assert selection.resolve_impl("layers.0.q_a_proj") == "qat"


def test_an_override_that_disables_quantization_falls_back_to_the_reference_row():
    """This is how "keep these layers in the model dtype" is spelled."""
    selection = OpsSelection.from_config_value(
        {
            "name": "liger_kernel",
            "low_precision": {"enable": True, "dtype": "fp8", "qat": True, "backend": "tilelang"},
            "overrides": [{"site": r"layers\.0\.mlp", "low_precision": {"enable": False}}],
        }
    )
    assert selection.resolve_impl("layers.5.mlp") == "qat"
    assert selection.resolve_impl("layers.0.mlp") == "liger_kernel"


def test_the_reverse_also_works_a_reference_default_with_quantized_sites():
    """The common shape: everything off, the routed experts on."""
    selection = OpsSelection.from_config_value(
        {
            "name": "fused_triton",
            "overrides": [
                {
                    "site": r"layers\.\d+\.experts",
                    "low_precision": {"enable": True, "dtype": "fp8", "qat": False, "backend": "deepgemm"},
                }
            ],
        }
    )
    assert selection.resolve_impl("layers.3.experts") == "deepgemm_fp8"
    assert selection.resolve_impl("layers.3.mlp") == "fused_triton"


def test_a_later_override_wins_over_an_earlier_one():
    """Ordering is the only conflict rule, so a narrow rule goes after a broad one."""
    selection = OpsSelection.from_config_value(
        {
            "name": "eager",
            "overrides": [
                {
                    "site": r"layers\.\d+\.q_a_proj",
                    "low_precision": {"enable": True, "dtype": "fp8", "qat": True, "backend": "tilelang"},
                },
                {"site": r"layers\.7\.q_a_proj", "name": "eager", "low_precision": {"enable": False}},
            ],
        }
    )
    assert selection.resolve_impl("layers.3.q_a_proj") == "qat"
    assert selection.resolve_impl("layers.7.q_a_proj") == "eager"


def test_a_partial_override_inherits_the_enclosing_block():
    """Only the fields a rule writes change; the rest comes from the outer block."""
    selection = OpsSelection.from_config_value(
        {
            "low_precision": {"enable": True, "dtype": "fp8", "qat": True, "backend": "tilelang"},
            "overrides": [
                {"site": r"layers\.\d+\.o_a_proj", "low_precision": {"dtype": "nvfp4", "backend": "torchao"}}
            ],
        }
    )
    # qat was inherited as True, so the override keeps fake quantization.
    assert selection.resolve_impl("layers.2.o_a_proj") == "torchao_nvfp4"
    assert selection.resolve_impl("layers.2.q_a_proj") == "qat"


def test_matching_is_anchored_so_a_suffix_does_not_leak():
    """``fullmatch``: ``layers.0.mlp`` must not match ``layers.10.mlp``."""
    selection = OpsSelection.from_config_value(
        {"name": "eager", "overrides": [{"site": r"layers\.0\.mlp", "name": "liger_kernel"}]}
    )
    assert selection.resolve_impl("layers.0.mlp") == "liger_kernel"
    assert selection.resolve_impl("layers.10.mlp") == "eager"


def test_all_impls_enumerates_every_row_the_field_can_produce():
    """Config-time validation has no site, so it clears every row the field names."""
    selection = OpsSelection.from_config_value(
        {
            "name": "liger_kernel",
            "low_precision": {"enable": True, "dtype": "fp8", "qat": True, "backend": "tilelang"},
            "overrides": [{"site": r"layers\.0\.mlp", "low_precision": {"enable": False}}],
        }
    )
    assert selection.all_impls() == ["qat", "liger_kernel"]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_an_unregistered_triple_is_refused_rather_than_downgraded():
    """A config asking for real NVFP4 must not quietly train something else."""
    with pytest.raises(ValueError, match="No registered row for dtype='nvfp4', qat=False, backend='torchao'"):
        low_precision_impl(LowPrecisionConfig(enable=True, dtype="nvfp4", qat=False, backend="torchao"))


def test_a_qat_only_backend_says_so():
    """``tilelang`` registers fake quantization only; the message names the fix."""
    with pytest.raises(ValueError, match="qat must be true"):
        low_precision_impl(LowPrecisionConfig(enable=True, dtype="fp8", qat=False, backend="tilelang"))


def test_enabling_without_a_backend_is_refused_at_construction():
    with pytest.raises(ValueError, match="requires a backend"):
        LowPrecisionConfig(enable=True, dtype="fp8")


def test_an_unknown_key_is_named_rather_than_ignored():
    """A typo in YAML would otherwise silently drop the setting it meant to write."""
    with pytest.raises(ValueError, match=r"Unknown low_precision keys: \['qatt'\]"):
        LowPrecisionConfig.from_dict({"enable": True, "dtype": "fp8", "qatt": True, "backend": "tilelang"})


def test_an_invalid_site_regex_is_refused():
    with pytest.raises(ValueError, match="is not a valid regex"):
        SiteOverride(site="layers.(0.mlp")


def test_an_override_without_a_site_is_refused():
    with pytest.raises(ValueError, match="requires a site pattern"):
        SiteOverride.from_dict({"name": "eager"})


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_veomni_op_accepts_a_selection_and_a_site():
    """The handle resolves at construction and keeps no site of its own."""
    selection = OpsSelection.from_config_value(
        {
            "name": "eager",
            "low_precision": {"enable": True, "dtype": "fp8", "qat": True, "backend": "tilelang"},
            "overrides": [{"site": r"layers\.0\..*", "low_precision": {"enable": False}}],
        }
    )
    # ``qat`` needs SM90, so the site that keeps it is checked by name only.
    plain = VeomniOp("linear", "standard", selection, site="layers.0.q_a_proj")
    assert plain.impl == "eager"


def test_sites_that_resolve_to_one_row_share_a_handle():
    """Intern is unchanged: the site decides the impl, and the impl is the key."""
    selection = OpsSelection.from_config_value("eager")
    first = VeomniOp("linear", "standard", selection, site="layers.3.q_a_proj")
    second = VeomniOp("linear", "standard", selection, site="layers.4.q_a_proj")
    assert first is second


def test_the_config_accepts_both_shapes_and_round_trips_the_mapping_one():
    """A bare string stays valid; the mapping form survives ``to_dict``."""
    plain = OpsImplementationConfig(linear_implementation="eager")
    assert plain.to_dict()["linear_implementation"] == "eager"

    mapping = {
        "low_precision": {"enable": True, "dtype": "fp8", "qat": True, "backend": "tilelang"},
        "overrides": [{"site": r"layers\.0\..*", "low_precision": {"enable": False}}],
    }
    configured = OpsImplementationConfig(linear_implementation=mapping)
    assert isinstance(configured.linear_implementation, OpsSelection)

    reloaded = OpsImplementationConfig(**configured.to_dict())
    assert reloaded.linear_implementation.all_impls() == configured.linear_implementation.all_impls()
