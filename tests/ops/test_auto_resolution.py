# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Tests for ``OpsImplementationConfig`` auto-resolution and legacy aliases.

Covers:
- ``"auto"`` resolves per device via the registered ``OpPolicy`` table.
- Legacy aliases (``moe_implementation="fused"`` -> ``"auto"``,
  ``cross_entropy_loss_implementation="npu"`` -> ``"chunk_loss"``) emit a
  deprecation warning and resolve correctly.
- Explicit values are honoured verbatim.
- When an auto-picked backend's software requirement is unmet (e.g.
  ``liger_kernel`` selected on a host without ``liger-kernel`` installed),
  the resolver falls back to ``"eager"`` with a ``WARNING`` — the contract of
  ``"auto"``.
- ``install_loss_mapping`` accepts the new ``"chunk_loss"`` name and still
  honours the legacy ``"npu"`` value.
"""

from __future__ import annotations

import logging
import unittest.mock as mock

import pytest

import veomni.ops  # noqa: F401 — trigger kernel registrations
from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.ops.config.auto_policy import list_op_policies


# veomni's logger sets ``propagate=False`` and attaches its own stdout
# StreamHandler at first import, so pytest's ``caplog`` (which patches the
# *root* logger) never sees the records. Attach a list-collecting handler
# directly to the ``"veomni"`` logger for the duration of one test.
@pytest.fixture
def veomni_log():
    """Yield a list that grows with formatted log records from ``veomni.*``."""
    records: list[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _ListHandler(level=logging.DEBUG)
    veomni_logger = logging.getLogger("veomni")
    veomni_logger.addHandler(handler)
    try:
        yield records
    finally:
        veomni_logger.removeHandler(handler)


# All __post_init__ assertions are run with a clean MODELING_BACKEND so the
# attention rewrite does not get in the way of the auto checks.

NON_ATTENTION_FIELDS = [
    "moe_implementation",
    "cross_entropy_loss_implementation",
    "rms_norm_implementation",
    "swiglu_mlp_implementation",
    "rotary_pos_emb_implementation",
    "load_balancing_loss_implementation",
]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_all_non_attention_defaults_are_auto():
    # Build the dataclass without going through __post_init__ so we can read
    # the raw defaults declared on the class.
    cls_defaults = {f: OpsImplementationConfig.__dataclass_fields__[f].default for f in NON_ATTENTION_FIELDS}
    for field, default in cls_defaults.items():
        assert default == "auto", f"{field} default should be 'auto', got {default!r}"


def test_attn_implementation_default_is_flash_attention_2():
    assert OpsImplementationConfig.__dataclass_fields__["attn_implementation"].default == "flash_attention_2"


# ---------------------------------------------------------------------------
# OpPolicy registrations
# ---------------------------------------------------------------------------


def test_every_non_attention_field_has_a_policy():
    registered = {p.config_field for p in list_op_policies()}
    for field in NON_ATTENTION_FIELDS:
        assert field in registered, f"{field} is missing an OpPolicy registration"


def test_legacy_aliases_present():
    by_field = {p.config_field: p for p in list_op_policies()}
    assert by_field["moe_implementation"].legacy_aliases.get("fused") == "auto"
    assert by_field["cross_entropy_loss_implementation"].legacy_aliases.get("npu") == "chunk_loss"


# ---------------------------------------------------------------------------
# Auto resolution per device
# ---------------------------------------------------------------------------


def _force_device(device_type: str):
    """Patch the static device probe to return *device_type* for one config."""
    return mock.patch.object(OpsImplementationConfig, "_current_device_type", return_value=device_type)


def _all_packages_available():
    """Return a fresh _PACKAGE_FLAGS dict with everything reported as installed."""
    return {
        "flash_attn": True,
        "liger_kernel": True,
        "torch_npu": True,
        "diffusers": True,
        "av": True,
        "librosa": True,
        "soundfile": True,
        "triton": True,
        "quack": True,
        "veomni_patch": True,
    }


def test_auto_resolves_to_gpu_picks():
    with _force_device("gpu"), mock.patch("veomni.utils.import_utils._PACKAGE_FLAGS", _all_packages_available()):
        cfg = OpsImplementationConfig()
        assert cfg.moe_implementation == "fused_triton"
        assert cfg.cross_entropy_loss_implementation == "chunk_loss"
        assert cfg.rms_norm_implementation == "liger_kernel"
        assert cfg.rotary_pos_emb_implementation == "liger_kernel"
        assert cfg.swiglu_mlp_implementation == "liger_kernel"
        assert cfg.load_balancing_loss_implementation == "triton"


def test_auto_resolves_to_npu_picks():
    with _force_device("npu"), mock.patch("veomni.utils.import_utils._PACKAGE_FLAGS", _all_packages_available()):
        cfg = OpsImplementationConfig()
        assert cfg.moe_implementation == "fused_npu"
        assert cfg.cross_entropy_loss_implementation == "chunk_loss"
        assert cfg.rms_norm_implementation == "npu"
        assert cfg.rotary_pos_emb_implementation == "npu"
        # No NPU entry in swiglu policy -> degrades to eager.
        assert cfg.swiglu_mlp_implementation == "eager"
        assert cfg.load_balancing_loss_implementation == "triton"


def test_auto_on_cpu_host_falls_back_to_eager_everywhere():
    """No accelerator means no auto entry -> every field resolves to 'eager'."""
    with _force_device("cpu"), mock.patch("veomni.utils.import_utils._PACKAGE_FLAGS", _all_packages_available()):
        cfg = OpsImplementationConfig()
        for field in NON_ATTENTION_FIELDS:
            assert getattr(cfg, field) == "eager", f"{field} should fall back to eager on CPU host"


# ---------------------------------------------------------------------------
# Best-effort fallback when chosen backend is unavailable
# ---------------------------------------------------------------------------


def test_auto_degrades_to_eager_when_triton_missing_on_npu(veomni_log):
    """NPU host without triton/triton-ascend -> load_balancing_loss falls back to eager.

    Regression for the NPU CI failure where the default config picked
    ``load_balancing_loss_implementation='triton'`` and crashed with
    ``ModuleNotFoundError: No module named 'triton'`` at apply time.
    """
    flags = _all_packages_available()
    flags["triton"] = False
    with _force_device("npu"), mock.patch("veomni.utils.import_utils._PACKAGE_FLAGS", flags):
        cfg = OpsImplementationConfig()
    assert cfg.load_balancing_loss_implementation == "eager"
    text = "\n".join(veomni_log)
    assert "load_balancing_loss_implementation" in text
    assert "falling back to 'eager'" in text


def test_explicit_triton_raises_when_unavailable():
    """Explicit values do NOT auto-degrade — they raise on misconfiguration."""
    flags = _all_packages_available()
    flags["triton"] = False
    with _force_device("npu"), mock.patch("veomni.utils.import_utils._PACKAGE_FLAGS", flags):
        with pytest.raises(ValueError, match="requires triton"):
            OpsImplementationConfig(load_balancing_loss_implementation="triton")


def test_auto_degrades_to_eager_when_liger_missing(veomni_log):
    """GPU host without liger-kernel -> liger picks degrade to eager + warning."""
    flags = _all_packages_available()
    flags["liger_kernel"] = False
    with _force_device("gpu"), mock.patch("veomni.utils.import_utils._PACKAGE_FLAGS", flags):
        cfg = OpsImplementationConfig()
    assert cfg.rms_norm_implementation == "eager"
    assert cfg.rotary_pos_emb_implementation == "eager"
    assert cfg.swiglu_mlp_implementation == "eager"
    text = "\n".join(veomni_log)
    assert "rms_norm_implementation" in text
    assert "falling back to 'eager'" in text


# ---------------------------------------------------------------------------
# Legacy aliases
# ---------------------------------------------------------------------------


def test_legacy_alias_fused_resolves_to_fused_triton_on_gpu(veomni_log):
    with _force_device("gpu"), mock.patch("veomni.utils.import_utils._PACKAGE_FLAGS", _all_packages_available()):
        cfg = OpsImplementationConfig(moe_implementation="fused")
    assert cfg.moe_implementation == "fused_triton"
    text = "\n".join(veomni_log)
    assert "moe_implementation='fused'" in text
    assert "deprecated" in text


def test_legacy_alias_fused_resolves_to_fused_npu_on_npu():
    with _force_device("npu"), mock.patch("veomni.utils.import_utils._PACKAGE_FLAGS", _all_packages_available()):
        cfg = OpsImplementationConfig(moe_implementation="fused")
    assert cfg.moe_implementation == "fused_npu"


def test_legacy_alias_npu_for_cross_entropy_resolves_to_chunk_loss(veomni_log):
    with _force_device("gpu"), mock.patch("veomni.utils.import_utils._PACKAGE_FLAGS", _all_packages_available()):
        cfg = OpsImplementationConfig(cross_entropy_loss_implementation="npu")
    assert cfg.cross_entropy_loss_implementation == "chunk_loss"
    text = "\n".join(veomni_log)
    assert "cross_entropy_loss_implementation='npu'" in text
    assert "deprecated" in text


# ---------------------------------------------------------------------------
# Explicit overrides bypass auto resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, value",
    [
        ("rms_norm_implementation", "eager"),
        ("rotary_pos_emb_implementation", "eager"),
        ("swiglu_mlp_implementation", "eager"),
        ("moe_implementation", "eager"),
        ("cross_entropy_loss_implementation", "eager"),
        ("load_balancing_loss_implementation", "eager"),
        ("rms_norm_implementation", "liger_kernel"),
        ("moe_implementation", "fused_triton"),
        ("cross_entropy_loss_implementation", "chunk_loss"),
    ],
)
def test_explicit_values_pass_through(field, value):
    with _force_device("gpu"), mock.patch("veomni.utils.import_utils._PACKAGE_FLAGS", _all_packages_available()):
        cfg = OpsImplementationConfig(**{field: value})
        assert getattr(cfg, field) == value


# ---------------------------------------------------------------------------
# install_loss_mapping accepts both "chunk_loss" and the legacy "npu"
# ---------------------------------------------------------------------------


def test_install_loss_mapping_chunk_loss():
    from veomni.ops.kernels.cross_entropy import install_loss_mapping

    label = install_loss_mapping("chunk_loss")
    assert label == "CrossEntropy (chunk_loss)"


def test_install_loss_mapping_npu_alias_emits_warning(veomni_log):
    from veomni.ops.kernels.cross_entropy import install_loss_mapping

    label = install_loss_mapping("npu")
    assert label == "CrossEntropy (chunk_loss)"
    assert any("deprecated" in line for line in veomni_log)


# ---------------------------------------------------------------------------
# Resolved-summary log
# ---------------------------------------------------------------------------


def test_resolved_summary_marks_auto_fields_only(veomni_log):
    """The summary tags fields resolved from auto, and leaves explicit ones unmarked."""
    with _force_device("gpu"), mock.patch("veomni.utils.import_utils._PACKAGE_FLAGS", _all_packages_available()):
        OpsImplementationConfig(moe_implementation="eager")  # explicit
    summary = next(line for line in veomni_log if line.startswith("Resolved ops implementation"))
    # Default attn + 5 auto fields + 1 explicit moe_implementation.
    assert "attn_implementation" in summary
    assert "moe_implementation                 = eager" in summary
    # No "(auto)" marker on the explicit field.
    assert "moe_implementation                 = eager (auto)" not in summary
    # Auto fields keep the marker.
    assert "rms_norm_implementation            = liger_kernel (auto)" in summary
    assert "cross_entropy_loss_implementation  = chunk_loss (auto)" in summary
