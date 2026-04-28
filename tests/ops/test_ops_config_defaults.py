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

"""Tests for OpsImplementationConfig defaults and device-compatibility validation.

Defaults are GPU-reasonable. Three things to lock down:

1. Each ``*_implementation`` field has the documented GPU default — a future
   accidental rename of ``"fused_triton"`` to ``"triton"`` etc. should fail
   here, not silently skip kernels.
2. ``moe_implementation: 'fused'`` (the pre-#678 alias that used to silently
   auto-pick by hardware) is rewritten to ``'fused_triton'`` with a
   deprecation warning, so old YAML keeps working.
3. On Ascend NPU, instantiating with default ``OpsImplementationConfig()``
   raises ``ValueError`` with a clear message and a suggested NPU value —
   we never silently fall back to a kernel that won't run on NPU.

The NPU branch is tested by mocking ``IS_NPU_AVAILABLE`` so the suite runs
on any CI host (mainline GPU CI, NPU CI, or a dev box).
"""

from unittest.mock import patch

import pytest


# Default per-field GPU-reasonable values. Kept as a single source of truth so
# adding a field to ``OpsImplementationConfig`` only needs one update here.
_EXPECTED_DEFAULTS = {
    "attn_implementation": "flash_attention_2",
    "moe_implementation": "fused_triton",
    "cross_entropy_loss_implementation": "liger_kernel",
    "rms_norm_implementation": "liger_kernel",
    "swiglu_mlp_implementation": "liger_kernel",
    "rotary_pos_emb_implementation": "liger_kernel",
    # Stays eager — see the dataclass docstring. ``apply_ops_config`` resolves
    # GLOBAL ops eagerly for every model, so a triton default would force
    # every dense run to depend on triton.
    "load_balancing_loss_implementation": "eager",
}


def _make_config_no_validation(**overrides):
    """Construct an ``OpsImplementationConfig`` while bypassing ``__post_init__``.

    Used by tests that only want to inspect the dataclass-level defaults
    without triggering the device / package availability checks (those are
    covered by their own dedicated tests below).
    """
    from veomni.arguments.arguments_types import OpsImplementationConfig

    with patch.object(OpsImplementationConfig, "__post_init__", lambda self: None):
        return OpsImplementationConfig(**overrides)


# ---------------------------------------------------------------------------
# 1. Defaults are GPU-reasonable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field_name,expected", list(_EXPECTED_DEFAULTS.items()))
def test_default_value_is_gpu_reasonable(field_name, expected):
    cfg = _make_config_no_validation()
    assert getattr(cfg, field_name) == expected, (
        f"{field_name} default changed; update _EXPECTED_DEFAULTS or the dataclass."
    )


def test_eager_defaults_classmethod_returns_eager_everywhere():
    """``eager_defaults()`` is the explicit-portable escape hatch.

    Used by the ``build_foundation_model`` standalone-script fallback and by
    tests that don't care about kernel selection — must keep every kernel
    field set to ``"eager"`` so it works on hosts without liger / triton."""
    from veomni.arguments.arguments_types import OpsImplementationConfig

    cfg = OpsImplementationConfig.eager_defaults()
    assert cfg.attn_implementation == "eager"
    assert cfg.moe_implementation == "eager"
    assert cfg.cross_entropy_loss_implementation == "eager"
    assert cfg.rms_norm_implementation == "eager"
    assert cfg.swiglu_mlp_implementation == "eager"
    assert cfg.rotary_pos_emb_implementation == "eager"
    assert cfg.load_balancing_loss_implementation == "eager"


def test_eager_defaults_classmethod_constructs_cleanly_on_npu():
    """``eager_defaults()`` must not raise on an NPU host.

    Regression test for the bug Gemini flagged in PR review: previously the
    classmethod inherited the dataclass default ``attn_implementation =
    "flash_attention_2"``, which (after the SP rewrite) is in
    ``_NPU_INCOMPATIBLE`` and would make ``__post_init__`` raise on NPU.
    The whole point of ``eager_defaults()`` is to be a portable escape
    hatch, so this path must construct without error on every device."""
    from veomni.arguments.arguments_types import OpsImplementationConfig

    with patch("veomni.utils.device.IS_NPU_AVAILABLE", True):
        cfg = OpsImplementationConfig.eager_defaults()
    assert cfg.attn_implementation == "eager"


# ---------------------------------------------------------------------------
# 2. Legacy alias rewrite for moe_implementation: 'fused' → 'fused_triton'
# ---------------------------------------------------------------------------


def test_legacy_moe_fused_alias_is_rewritten():
    """``veomni.utils.logging`` configures a non-propagating root logger that
    writes via a stdout handler captured at import time, so neither
    ``caplog`` nor ``capsys`` see the warning. We attach our own handler to
    the ``veomni`` logger to capture the warning emitted by
    ``_rewrite_legacy_aliases``."""
    import logging as stdlib_logging

    from veomni.arguments.arguments_types import OpsImplementationConfig

    records: list[stdlib_logging.LogRecord] = []

    class _ListHandler(stdlib_logging.Handler):
        def emit(self, record):
            records.append(record)

    veomni_logger = stdlib_logging.getLogger("veomni")
    handler = _ListHandler(level=stdlib_logging.WARNING)
    veomni_logger.addHandler(handler)
    try:
        # Force GPU branch so device validation doesn't preempt the alias rewrite.
        with patch("veomni.utils.device.IS_NPU_AVAILABLE", False):
            cfg = OpsImplementationConfig.eager_defaults()
            # eager_defaults sets moe_implementation='eager'; flip back to the
            # legacy value to exercise the alias rewrite. Re-running __post_init__
            # so the warning fires.
            cfg.moe_implementation = "fused"
            cfg.__post_init__()
    finally:
        veomni_logger.removeHandler(handler)

    assert cfg.moe_implementation == "fused_triton"
    deprecation_warnings = [r for r in records if "deprecated" in r.getMessage() and "fused" in r.getMessage()]
    assert deprecation_warnings, f"Expected a deprecation warning; got records: {[r.getMessage() for r in records]}"


# ---------------------------------------------------------------------------
# 3. NPU device-compatibility validation
# ---------------------------------------------------------------------------


# (field, GPU-only value, expected suggestion in the error message)
_NPU_INCOMPATIBLE_CASES = [
    ("attn_implementation", "flash_attention_2", "sdpa"),
    ("attn_implementation", "flash_attention_3", "sdpa"),
    ("attn_implementation", "veomni_flash_attention_2_with_sp", "sdpa"),
    ("moe_implementation", "fused_triton", "fused_npu"),
    ("moe_implementation", "fused_quack", "fused_npu"),
    ("cross_entropy_loss_implementation", "liger_kernel", "npu"),
    ("rms_norm_implementation", "liger_kernel", "npu"),
    ("rms_norm_implementation", "triton", "npu"),
    ("rotary_pos_emb_implementation", "liger_kernel", "npu"),
    ("rotary_pos_emb_implementation", "triton", "npu"),
    ("swiglu_mlp_implementation", "liger_kernel", "eager"),
]


@pytest.mark.parametrize("field_name,gpu_value,suggested", _NPU_INCOMPATIBLE_CASES)
def test_npu_default_raises_with_clear_suggestion(field_name, gpu_value, suggested):
    """On NPU, every GPU-only default value raises with a suggestion.

    The error message must:
    - name the offending field and value,
    - state that the current device is NPU,
    - include the suggested replacement value verbatim, so a user can copy
      it into their config without further guessing.
    """
    from veomni.arguments.arguments_types import OpsImplementationConfig

    # Start every field on a NPU-friendly value so the parametrized field is
    # the only one that can trip the validator. ``attn_implementation`` is
    # also in ``_NPU_INCOMPATIBLE`` (default ``flash_attention_2`` is GPU-only)
    # so it has to be explicitly set to a portable value here too.
    overrides = dict.fromkeys(_EXPECTED_DEFAULTS, "eager")
    overrides["attn_implementation"] = "sdpa"
    overrides[field_name] = gpu_value

    with patch("veomni.utils.device.IS_NPU_AVAILABLE", True):
        with pytest.raises(ValueError) as excinfo:
            OpsImplementationConfig(**overrides)

    message = str(excinfo.value)
    assert field_name in message
    assert gpu_value in message
    assert "NPU" in message
    assert suggested in message


def test_npu_default_constructor_raises_on_first_incompatible_field():
    """``OpsImplementationConfig()`` on NPU raises before any kernel is bound.

    The exact field that trips first depends on dict iteration order in
    ``_NPU_INCOMPATIBLE``; we just assert that *some* GPU-only default fires
    a ``ValueError`` (i.e. the user gets an early, actionable error rather
    than a runtime crash inside the kernel).
    """
    from veomni.arguments.arguments_types import OpsImplementationConfig

    with patch("veomni.utils.device.IS_NPU_AVAILABLE", True):
        with pytest.raises(ValueError, match="GPU-only"):
            OpsImplementationConfig()


def test_npu_with_explicit_npu_values_constructs_cleanly():
    """A correctly-overridden NPU config validates without raising.

    This is the path NPU users are pushed onto by the validator: explicit
    ``npu`` / ``fused_npu`` / ``eager`` per field.
    """
    from veomni.arguments.arguments_types import OpsImplementationConfig

    with patch("veomni.utils.device.IS_NPU_AVAILABLE", True):
        # ``_validate_implementations`` on NPU still checks ``torch_npu``
        # availability; mock the package check so the test runs on a dev box
        # that doesn't have torch_npu installed.
        with patch("veomni.utils.import_utils.is_torch_npu_available", return_value=True):
            cfg = OpsImplementationConfig(
                attn_implementation="eager",
                moe_implementation="fused_npu",
                cross_entropy_loss_implementation="npu",
                rms_norm_implementation="npu",
                swiglu_mlp_implementation="eager",
                rotary_pos_emb_implementation="npu",
                load_balancing_loss_implementation="triton",
            )
    assert cfg.moe_implementation == "fused_npu"


def test_load_balancing_triton_is_universal_on_npu():
    """``load_balancing_loss: triton`` works on NPU via ``triton-ascend``.

    This is the one field whose GPU default is *also* the NPU pick — the
    package name ``triton`` is shared by both stacks. The validator must
    not raise on NPU for this value; otherwise NPU users would have to
    override a default that already does the right thing.
    """
    from veomni.arguments.arguments_types import OpsImplementationConfig

    with patch("veomni.utils.device.IS_NPU_AVAILABLE", True):
        with patch("veomni.utils.import_utils.is_torch_npu_available", return_value=True):
            cfg = OpsImplementationConfig(
                attn_implementation="eager",
                moe_implementation="eager",
                cross_entropy_loss_implementation="eager",
                rms_norm_implementation="eager",
                swiglu_mlp_implementation="eager",
                rotary_pos_emb_implementation="eager",
                load_balancing_loss_implementation="triton",  # universal
            )
    assert cfg.load_balancing_loss_implementation == "triton"


def test_load_balancing_loss_triton_default_requires_triton_package():
    """``load_balancing_loss_implementation: triton`` is the new default.

    ``apply_global_ops`` resolves GLOBAL ops eagerly for *every* model
    (including dense models that never call this loss), so the backend
    declares ``requires=("triton",)``. On a host without triton (or
    triton-ascend), the validator must raise at config-parse time with a
    clear message — not crash later inside ``apply_global_ops`` when it
    imports the kernel module.

    Regression test for the second issue Codex flagged in PR review.
    """
    from veomni.arguments.arguments_types import OpsImplementationConfig

    # GPU host so the device-compatibility check is skipped.
    with patch("veomni.utils.device.IS_NPU_AVAILABLE", False):
        with patch("veomni.utils.import_utils.is_package_available", return_value=False):
            with pytest.raises(ValueError, match="triton.*triton-ascend"):
                # Have to bypass the liger validator too, otherwise the test
                # would fail there first on hosts without liger-kernel.
                with patch("veomni.arguments.arguments_types.is_liger_kernel_available", return_value=True):
                    OpsImplementationConfig(load_balancing_loss_implementation="triton")


def test_gpu_host_skips_npu_validation():
    """``_validate_device_compatibility`` is a no-op on a GPU host.

    Default GPU values must construct cleanly — this is the happy path that
    GPU users hit out of the box."""
    from veomni.arguments.arguments_types import OpsImplementationConfig

    with patch("veomni.utils.device.IS_NPU_AVAILABLE", False):
        # ``_validate_implementations`` calls ``is_liger_kernel_available`` via
        # the local re-export bound at module import time
        # (``from ..utils.import_utils import is_liger_kernel_available``).
        # Patch the local name on ``arguments_types`` so the mock is visible to
        # the validator on hosts without liger-kernel installed.
        with patch("veomni.arguments.arguments_types.is_liger_kernel_available", return_value=True):
            cfg = OpsImplementationConfig()
    # All GPU-reasonable defaults preserved.
    assert cfg.moe_implementation == "fused_triton"
    assert cfg.cross_entropy_loss_implementation == "liger_kernel"
    assert cfg.rms_norm_implementation == "liger_kernel"
    assert cfg.rotary_pos_emb_implementation == "liger_kernel"
    assert cfg.swiglu_mlp_implementation == "liger_kernel"
    # load_balancing_loss stays eager — see the dataclass docstring.
    assert cfg.load_balancing_loss_implementation == "eager"
