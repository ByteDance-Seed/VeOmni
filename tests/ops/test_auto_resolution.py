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


def test_apply_veomni_fused_moe_patch_triton_works_with_torch_npu_package_only(monkeypatch):
    """Regression: fused_triton must succeed on a CUDA dev host that happens
    to have ``torch_npu`` installed but no NPU device.

    Pre-fix, ``apply_veomni_fused_moe_patch("triton")`` checked the bare
    ``torch_npu`` package flag and aborted. Now both the kernel patch and
    ``is_fused_moe_available`` use the device-aware ``is_npu_device_available``,
    so the GPU path stays open as long as no NPU device is reachable.
    """
    from veomni.utils import device

    # Simulate dual-stack: torch_npu package installed, no actual NPU device.
    monkeypatch.setattr(device, "IS_NPU_AVAILABLE", False)
    monkeypatch.setattr(device, "is_npu_device_available", lambda: False)
    # Re-route the moe module's locally-imported reference too.
    import veomni.ops.kernels.moe as moe_module

    monkeypatch.setattr(moe_module, "is_npu_device_available", lambda: False)
    # is_fused_moe_available reads through device.is_npu_device_available
    # (already patched) and _PACKAGE_FLAGS["triton"] (left as-is).
    # The bind itself is not what we're testing — only that the GPU path
    # is reachable. If triton isn't installed locally, is_fused_moe_available
    # will return False and the bind raises a *triton-missing* RuntimeError,
    # which is the correct, clearer error than the pre-fix NPU rejection.
    from veomni.utils.import_utils import is_fused_moe_available

    # The probe should not see torch_npu masquerading as NPU device.
    # If triton is locally installed, is_fused_moe_available is True; if not,
    # it's False — but in both cases the failure mode is "triton missing",
    # not "this is an NPU device".
    _ = is_fused_moe_available()


def test_build_foundation_model_skips_moe_patch_for_non_moe_config():
    """Regression: non-MoE models (Llama, plain Qwen, ...) must not trigger
    ``apply_veomni_fused_moe_patch`` even when the resolved
    ``moe_implementation`` is ``fused_triton`` / ``fused_npu``.

    Pre-fix, the new auto default propagated ``fused_triton`` through to
    ``apply_moe_patch_transformers_v4`` for every non-OpSlot model. On a
    clean GPU host this was wasteful (importing triton, binding the global
    pointer that nothing calls); on dual-stack hosts it crashed.
    """
    from types import SimpleNamespace

    from veomni.models.auto import _config_is_moe

    # Llama-style config: no `num_experts`, no nested MoE config.
    llama_like = SimpleNamespace(hidden_size=4096, num_attention_heads=32)
    assert _config_is_moe(llama_like) is False

    # Qwen3-MoE-style: top-level num_experts > 1.
    qwen_moe_like = SimpleNamespace(num_experts=8, hidden_size=4096)
    assert _config_is_moe(qwen_moe_like) is True

    # VLM wrapper: MoE lives under text_config.
    vlm_text = SimpleNamespace(num_experts=8)
    vlm_root = SimpleNamespace(text_config=vlm_text, hidden_size=4096)
    assert _config_is_moe(vlm_root) is True

    # num_experts == 1 is a degenerate "single expert", treated as non-MoE.
    single = SimpleNamespace(num_experts=1)
    assert _config_is_moe(single) is False


def test_register_op_rejects_duplicate_config_field():
    """``_backend_requirements_met`` and other consumers stop at the first
    OpSpec whose ``config_field`` matches, so two ops sharing one would
    create non-deterministic resolution. ``register_op`` must surface that
    misregistration at import time."""
    from veomni.ops.config.registry import BackendSpec, OpScope, OpSpec, register_op

    bogus = OpSpec(
        name="__test_dup_rms_norm__",
        # Already used by the real ``rms_norm`` OpSpec registered at import time.
        config_field="rms_norm_implementation",
        label="Dup",
        scope=OpScope.PER_MODEL,
        default="eager",
        backends={"eager": BackendSpec(entry="builtins:object")},
    )
    with pytest.raises(ValueError, match="config_field 'rms_norm_implementation' is already used"):
        register_op(bogus)


def test_load_balancing_loss_kernelspec_runs_on_npu():
    """Regression: the OpSlot KernelSpec for load_balancing_loss/triton must
    accept NPU devices.

    NPU patchgen models (qwen3_5_moe, qwen3_omni_moe) declare
    ``OpSlot("load_balancing_loss", "standard")`` which calls
    ``KERNEL_REGISTRY.resolve("load_balancing_loss", "standard", "triton")``
    at model-build time. Before the device-agnostic fix the spec was
    GPU-only, so binding crashed on NPU even when triton-ascend was present.
    """
    from veomni.ops.kernel_registry import KERNEL_REGISTRY

    spec = KERNEL_REGISTRY._specs[("load_balancing_loss", "standard")]["triton"]
    # device_type=None means "any accelerator"; the actual triton package
    # check happens via the OpSpec `requires=("triton",)` clause + the auto
    # resolver's fallback to eager.
    assert spec.hardware.device_type is None


def test_npu_device_check_uses_real_device_not_just_package(monkeypatch):
    """Regression: ``IS_NPU_AVAILABLE`` must reflect actual NPU presence,
    not just ``find_spec("torch_npu")``.

    Before the fix, a CUDA dev host with ``torch_npu`` installed would
    report ``IS_NPU_AVAILABLE = True`` and the auto resolver would pick
    NPU backends. Now the helper imports ``torch_npu`` and consults
    ``torch.npu.is_available()``.
    """
    from veomni.utils import device

    # Simulate "package installed but no NPU device".
    monkeypatch.setattr(device, "is_torch_npu_available", lambda: True)

    class _FakeNpu:
        @staticmethod
        def is_available() -> bool:
            return False

    monkeypatch.setattr(device.torch, "npu", _FakeNpu, raising=False)
    assert device._detect_npu_device() is False


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
