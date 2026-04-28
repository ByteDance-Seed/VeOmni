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

"""Hardware-gate tests for MoE kernel selection.

Covers the two dispatch paths that select a fused MoE kernel from
``OpsImplementationConfig.moe_implementation`` and verifies that a
kernel-vs-hardware mismatch raises with a clear message at model-build time
rather than silently falling back to another backend.

Two paths:
1. Legacy: ``apply_veomni_fused_moe_patch`` (qwen3_moe, deepseek_v3, etc.)
2. OpSlot: ``KERNEL_REGISTRY.resolve`` via ``HardwareRequirement`` (qwen3_5_moe)
3. Auto: ``moe_implementation='fused'`` resolves to the accelerator-specific
   concrete fused backend before either path binds a kernel.

We mock the hardware-detection helpers so the same test suite runs on any
CI host.
"""

from unittest.mock import patch

import pytest

import veomni.ops  # noqa: F401 — trigger KERNEL_REGISTRY registrations
from veomni.ops.dispatch import OpSlot
from veomni.ops.kernel_registry import KERNEL_REGISTRY
from veomni.ops.kernels.moe import apply_veomni_fused_moe_patch


# ---------------------------------------------------------------------------
# 1) Legacy path — apply_veomni_fused_moe_patch
# ---------------------------------------------------------------------------

_MOE_MODULE = "veomni.ops.kernels.moe"


@patch(f"{_MOE_MODULE}.is_torch_npu_available", return_value=True)
def test_legacy_fused_quack_on_npu_raises(_mock_npu):
    with pytest.raises(RuntimeError, match="quack.*GPU-only"):
        apply_veomni_fused_moe_patch(fused_moe_kernel="quack")


@patch(f"{_MOE_MODULE}.is_torch_npu_available", return_value=False)
@patch(f"{_MOE_MODULE}.is_quack_gemm_available", return_value=False)
def test_legacy_fused_quack_without_sm90_raises(_mock_quack, _mock_npu):
    """``is_quack_gemm_available()`` returns False on sub-SM90 GPUs (e.g. A100)."""
    with pytest.raises(RuntimeError, match="quack.*SM90\\+"):
        apply_veomni_fused_moe_patch(fused_moe_kernel="quack")


@patch(f"{_MOE_MODULE}.is_torch_npu_available", return_value=True)
def test_legacy_fused_triton_on_npu_raises(_mock_npu):
    with pytest.raises(RuntimeError, match="triton.*GPU-only"):
        apply_veomni_fused_moe_patch(fused_moe_kernel="triton")


@patch(f"{_MOE_MODULE}.is_torch_npu_available", return_value=False)
def test_legacy_fused_npu_on_gpu_raises(_mock_npu):
    with pytest.raises(RuntimeError, match="npu.*requires torch_npu"):
        apply_veomni_fused_moe_patch(fused_moe_kernel="npu")


def test_legacy_invalid_kernel_name_raises():
    with pytest.raises(ValueError, match="Invalid fused_moe_kernel"):
        apply_veomni_fused_moe_patch(fused_moe_kernel="bogus")


# ---------------------------------------------------------------------------
# 2) OpSlot path — KERNEL_REGISTRY.resolve → HardwareRequirement.is_satisfied
# ---------------------------------------------------------------------------

_REGISTRY_MODULE = "veomni.ops.kernel_registry"


@patch(f"{_REGISTRY_MODULE}.IS_CUDA_AVAILABLE", True)
@patch(f"{_REGISTRY_MODULE}.IS_NPU_AVAILABLE", False)
@patch(f"{_REGISTRY_MODULE}.get_gpu_compute_capability", return_value=80)
def test_opslot_fused_quack_on_sm80_raises(_mock_cc):
    """A100-class GPU (SM80) should fail the SM90 min_compute_capability gate."""
    slot = OpSlot("moe_experts", "standard")
    with pytest.raises(RuntimeError, match="compute_capability>=90"):
        slot.bind("quack")


@patch(f"{_REGISTRY_MODULE}.IS_CUDA_AVAILABLE", False)
@patch(f"{_REGISTRY_MODULE}.IS_NPU_AVAILABLE", True)
def test_opslot_fused_quack_on_npu_raises():
    slot = OpSlot("moe_experts", "standard")
    with pytest.raises(RuntimeError, match="device_type='gpu'"):
        slot.bind("quack")


@patch(f"{_REGISTRY_MODULE}.IS_CUDA_AVAILABLE", False)
@patch(f"{_REGISTRY_MODULE}.IS_NPU_AVAILABLE", True)
def test_opslot_fused_triton_on_npu_raises():
    slot = OpSlot("moe_experts", "standard")
    with pytest.raises(RuntimeError, match="device_type='gpu'"):
        slot.bind("triton")


@patch(f"{_REGISTRY_MODULE}.IS_CUDA_AVAILABLE", True)
@patch(f"{_REGISTRY_MODULE}.IS_NPU_AVAILABLE", False)
def test_opslot_fused_npu_on_gpu_raises():
    slot = OpSlot("moe_experts", "standard")
    with pytest.raises(RuntimeError, match="device_type='npu'"):
        slot.bind("npu")


def test_opslot_eager_skips_hw_check():
    """'eager' resolves to None without touching HardwareRequirement."""
    slot = OpSlot("moe_experts", "standard")
    slot.bind("eager")
    assert not slot.use_non_eager_impl


def test_opslot_unknown_kernel_name_raises():
    slot = OpSlot("moe_experts", "standard")
    with pytest.raises(KeyError, match="Unknown kernel 'bogus'"):
        slot.bind("bogus")


# ---------------------------------------------------------------------------
# 3) End-to-end — _bind_veomni_ops wires config → OpSlot with fused_ prefix
#    stripped, so the HardwareRequirement gate sees the right impl_name.
# ---------------------------------------------------------------------------


@patch(f"{_REGISTRY_MODULE}.IS_CUDA_AVAILABLE", True)
@patch(f"{_REGISTRY_MODULE}.IS_NPU_AVAILABLE", False)
@patch(f"{_REGISTRY_MODULE}.get_gpu_compute_capability", return_value=80)
def test_bind_veomni_ops_translates_moe_kernel_and_checks_hw(_mock_cc):
    """Reproducer for the silent-fallback regression:

    User sets ``moe_implementation='fused_quack'`` on an A100. The binding
    must route ``fused_quack`` → KERNEL_REGISTRY lookup ``quack`` and raise
    at bind time, not silently stay eager.
    """
    from types import SimpleNamespace

    from veomni.arguments.arguments_types import OpsImplementationConfig
    from veomni.models.auto import _bind_veomni_ops

    ops_config = OpsImplementationConfig(moe_implementation="fused_quack")
    # Simulate a patchgen'd modeling module with a moe_experts OpSlot.
    fake_module = SimpleNamespace(veomni_moe_experts_forward=OpSlot("moe_experts", "standard"))

    with pytest.raises(RuntimeError, match="compute_capability>=90"):
        _bind_veomni_ops(fake_module, ops_config)


@patch("veomni.models.auto.IS_NPU_AVAILABLE", True)
@patch("veomni.models.auto.IS_CUDA_AVAILABLE", False)
def test_select_moe_kernel_fused_auto_selects_npu():
    from veomni.models.auto import _select_moe_kernel_by_device

    resolved = _select_moe_kernel_by_device("fused")
    assert resolved.config_value == "fused"
    assert resolved.kernel_name == "npu"


@patch("veomni.models.auto.IS_NPU_AVAILABLE", False)
@patch("veomni.models.auto.IS_CUDA_AVAILABLE", True)
@patch("veomni.models.auto.get_gpu_compute_capability", return_value=100)
def test_select_moe_kernel_fused_auto_selects_quack_on_sm100(_mock_cc):
    from veomni.models.auto import _select_moe_kernel_by_device

    resolved = _select_moe_kernel_by_device("fused")
    assert resolved.kernel_name == "quack"


@patch("veomni.models.auto.IS_NPU_AVAILABLE", False)
@patch("veomni.models.auto.IS_CUDA_AVAILABLE", True)
@patch("veomni.models.auto.get_gpu_compute_capability", return_value=90)
def test_select_moe_kernel_fused_auto_selects_triton_below_sm100(_mock_cc):
    from veomni.models.auto import _select_moe_kernel_by_device

    resolved = _select_moe_kernel_by_device("fused")
    assert resolved.kernel_name == "triton"


@patch("veomni.models.auto.IS_NPU_AVAILABLE", False)
@patch("veomni.models.auto.IS_CUDA_AVAILABLE", False)
def test_select_moe_kernel_fused_auto_requires_accelerator():
    from veomni.models.auto import _select_moe_kernel_by_device

    with pytest.raises(RuntimeError, match="requires a CUDA GPU or NPU"):
        _select_moe_kernel_by_device("fused")


def test_select_moe_kernel_rejects_unknown_fused_backend():
    from veomni.models.auto import _select_moe_kernel_by_device

    with pytest.raises(ValueError, match="Expected one of"):
        _select_moe_kernel_by_device("fused_unit_test_custom")


@patch("veomni.models.auto.is_transformers_version_greater_or_equal_to", return_value=False)
def test_opslot_moe_dispatch_ignores_transformers_v4(_mock_tf_version):
    from types import SimpleNamespace

    from veomni.models.auto import _module_uses_opslot_moe_dispatch

    fake_module = SimpleNamespace(veomni_moe_experts_forward=OpSlot("moe_experts", "standard"))
    assert not _module_uses_opslot_moe_dispatch(fake_module)


@patch("veomni.models.auto.is_transformers_version_greater_or_equal_to", return_value=True)
def test_opslot_moe_dispatch_requires_moe_experts_slot(_mock_tf_version):
    from types import SimpleNamespace

    from veomni.models.auto import _module_uses_opslot_moe_dispatch

    assert not _module_uses_opslot_moe_dispatch(SimpleNamespace())
    fake_module = SimpleNamespace(veomni_moe_experts_forward=OpSlot("moe_experts", "standard"))
    assert _module_uses_opslot_moe_dispatch(fake_module)


# KERNEL_REGISTRY is a module-level singleton. Assert the registrations the
# tests rely on are present so a future registry reshuffle trips this early.
@pytest.mark.parametrize("impl_name", ["triton", "quack", "npu"])
def test_moe_experts_registry_has_kernel(impl_name):
    assert impl_name in KERNEL_REGISTRY.list_available("moe_experts", "standard")
