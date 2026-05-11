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


# ``OpsImplementationConfig.__post_init__`` runs ``_validate_implementations``
# which calls ``veomni.utils.import_utils.is_torch_npu_available`` directly —
# the registry-level IS_NPU_AVAILABLE patches above don't reach it. Mock the
# validator's NPU detection so this GPU-scenario test can construct a config
# with ``fused_quack`` on the NPU CI runner without the validator firing
# before the bind step we actually want to exercise.
@patch("veomni.utils.import_utils.is_torch_npu_available", return_value=False)
@patch(f"{_REGISTRY_MODULE}.IS_CUDA_AVAILABLE", True)
@patch(f"{_REGISTRY_MODULE}.IS_NPU_AVAILABLE", False)
@patch(f"{_REGISTRY_MODULE}.get_gpu_compute_capability", return_value=80)
def test_bind_veomni_ops_translates_moe_implementation_and_checks_hw(_mock_cc, _mock_npu):
    """Reproducer for the silent-fallback regression:

    User sets ``moe_implementation='fused_quack'`` on an A100. The binding
    must route ``fused_quack`` → KERNEL_REGISTRY lookup ``quack`` and raise
    at bind time, not silently stay eager.
    """
    from types import SimpleNamespace

    from veomni.arguments.arguments_types import OpsImplementationConfig
    from veomni.models.auto import _bind_veomni_ops

    # Start from all-eager so this test exercises only the moe_experts
    # OpSlot binding, not the GPU-optimal defaults (which would require
    # liger-kernel + triton in the test environment).
    ops_config = OpsImplementationConfig(
        attn_implementation="eager",
        moe_implementation="fused_quack",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
        load_balancing_loss_implementation="eager",
        rms_norm_gated_implementation="eager",
        causal_conv1d_implementation="eager",
        chunk_gated_delta_rule_implementation="eager",
    )
    # Simulate a patchgen'd modeling module with a moe_experts OpSlot.
    fake_module = SimpleNamespace(veomni_moe_experts_forward=OpSlot("moe_experts", "standard"))

    with pytest.raises(RuntimeError, match="compute_capability>=90"):
        _bind_veomni_ops(fake_module, ops_config)


# KERNEL_REGISTRY is a module-level singleton. Assert the registrations the
# tests rely on are present so a future registry reshuffle trips this early.
@pytest.mark.parametrize("impl_name", ["triton", "quack", "npu"])
def test_moe_experts_registry_has_kernel(impl_name):
    assert impl_name in KERNEL_REGISTRY.list_available("moe_experts", "standard")
