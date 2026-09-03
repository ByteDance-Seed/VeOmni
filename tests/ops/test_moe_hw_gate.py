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

"""Hardware-gate tests for leftover ``apply_veomni_fused_moe_patch``.

``moe_experts`` OpSlot rows are gone. Production ``models/`` still binds the
legacy pointer. Kernels resolve gates live in ``tests/kernels/test_moe_experts.py``.
"""

from unittest.mock import patch

import pytest

from veomni.ops.dispatch import OpsConfigSlot
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


def test_bind_veomni_ops_binds_model_registered_config_slots():
    from types import SimpleNamespace

    from veomni.arguments.arguments_types import OpsImplementationConfig
    from veomni.models.auto import _bind_veomni_ops

    ops_config = OpsImplementationConfig(
        attn_implementation="eager",
        moe_implementation="eager",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
        load_balancing_loss_implementation="eager",
        rms_norm_gated_implementation="eager",
        causal_conv1d_implementation="eager",
        chunk_gated_delta_rule_implementation="eager",
        dsa_indexer_implementation="cudnn",
        dsa_attention_implementation="flashmla_cudnn",
    )
    indexer_slot = OpsConfigSlot("dsa_indexer_implementation")
    attention_slot = OpsConfigSlot("dsa_attention_implementation")
    fake_module = SimpleNamespace(
        veomni_dsa_indexer_implementation=indexer_slot,
        veomni_dsa_attention_implementation=attention_slot,
    )

    assert _bind_veomni_ops(fake_module, ops_config)
    assert indexer_slot.value == "cudnn"
    assert attention_slot.value == "flashmla_cudnn"


def test_bind_veomni_ops_rejects_unknown_config_slot():
    from types import SimpleNamespace

    from veomni.arguments.arguments_types import OpsImplementationConfig
    from veomni.models.auto import _bind_veomni_ops

    fake_module = SimpleNamespace(veomni_unknown_backend=OpsConfigSlot("missing_backend"))
    ops_config = OpsImplementationConfig(
        attn_implementation="eager",
        moe_implementation="eager",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
        load_balancing_loss_implementation="eager",
        rms_norm_gated_implementation="eager",
        causal_conv1d_implementation="eager",
        chunk_gated_delta_rule_implementation="eager",
    )

    with pytest.raises(AttributeError, match="missing_backend"):
        _bind_veomni_ops(fake_module, ops_config)
