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

"""Hardware-gate tests for Qwen3.5 gated delta-rule kernel selection.

Covers the ``causal_conv1d`` and ``chunk_gated_delta_rule`` OpSlots, which now
each expose a GPU (``fla``) and an NPU (``npu``) backend. The point of the
registry refactor is that selecting a backend whose ``HardwareRequirement`` is
not met raises at ``OpSlot.bind()`` time (via ``KERNEL_REGISTRY.resolve``)
rather than silently binding the wrong kernel — the exact guarantee the old
hard-coded path bypassed.

Failure directions reject before importing kernel dependencies. The FLA-on-NPU
success direction replaces only the lazy factory, preserving the real hardware
requirement. Hardware selection tests run without optional Triton/FLA packages.
"""

import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import veomni.ops  # noqa: F401 — trigger KERNEL_REGISTRY registrations
from veomni.ops.dispatch import OpSlot
from veomni.ops.kernel_registry import KERNEL_REGISTRY
from veomni.ops.kernels.gated_delta_rule.npu_hardware import (
    _get_hidden_state_block_value,
    get_hidden_state_block_value,
    select_hidden_state_block_value,
)
from veomni.utils.import_utils import is_torch_npu_available


_REGISTRY_MODULE = "veomni.ops.kernel_registry"

_GDN_OPS = ["causal_conv1d", "chunk_gated_delta_rule"]


# ---------------------------------------------------------------------------
# npu backend requested on a GPU host → device_type='npu' gate fails
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", _GDN_OPS)
@patch(f"{_REGISTRY_MODULE}.IS_CUDA_AVAILABLE", True)
@patch(f"{_REGISTRY_MODULE}.IS_NPU_AVAILABLE", False)
def test_opslot_npu_backend_on_gpu_raises(op_name):
    slot = OpSlot(op_name, "standard")
    with pytest.raises(RuntimeError, match="device_type='npu'"):
        slot.bind("npu")


# ---------------------------------------------------------------------------
# fla backend requested on an NPU host binds successfully
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", _GDN_OPS)
@patch(f"{_REGISTRY_MODULE}.IS_CUDA_AVAILABLE", False)
@patch(f"{_REGISTRY_MODULE}.IS_NPU_AVAILABLE", True)
@patch(f"{_REGISTRY_MODULE}.IS_MLU_AVAILABLE", False)
def test_opslot_fla_backend_on_npu_binds(op_name, monkeypatch):
    bucket = KERNEL_REGISTRY._specs[(op_name, "standard")]

    def sentinel():
        return None

    monkeypatch.setitem(bucket, "fla", replace(bucket["fla"], factory=lambda: sentinel))
    slot = OpSlot(op_name, "standard")
    slot.bind("fla")
    assert slot.use_non_eager_impl


# ---------------------------------------------------------------------------
# eager path never touches HardwareRequirement (resolves to None)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", _GDN_OPS)
def test_opslot_eager_skips_hw_check(op_name):
    slot = OpSlot(op_name, "standard")
    slot.bind("eager")
    assert not slot.use_non_eager_impl


# ---------------------------------------------------------------------------
# unknown backend name is a KeyError, listing the available options
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", _GDN_OPS)
def test_opslot_unknown_backend_raises(op_name):
    slot = OpSlot(op_name, "standard")
    with pytest.raises(KeyError, match="Unknown kernel 'bogus'"):
        slot.bind("bogus")


# ---------------------------------------------------------------------------
# Registry presence — a future reshuffle that drops a backend trips this early.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", _GDN_OPS)
def test_gdn_registry_has_fla_and_npu(op_name):
    available = KERNEL_REGISTRY.list_available(op_name, "standard")
    assert "fla" in available
    assert "npu" in available


@pytest.mark.parametrize("device_name", ["Ascend910B4", "Ascend910B4-1"])
def test_ascend_910b4_uses_bv64(device_name):
    assert select_hidden_state_block_value(device_name) == 64


@pytest.mark.parametrize(
    "device_name",
    ["Ascend910B1", "Ascend910B2", "Ascend910B2C", "Ascend910B3", "Ascend910_95", "Ascend950", "", "FutureAscend"],
)
def test_other_ascend_devices_keep_bv128(device_name):
    assert select_hidden_state_block_value(device_name) == 128


def test_runtime_selector_caches_by_explicit_device_index(monkeypatch):
    calls = []
    device_names = {0: "Ascend910B4-1", 1: "Ascend910B2C"}

    def get_device_name(device_index):
        calls.append(device_index)
        return device_names[device_index]

    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(npu=SimpleNamespace(get_device_name=get_device_name))
    )
    _get_hidden_state_block_value.cache_clear()
    try:
        assert get_hidden_state_block_value(0) == 64
        assert get_hidden_state_block_value(1) == 128
        assert get_hidden_state_block_value(0) == 64
        assert calls == [0, 1]
    finally:
        _get_hidden_state_block_value.cache_clear()


# ---------------------------------------------------------------------------
# npu_ascendc — the second NPU backend on chunk_gated_delta_rule only
# ---------------------------------------------------------------------------


def test_chunk_gdr_registry_has_npu_ascendc():
    available = KERNEL_REGISTRY.list_available("chunk_gated_delta_rule", "standard")
    assert "npu_ascendc" in available
    # npu_ascendc is scoped to chunk_gated_delta_rule; causal_conv1d keeps a single npu backend.
    assert "npu_ascendc" not in KERNEL_REGISTRY.list_available("causal_conv1d", "standard")


@patch(f"{_REGISTRY_MODULE}.IS_CUDA_AVAILABLE", True)
@patch(f"{_REGISTRY_MODULE}.IS_NPU_AVAILABLE", False)
def test_opslot_npu_ascendc_backend_on_gpu_raises():
    slot = OpSlot("chunk_gated_delta_rule", "standard")
    with pytest.raises(RuntimeError, match="device_type='npu'"):
        slot.bind("npu_ascendc")


@pytest.mark.skipif(
    is_torch_npu_available(),
    reason="guard only fires when torch_npu/fla_npu/triton are absent; on NPU the factory imports the real kernel",
)
def test_npu_ascendc_factory_missing_dep_raises_actionable():
    """Absent ``fla_npu`` / ``torch_npu`` / ``triton-ascend`` surfaces a RuntimeError with
    install guidance, not a bare ModuleNotFoundError from the transitive imports."""
    from veomni.ops.kernels.gated_delta_rule import _npu_ascendc_chunk_gated_delta_rule_factory

    with pytest.raises(RuntimeError, match="npu_ascendc"):
        _npu_ascendc_chunk_gated_delta_rule_factory()


def test_runtime_selector_resolves_current_device_before_caching(monkeypatch):
    current = [0]
    calls = []

    def get_device_name(index):
        calls.append(index)
        return {0: "Ascend910B4-1", 1: "Ascend910B2C"}[index]

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu=SimpleNamespace(get_device_name=get_device_name, current_device=lambda: current[0])),
    )
    _get_hidden_state_block_value.cache_clear()
    try:
        assert get_hidden_state_block_value(None) == 64
        current[0] = 1
        assert get_hidden_state_block_value(None) == 128
        assert get_hidden_state_block_value(0) == 64
        assert calls == [0, 1]
    finally:
        _get_hidden_state_block_value.cache_clear()


def test_910b4_gate_rejects_diagnostic_profile_when_strict_is_required(monkeypatch):
    import importlib.util
    from pathlib import Path

    gate_path = Path(__file__).resolve().parents[2] / "scripts/npu/validate_910b4_gdn_training.py"
    spec = importlib.util.spec_from_file_location("veomni_910b4_gate_test", gate_path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    monkeypatch.setattr(
        gate,
        "_parse_args",
        lambda: SimpleNamespace(
            sp_size=8, tokens=128, reference_tokens=127, atol=0.05, rtol=0.05, require_strict=True
        ),
    )
    with pytest.raises(ValueError, match="strict production profile"):
        gate.main()
