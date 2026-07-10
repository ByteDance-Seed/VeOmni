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

Only the *failure* direction is exercised (npu-on-GPU, fla-on-NPU): the
hardware check fires inside ``resolve`` before ``spec.factory()`` runs, so
these tests never import the Triton kernels and run on any CI host without
``triton-ascend`` / ``flash-linear-attention``.
"""

from unittest.mock import patch

import pytest

import veomni.ops  # noqa: F401 — trigger KERNEL_REGISTRY registrations
from veomni.ops.dispatch import OpSlot
from veomni.ops.kernel_registry import KERNEL_REGISTRY


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
# fla (GPU) backend requested on an NPU host → device_type='gpu' gate fails
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", _GDN_OPS)
@patch(f"{_REGISTRY_MODULE}.IS_CUDA_AVAILABLE", False)
@patch(f"{_REGISTRY_MODULE}.IS_NPU_AVAILABLE", True)
def test_opslot_fla_backend_on_npu_raises(op_name):
    slot = OpSlot(op_name, "standard")
    with pytest.raises(RuntimeError, match="device_type='gpu'"):
        slot.bind("fla")


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
