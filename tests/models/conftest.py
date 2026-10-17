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

"""Shared fixtures for model construction, integration, and parity tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tests.models.compare import ops_config_scope
from veomni.ops import VeomniOp
from veomni.ops import registry as op_registry
from veomni.ops.config import get_ops_config
from veomni.ops.platform import GpuKernelRequirement, NvidiaGpuPlatform
from veomni.utils.device import IS_NPU_AVAILABLE


@pytest.fixture(autouse=True)
def preserve_ops_config():
    """Keep each model test's installed config from leaking into the next test."""
    with ops_config_scope(get_ops_config()):
        yield


@pytest.fixture(autouse=True)
def cpu_safe_diffusers_rms_norm(monkeypatch):
    """Keep Diffusers RMSNorm on CPU math when tensors are not on NPU.

    Upstream ``RMSNorm.forward`` calls ``npu_rms_norm`` whenever torch_npu is
    importable, including CPU unit tests on Ascend hosts. Qwen-Image inherits
    that module as ``txt_norm``.
    """
    if not IS_NPU_AVAILABLE:
        yield
        return

    from diffusers.models.normalization import RMSNorm

    original = RMSNorm.forward

    def forward(self, hidden_states):
        if hidden_states.device.type == "npu":
            return original(self, hidden_states)
        weight = self.weight
        if weight is not None and weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(weight.dtype)
        hidden_states = F.rms_norm(hidden_states, hidden_states.shape[-1:], weight, self.eps)
        if self.bias is not None:
            hidden_states = hidden_states + self.bias
        return hidden_states

    monkeypatch.setattr(RMSNorm, "forward", forward)
    yield


@pytest.fixture
def available_nvidia_ops(monkeypatch):
    """Make static NVIDIA/package gates pass without executing a GPU kernel."""
    previous_handles = dict(VeomniOp._intern)
    VeomniOp._intern.clear()
    monkeypatch.setattr(op_registry, "get_device_type", lambda: GpuKernelRequirement.device)
    monkeypatch.setattr(NvidiaGpuPlatform, "matches", lambda self: True)
    monkeypatch.setattr(op_registry, "is_package_available", lambda _package: True)
    yield
    VeomniOp._intern.clear()
    VeomniOp._intern.update(previous_handles)
