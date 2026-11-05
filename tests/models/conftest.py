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

import os

import pytest
import torch

from tests.tools.model_fixtures import cpu_model_attention_kernel_guard as cpu_model_attention_kernel_guard
from veomni.ops import VeomniOp
from veomni.ops import registry as op_registry
from veomni.ops.platform import GpuKernelRequirement, NvidiaGpuPlatform
from veomni.utils.device import IS_NPU_AVAILABLE


@pytest.fixture(autouse=True)
def restore_modeling_backend():
    """Undo ``MODELING_BACKEND`` writes so test order cannot change behaviour.

    ``utils.set_environ_param`` flips the variable with a raw ``os.environ``
    assignment mid-test (it has to outlive the model build inside the same test),
    which otherwise persists for the whole pytest process. Every later test module
    then reads the last case's backend — and under ``hf`` the veomni-only config
    normalization (``flex_attention`` -> ``veomni_flex_attention``) stops
    happening, so unrelated suites fail only when run after this directory.
    """
    previous = os.environ.get("MODELING_BACKEND")
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MODELING_BACKEND", None)
        else:
            os.environ["MODELING_BACKEND"] = previous


@pytest.fixture(autouse=True)
def cpu_safe_diffusers_rms_norm(monkeypatch, request):
    """Keep Diffusers RMSNorm on CPU math when tensors are not on NPU.

    Upstream ``RMSNorm.forward`` calls ``npu_rms_norm`` whenever torch_npu is
    importable, including CPU unit tests on Ascend hosts. Qwen-Image inherits
    that module as ``txt_norm``.
    """
    # Indirect parametrization exercises the Ascend-host CPU path on any host.
    if not IS_NPU_AVAILABLE and not getattr(request, "param", False):
        yield
        return

    from diffusers.models.normalization import RMSNorm

    original = RMSNorm.forward

    def forward(self, hidden_states):
        if hidden_states.device.type == "npu":
            return original(self, hidden_states)
        # Preserve Diffusers' native CPU math, including mixed-dtype promotion.
        # F.rms_norm keeps a BF16 input in BF16 even with an FP32 weight.
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        weight = self.weight
        if weight is not None:
            if weight.dtype in (torch.float16, torch.bfloat16):
                hidden_states = hidden_states.to(weight.dtype)
            hidden_states = hidden_states * weight
            if self.bias is not None:
                hidden_states = hidden_states + self.bias
        else:
            hidden_states = hidden_states.to(input_dtype)
        return hidden_states

    monkeypatch.setattr(RMSNorm, "forward", forward)
    yield


@pytest.fixture(
    params=[
        pytest.param(("gpu", "eager"), id="gpu-eager"),
        pytest.param(("gpu", "liger_kernel"), id="gpu-liger"),
        pytest.param(("npu", "eager"), id="npu-eager"),
        pytest.param(("npu", "torch_npu"), id="npu-native"),
        pytest.param(("mlu", "eager"), id="mlu-eager"),
        pytest.param(("mlu", "liger_kernel"), id="mlu-liger"),
    ]
)
def selected_rope_impl(monkeypatch, request):
    """Use valid host selections while the consumer stubs kernel binding."""
    host, impl = request.param
    monkeypatch.setattr("veomni.utils.device.IS_NPU_AVAILABLE", host == "npu")
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_mlu_available", lambda: host == "mlu")
    monkeypatch.setattr("veomni.utils.import_utils.is_liger_kernel_mlu_available", lambda: host == "mlu")
    return impl


@pytest.fixture
def available_nvidia_ops(monkeypatch):
    """Make static NVIDIA/package gates pass without executing a GPU kernel."""
    previous_handles = dict(VeomniOp._intern)
    VeomniOp._intern.clear()
    monkeypatch.setattr("veomni.utils.device.IS_NPU_AVAILABLE", False)
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_mlu_available", lambda: False)
    monkeypatch.setattr(op_registry, "get_device_type", lambda: GpuKernelRequirement.device)
    monkeypatch.setattr(NvidiaGpuPlatform, "matches", lambda self: True)
    monkeypatch.setattr(op_registry, "is_package_available", lambda _package: True)
    monkeypatch.setattr("transformers.modeling_utils.lazy_import_flash_attention", lambda *_args, **_kwargs: None)
    yield
    VeomniOp._intern.clear()
    VeomniOp._intern.update(previous_handles)
