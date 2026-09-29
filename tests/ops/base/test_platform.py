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

"""GPU-platform and hardware-requirement tests."""

from __future__ import annotations

import pytest

from veomni.ops.platform import (
    NVIDIA_SM70_PLUS,
    ROCM_GPU,
    GpuKernelRequirement,
    NvidiaGpuPlatform,
    RocmGpuPlatform,
)


def test_nvidia_platform_rejects_rocm_without_reading_cuda_cc(monkeypatch):
    """ROCm must not be classified by NVIDIA compute capability."""
    monkeypatch.setattr("veomni.ops.platform.gpu.IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr("veomni.ops.platform.gpu.torch.version.hip", "6.0", raising=False)

    def unexpected_cc_read():
        raise AssertionError("NVIDIA compute capability must not be read on ROCm")

    monkeypatch.setattr("veomni.ops.platform.gpu.get_gpu_compute_capability", unexpected_cc_read)

    assert not NvidiaGpuPlatform(min_cc=90).matches()


def test_rocm_platform_uses_pytorch_cuda_namespace(monkeypatch):
    """ROCm matches when PyTorch exposes an available HIP-backed cuda device."""
    monkeypatch.setattr("veomni.ops.platform.gpu.IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr("veomni.ops.platform.gpu.torch.version.hip", "6.0", raising=False)

    assert RocmGpuPlatform().matches()


def test_default_gpu_requirement_supports_nvidia_and_rocm():
    """Default GPU requirements cover both platforms without architecture bounds."""
    requirement = GpuKernelRequirement()

    assert requirement.platforms == (NvidiaGpuPlatform(), RocmGpuPlatform())


def test_mixed_gpu_requirement_accepts_rocm_without_nvidia_cc(monkeypatch):
    """NVIDIA CC bounds do not constrain the ROCm branch of a shared kernel."""
    monkeypatch.setattr("veomni.ops.platform.gpu.IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr("veomni.ops.platform.gpu.torch.version.hip", "6.0", raising=False)
    requirement = GpuKernelRequirement(platforms=(NVIDIA_SM70_PLUS, ROCM_GPU))

    assert requirement.matches()


@pytest.mark.parametrize(
    ("cc", "expected"),
    ((79, False), (80, True), (90, True), (91, False)),
)
def test_nvidia_platform_compute_capability_range(monkeypatch, cc, expected):
    """NVIDIA CC bounds are inclusive and apply only to NVIDIA CUDA."""
    monkeypatch.setattr("veomni.ops.platform.gpu.IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr("veomni.ops.platform.gpu.torch.version.hip", None, raising=False)
    monkeypatch.setattr("veomni.ops.platform.gpu.get_gpu_compute_capability", lambda: cc)

    assert NvidiaGpuPlatform(min_cc=80, max_cc=90).matches() is expected


def test_gpu_requirement_error_describes_supported_platforms(monkeypatch):
    """Requirement errors expose platform and architecture constraints."""
    monkeypatch.setattr("veomni.ops.platform.gpu.IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr("veomni.ops.platform.gpu.torch.version.hip", "6.0", raising=False)
    requirement = GpuKernelRequirement(platforms=(NvidiaGpuPlatform(min_cc=90),))

    with pytest.raises(RuntimeError, match="NVIDIA CUDA with compute capability >= 90"):
        requirement.check()


@pytest.mark.parametrize(
    "kwargs",
    (
        {"min_cc": -1},
        {"max_cc": -1},
        {"min_cc": 90, "max_cc": 80},
    ),
)
def test_nvidia_platform_rejects_invalid_cc_range(kwargs):
    """Invalid NVIDIA CC intervals fail at registration construction time."""
    with pytest.raises(ValueError):
        NvidiaGpuPlatform(**kwargs)


def test_gpu_requirement_requires_platforms():
    """An empty GPU requirement cannot match any platform and is rejected."""
    with pytest.raises(ValueError, match="at least one platform"):
        GpuKernelRequirement(platforms=())
