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

"""Hardware-platform constraints for registered operations."""

from .gpu import (
    NVIDIA_GPU,
    NVIDIA_SM70_PLUS,
    NVIDIA_SM90_PLUS,
    ROCM_GPU,
    GpuPlatform,
    NvidiaGpuPlatform,
    RocmGpuPlatform,
)
from .requirement import (
    ANY_DEVICE,
    GpuKernelRequirement,
    KernelRequirement,
    MluKernelRequirement,
    NpuKernelRequirement,
)


__all__ = [
    "ANY_DEVICE",
    "GpuKernelRequirement",
    "GpuPlatform",
    "KernelRequirement",
    "MluKernelRequirement",
    "NVIDIA_GPU",
    "NVIDIA_SM70_PLUS",
    "NVIDIA_SM90_PLUS",
    "NpuKernelRequirement",
    "NvidiaGpuPlatform",
    "ROCM_GPU",
    "RocmGpuPlatform",
]
