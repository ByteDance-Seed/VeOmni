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

"""Hardware requirements for kernel rows.

``device`` is the fourth registry key. GPU rows retain ``"cuda"`` because
PyTorch exposes both NVIDIA CUDA and AMD ROCm tensors through that device
namespace; ``GpuPlatform`` objects express the actual supported platforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol

from ...utils.device import IS_MLU_AVAILABLE, IS_NPU_AVAILABLE
from .gpu import NVIDIA_GPU, ROCM_GPU, GpuPlatform


ANY_DEVICE = "any"


class KernelRequirement(Protocol):
    """Predicate that decides whether this machine can run a registered row."""

    device: str

    def matches(self) -> bool:
        """Return whether this machine can run the row."""
        ...

    def check(self) -> None:
        """Raise if ``matches`` is false."""
        ...


@dataclass(frozen=True)
class GpuKernelRequirement:
    """One or more supported GPU platforms under PyTorch's ``cuda`` device key."""

    platforms: tuple[GpuPlatform, ...] = (NVIDIA_GPU, ROCM_GPU)
    device: ClassVar[str] = "cuda"

    def __post_init__(self) -> None:
        """Require at least one concrete GPU platform."""
        if not self.platforms:
            raise ValueError("GpuKernelRequirement requires at least one platform")
        if not all(isinstance(platform, GpuPlatform) for platform in self.platforms):
            raise TypeError("GpuKernelRequirement platforms must be GpuPlatform instances")

    def matches(self) -> bool:
        """Return whether any supported GPU platform matches this machine."""
        return any(platform.matches() for platform in self.platforms)

    def check(self) -> None:
        """Raise when none of the supported GPU platforms matches this machine."""
        if self.matches():
            return
        supported = ", ".join(platform.describe() for platform in self.platforms)
        raise RuntimeError(f"GpuKernelRequirement is not satisfied (requires one of: {supported})")


@dataclass(frozen=True)
class NpuKernelRequirement:
    """torch_npu device is available."""

    device: ClassVar[str] = "npu"

    def matches(self) -> bool:
        """Return whether a torch_npu device is available."""
        return IS_NPU_AVAILABLE

    def check(self) -> None:
        """Raise if a torch_npu device is not available."""
        if self.matches():
            return
        raise RuntimeError("NpuKernelRequirement is not satisfied (torch_npu device is unavailable)")


@dataclass(frozen=True)
class MluKernelRequirement:
    """torch_mlu device is available."""

    device: ClassVar[str] = "mlu"

    def matches(self) -> bool:
        """Return whether a torch_mlu device is available."""
        return IS_MLU_AVAILABLE

    def check(self) -> None:
        """Raise if a torch_mlu device is not available."""
        if self.matches():
            return
        raise RuntimeError("MluKernelRequirement is not satisfied (torch_mlu device is unavailable)")
