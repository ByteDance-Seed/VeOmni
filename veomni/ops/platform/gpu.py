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

"""GPU platforms exposed through PyTorch's ``cuda`` device namespace."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from ...utils.device import IS_CUDA_AVAILABLE, get_gpu_compute_capability


class GpuPlatform(ABC):
    """Platform-specific availability predicate for a GPU kernel."""

    @abstractmethod
    def matches(self) -> bool:
        """Return whether the current PyTorch GPU platform satisfies this constraint."""
        ...

    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable description for requirement errors."""
        ...


@dataclass(frozen=True)
class NvidiaGpuPlatform(GpuPlatform):
    """NVIDIA CUDA with optional inclusive compute-capability bounds."""

    min_cc: int | None = None
    max_cc: int | None = None

    def __post_init__(self) -> None:
        """Validate the compute-capability interval."""
        if self.min_cc is not None and self.min_cc < 0:
            raise ValueError("min_cc must be non-negative")
        if self.max_cc is not None and self.max_cc < 0:
            raise ValueError("max_cc must be non-negative")
        if self.min_cc is not None and self.max_cc is not None and self.min_cc > self.max_cc:
            raise ValueError("min_cc must not exceed max_cc")

    def matches(self) -> bool:
        """Return whether an NVIDIA CUDA GPU in the requested CC range is available."""
        if not IS_CUDA_AVAILABLE or torch.version.hip is not None:
            return False
        if self.min_cc is None and self.max_cc is None:
            return True

        cc = get_gpu_compute_capability()
        if self.min_cc is not None and cc < self.min_cc:
            return False
        if self.max_cc is not None and cc > self.max_cc:
            return False
        return True

    def describe(self) -> str:
        """Return the NVIDIA CUDA and compute-capability requirement."""
        if self.min_cc is not None and self.max_cc == self.min_cc:
            return f"NVIDIA CUDA with compute capability == {self.min_cc}"
        if self.min_cc is not None and self.max_cc is not None:
            return f"NVIDIA CUDA with {self.min_cc} <= compute capability <= {self.max_cc}"
        if self.min_cc is not None:
            return f"NVIDIA CUDA with compute capability >= {self.min_cc}"
        if self.max_cc is not None:
            return f"NVIDIA CUDA with compute capability <= {self.max_cc}"
        return "NVIDIA CUDA"


@dataclass(frozen=True)
class RocmGpuPlatform(GpuPlatform):
    """AMD ROCm/HIP as exposed through PyTorch's ``cuda`` namespace."""

    def matches(self) -> bool:
        """Return whether an AMD ROCm GPU is available."""
        return IS_CUDA_AVAILABLE and torch.version.hip is not None

    def describe(self) -> str:
        """Return the AMD ROCm platform description."""
        return "AMD ROCm"


NVIDIA_GPU = NvidiaGpuPlatform()
ROCM_GPU = RocmGpuPlatform()
NVIDIA_SM70_PLUS = NvidiaGpuPlatform(min_cc=70)
NVIDIA_SM90_PLUS = NvidiaGpuPlatform(min_cc=90)
