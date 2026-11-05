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

from ...utils.device import IS_CUDA_AVAILABLE, get_gpu_compute_capability, get_rocm_gcn_arch


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
    """AMD ROCm/HIP as exposed through PyTorch's ``cuda`` namespace.

    ``archs`` optionally restricts the match to GPU architectures, named as in
    ``gcnArchName`` without its feature suffix (``"gfx950"``, not ``"gfx950:sramecc+:xnack-"``).
    """

    archs: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the architecture list."""
        if self.archs is not None and not self.archs:
            raise ValueError("archs must be None or a non-empty tuple")

    def matches(self) -> bool:
        """Return whether an AMD ROCm GPU of a requested architecture is available."""
        if not IS_CUDA_AVAILABLE or torch.version.hip is None:
            return False
        return self.archs is None or get_rocm_gcn_arch() in self.archs

    def describe(self) -> str:
        """Return the AMD ROCm platform description."""
        if self.archs is None:
            return "AMD ROCm"
        return f"AMD ROCm {'/'.join(self.archs)}"


NVIDIA_GPU = NvidiaGpuPlatform()
ROCM_GPU = RocmGpuPlatform()
NVIDIA_SM70_PLUS = NvidiaGpuPlatform(min_cc=70)
NVIDIA_SM90_PLUS = NvidiaGpuPlatform(min_cc=90)
NVIDIA_SM100_PLUS = NvidiaGpuPlatform(min_cc=100)
# gfx942 (64 KB LDS, FP8 is e4m3fnuz) and gfx950 (160 KB LDS).
ROCM_GFX942_GFX950 = RocmGpuPlatform(archs=("gfx942", "gfx950"))
# GPUs that run the DeepSeek-V4 TileLang kernels: DSA attention / indexer, mHC and the QAT quantizers.
TILELANG_GPUS = (NVIDIA_SM90_PLUS, ROCM_GFX942_GFX950)
