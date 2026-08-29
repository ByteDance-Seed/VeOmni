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
"""Hardware requirements for kernel rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE, get_gpu_compute_capability


class KernelRequirement(Protocol):
    def matches(self) -> bool: ...

    def check(self) -> None: ...


@dataclass(frozen=True)
class CudaKernelRequirement:
    min_cc: int | None = None
    max_cc: int | None = None

    def matches(self) -> bool:
        if not IS_CUDA_AVAILABLE:
            return False

        cc = get_gpu_compute_capability()
        if self.min_cc is not None and cc < self.min_cc:
            return False
        if self.max_cc is not None and cc > self.max_cc:
            return False
        return True

    def check(self) -> None:
        if self.matches():
            return
        cc = get_gpu_compute_capability()
        raise RuntimeError(
            "CudaKernelRequirement is not satisfied "
            f"(min_cc={self.min_cc}, max_cc={self.max_cc}, current_cc={cc}, cuda={IS_CUDA_AVAILABLE})"
        )


@dataclass(frozen=True)
class NpuKernelRequirement:
    def matches(self) -> bool:
        return IS_NPU_AVAILABLE

    def check(self) -> None:
        if self.matches():
            return
        raise RuntimeError("NpuKernelRequirement is not satisfied (torch_npu device is unavailable)")
