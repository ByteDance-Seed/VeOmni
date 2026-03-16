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
"""
Kernel registry for OpSlot-based dispatch.

Provides a global registry of kernel implementations keyed by
(op_name, variant, impl_name). Each kernel is described by a KernelSpec
that includes a lazy factory, hardware requirements, and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from ..utils import logging


logger = logging.get_logger(__name__)


@dataclass(frozen=True)
class HardwareRequirement:
    """Describes hardware constraints for a kernel."""

    device_type: str  # "cuda" | "npu"
    min_compute_capability: int | None = None  # e.g. 70, 80, 90

    def is_satisfied(self) -> bool:
        if self.device_type == "cuda":
            if not torch.cuda.is_available():
                return False
            if self.min_compute_capability is not None:
                major, minor = torch.cuda.get_device_capability()
                cc = major * 10 + minor
                if cc < self.min_compute_capability:
                    return False
            return True
        elif self.device_type == "npu":
            try:
                import torch_npu  # noqa: F401

                return True
            except ImportError:
                return False
        return False


@dataclass(frozen=True)
class KernelSpec:
    """Describes a single kernel implementation."""

    name: str  # e.g. "triton_group_gemm"
    op_name: str  # e.g. "moe_experts"
    variant: str  # e.g. "standard"
    factory: Callable[[], Callable]  # lazy import: () -> callable
    hardware: HardwareRequirement
    description: str = ""


class KernelRegistry:
    """Global registry mapping (op_name, variant) -> {impl_name: KernelSpec}."""

    def __init__(self):
        self._specs: dict[tuple[str, str], dict[str, KernelSpec]] = {}

    def register(self, spec: KernelSpec) -> None:
        key = (spec.op_name, spec.variant)
        if key not in self._specs:
            self._specs[key] = {}
        self._specs[key][spec.name] = spec

    def resolve(self, op_name: str, variant: str, impl_name: str) -> Callable | None:
        """Resolve an implementation by name.

        Returns ``None`` when *impl_name* is ``"eager"`` (meaning: use the
        original HF code path).

        Raises ``KeyError`` if *impl_name* is unknown, and ``RuntimeError``
        if the hardware requirement is not satisfied.
        """
        if impl_name == "eager":
            return None

        key = (op_name, variant)
        bucket = self._specs.get(key, {})
        if impl_name not in bucket:
            available = list(bucket.keys()) + ["eager"]
            raise KeyError(
                f"Unknown kernel '{impl_name}' for op='{op_name}', variant='{variant}'. Available: {available}"
            )

        spec = bucket[impl_name]
        if not spec.hardware.is_satisfied():
            raise RuntimeError(
                f"Kernel '{impl_name}' for op='{op_name}' requires "
                f"device_type='{spec.hardware.device_type}'"
                + (
                    f", compute_capability>={spec.hardware.min_compute_capability}"
                    if spec.hardware.min_compute_capability
                    else ""
                )
                + ", but the current hardware does not satisfy this."
            )

        return spec.factory()

    def list_available(self, op_name: str, variant: str) -> list[str]:
        key = (op_name, variant)
        return list(self._specs.get(key, {}).keys())


KERNEL_REGISTRY = KernelRegistry()
