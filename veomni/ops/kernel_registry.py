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

from ..utils import logging
from ..utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE, get_gpu_compute_capability


logger = logging.get_logger(__name__)


@dataclass(frozen=True)
class HardwareRequirement:
    """Describes hardware constraints for a kernel.

    ``device_type=None`` marks a device-agnostic kernel (e.g. the chunked
    cross-entropy loss, which is pure ``torch.func`` on top of the eager CE
    kernel and runs on either accelerator). It still requires *some*
    accelerator to be present.
    """

    device_type: str | None  # "gpu" | "npu" | None (any accelerator)
    min_compute_capability: int | None = None  # e.g. 70, 80, 90

    def is_satisfied(self) -> bool:
        if self.device_type is None:
            return IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE
        if self.device_type == "gpu":
            if not IS_CUDA_AVAILABLE:
                return False
            if self.min_compute_capability is not None:
                if get_gpu_compute_capability() < self.min_compute_capability:
                    return False
            return True
        if self.device_type == "npu":
            # IS_NPU_AVAILABLE here is the device-presence check from
            # veomni.utils.device — torch_npu installed AND an Ascend NPU is
            # actually reachable. ``is_torch_npu_available()`` alone only
            # checks the package, which would pass on dev hosts that merely
            # have the library and let the resolver pick NPU kernels in error.
            return IS_NPU_AVAILABLE
        raise ValueError(f"Unknown device_type: {self.device_type!r} (expected 'gpu', 'npu', or None)")


@dataclass(frozen=True)
class KernelSpec:
    """Describes a single kernel implementation registered under an op/variant.

    Attributes:
        name: Identifier exposed to users via the matching
            ``OpsImplementationConfig`` field (e.g. ``"liger_kernel"``,
            ``"triton"``, ``"quack"``). Must be unique within a given
            ``(op_name, variant)`` bucket.
        op_name: The logical op that this kernel implements (e.g.
            ``"rms_norm"``, ``"moe_experts"``). Matches the ``OpSlot``'s
            ``op_name``.
        variant: Sub-variant of the op, used when a single op has multiple
            forward-compatible shapes (e.g. ``"standard"`` vs ``"qwen3_5"``
            RMSNorm). Kernels for different variants never collide.
        factory: Zero-argument callable returning the concrete kernel
            callable. Kept lazy so optional imports (Liger, Triton, etc.)
            only load on demand.
        hardware: Hardware gate enforced at ``resolve()`` time; raises
            ``RuntimeError`` early when the requested kernel cannot run on
            the current accelerator.
        description: Free-form human-readable description, surfaced in
            registry listings.
    """

    name: str
    op_name: str
    variant: str
    factory: Callable[[], Callable]
    hardware: HardwareRequirement
    description: str = ""


class KernelRegistry:
    """Global registry mapping (op_name, variant) -> {impl_name: KernelSpec}."""

    def __init__(self):
        self._specs: dict[tuple[str, str], dict[str, KernelSpec]] = {}

    def register(self, spec: KernelSpec) -> None:
        key = (spec.op_name, spec.variant)
        bucket = self._specs.setdefault(key, {})
        if spec.name in bucket:
            raise ValueError(
                f"Duplicate kernel registration: op='{spec.op_name}', variant='{spec.variant}', name='{spec.name}'"
            )
        bucket[spec.name] = spec

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
