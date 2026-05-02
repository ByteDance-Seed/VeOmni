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
    """Describes hardware constraints for a kernel."""

    device_type: str  # "gpu" | "npu"
    min_compute_capability: int | None = None  # e.g. 70, 80, 90
    # Inclusive upper bound for kernels that don't yet support newer arches
    # (e.g. FlashQLA today only ships SM90; SM100/SM120 wheels are WIP per
    # https://github.com/QwenLM/FlashQLA/issues/2). Drop this once the kernel
    # adds forward-compatibility for higher arches.
    max_compute_capability: int | None = None

    def is_satisfied(self) -> bool:
        if self.device_type == "gpu":
            if not IS_CUDA_AVAILABLE:
                return False
            cc = get_gpu_compute_capability()
            if self.min_compute_capability is not None and cc < self.min_compute_capability:
                return False
            if self.max_compute_capability is not None and cc > self.max_compute_capability:
                return False
            return True
        if self.device_type == "npu":
            # IS_NPU_AVAILABLE == is_torch_npu_available(): requires both the
            # torch_npu package AND an actual NPU device (unlike a bare import
            # check, which passes on dev boxes that merely have the library).
            return IS_NPU_AVAILABLE
        if self.device_type == "any":
            # Hardware-agnostic kernel (pure PyTorch). Used e.g. by chunk_loss
            # (F.linear + eager_cross_entropy in a chunked autograd Function),
            # which has no device-specific calls. Always satisfied — including
            # on CPU-only hosts (unit tests, weight materialization, dev boxes
            # without an accelerator).
            return True
        raise ValueError(f"Unknown device_type: {self.device_type!r} (expected 'gpu' | 'npu' | 'any')")


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
            cc_min = spec.hardware.min_compute_capability
            cc_max = spec.hardware.max_compute_capability
            if cc_min is not None and cc_max is not None and cc_min == cc_max:
                cc_clause = f", compute_capability=={cc_min}"
            elif cc_min is not None and cc_max is not None:
                cc_clause = f", {cc_min}<=compute_capability<={cc_max}"
            elif cc_min is not None:
                cc_clause = f", compute_capability>={cc_min}"
            elif cc_max is not None:
                cc_clause = f", compute_capability<={cc_max}"
            else:
                cc_clause = ""
            raise RuntimeError(
                f"Kernel '{impl_name}' for op='{op_name}' requires "
                f"device_type='{spec.hardware.device_type}'"
                + cc_clause
                + ", but the current hardware does not satisfy this."
            )

        return spec.factory()

    def list_available(self, op_name: str, variant: str) -> list[str]:
        key = (op_name, variant)
        return list(self._specs.get(key, {}).keys())


KERNEL_REGISTRY = KernelRegistry()
