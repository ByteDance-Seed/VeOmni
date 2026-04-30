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

import torch

from ....utils import logging
from ....utils.import_utils import (
    is_fused_moe_available,
    is_quack_gemm_available,
    is_torch_npu_available,
)


logger = logging.get_logger(__name__)

_fused_moe_forward = None


def fused_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor | None,
    fc1_2_weight: torch.Tensor | None,
    fc2_weight: torch.Tensor,
    fc1_1_2_weight: torch.Tensor | None = None,
):
    if _fused_moe_forward is None:
        raise NotImplementedError("No fused MoE kernel is available. Please check your environment.")

    assert routing_weights.dtype in [torch.bfloat16, torch.float16], (
        f"routing_weights dtype must be bfloat16 or float16 for fused MoE kernel, but got {routing_weights.dtype}"
    )
    assert hidden_states.dtype in [torch.bfloat16, torch.float16], (
        f"hidden_states dtype must be bfloat16 or float16 for fused MoE kernel, but got {hidden_states.dtype}"
    )

    return _fused_moe_forward(
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
        fc1_1_2_weight,
    )


def apply_veomni_fused_moe_patch(fused_moe_kernel: str = "triton") -> None:
    """Bind the global ``_fused_moe_forward`` function pointer.

    Args:
        fused_moe_kernel: Which fused MoE kernel to activate. OSS values:
            ``"triton"`` (Triton group-gemm, GPU, SM70+),
            ``"quack"`` (Quack CUTLASS/CuTe, GPU, SM90+),
            ``"npu"`` (NPU group-gemm, requires torch_npu).
            The kernel must match the hardware; mismatches raise here rather
            than silently falling back to a different backend.

            Typed as plain ``str`` (not ``Literal``) so third-party backends
            (e.g. seed-kernels) can monkey-patch this function to intercept
            their own name and delegate unknown names to the OSS dispatch
            below. Unknown names that reach the OSS dispatch raise
            ``ValueError``.
    """
    global _fused_moe_forward
    if fused_moe_kernel == "npu":
        if not is_torch_npu_available():
            raise RuntimeError(
                "fused_moe_kernel='npu' requires torch_npu and an NPU device. On GPU, use 'triton' or 'quack' instead."
            )
        from .npu_group_gemm import npu_fused_moe_forward

        _fused_moe_forward = npu_fused_moe_forward
    elif fused_moe_kernel == "quack":
        if is_torch_npu_available():
            raise RuntimeError("fused_moe_kernel='quack' is GPU-only. Use 'npu' on NPU devices.")
        if not is_quack_gemm_available():
            raise RuntimeError(
                "fused_moe_kernel='quack' requires the quack package and an SM90+ GPU. "
                "Please install quack or use fused_moe_kernel='triton'."
            )
        from .quack_gemm import quack_gemm_fused_moe_forward

        _fused_moe_forward = quack_gemm_fused_moe_forward
    elif fused_moe_kernel == "triton":
        if is_torch_npu_available():
            raise RuntimeError("fused_moe_kernel='triton' is GPU-only. Use 'npu' on NPU devices.")
        if not is_fused_moe_available():
            raise RuntimeError("fused_moe_kernel='triton' requires triton to be installed and a supported GPU.")
        from .group_gemm import group_gemm_fused_moe_forward

        _fused_moe_forward = group_gemm_fused_moe_forward
    else:
        raise ValueError(f"Invalid fused_moe_kernel: {fused_moe_kernel!r}. Expected one of: 'triton', 'quack', 'npu'.")


# ── OpSlot kernel registrations ──────────────────────────────────────────────

from ...kernel_registry import KERNEL_REGISTRY, HardwareRequirement, KernelSpec


def _make_moe_experts_adapter(raw_forward):
    """Adapt the raw fused MoE kernel to the OpSlot call signature.

    The generated modeling code calls the slot as a bound method of the
    HF experts module::

        veomni_moe_experts_forward(self, hidden_states, top_k_index, top_k_weights)

    The raw kernels (``group_gemm_fused_moe_forward`` /
    ``quack_gemm_fused_moe_forward``) instead take the flat tensor-level
    signature ``(num_experts, routing_weights, selected_experts,
    hidden_states, fc1_1_weight, fc1_2_weight, fc2_weight,
    fc1_1_2_weight)``. This adapter pulls ``num_experts``/``gate_up_proj``/
    ``down_proj`` off ``self`` and forwards everything else positionally so
    the OpSlot stays a drop-in replacement for the HF ``forward``.
    """

    def adapter(self, hidden_states, top_k_index, top_k_weights):
        return raw_forward(
            num_experts=self.num_experts,
            routing_weights=top_k_weights.to(hidden_states.dtype),
            selected_experts=top_k_index,
            hidden_states=hidden_states,
            fc1_1_weight=None,
            fc1_2_weight=None,
            fc2_weight=self.down_proj,
            fc1_1_2_weight=self.gate_up_proj,
        )

    return adapter


def _triton_kernel_factory():
    from .group_gemm import group_gemm_fused_moe_forward

    return _make_moe_experts_adapter(group_gemm_fused_moe_forward)


KERNEL_REGISTRY.register(
    KernelSpec(
        name="triton",
        op_name="moe_experts",
        variant="standard",
        factory=_triton_kernel_factory,
        hardware=HardwareRequirement(device_type="gpu", min_compute_capability=70),
        description="Triton group-gemm fused MoE forward",
    )
)


def _quack_kernel_factory():
    from .quack_gemm import quack_gemm_fused_moe_forward

    return _make_moe_experts_adapter(quack_gemm_fused_moe_forward)


KERNEL_REGISTRY.register(
    KernelSpec(
        name="quack",
        op_name="moe_experts",
        variant="standard",
        factory=_quack_kernel_factory,
        hardware=HardwareRequirement(device_type="gpu", min_compute_capability=90),
        description="Quack CUTLASS/CuTe fused MoE forward (SM90+)",
    )
)


def _npu_kernel_factory():
    from .npu_group_gemm import npu_fused_moe_forward

    return _make_moe_experts_adapter(npu_fused_moe_forward)


KERNEL_REGISTRY.register(
    KernelSpec(
        name="npu",
        op_name="moe_experts",
        variant="standard",
        factory=_npu_kernel_factory,
        hardware=HardwareRequirement(device_type="npu"),
        description="NPU group-gemm fused MoE forward",
    )
)
