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

from typing import Literal

import torch

from ...utils import logging
from ...utils.env import get_env
from ...utils.import_utils import (
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


def apply_veomni_fused_moe_patch(
    moe_implementation: Literal["fused", "fused_quack"] = "fused",
):
    """Bind the global ``_fused_moe_forward`` function pointer.

    Args:
        moe_implementation: Which fused MoE kernel to activate.
            ``"fused"`` uses the Triton group-gemm kernels (default).
            ``"fused_quack"`` uses the Quack CUTLASS/CuTe kernels (SM90+).
            On NPU devices the parameter is ignored and the NPU kernel is
            always selected.
    """
    global _fused_moe_forward
    if is_torch_npu_available():
        from .npu_group_gemm import npu_fused_moe_forward

        _fused_moe_forward = npu_fused_moe_forward
    elif moe_implementation == "fused_quack":
        if not is_quack_gemm_available():
            raise RuntimeError(
                "moe_implementation='fused_quack' requires the quack package and an SM90+ GPU. "
                "Please install quack or use moe_implementation='fused'."
            )
        from .quack_gemm import quack_gemm_fused_moe_forward

        _fused_moe_forward = quack_gemm_fused_moe_forward
    elif moe_implementation == "fused" and is_fused_moe_available() and get_env("USE_GROUP_GEMM") == "1":
        from .group_gemm import group_gemm_fused_moe_forward

        _fused_moe_forward = group_gemm_fused_moe_forward
    else:
        _fused_moe_forward = None


# ── OpSlot kernel registrations ──────────────────────────────────────────────

from ..kernel_registry import KERNEL_REGISTRY, HardwareRequirement, KernelSpec


def _make_moe_experts_adapter(raw_forward):
    """Adapt the raw fused MoE kernel to the OpSlot call signature.

    OpSlot invokes with ``(self, hidden_states, top_k_index, top_k_weights)``
    but the raw kernel uses ``(num_experts, routing_weights, ...)``.
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


def _fused_moe_kernel_factory():
    from .group_gemm import group_gemm_fused_moe_forward

    return _make_moe_experts_adapter(group_gemm_fused_moe_forward)


KERNEL_REGISTRY.register(
    KernelSpec(
        name="fused",
        op_name="moe_experts",
        variant="standard",
        factory=_fused_moe_kernel_factory,
        hardware=HardwareRequirement(device_type="gpu", min_compute_capability=70),
        description="Triton group-gemm fused MoE forward",
    )
)


def _fused_quack_kernel_factory():
    from .quack_gemm import quack_gemm_fused_moe_forward

    return _make_moe_experts_adapter(quack_gemm_fused_moe_forward)


KERNEL_REGISTRY.register(
    KernelSpec(
        name="fused_quack",
        op_name="moe_experts",
        variant="standard",
        factory=_fused_quack_kernel_factory,
        hardware=HardwareRequirement(device_type="gpu", min_compute_capability=90),
        description="Quack CUTLASS/CuTe fused MoE forward (SM90+)",
    )
)

if is_torch_npu_available():
    def _make_npu_moe_experts_adapter(raw_forward):
        """Adapt the raw NPU fused MoE kernel to the OpSlot call signature.

        OpSlot invokes with ``(self, hidden_states, top_k_index, top_k_weights)``
        but the raw kernel uses ``(num_experts, routing_weights, ...)``.
        """

        def adapter(self, hidden_states, top_k_index, top_k_weights):
            return raw_forward(
                num_experts=self.num_experts,
                routing_weights=top_k_weights.to(hidden_states.dtype),
                selected_experts=top_k_index.to(torch.int64),
                hidden_states=hidden_states,
                fc1_1_weight=self.gate_up_proj[:, :self.intermediate_dim, :].contiguous(),
                fc1_2_weight=self.gate_up_proj[:, self.intermediate_dim:, :].contiguous(),
                fc2_weight=self.down_proj,
                fc1_1_2_weight=None,
            )

        return adapter
    
    def _npu_fused_moe_kernel_factory():
        from .npu_group_gemm import npu_fused_moe_forward
        return _make_npu_moe_experts_adapter(npu_fused_moe_forward)

    KERNEL_REGISTRY.register(
        KernelSpec(
            name="npu_fused",
            op_name="moe_experts",
            variant="standard",
            factory=_npu_fused_moe_kernel_factory,
            hardware=HardwareRequirement(device_type="npu"),
            description="NPU group-gemm fused MoE forward",
        )
    )
