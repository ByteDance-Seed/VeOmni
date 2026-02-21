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

import os

import torch

from ...utils import logging
from ...utils.env import get_env
from ...utils.import_utils import (
    is_fused_moe_available,
    is_torch_npu_available,
    is_transformers_version_greater_or_equal_to,
)


logger = logging.get_logger(__name__)

_fused_moe_forward = None
VEOMNI_FUSED_MOE_EXPERTS_IMPL = "veomni_fused_moe"


def fused_moe_forward(
    module: torch.nn.Module,
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor,
    fc1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
):
    if _fused_moe_forward is None:
        raise NotImplementedError("No fused MoE kernel is available. Please check your environment.")

    assert routing_weights.dtype in [torch.bfloat16, torch.float16], (
        f"routing_weights dtype must be bfloat16 or float16 for triton kernel, but got {routing_weights.dtype}"
    )
    assert hidden_states.dtype in [torch.bfloat16, torch.float16], (
        f"hidden_states dtype must be bfloat16 or float16 for triton kernel, but got {hidden_states.dtype}"
    )

    return _fused_moe_forward(
        module,
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
    )


def veomni_fused_moe_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    if not hasattr(self, "gate_up_proj") or not hasattr(self, "down_proj"):
        raise ValueError(
            "VeOmni fused MoE experts backend requires merged expert weights: `gate_up_proj` and `down_proj`."
        )

    assert not getattr(self, "is_transposed", False), (
        "VeOmni fused MoE experts backend does not support `is_transposed=True` yet. "
        "In Transformers, only GPT-OSS currently uses `is_transposed=True`."
    )
    assert not getattr(self, "has_bias", False), (
        "VeOmni fused MoE experts backend does not support expert bias weights."
    )

    gate_up_proj = self.gate_up_proj
    down_proj = self.down_proj
    if gate_up_proj.ndim != 3 or down_proj.ndim != 3:
        raise ValueError(
            "VeOmni fused MoE experts backend expects 3D expert weights, "
            f"got gate_up_proj.ndim={gate_up_proj.ndim}, down_proj.ndim={down_proj.ndim}."
        )
    if gate_up_proj.shape[1] % 2 != 0:
        raise ValueError(
            "VeOmni fused MoE experts backend expects the merged `gate_up_proj` middle dimension to be even, "
            f"got {gate_up_proj.shape[1]}."
        )

    expert_dim = gate_up_proj.shape[1] // 2
    # TODO(yifan.pi): Remove/relax this materialization when fused kernels can efficiently consume strided views.
    # See docs/transformers_v5/moe_experts_registration.md for why split views from merged `gate_up_proj`
    # currently need `contiguous()` before entering VeOmni fused kernels.
    fc1_1_weight = gate_up_proj[:, :expert_dim, :].contiguous()
    fc1_2_weight = gate_up_proj[:, expert_dim:, :].contiguous()
    fc2_weight = down_proj.contiguous()
    routing_weights = top_k_weights.to(hidden_states.dtype)

    out = fused_moe_forward(
        module=self,
        num_experts=self.num_experts,
        routing_weights=routing_weights,
        selected_experts=top_k_index,
        hidden_states=hidden_states,
        fc1_1_weight=fc1_1_weight,
        fc1_2_weight=fc1_2_weight,
        fc2_weight=fc2_weight,
    )
    return out.to(hidden_states.dtype)


def _register_veomni_fused_moe_experts_impl():
    if not is_transformers_version_greater_or_equal_to("5.0.0"):
        return
    try:
        from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    except ImportError as e:
        logger.warning_rank0(f"Failed to register `{VEOMNI_FUSED_MOE_EXPERTS_IMPL}` experts backend: {e}")
        return

    ALL_EXPERTS_FUNCTIONS.register(VEOMNI_FUSED_MOE_EXPERTS_IMPL, veomni_fused_moe_experts_forward)


def apply_veomni_fused_moe_patch():
    global _fused_moe_forward
    if is_torch_npu_available():
        from .npu_group_gemm import npu_fused_moe_forward

        _fused_moe_forward = npu_fused_moe_forward
    elif is_fused_moe_available() and get_env("USE_GROUP_GEMM") == "1":
        from .group_gemm import group_gemm_fused_moe_forward

        _fused_moe_forward = group_gemm_fused_moe_forward
    else:
        _fused_moe_forward = None

    _register_veomni_fused_moe_experts_impl()
