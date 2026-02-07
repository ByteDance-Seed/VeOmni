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
from ...utils.import_utils import is_fused_moe_available, is_torch_npu_available


logger = logging.get_logger(__name__)

_fused_moe_forward = None


def fused_moe_forward(
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
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
    )


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
