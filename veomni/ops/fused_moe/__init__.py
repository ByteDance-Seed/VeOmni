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

from ...utils import logging
from ...utils.env import get_env
from ...utils.import_utils import (
    is_fused_moe_available,
    is_torch_npu_available,
)


logger = logging.get_logger(__name__)

_fused_moe_forward = None


def resolve_fc1_weights(
    fc1_1_weight: torch.Tensor | None,
    fc1_2_weight: torch.Tensor | None,
    fc1_1_2_weight: torch.Tensor | None,
    *,
    return_merged_fc1: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """Normalize fc1 weight inputs into the format required by the backend.

    Callers may pass *either* split weights (``fc1_1_weight``, ``fc1_2_weight``)
    or a single merged weight (``fc1_1_2_weight`` with shape ``[E, 2*I, H]``).
    This helper validates the inputs and converts to whichever format the
    downstream kernel expects.

    Args:
        return_merged_fc1: When *False* (default) return a ``(fc1_1, fc1_2)``
            tuple of split, contiguous tensors – suitable for Triton / torch
            grouped-gemm kernels.  When *True* return a single merged tensor
            ``[E, 2*I, H]`` – suitable for NPU kernels that concatenate
            gate+up before computing.
    """
    has_split = fc1_1_weight is not None or fc1_2_weight is not None
    has_merged = fc1_1_2_weight is not None

    if has_split and has_merged:
        raise ValueError("Provide either split fc1 weights (fc1_1 + fc1_2) or merged fc1_1_2_weight, not both.")
    if not has_split and not has_merged:
        raise ValueError("Either split fc1 weights (fc1_1 + fc1_2) or merged fc1_1_2_weight must be provided.")

    if return_merged_fc1:
        if has_merged:
            return fc1_1_2_weight
        assert fc1_1_weight is not None and fc1_2_weight is not None
        return torch.cat([fc1_1_weight, fc1_2_weight], dim=1)
    else:
        if has_split:
            if fc1_1_weight is None or fc1_2_weight is None:
                raise ValueError("Split fc1 mode requires both fc1_1_weight and fc1_2_weight.")
            return fc1_1_weight, fc1_2_weight
        assert fc1_1_2_weight is not None
        intermediate_dim = fc1_1_2_weight.shape[1] // 2
        return (
            fc1_1_2_weight[:, :intermediate_dim, :].contiguous(),
            fc1_1_2_weight[:, intermediate_dim:, :].contiguous(),
        )


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
        fc1_1_2_weight,
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
