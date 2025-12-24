import os

import torch

from ...utils import logging
from ...utils.import_utils import is_fused_moe_available, is_torch_npu_available


logger = logging.get_logger(__name__)

_fused_moe_forward = None


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
    # bf16/fp16 for triton
    assert routing_weights.dtype in [torch.bfloat16, torch.float16], (
        f"routing_weights dtype must be bfloat16 or float16, but got {routing_weights.dtype}"
    )
    assert hidden_states.dtype in [torch.bfloat16, torch.float16], (
        f"hidden_states dtype must be bfloat16 or float16, but got {hidden_states.dtype}"
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


def apply_veomni_fused_moe_patch():
    global _fused_moe_forward
    if is_torch_npu_available():
        from .npu_fused_moe import npu_fused_moe_forward

        _fused_moe_forward = npu_fused_moe_forward
    elif is_fused_moe_available() and os.environ.get("USE_GROUP_GEMM", "1") == "1":
        from .group_gemm_fused_moe import group_gemm_fused_moe_forward

        _fused_moe_forward = group_gemm_fused_moe_forward
    else:
        _fused_moe_forward = None

    kernel_name = _fused_moe_forward.__name__ if _fused_moe_forward is not None else "None"
    logger.info_rank0(f"✅ using {kernel_name} for fused moe kernel")
