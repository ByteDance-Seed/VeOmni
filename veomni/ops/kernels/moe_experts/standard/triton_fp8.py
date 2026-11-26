# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Independent Blackwell MXFP8 standard MoE backend.

Like the DeepGEMM backend, routing composes with two grouped-linear
nodes and a recomputed activation through regular PyTorch autograd.
"""

import torch
from torch import Tensor

from .....distributed.parallel_state import get_parallel_state
from ..shared.activation import is_silu


def _experts(x, token_ends, fc1, fc2, swiglu_limit):
    from ..shared.mxfp8 import GroupedLinear, RecomputedSwiGLU

    projected = GroupedLinear.apply(x, fc1, token_ends)
    activated = RecomputedSwiGLU.apply(projected, swiglu_limit)
    return GroupedLinear.apply(activated, fc2, token_ends)


def wrapper(
    hidden_states: Tensor,
    routing_weights: Tensor,
    selected_experts: Tensor,
    fc1_1_weight: Tensor,
    fc1_2_weight: Tensor,
    fc2_weight: Tensor,
    fc1_1_2_weight: Tensor,
    *,
    num_experts: int,
    swiglu_limit: float | None = None,
    assume_distinct_experts: bool = False,
    block_size: int = 128,
    dtype: str = "fp8",
    activation: str = "silu",
) -> Tensor:
    """Run both merged expert projections in MXFP8 forward, dgrad and wgrad.

    Requires BF16 operands, merged gate/up, SiLU/SwiGLU and EP1 on SM10x.
    Routing weights multiply the down-projection output in the model dtype,
    matching fused_triton. Duplicate routes remain distinct rows. The generic
    block_size hint does not change this fixed E4M3/E8M0 K32 recipe.
    Host expert-boundary validation prevents CUDA graph/full-graph capture.
    """
    del block_size, assume_distinct_experts
    if dtype != "fp8" or not is_silu(activation):
        raise ValueError("fused_triton_fp8 requires FP8 and SiLU/SwiGLU")
    if get_parallel_state().ep_enabled:
        raise ValueError("Triton MXFP8 requires EP1")
    if fc1_1_weight.numel() or fc1_2_weight.numel() or not fc1_1_2_weight.numel():
        raise ValueError("Triton MXFP8 requires merged gate/up weights")
    if hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
        raise ValueError("MXFP8 hidden_states must be BF16 [tokens, hidden]")
    if (
        selected_experts.ndim != 2
        or selected_experts.shape[1] == 0
        or selected_experts.shape[0] != hidden_states.shape[0]
        or routing_weights.shape != selected_experts.shape
        or selected_experts.dtype != torch.int64
        or not routing_weights.is_floating_point()
        or selected_experts.device != hidden_states.device
        or routing_weights.device != hidden_states.device
    ):
        raise ValueError("Routing tensors must be matching [tokens, top_k] on the input device, with int64 indices")
    fc1, fc2 = fc1_1_2_weight, fc2_weight
    if fc1.ndim != 3 or fc2.ndim != 3:
        raise ValueError("MXFP8 weights must be [E,2*I,H] and [E,H,I]")
    e, h, intermediate = fc2.shape
    if (
        e != num_experts
        or min(e, h, intermediate) == 0
        or h % 128
        or intermediate % 128
        or fc1.shape != (e, 2 * intermediate, h)
        or hidden_states.shape[1] != h
    ):
        raise ValueError("MXFP8 requires matching expert weights and positive hidden/intermediate multiples of 128")
    for weight in (fc1, fc2):
        if weight.dtype != torch.bfloat16 or weight.device != hidden_states.device:
            raise ValueError("MXFP8 weights must be BF16 on the input device")
    from ..shared.mxfp8_group_gemm import _check_device

    _check_device(hidden_states, fc2)
    routing_weights = routing_weights.to(hidden_states.dtype)
    order = torch.argsort(selected_experts.reshape(-1), stable=True)
    token_indices = torch.div(order, selected_experts.shape[1], rounding_mode="floor")
    counts = torch.bincount(selected_experts.reshape(-1), minlength=num_experts)
    if counts.numel() != num_experts:
        raise ValueError("Expert indices must be less than num_experts")
    token_ends = counts.cumsum(0)
    tokens = hidden_states.index_select(0, token_indices)
    output = _experts(tokens, token_ends, fc1.contiguous(), fc2.contiguous(), swiglu_limit)
    weights = routing_weights.reshape(-1).index_select(0, order)
    weighted = output * weights[:, None]
    return torch.zeros_like(hidden_states).index_add(0, token_indices, weighted)
