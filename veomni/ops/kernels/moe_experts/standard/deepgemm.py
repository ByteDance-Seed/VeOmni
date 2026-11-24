# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Hopper/Blackwell DeepGEMM FP8 routed experts, including training backward.

The public tensor contract is the standard BF16 MoE interface. EP communication,
routing and activations retain their existing precision. FSDP may supply
block-quantized expert weights directly through its optional FP8 all-gather.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .....distributed.moe import cumsum_to_counts
from .....distributed.parallel_state import get_parallel_state
from ..shared.activation import is_silu


def _experts(x, counts, fc1, fc2, swiglu_limit, recompute_weights=(True, True)):
    from ..shared.deepgemm import GroupedLayout, GroupedLinear, RecomputedSwiGLU

    layout = GroupedLayout.from_counts(counts, x.device)
    projected = GroupedLinear.apply(x, fc1, layout, recompute_weights[0])
    return GroupedLinear.apply(RecomputedSwiGLU.apply(projected, swiglu_limit), fc2, layout, recompute_weights[1])


def _deepgemm_expert_math(tokens, cumsum, fc1, fc2, swiglu_limit, recompute_weights=(True, True)):
    """The per-rank FP8 expert GEMMs, in the shape ``dispatch_to_ep_class`` provides.

    ``dispatch_to_ep_class`` hands over a *cumulative* token count, while
    ``GroupedLayout.from_counts`` wants each expert's own count -- so this subtracts
    the running total back out. Everything else is unchanged: the grouped FP8 GEMMs
    and ``RecomputedSwiGLU`` own their autograd nodes.
    """
    counts = cumsum_to_counts(cumsum)
    return _experts(tokens, counts, fc1, fc2, swiglu_limit, recompute_weights)


def _ep_forward(
    x, routing_weights, selected_experts, fc1, fc2, num_experts, swiglu_limit, recompute_weights=(True, True)
):
    """Dispatch tokens across EP ranks, run FP8 local experts, and combine outputs."""
    from .....distributed.moe import dispatch_to_ep_class

    return dispatch_to_ep_class(
        _deepgemm_expert_math,
        num_experts,
        routing_weights,
        selected_experts,
        x,
        fc1,
        fc2,
        swiglu_limit,
        recompute_weights,
    )


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
    # DeepGEMM currently implements FP8 only; any other dtype raises below.
    dtype: str = "fp8",
    activation: str = "silu",
) -> Tensor:
    """Run FP8 expert GEMMs with BF16 inputs, outputs and first-order gradients.

    Supports merged or split gate/up weights, zero-token experts, and EP with
    distinct or repeated expert ids. Hidden and intermediate sizes must be
    multiples of 128. Unsupported hardware,
    dtypes or layouts fail explicitly. CPU count metadata prevents CUDA graph
    capture. Routing weights are applied after the down projection, matching the
    eager expert contract rather than moving them through FP8 quantization.
    """
    if dtype != "fp8":
        raise ValueError(f"moe_implementation='deepgemm' currently supports dtype='fp8' only, got {dtype!r}.")
    if not is_silu(activation):
        # Refused here, before any of the FP8 math, because this is where the
        # quantization is decided. ``RecomputedSwiGLU`` computes SiLU inside the
        # grouped GEMM and recomputes it in the backward instead of saving it, so
        # there is no seam to substitute another activation into. Handing the call
        # to a reference row -- what the bf16 fused rows do, losing nothing -- would
        # skip the FP8 GEMMs and train the experts in bf16 under a config that says
        # ``deepgemm``.
        raise ValueError(
            f"moe_implementation='deepgemm' computes SiLU only, and the model asks "
            f"for {activation!r}. Its grouped GEMM runs in FP8 and fuses the activation, "
            f"so another activation cannot be substituted without dropping the "
            f"quantization. Use hidden_act='silu', or moe_implementation='eager' to train "
            f"the experts in the model dtype."
        )
    if (
        hidden_states.device.type != "cuda"
        or torch.version.hip is not None
        or torch.cuda.get_device_capability(hidden_states.device) not in ((9, 0), (10, 0))
    ):
        raise RuntimeError("deepgemm requires NVIDIA Hopper SM90 or Blackwell SM100.")
    if hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
        raise ValueError("deepgemm requires BF16 hidden_states with shape [tokens, hidden].")
    if (
        selected_experts.ndim != 2
        or selected_experts.shape[1] == 0
        or routing_weights.shape != selected_experts.shape
        or selected_experts.shape[0] != hidden_states.shape[0]
    ):
        raise ValueError("Routing weights and expert indices must have matching [tokens, top_k] shapes.")
    if selected_experts.dtype != torch.int64 or not routing_weights.is_floating_point():
        raise ValueError("Expert indices must be int64 and routing weights must be floating point.")
    if selected_experts.device != hidden_states.device or routing_weights.device != hidden_states.device:
        raise ValueError("Routing tensors must be on the same device as hidden_states.")
    merged = fc1_1_2_weight.numel() != 0
    split = fc1_1_weight.numel() != 0 or fc1_2_weight.numel() != 0
    if merged == split or (split and (fc1_1_weight.numel() == 0 or fc1_2_weight.numel() == 0)):
        raise ValueError("Provide either both split fc1 weights or merged fc1 weights, not both or neither.")
    fc1 = fc1_1_2_weight if merged else torch.cat((fc1_1_weight, fc1_2_weight), dim=1)
    if fc1.ndim != 3 or fc2_weight.ndim != 3:
        raise ValueError("Expert weights must have shapes [E,2*I,H] and [E,H,I].")
    e, h, intermediate = fc2_weight.shape
    if fc1.shape != (e, 2 * intermediate, h) or hidden_states.shape[1] != h:
        raise ValueError("Incompatible expert weights or hidden dimension.")
    if min(e, h, intermediate) == 0 or h % 128 or intermediate % 128:
        raise ValueError("FP8 MoE requires positive dimensions and hidden/intermediate multiples of 128.")
    for weight in (fc1, fc2_weight):
        if weight.dtype != torch.bfloat16 or weight.device != hidden_states.device:
            raise ValueError("FP8 MoE weights must be BF16 and on the same device as hidden_states.")
    # Reuse only parameter storage: casts, QAT, concat and contiguous copies
    # can otherwise keep a larger BF16 temporary alive than the FP8 cache.
    # FSDP2 repopulates its unsharded Parameter before the backward hook.
    recompute_weights = tuple(isinstance(w, torch.nn.Parameter) and w.is_contiguous() for w in (fc1, fc2_weight))
    fc1, fc2 = fc1.contiguous(), fc2_weight.contiguous()
    if get_parallel_state().ep_enabled:
        if e * get_parallel_state().ep_size != num_experts:
            raise ValueError("The EP-local expert weight count does not match num_experts / ep_size.")
        return _ep_forward(
            hidden_states, routing_weights, selected_experts, fc1, fc2, num_experts, swiglu_limit, recompute_weights
        )
    if e != num_experts:
        raise ValueError("Expert weights must contain num_experts experts when EP is disabled.")

    # Each top-k slot remains a separate row, including duplicate expert ids.
    order = torch.argsort(selected_experts.reshape(-1), stable=True)
    token_indices = torch.div(order, selected_experts.shape[1], rounding_mode="floor")
    counts = torch.bincount(selected_experts.reshape(-1), minlength=num_experts)
    if counts.numel() != num_experts:
        raise ValueError("Expert indices must be less than num_experts.")
    x = hidden_states.index_select(0, token_indices)
    y = _experts(x, counts, fc1, fc2, swiglu_limit, recompute_weights)
    weights = routing_weights.reshape(-1).index_select(0, order)
    weighted = (y * weights[:, None]).to(hidden_states.dtype)
    return torch.zeros_like(hidden_states).index_add(0, token_indices, weighted)
