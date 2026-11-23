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


"""standard MoE experts npu implementation."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from .....distributed.moe import cumsum_to_counts, dispatch_to_ep_class
from .....distributed.parallel_state import get_parallel_state
from ..shared.activation import is_silu


class GmmFunction(torch.autograd.Function):
    """NPU grouped matmul with dgrad and wgrad."""

    @staticmethod
    def forward(ctx, x, weight, group_list):
        """Grouped matmul ``x @ weight`` per expert."""
        import torch_npu

        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list

        fwd_output = torch_npu.npu_grouped_matmul(
            [x], [weight], bias=None, group_list=group_list, split_item=2, group_type=0, group_list_type=1
        )[0]
        return fwd_output

    @staticmethod
    def backward(ctx, grad_output):
        """Grouped dgrad and wgrad via ``npu_grouped_matmul``."""
        import torch_npu

        input_tensor, weight = ctx.saved_tensors
        group_list = ctx.group_list

        weight = torch.transpose(weight, 1, 2)
        grad_input = torch_npu.npu_grouped_matmul(
            [grad_output], [weight], bias=None, group_list=group_list, split_item=2, group_type=0, group_list_type=1
        )[0]

        grad_weight = torch_npu.npu_grouped_matmul(
            [input_tensor.T],
            [grad_output],
            bias=None,
            group_list=group_list,
            split_item=3,
            group_type=2,
            group_list_type=1,
        )[0]

        return grad_input, grad_weight, None


def npu_group_gemm(x, weight, group_list):
    """Apply ``GmmFunction`` to ``(x, weight, group_list)``."""
    output = GmmFunction.apply(x, weight, group_list)
    return output


def _eager_clamped_swiglu(x: torch.Tensor, limit: float) -> torch.Tensor:
    """Apply clamped SwiGLU with eager PyTorch operations."""
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


def _fc1_weight(
    fc1_1_weight: torch.Tensor | None,
    fc1_2_weight: torch.Tensor | None,
    fc1_1_2_weight: torch.Tensor | None,
) -> torch.Tensor:
    """Resolve split vs merged fc1. Same contract as the eager wrapper."""
    has_merged = fc1_1_2_weight is not None
    has_split = fc1_1_weight is not None or fc1_2_weight is not None
    if has_split == has_merged:
        raise ValueError("Provide either split fc1 weights or merged fc1_1_2_weight, not both or neither.")
    if has_merged:
        return fc1_1_2_weight
    if fc1_1_weight is None or fc1_2_weight is None:
        raise ValueError("Split fc1 mode requires both fc1_1_weight and fc1_2_weight.")
    return torch.cat([fc1_1_weight, fc1_2_weight], dim=1)


def _is_triton_ascend_available() -> bool:
    """Return whether the installed Triton runtime includes the Ascend backend."""
    try:
        from triton._C import libtriton
    except ImportError:
        return False
    return hasattr(libtriton, "ascend")


def _clamped_swiglu(x: torch.Tensor, limit: float) -> torch.Tensor:
    """Use Ascend Triton when available and preserve the eager training fallback."""
    if _is_triton_ascend_available():
        from ..shared.npu_clamped_swiglu import npu_triton_clamped_swiglu

        return npu_triton_clamped_swiglu(x, limit)

    return _eager_clamped_swiglu(x, limit)


def _swiglu(x: torch.Tensor, swiglu_limit: float | None) -> torch.Tensor:
    """NPU SwiGLU, or the clamped path when ``swiglu_limit`` is set."""
    import torch_npu

    if swiglu_limit is None:
        return torch_npu.npu_swiglu(x, dim=-1)
    return _clamped_swiglu(x, swiglu_limit)


def _npu_fused_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor | None,
    fc1_2_weight: torch.Tensor | None,
    fc2_weight: torch.Tensor,
    fc1_1_2_weight: torch.Tensor | None = None,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """NPU single-device fused MoE forward pass (non-EP).

    Accepts either split fc1 weights or a merged fc1_1_2_weight tensor.
    Weights are merged and transposed for the NPU group-gemm kernel.
    """
    import torch_npu

    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(
        hidden_states, selected_experts.to(torch.int32)
    )
    tokens_per_expert = torch.histc(selected_experts, bins=num_experts, min=0, max=num_experts)

    fc1_weight = _fc1_weight(fc1_1_weight, fc1_2_weight, fc1_1_2_weight).transpose(1, 2)
    intermediate_hidden_states = npu_group_gemm(permuted_hidden_states, fc1_weight, tokens_per_expert)
    intermediate_activations = _swiglu(intermediate_hidden_states, swiglu_limit)
    output = npu_group_gemm(intermediate_activations, fc2_weight.transpose(1, 2), tokens_per_expert)
    # NPU unpermute applies routing after fc2. Equivalent for a biasless down
    # projection; bf16 rounding can differ from the pre-fc2 fused order.
    hidden_states = torch_npu.npu_moe_token_unpermute(output, row_ids_map, probs=routing_weights)
    return hidden_states


def _npu_expert_math(permute_tokens, cumsum, fc1_weight, fc2_weight, swiglu_limit=None):
    """Two Ascend grouped GEMMs with the gated activation between them.

    Takes the shape ``dispatch_to_ep_class`` provides, so the all-to-all plumbing is
    shared with the other rows. No custom backward is needed: ``npu_group_gemm`` is
    Ascend's grouped matmul wrapped in ``GmmFunction``, which builds the graph itself.

    The shared helper hands over a *cumulative* count, but ``npu_group_gemm`` pins
    ``group_list_type=1`` -- per-group sizes, whose sum must equal the row count. So
    the running total is subtracted back out here, exactly as ``MLUGroupGemm`` does
    internally and ``_deepgemm_expert_math`` does explicitly.
    """
    counts = cumsum_to_counts(cumsum)
    intermediate = npu_group_gemm(permute_tokens, fc1_weight, counts)
    activated = _swiglu(intermediate, swiglu_limit)
    return npu_group_gemm(activated, fc2_weight, counts)


def npu_ep_fused_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor | None,
    fc1_2_weight: torch.Tensor | None,
    fc2_weight: torch.Tensor,
    fc1_1_2_weight: torch.Tensor | None = None,
    ep_group: dist.ProcessGroup | None = None,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """NPU expert-parallel fused MoE forward pass.

    Accepts either split fc1 weights or a merged fc1_1_2_weight tensor. The all-to-all
    plumbing is shared with every other fused row; only the per-rank math is Ascend's.
    """
    fc1_weight = _fc1_weight(fc1_1_weight, fc1_2_weight, fc1_1_2_weight).transpose(1, 2)
    return dispatch_to_ep_class(
        _npu_expert_math,
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_weight,
        fc2_weight.transpose(1, 2),
        swiglu_limit,
    )


def npu_fused_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor | None,
    fc1_2_weight: torch.Tensor | None,
    fc2_weight: torch.Tensor,
    fc1_1_2_weight: torch.Tensor | None = None,
    swiglu_limit: float | None = None,
    assume_distinct_experts: bool = False,  # intentionally unused; see the del below
):
    """NPU fused MoE entry. Branches on EP."""
    # Router scores stay float32. Cast before the fused scale so the NPU GEMM
    # does not see an fp32 activation against bf16 weights.
    routing_weights = routing_weights.to(dtype=hidden_states.dtype)
    # ``assume_distinct_experts`` only tightens the grouped-GEMM ``max_M``
    # launch bound in the Triton backend. The NPU group GEMM has no such launch
    # bound to tighten, so the flag is a no-op here.
    del assume_distinct_experts
    # EP comm is outside the Function so all2all is not under no_grad.
    if get_parallel_state().ep_enabled:
        final_hidden_states = npu_ep_fused_moe_forward(
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_2_weight,
            ep_group=get_parallel_state().ep_group,
            swiglu_limit=swiglu_limit,
        )
    else:
        final_hidden_states = _npu_fused_moe_forward(
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_2_weight,
            swiglu_limit=swiglu_limit,
        )
    return final_hidden_states


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
    """Call the NPU fused MoE path. Empty weights are ``None``."""
    if not is_silu(activation):
        # This kernel computes SiLU inside the grouped GEMM, so another activation has
        # no seam here. The ``eager`` row runs the reference expert loop, which also
        # owns the expert-parallel dispatch those sharded weights need.
        from . import eager as _eager

        return _eager.wrapper(
            hidden_states,
            routing_weights,
            selected_experts,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_2_weight,
            num_experts=num_experts,
            swiglu_limit=swiglu_limit,
            assume_distinct_experts=assume_distinct_experts,
            block_size=block_size,
            dtype=dtype,
            activation=activation,
        )
    del block_size, dtype
    # Empty tensor = unused layout. Registry args stay tensors.
    return npu_fused_moe_forward(
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_weight if fc1_1_weight.numel() else None,
        fc1_2_weight if fc1_2_weight.numel() else None,
        fc2_weight,
        fc1_1_2_weight if fc1_1_2_weight.numel() else None,
        swiglu_limit=swiglu_limit,
        assume_distinct_experts=assume_distinct_experts,
    )
