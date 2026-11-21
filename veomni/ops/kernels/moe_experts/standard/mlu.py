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

"""standard MoE experts MLU Apex group-gemm implementation.

Apex grouped GEMM. Apex is imported lazily so GPU hosts can register
this row without ``apex`` / ``torch_mlu``. See ByteDance-Seed/VeOmni#903.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .....distributed.moe import cumsum_to_counts
from .....distributed.parallel_state import get_parallel_state
from ..shared.activation import is_silu
from ..shared.swiglu import apply_swiglu_clamp


def _apex_gmm(**kwargs):
    """Call Apex grouped GEMM. Lazy-imports ``apex``."""
    from apex.contrib.grouped_gemm.ops import backend as apex_backend

    return apex_backend.gmm(**kwargs)


class MLUGroupGemm(torch.autograd.Function):
    """Split-fc1 MLU grouped GEMM with SwiGLU and fc2."""

    @staticmethod
    def forward(ctx, permute_tokens, cumsum, fc1_1_weight, fc1_2_weight, fc2_weight, swiglu_limit=None):
        """Permute tokens through split fc1, SwiGLU, and fc2."""
        batch_sizes = cumsum_to_counts(cumsum)
        fc1_1_output = _apex_gmm(
            a=permute_tokens, b=fc1_1_weight, batch_sizes=batch_sizes, trans_a=False, trans_b=True
        )
        fc1_2_output = _apex_gmm(
            a=permute_tokens, b=fc1_2_weight, batch_sizes=batch_sizes, trans_a=False, trans_b=True
        )
        fc1_1_output, fc1_2_output, mask_fc1_1, mask_fc1_2 = apply_swiglu_clamp(
            fc1_1_output, fc1_2_output, swiglu_limit
        )
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_output = fc1_1_activation * fc1_2_output
        fc2_output = _apex_gmm(a=fc1_output, b=fc2_weight, batch_sizes=batch_sizes, trans_a=False, trans_b=True)
        ctx.swiglu_limit = swiglu_limit
        ctx.save_for_backward(
            permute_tokens,
            cumsum,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            mask_fc1_1 if mask_fc1_1 is not None else torch.empty(0, device=permute_tokens.device),
            mask_fc1_2 if mask_fc1_2 is not None else torch.empty(0, device=permute_tokens.device),
        )
        return fc2_output

    @staticmethod
    def backward(ctx, grad_output):
        """Return grads for permute tokens and the three weight tensors."""
        (
            permute_tokens,
            cumsum,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            mask_fc1_1,
            mask_fc1_2,
        ) = ctx.saved_tensors
        swiglu_limit = ctx.swiglu_limit
        batch_sizes = cumsum_to_counts(cumsum)
        grad_fc1_output = _apex_gmm(a=grad_output, b=fc2_weight, batch_sizes=batch_sizes, trans_a=False, trans_b=False)
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_output = fc1_1_activation * fc1_2_output
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = _apex_gmm(
                a=grad_output, b=fc1_output, batch_sizes=batch_sizes, trans_a=True, trans_b=False
            )
        grad_fc1_2_output = fc1_1_activation * grad_fc1_output
        grad_fc1_1_activation = grad_fc1_output * fc1_2_output
        if swiglu_limit is not None:
            grad_fc1_2_output.masked_fill_(~mask_fc1_2, 0)
        grad_scatter_output_2 = _apex_gmm(
            a=grad_fc1_2_output, b=fc1_2_weight, batch_sizes=batch_sizes, trans_a=False, trans_b=False
        )
        grad_fc1_2_weight = None
        if fc1_2_weight.requires_grad:
            grad_fc1_2_weight = _apex_gmm(
                a=grad_fc1_2_output, b=permute_tokens, batch_sizes=batch_sizes, trans_a=True, trans_b=False
            )
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)
        if swiglu_limit is not None:
            grad_fc1_1_output.masked_fill_(~mask_fc1_1, 0)
        grad_scatter_output_1 = _apex_gmm(
            a=grad_fc1_1_output, b=fc1_1_weight, batch_sizes=batch_sizes, trans_a=False, trans_b=False
        )
        grad_fc1_1_weight = None
        if fc1_1_weight.requires_grad:
            grad_fc1_1_weight = _apex_gmm(
                a=grad_fc1_1_output, b=permute_tokens, batch_sizes=batch_sizes, trans_a=True, trans_b=False
            )
        return (
            grad_scatter_output_1 + grad_scatter_output_2,
            None,
            grad_fc1_1_weight,
            grad_fc1_2_weight,
            grad_fc2_weight,
            None,
        )


class MLUMergedFc1GroupGemm(torch.autograd.Function):
    """Merged-fc1 MLU grouped GEMM with SwiGLU and fc2."""

    @staticmethod
    def forward(ctx, permute_tokens, cumsum, fc1_1_2_weight, fc2_weight, swiglu_limit=None):
        """Permute tokens through merged fc1, SwiGLU, and fc2."""
        if fc1_1_2_weight.shape[1] % 2 != 0:
            raise ValueError(f"Merged fc1_1_2_weight dim 1 must be even, got {fc1_1_2_weight.shape[1]}")
        batch_sizes = cumsum_to_counts(cumsum)
        fc1_output = _apex_gmm(
            a=permute_tokens, b=fc1_1_2_weight, batch_sizes=batch_sizes, trans_a=False, trans_b=True
        )
        fc1_1_output, fc1_2_output = fc1_output.chunk(2, dim=-1)
        fc1_1_output, fc1_2_output, mask_fc1_1, mask_fc1_2 = apply_swiglu_clamp(
            fc1_1_output, fc1_2_output, swiglu_limit
        )
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_result = fc1_1_activation * fc1_2_output
        fc2_output = _apex_gmm(a=fc1_result, b=fc2_weight, batch_sizes=batch_sizes, trans_a=False, trans_b=True)
        ctx.swiglu_limit = swiglu_limit
        ctx.save_for_backward(
            permute_tokens,
            cumsum,
            fc1_1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            mask_fc1_1 if mask_fc1_1 is not None else torch.empty(0, device=permute_tokens.device),
            mask_fc1_2 if mask_fc1_2 is not None else torch.empty(0, device=permute_tokens.device),
        )
        return fc2_output

    @staticmethod
    def backward(ctx, grad_output):
        """Return grads for permute tokens and the two weight tensors."""
        (
            permute_tokens,
            cumsum,
            fc1_1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            mask_fc1_1,
            mask_fc1_2,
        ) = ctx.saved_tensors
        swiglu_limit = ctx.swiglu_limit
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_result = fc1_1_activation * fc1_2_output
        batch_sizes = cumsum_to_counts(cumsum)
        grad_fc1_result = _apex_gmm(a=grad_output, b=fc2_weight, batch_sizes=batch_sizes, trans_b=False)
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = _apex_gmm(
                a=grad_output, b=fc1_result, batch_sizes=batch_sizes, trans_a=True, trans_b=False
            )
        grad_fc1_2_output = fc1_1_activation * grad_fc1_result
        grad_fc1_1_activation = grad_fc1_result * fc1_2_output
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)
        if swiglu_limit is not None:
            grad_fc1_1_output.masked_fill_(~mask_fc1_1, 0)
            grad_fc1_2_output.masked_fill_(~mask_fc1_2, 0)
        grad_fc1_output = torch.cat([grad_fc1_1_output, grad_fc1_2_output], dim=-1)
        grad_permute_tokens = _apex_gmm(a=grad_fc1_output, b=fc1_1_2_weight, batch_sizes=batch_sizes, trans_b=False)
        grad_fc1_1_2_weight = None
        if fc1_1_2_weight.requires_grad:
            grad_fc1_1_2_weight = _apex_gmm(
                a=grad_fc1_output, b=permute_tokens, batch_sizes=batch_sizes, trans_a=True, trans_b=False
            )
        return grad_permute_tokens, None, grad_fc1_1_2_weight, grad_fc2_weight, None


def mlu_group_gemm_fused_moe_forward(
    num_experts: int,
    routing_weights: Tensor,
    selected_experts: Tensor,
    hidden_states: Tensor,
    fc1_1_weight: Tensor | None,
    fc1_2_weight: Tensor | None,
    fc2_weight: Tensor,
    fc1_1_2_weight: Tensor | None = None,
    swiglu_limit: float | None = None,
    assume_distinct_experts: bool = False,
    block_size: int = 128,
    dtype: str = "fp8",
) -> Tensor:
    """MLU grouped-gemm fused MoE. Empty weights are ``None``.

    Split or merged fc1. EP comm stays outside the Function.
    ``assume_distinct_experts`` only tightens the Triton grouped-GEMM
    launch bound, so the flag is a no-op here.
    """
    # Router scores stay float32. Cast before the fused scale so the MLU GEMM
    # does not see an fp32 activation against bf16 weights.
    routing_weights = routing_weights.to(dtype=hidden_states.dtype)
    del assume_distinct_experts
    if get_parallel_state().ep_enabled:
        # The all-to-all plumbing is the same for every fused row, so this shares it
        # with ``fused_triton`` and ``fused_quack`` instead of keeping its own copy.
        from .....distributed.moe import dispatch_to_ep_class

        if fc1_1_2_weight is not None:
            if fc1_1_weight is not None or fc1_2_weight is not None:
                raise ValueError("Provide either split fc1 weights or merged fc1_1_2_weight, not both.")
            return dispatch_to_ep_class(
                MLUMergedFc1GroupGemm,
                num_experts,
                routing_weights,
                selected_experts,
                hidden_states,
                fc1_1_2_weight,
                fc2_weight,
                swiglu_limit,
            )
        if fc1_1_weight is None or fc1_2_weight is None:
            raise ValueError("EP requires split fc1 weights (fc1_1_weight and fc1_2_weight).")
        return dispatch_to_ep_class(
            MLUGroupGemm,
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            swiglu_limit,
        )

    from apex.contrib.permute import permute
    from apex.contrib.unpermute import unpermute

    permute_tokens, sorted_indice = permute(hidden_states, selected_experts, -1)
    splits = torch.bincount(selected_experts.view(-1), minlength=num_experts)
    cumsum = torch.cumsum(splits, dim=0)

    if fc1_1_2_weight is not None:
        if fc1_1_weight is not None or fc1_2_weight is not None:
            raise ValueError("Provide either split fc1 weights or merged fc1_1_2_weight, not both.")
        final_permute_tokens = MLUMergedFc1GroupGemm.apply(
            permute_tokens, cumsum, fc1_1_2_weight, fc2_weight, swiglu_limit
        )
    else:
        if fc1_1_weight is None or fc1_2_weight is None:
            raise ValueError("EP requires split fc1 weights (fc1_1_weight and fc1_2_weight).")
        final_permute_tokens = MLUGroupGemm.apply(
            permute_tokens, cumsum, fc1_1_weight, fc1_2_weight, fc2_weight, swiglu_limit
        )

    return unpermute(final_permute_tokens, sorted_indice, routing_weights)


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
    """Call the MLU fused MoE path. Empty weights are ``None``."""
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
    return mlu_group_gemm_fused_moe_forward(
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
