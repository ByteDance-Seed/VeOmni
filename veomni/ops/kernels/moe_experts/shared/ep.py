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

"""EP-local Triton grouped GEMM shared by moe_experts impls.

These Functions run after ``token_pre_all2all`` and before
``tokens_post_all2all``. Routing is applied by the combine step, not here.
"""

from __future__ import annotations

import torch

from ....activation import apply_activation
from . import swiglu
from .activation import activation_backward, is_silu
from .swiglu import apply_swiglu_clamp


def _expert_activation(gate, activation):
    """Evaluate the configured activation, whichever name it is."""
    from ....activation import resolve, supports

    return apply_activation(gate, activation) if supports(activation) else resolve(activation)(gate)


class EPGroupGemm(torch.autograd.Function):
    """EP-local split-fc1 Triton experts on already-permuted tokens."""

    @staticmethod
    def forward(
        ctx,
        permute_tokens,
        cumsum,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
        swiglu_limit=None,
        activation="silu",
    ):
        """Split fc1 GEMMs, gated activation, and fc2. No routing scale."""
        from .group_gemm import group_gemm_same_nk, same_nk_tile_schedule

        tile_cumsum, tile_total_M = same_nk_tile_schedule(cumsum, permute_tokens.shape[0])
        fc1_1_output = group_gemm_same_nk(
            a=permute_tokens,
            b=fc1_1_weight,
            cumsum_M=cumsum,
            max_M=permute_tokens.shape[0],
            transpose_a=False,
            transpose_b=True,
            tile_cumsum=tile_cumsum,
            tile_total_M=tile_total_M,
        )
        fc1_2_output = group_gemm_same_nk(
            a=permute_tokens,
            b=fc1_2_weight,
            cumsum_M=cumsum,
            max_M=permute_tokens.shape[0],
            transpose_a=False,
            transpose_b=True,
            tile_cumsum=tile_cumsum,
            tile_total_M=tile_total_M,
        )
        fc1_1_output, fc1_2_output, mask_fc1_1, mask_fc1_2 = apply_swiglu_clamp(
            fc1_1_output, fc1_2_output, swiglu_limit
        )
        fc1_1_activation = _expert_activation(fc1_1_output, activation)
        fc1_output = fc1_1_activation * fc1_2_output
        fc2_output = group_gemm_same_nk(
            a=fc1_output,
            b=fc2_weight,
            cumsum_M=cumsum,
            max_M=permute_tokens.shape[0],
            transpose_a=False,
            transpose_b=True,
            tile_cumsum=tile_cumsum,
            tile_total_M=tile_total_M,
        )

        ctx.swiglu_limit = swiglu_limit
        ctx.activation = activation
        ctx.tile_prefix_total = tile_total_M
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
            tile_cumsum,
        )
        return fc2_output

    @staticmethod
    def backward(ctx, grad_output):
        """Split-fc1 dgrad and wgrad on permuted tokens."""
        from .group_gemm import group_gemm_same_mn, group_gemm_same_nk

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
            tile_cumsum,
        ) = ctx.saved_tensors
        tile_total_M = ctx.tile_prefix_total
        swiglu_limit = ctx.swiglu_limit

        grad_fc1_output = group_gemm_same_nk(
            a=grad_output,
            b=fc2_weight,
            cumsum_M=cumsum,
            max_M=grad_output.shape[0],
            transpose_b=False,
            tile_cumsum=tile_cumsum,
            tile_total_M=tile_total_M,
        )
        fc1_1_activation = _expert_activation(fc1_1_output, ctx.activation)
        fc1_output = fc1_1_activation * fc1_2_output

        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = torch.empty_like(fc2_weight)
            group_gemm_same_mn(
                a=grad_output,
                b=fc1_output,
                c=grad_fc2_weight,
                cumsum_K=cumsum,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        grad_fc1_2_output = fc1_1_activation * grad_fc1_output
        grad_fc1_1_activation = grad_fc1_output * fc1_2_output
        if swiglu_limit is not None:
            grad_fc1_2_output.masked_fill_(~mask_fc1_2, 0)

        grad_scatter_output_2 = group_gemm_same_nk(
            a=grad_fc1_2_output,
            b=fc1_2_weight,
            cumsum_M=cumsum,
            max_M=grad_output.shape[0],
            transpose_b=False,
            tile_cumsum=tile_cumsum,
            tile_total_M=tile_total_M,
        )
        grad_fc1_2_weight = None
        if fc1_2_weight.requires_grad:
            grad_fc1_2_weight = torch.empty_like(fc1_2_weight)
            group_gemm_same_mn(
                a=grad_fc1_2_output,
                b=permute_tokens,
                c=grad_fc1_2_weight,
                cumsum_K=cumsum,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        grad_fc1_1_output = activation_backward(grad_fc1_1_activation, fc1_1_output, ctx.activation)
        if swiglu_limit is not None:
            grad_fc1_1_output.masked_fill_(~mask_fc1_1, 0)

        grad_scatter_output_1 = group_gemm_same_nk(
            a=grad_fc1_1_output,
            b=fc1_1_weight,
            cumsum_M=cumsum,
            max_M=grad_output.shape[0],
            transpose_b=False,
            tile_cumsum=tile_cumsum,
            tile_total_M=tile_total_M,
        )
        grad_fc1_1_weight = None
        if fc1_1_weight.requires_grad:
            grad_fc1_1_weight = torch.empty_like(fc1_1_weight)
            group_gemm_same_mn(
                a=grad_fc1_1_output,
                b=permute_tokens,
                c=grad_fc1_1_weight,
                cumsum_K=cumsum,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        return (
            grad_scatter_output_1 + grad_scatter_output_2,
            None,
            grad_fc1_1_weight,
            grad_fc1_2_weight,
            grad_fc2_weight,
            None,
            None,  # activation
        )


class EPMergedFc1GroupGemm(torch.autograd.Function):
    """EP-local merged-fc1 Triton experts on already-permuted tokens."""

    @staticmethod
    def forward(
        ctx,
        permute_tokens,
        cumsum,
        fc1_1_2_weight,
        fc2_weight,
        swiglu_limit=None,
        activation="silu",
        *,
        replica_fc1_weight=None,
        replica_fc2_weight=None,
    ):
        """Merged fc1 GEMM, SwiGLU, and fc2. No routing scale."""
        from .group_gemm import group_gemm_same_nk, same_nk_tile_schedule

        assert fc1_1_2_weight.shape[1] % 2 == 0, (
            f"Merged fc1_1_2_weight dim 1 must be even, got {fc1_1_2_weight.shape[1]}"
        )
        tile_cumsum, tile_total_M = same_nk_tile_schedule(cumsum, permute_tokens.shape[0])
        max_tokens = getattr(ctx, "max_tokens_per_expert", permute_tokens.shape[0])
        fc1_output = group_gemm_same_nk(
            a=permute_tokens,
            b=fc1_1_2_weight,
            cumsum_M=cumsum,
            max_M=max_tokens,
            transpose_a=False,
            transpose_b=True,
            tile_cumsum=tile_cumsum,
            tile_total_M=tile_total_M,
            b_extra=replica_fc1_weight,
        )
        fc1_1_output, fc1_2_output = fc1_output.chunk(2, dim=-1)
        fc1_1_output, fc1_2_output, mask_fc1_1, mask_fc1_2 = apply_swiglu_clamp(
            fc1_1_output, fc1_2_output, swiglu_limit
        )
        if is_silu(activation):
            fc1_result = swiglu.merged_swiglu_forward(fc1_1_output, fc1_2_output)
        else:
            fc1_result = _expert_activation(fc1_1_output, activation) * fc1_2_output
        fc2_output = group_gemm_same_nk(
            a=fc1_result,
            b=fc2_weight,
            cumsum_M=cumsum,
            max_M=max_tokens,
            transpose_a=False,
            transpose_b=True,
            tile_cumsum=tile_cumsum,
            tile_total_M=tile_total_M,
            b_extra=replica_fc2_weight,
        )

        ctx.swiglu_limit = swiglu_limit
        ctx.activation = activation
        ctx.tile_prefix_total = tile_total_M
        ctx.save_for_backward(
            permute_tokens,
            cumsum,
            fc1_1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            mask_fc1_1 if mask_fc1_1 is not None else torch.empty(0, device=permute_tokens.device),
            mask_fc1_2 if mask_fc1_2 is not None else torch.empty(0, device=permute_tokens.device),
            tile_cumsum,
            *(
                (replica_fc1_weight, replica_fc2_weight)
                if (replica_fc1_weight is not None or replica_fc2_weight is not None)
                else ()
            ),
        )
        return fc2_output

    @staticmethod
    def backward(ctx, grad_output, *, weight_gradient_dtype=None, weight_needs_grad=None, weight_grad_fn=None):
        """Merged-fc1 dgrad and wgrad on permuted tokens."""
        from .group_gemm import group_gemm_same_mn, group_gemm_same_nk

        (
            permute_tokens,
            cumsum,
            fc1_1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            mask_fc1_1,
            mask_fc1_2,
            tile_cumsum,
            *replica_weights,
        ) = ctx.saved_tensors
        replica_fc1_weight, replica_fc2_weight = replica_weights or (None, None)
        tile_total_M = ctx.tile_prefix_total
        swiglu_limit = ctx.swiglu_limit
        weight_grad_fn = group_gemm_same_mn if weight_grad_fn is None else weight_grad_fn
        max_tokens = getattr(ctx, "max_tokens_per_expert", grad_output.shape[0])

        grad_fc1_result = group_gemm_same_nk(
            a=grad_output,
            b=fc2_weight,
            cumsum_M=cumsum,
            max_M=max_tokens,
            transpose_b=False,
            tile_cumsum=tile_cumsum,
            tile_total_M=tile_total_M,
            b_extra=replica_fc2_weight,
        )

        if is_silu(ctx.activation):
            fc1_result, grad_fc1_output = swiglu.merged_swiglu_backward(
                fc1_1_output,
                fc1_2_output,
                grad_fc1_result,
                mask_fc1_1 if swiglu_limit is not None else None,
                mask_fc1_2 if swiglu_limit is not None else None,
            )
        else:
            gate_grad = activation_backward(grad_fc1_result * fc1_2_output, fc1_1_output, ctx.activation)
            up_grad = grad_fc1_result * _expert_activation(fc1_1_output, ctx.activation)
            # Gate on ``swiglu_limit``: forward saves an empty float tensor where a
            # clamp mask would go, so "is not None" is always true and ``~`` would be
            # applied to a float.
            if swiglu_limit is not None:
                gate_grad.masked_fill_(~mask_fc1_1, 0)
                up_grad.masked_fill_(~mask_fc1_2, 0)
            fc1_result = _expert_activation(fc1_1_output, ctx.activation) * fc1_2_output
            grad_fc1_output = torch.cat([gate_grad, up_grad], dim=-1)
        del grad_fc1_result

        def empty_weight_grad(weight):
            if weight_gradient_dtype is None:
                return torch.empty_like(weight)
            return torch.empty((len(cumsum), *weight.shape[1:]), dtype=weight_gradient_dtype, device=weight.device)

        grad_fc2_weight = pending_fc2 = None
        if fc2_weight.requires_grad if weight_needs_grad is None else weight_needs_grad[1]:
            grad_fc2_weight = empty_weight_grad(fc2_weight)
            start_weight_grad = getattr(weight_grad_fn, "start", None)
            launch_weight_grad = start_weight_grad if start_weight_grad is not None else weight_grad_fn
            result = launch_weight_grad(
                a=grad_output,
                b=fc1_result,
                c=grad_fc2_weight,
                cumsum_K=cumsum,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )
            if start_weight_grad is not None:
                pending_fc2 = result
        del fc1_result

        try:
            grad_permute_tokens = group_gemm_same_nk(
                a=grad_fc1_output,
                b=fc1_1_2_weight,
                cumsum_M=cumsum,
                max_M=max_tokens,
                transpose_b=False,
                tile_cumsum=tile_cumsum,
                tile_total_M=tile_total_M,
                b_extra=replica_fc1_weight,
            )
        finally:
            # Drain the first ordered prefix before starting the other chain.
            if pending_fc2 is not None:
                pending_fc2.wait()

        grad_fc1_1_2_weight = None
        if fc1_1_2_weight.requires_grad if weight_needs_grad is None else weight_needs_grad[0]:
            grad_fc1_1_2_weight = empty_weight_grad(fc1_1_2_weight)
            weight_grad_fn(
                a=grad_fc1_output,
                b=permute_tokens,
                c=grad_fc1_1_2_weight,
                cumsum_K=cumsum,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        return (
            grad_permute_tokens,
            None,
            grad_fc1_1_2_weight,
            grad_fc2_weight,
            None,
            None,  # activation
        )


def _record_transfer_uses(transfers):
    if transfers is not None:
        gate, down = transfers
        try:
            gate.record_stream_use()
        finally:
            down.record_stream_use()


class BalancedEPMergedFc1GroupGemm(torch.autograd.Function):
    """Extend the standard merged EP row with temporary weights and ordered FP32 wgrad."""

    @staticmethod
    def forward(
        ctx, tokens, cumsum, gate, down, replica_gate, replica_down, plan, group, limit, activation, transfers
    ):
        from .....distributed.moe.ep_load_balance import _validate_replica_weight

        ctx.replica_plan, ctx.replica_group = plan, group
        ctx.original_weight_dtype = gate.dtype
        ctx.replica_transfers = transfers
        ctx.max_tokens_per_expert = int(plan.tokens_per_local_physical_expert.sum(dim=0).max())
        try:
            _validate_replica_weight(gate, replica_gate, plan, group)
            _validate_replica_weight(down, replica_down, plan, group)
            if transfers is None:
                return EPMergedFc1GroupGemm.forward(
                    ctx,
                    tokens,
                    cumsum,
                    torch.cat((gate, replica_gate)),
                    torch.cat((down, replica_down)),
                    limit,
                    activation,
                )
            return EPMergedFc1GroupGemm.forward(
                ctx,
                tokens,
                cumsum,
                gate,
                down,
                limit,
                activation,
                replica_fc1_weight=replica_gate,
                replica_fc2_weight=replica_down,
            )
        finally:
            # The owner stays with ctx across retained backward graphs. Also
            # record submitted work if a later kernel launch raises.
            _record_transfer_uses(transfers)

    @staticmethod
    def backward(ctx, grad_output):
        from .....distributed.moe.ep_grad import make_ordered_weight_grad
        from .....distributed.moe.ep_load_balance import _merge_replica_weight_gradients

        plan, group = ctx.replica_plan, ctx.replica_group
        need_weights = ctx.needs_input_grad[2:4]
        ordered = grad_output.is_cuda
        weight_grad = (
            make_ordered_weight_grad(plan, group, grad_output.device) if ordered and any(need_weights) else None
        )
        try:
            dx, _, dgate, ddown, _, _ = EPMergedFc1GroupGemm.backward(
                ctx,
                grad_output,
                weight_gradient_dtype=torch.float32,
                weight_needs_grad=need_weights,
                weight_grad_fn=weight_grad,
            )

            def finish(grad):
                if grad is None:
                    return None
                owned = (
                    grad[: plan.num_local_experts] if ordered else _merge_replica_weight_gradients(grad, plan, group)
                )
                return owned.to(ctx.original_weight_dtype)

            return dx, None, finish(dgate), finish(ddown), None, None, None, None, None, None, None
        finally:
            _record_transfer_uses(ctx.replica_transfers)
