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

from .swiglu import apply_swiglu_clamp


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
    ):
        """Split fc1 GEMMs, SwiGLU, and fc2. No routing scale."""
        from .group_gemm import group_gemm_same_nk

        fc1_1_output = group_gemm_same_nk(
            a=permute_tokens,
            b=fc1_1_weight,
            cumsum_M=cumsum,
            max_M=permute_tokens.shape[0],
            transpose_a=False,
            transpose_b=True,
        )
        fc1_2_output = group_gemm_same_nk(
            a=permute_tokens,
            b=fc1_2_weight,
            cumsum_M=cumsum,
            max_M=permute_tokens.shape[0],
            transpose_a=False,
            transpose_b=True,
        )
        fc1_1_output, fc1_2_output, mask_fc1_1, mask_fc1_2 = apply_swiglu_clamp(
            fc1_1_output, fc1_2_output, swiglu_limit
        )
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_output = fc1_1_activation * fc1_2_output
        fc2_output = group_gemm_same_nk(
            a=fc1_output,
            b=fc2_weight,
            cumsum_M=cumsum,
            max_M=permute_tokens.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

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
        ) = ctx.saved_tensors
        swiglu_limit = ctx.swiglu_limit

        grad_fc1_output = group_gemm_same_nk(
            a=grad_output,
            b=fc2_weight,
            cumsum_M=cumsum,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
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

        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)
        if swiglu_limit is not None:
            grad_fc1_1_output.masked_fill_(~mask_fc1_1, 0)

        grad_scatter_output_1 = group_gemm_same_nk(
            a=grad_fc1_1_output,
            b=fc1_1_weight,
            cumsum_M=cumsum,
            max_M=grad_output.shape[0],
            transpose_b=False,
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
    ):
        """Merged fc1 GEMM, SwiGLU, and fc2. No routing scale."""
        from .group_gemm import group_gemm_same_nk

        assert fc1_1_2_weight.shape[1] % 2 == 0, (
            f"Merged fc1_1_2_weight dim 1 must be even, got {fc1_1_2_weight.shape[1]}"
        )
        fc1_output = group_gemm_same_nk(
            a=permute_tokens,
            b=fc1_1_2_weight,
            cumsum_M=cumsum,
            max_M=permute_tokens.shape[0],
            transpose_a=False,
            transpose_b=True,
        )
        fc1_1_output, fc1_2_output = fc1_output.chunk(2, dim=-1)
        fc1_1_output, fc1_2_output, mask_fc1_1, mask_fc1_2 = apply_swiglu_clamp(
            fc1_1_output, fc1_2_output, swiglu_limit
        )
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_result = fc1_1_activation * fc1_2_output
        fc2_output = group_gemm_same_nk(
            a=fc1_result,
            b=fc2_weight,
            cumsum_M=cumsum,
            max_M=permute_tokens.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

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
        ) = ctx.saved_tensors
        swiglu_limit = ctx.swiglu_limit

        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_result = fc1_1_activation * fc1_2_output
        grad_fc1_result = group_gemm_same_nk(
            a=grad_output,
            b=fc2_weight,
            cumsum_M=cumsum,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )

        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = torch.empty_like(fc2_weight)
            group_gemm_same_mn(
                a=grad_output,
                b=fc1_result,
                c=grad_fc2_weight,
                cumsum_K=cumsum,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        grad_fc1_2_output = fc1_1_activation * grad_fc1_result
        grad_fc1_1_activation = grad_fc1_result * fc1_2_output
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)
        if swiglu_limit is not None:
            grad_fc1_1_output.masked_fill_(~mask_fc1_1, 0)
            grad_fc1_2_output.masked_fill_(~mask_fc1_2, 0)

        grad_fc1_output = torch.cat([grad_fc1_1_output, grad_fc1_2_output], dim=-1)
        grad_permute_tokens = group_gemm_same_nk(
            a=grad_fc1_output,
            b=fc1_1_2_weight,
            cumsum_M=cumsum,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )

        grad_fc1_1_2_weight = None
        if fc1_1_2_weight.requires_grad:
            grad_fc1_1_2_weight = torch.empty_like(fc1_1_2_weight)
            group_gemm_same_mn(
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
        )
