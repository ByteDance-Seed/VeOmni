# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing limitations
# under the License.

"""CUDA FA4 autograd wrapper and MagiAttention entry point."""

from contextlib import nullcontext

import torch

from ._kernel import CUDA_DEVICE_TYPE, KERNEL_CUTLASS, prepare_kernel, validate_cutlass_inputs
from ._metadata import get_or_prepare_attn_arg


def cuda_device_context(device: torch.device):
    """Enter a CUDA device context, or a no-op when the tensor is not on CUDA."""
    if device.type == CUDA_DEVICE_TYPE:
        return torch.cuda.device(device)
    return nullcontext()


class _MagiFA4Function(torch.autograd.Function):
    """Run FA4 autograd with an explicit prepared FA4AttnArg."""

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_ranges: torch.Tensor,
        k_ranges: torch.Tensor,
        attn_type_map: torch.Tensor | None,
        softmax_scale: float | None,
        softcap: float,
        attn_arg: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        softmax_scale = query.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        with cuda_device_context(query.device):
            from magi_attention.functional.fa4 import fa4_fwd

            output, lse = fa4_fwd(
                q=query,
                k=key,
                v=value,
                sink=None,
                attn_arg=attn_arg,
                softmax_scale=softmax_scale,
                softcap=softcap,
            )

        ctx.save_for_backward(query, key, value, output, lse, q_ranges, k_ranges, attn_type_map)
        ctx.softmax_scale = softmax_scale
        ctx.softcap = softcap
        ctx.attn_arg = attn_arg
        ctx.mark_non_differentiable(lse)
        return output, lse

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, *args: object) -> tuple[torch.Tensor | None, ...]:
        query, key, value, output, lse, _, _, _ = ctx.saved_tensors
        with cuda_device_context(query.device):
            from magi_attention.functional.fa4 import fa4_bwd

            grad_query, grad_key, grad_value, _ = fa4_bwd(
                do=grad_output,
                q=query,
                k=key,
                v=value,
                sink=None,
                o=output,
                lse=lse,
                attn_arg=ctx.attn_arg,
                softmax_scale=ctx.softmax_scale,
                softcap=ctx.softcap,
                deterministic=False,
            )

        return grad_query, grad_key, grad_value, None, None, None, None, None, None


def _fa4_cuda_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor | None,
    *,
    softmax_scale: float | None,
    softcap: float,
):
    """Run the SM-specific CUDA FA4 backend with prepared mask metadata."""
    kernel_mode, build_flags = prepare_kernel(query.device)
    metadata_head_dim = None
    if kernel_mode == KERNEL_CUTLASS:
        if build_flags is None:
            raise RuntimeError("MagiAttention's SM90 CUTLASS backend did not provide build configuration.")
        # CUTLASS arbitrary-mask tiles follow the compiled bucket rather than
        # the smaller runtime head dimension served by that bucket.
        metadata_head_dim = validate_cutlass_inputs(query, value, softcap, build_flags)

    try:
        from magi_attention.api import AttnForwardMeta
    except ImportError as error:
        raise ImportError(
            "VeOmni `magi_attention` requires the optional `magi-attention` package. "
            "Install VeOmni with the `gpu` extra."
        ) from error

    attn_arg = get_or_prepare_attn_arg(
        query,
        key,
        q_ranges,
        k_ranges,
        attn_type_map,
        metadata_head_dim,
    )
    output, lse = _MagiFA4Function.apply(
        query,
        key,
        value,
        q_ranges,
        k_ranges,
        attn_type_map,
        softmax_scale,
        softcap,
        attn_arg,
    )
    return output, AttnForwardMeta(lse=lse, max_logits=None)
