# Copyright (c) 2025, Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from typing import Optional

import torch

from .triton.convolution import (
    causal_conv1d_fwd_impl,
    causal_conv1d_bwd_impl,
)

# save_for_backward 不接受 None，用空 tensor 占位
_PLACEHOLDER = torch.empty(0)


class CausalConv1dFunction(torch.autograd.Function):
    """
    Differentiable wrapper around causal_conv1d_fwd_impl / causal_conv1d_bwd_impl.

    forward 签名:
        x:                [B, T, D]  或 varlen 下 [1, total_T, D]
        weight:           [W, D]
        bias:             [D] or None
        residual:         [B, T, D] or None
        initial_state:    [B, D, W] or None
        activation:       str or None  ("silu" / "swish" / None)
        cu_seqlens:       [N+1] int32 or None
        output_final_state: bool

    Returns:
        y:            same shape as x
        final_state:  [B, D, W] or None
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        activation: str = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
    ):
        y, final_state = causal_conv1d_fwd_impl(
            x=x,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            output_final_state=output_final_state,
        )

        # save_for_backward 不接受 None，用 _PLACEHOLDER 占位
        ctx.save_for_backward(
            x, weight,
            bias if bias is not None else _PLACEHOLDER,
            residual if residual is not None else _PLACEHOLDER,
            initial_state if initial_state is not None else _PLACEHOLDER,
            cu_seqlens if cu_seqlens is not None else _PLACEHOLDER,
        )
        ctx.has_bias = bias is not None
        ctx.has_residual = residual is not None
        ctx.has_initial_state = initial_state is not None
        ctx.has_cu_seqlens = cu_seqlens is not None
        ctx.activation = activation

        return y, final_state

    @staticmethod
    def backward(ctx, dy: torch.Tensor, d_final_state: Optional[torch.Tensor] = None):
        x, weight, bias, residual, initial_state, cu_seqlens = ctx.saved_tensors

        # 还原 None
        bias = bias if ctx.has_bias else None
        residual = residual if ctx.has_residual else None
        initial_state = initial_state if ctx.has_initial_state else None
        cu_seqlens = cu_seqlens if ctx.has_cu_seqlens else None

        # bwd_impl 有 @input_guard(make_contiguous=True)，无需手动处理
        dx, dw, db, dr, dh0 = causal_conv1d_bwd_impl(
            x=x,
            dy=dy,
            dht=d_final_state,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            activation=ctx.activation,
            cu_seqlens=cu_seqlens,
        )

        # 返回顺序严格对应 forward 的参数:
        # x, weight, bias, residual, initial_state, activation, cu_seqlens, output_final_state
        return dx, dw, db, dr, dh0, None, None, None


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    activation: str = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
):
    """Functional API，直接调用即可自动支持 autograd。"""
    return CausalConv1dFunction.apply(
        x, weight, bias, residual, initial_state,
        activation, cu_seqlens, output_final_state,
    )

