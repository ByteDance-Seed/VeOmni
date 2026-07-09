# Copyright (c) 2026, Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from typing import Optional

import torch

from .triton.convolution import (
    causal_conv1d_fwd_impl,
    causal_conv1d_bwd_impl,
)

from .triton.utils import is_arch35

__all__ = ["CausalConv1dFunction", "causal_conv1d"]


# Placeholder used in ctx.save_for_backward since it does not accept None
_PLACEHOLDER = torch.empty(0)


class CausalConv1dFunction(torch.autograd.Function):
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
        if is_arch35():
            raise NotImplementedError("causal_conv1d is not supported on arch35")

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

        # save_for_backward does not accept None — use _PLACEHOLDER instead
        ctx.save_for_backward(
            x,
            weight,
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
        if is_arch35():
            raise NotImplementedError("causal_conv1d is not supported on arch35")

        x, weight, bias, residual, initial_state, cu_seqlens = ctx.saved_tensors

        # Restore None placeholders
        bias = bias if ctx.has_bias else None
        residual = residual if ctx.has_residual else None
        initial_state = initial_state if ctx.has_initial_state else None
        cu_seqlens = cu_seqlens if ctx.has_cu_seqlens else None

        # bwd_impl has @input_guard(make_contiguous=True); no manual handling needed
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

        # Return order must match forward args:
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
    return CausalConv1dFunction.apply(
        x,
        weight,
        bias,
        residual,
        initial_state,
        activation,
        cu_seqlens,
        output_final_state,
    )
