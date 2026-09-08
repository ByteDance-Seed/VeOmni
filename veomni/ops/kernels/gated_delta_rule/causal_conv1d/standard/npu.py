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

"""standard causal_conv1d NPU vendored-Triton pair."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .....registry import SavedState
from ...optional import optional_tensor, unused_like


@dataclass(frozen=True)
class _Meta:
    """Activation, which optionals were real, and how many apply inputs to grad."""

    activation: str | None
    has_bias: bool
    has_cu_seqlens: bool
    n_in: int


def forward(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    *,
    activation: str | None = "silu",
    seq_idx: Tensor | None = None,
    backend: str | None = None,
) -> tuple[Tensor, SavedState]:
    """NPU causal conv1d. *weight* is FLA ``[D, W]``; the kernel wants ``[W, D]``."""
    from ...vendor.triton.convolution import causal_conv1d_fwd_impl
    from ...vendor.triton.utils import is_arch35

    del seq_idx, backend
    n_in = 2 + (bias is not None) + (cu_seqlens is not None)
    if bias is None:
        bias = unused_like(weight)
    if cu_seqlens is None:
        cu_seqlens = unused_like(x, dtype=torch.int32)

    if is_arch35():
        raise NotImplementedError("causal_conv1d is not supported on arch35")

    weight_wd = weight.transpose(0, 1).contiguous()
    bias_opt = optional_tensor(bias)
    cu_opt = optional_tensor(cu_seqlens)
    output, _final_state = causal_conv1d_fwd_impl(
        x=x,
        weight=weight_wd,
        bias=bias_opt,
        residual=None,
        initial_state=None,
        activation=activation,
        cu_seqlens=cu_opt,
        output_final_state=False,
    )
    return output, SavedState(
        (x, weight_wd, bias, cu_seqlens),
        _Meta(activation, bias_opt is not None, cu_opt is not None, n_in),
    )


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor | None, ...]:
    """Return ``(grad_x, grad_weight, grad_bias, grad_cu_seqlens)``."""
    from ...vendor.triton.convolution import causal_conv1d_bwd_impl
    from ...vendor.triton.utils import is_arch35

    if is_arch35():
        raise NotImplementedError("causal_conv1d is not supported on arch35")

    meta = saved.metadata
    assert isinstance(meta, _Meta)
    x, weight_wd, bias, cu_seqlens = saved.tensors
    bias_opt = bias if meta.has_bias else None
    cu_opt = cu_seqlens if meta.has_cu_seqlens else None
    grad_x, grad_weight_wd, grad_bias, _grad_residual, _grad_h0 = causal_conv1d_bwd_impl(
        x=x,
        dy=grad_output,
        dht=None,
        weight=weight_wd,
        bias=bias_opt,
        residual=None,
        initial_state=None,
        activation=meta.activation,
        cu_seqlens=cu_opt,
    )
    grad_weight = None if grad_weight_wd is None else grad_weight_wd.transpose(0, 1)
    if not meta.has_bias:
        grad_bias = None
    return (grad_x, grad_weight, grad_bias, None)[: meta.n_in]
