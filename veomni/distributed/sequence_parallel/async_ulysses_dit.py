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

"""Thin facades for DiT async Ulysses. Math lives in ``veomni.kernels``."""

from typing import Optional

import torch
from torch import Tensor
from torch.distributed import ProcessGroup

from veomni.kernels import VeomniKernel
from veomni.kernels.compound import InnerHandle

from .ulysses import _all_to_all_single


def divide_qkv_linear_weight(weight: Tensor, dim: int):
    """Split a fused QKV weight along ``dim`` into Q, K, and V."""
    return weight.chunk(3, dim=dim)


def divide_qkv_linear_bias(bias: Tensor, dim: int):
    """Split a fused QKV bias along ``dim``, or return three ``None``s."""
    if bias is not None:
        return bias.chunk(3, dim=dim)
    return None, None, None


def async_ulysses_qkv_projection(
    hidden_states: Tensor = None,
    seq_dimension: int = None,
    head_dimension: int = None,
    q_weight: Tensor = None,
    q_bias: Optional[Tensor] = None,
    k_weight: Tensor = None,
    k_bias: Optional[Tensor] = None,
    v_weight: Tensor = None,
    v_bias: Optional[Tensor] = None,
    norm_type: str = None,
    norm_q_weight: Optional[Tensor] = None,
    norm_q_bias: Optional[Tensor] = None,
    norm_k_weight: Optional[Tensor] = None,
    norm_k_bias: Optional[Tensor] = None,
    normalized_shape: Optional[int] = None,
    eps: Optional[float] = None,
    unpadded_dim_size: int = None,
    head_dim: int = None,
    group: Optional[ProcessGroup] = None,
    rms_norm: InnerHandle = None,
):
    """Call the registered ``async_ulysses_qkv`` / ``dit`` wrapper."""
    return VeomniKernel("async_ulysses_qkv", "dit")(
        hidden_states,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        norm_q_weight,
        norm_q_bias,
        norm_k_weight,
        norm_k_bias,
        seq_dimension=seq_dimension,
        head_dimension=head_dimension,
        unpadded_dim_size=unpadded_dim_size,
        head_dim=head_dim,
        group=group,
        norm_type=norm_type,
        normalized_shape=normalized_shape,
        eps=eps,
        rms_norm=rms_norm,
    )


def async_ulysses_output_projection(
    hidden_states: Optional[Tensor] = None,
    seq_dimension: int = None,
    head_dimension: int = None,
    proj_weight: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    unpadded_dim_size: Optional[int] = None,
    group: Optional[ProcessGroup] = None,
):
    """Call the registered ``async_ulysses_o`` / ``dit`` wrapper."""
    return VeomniKernel("async_ulysses_o", "dit")(
        hidden_states,
        proj_weight,
        proj_bias,
        seq_dimension=seq_dimension,
        head_dimension=head_dimension,
        unpadded_dim_size=unpadded_dim_size,
        group=group,
    )


class _AsyncA2A(torch.autograd.Function):
    """Wait on an async all-to-all started by ``_all_to_all_single(async_op=True)``.

    ``x`` only anchors the gradient graph: backward performs the inverse
    exchange on the incoming gradient, matching the exchange that ``x`` went
    through, so gradients reach the pre-exchange tensor.
    """

    @staticmethod
    def forward(ctx, wait_fn, x, scatter_dim, gather_dim, group):
        ctx.group, ctx.scatter_dim, ctx.gather_dim = group, scatter_dim, gather_dim
        return wait_fn()

    @staticmethod
    def backward(ctx, grad_output):
        return (
            None,
            _all_to_all_single(grad_output, ctx.gather_dim, ctx.scatter_dim, ctx.group),
            None,
            None,
            None,
        )
