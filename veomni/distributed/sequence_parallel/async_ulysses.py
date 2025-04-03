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

import importlib
import numbers
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup

from .comm import get_ulysses_sequence_parallel_group
from .ulysses import all_to_all_tensor
from .utils import padding_tensor_for_seqeunce_parallel, unpadding_tensor_for_seqeunce_parallel


fused_layer_norm_cuda = None


def divide_qkv_linear_weight(weight: Tensor, dim: int):
    return weight.chunk(3, dim=dim)


def divide_qkv_linear_bias(bias: Tensor, dim: int):
    if bias is not None:
        return bias.chunk(3, dim=dim)
    else:
        return None, None, None


class AsyncUlyssesQKVProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: Tensor,
        seq_dimension: int,
        head_dimension: int,
        q_weight: Tensor,
        q_bias: Tensor,
        k_weight: Tensor,
        k_bias: Tensor,
        v_weight: Tensor,
        v_bias: Tensor,
        norm_type: str,
        norm_q_weight: Tensor,
        norm_q_bias: Tensor,
        norm_k_weight: Tensor,
        norm_k_bias: Tensor,
        normalized_shape: int,
        eps: float,
        unpadded_dim_size: int,
        head_dim: int,
        group: ProcessGroup,
    ):
        sp_group = get_ulysses_sequence_parallel_group() if group is None else group

        # q projection
        q = F.linear(hidden_states, q_weight, q_bias)

        # q communication launch
        q_res = all_to_all_tensor(
            q, scatter_dim=head_dimension, gather_dim=seq_dimension, group=sp_group, async_op=True
        )

        # k projection
        k = F.linear(hidden_states, k_weight, k_bias)

        # k communication launch
        k_res = all_to_all_tensor(
            k, scatter_dim=head_dimension, gather_dim=seq_dimension, group=sp_group, async_op=True
        )

        # v projection
        v = F.linear(hidden_states, v_weight, v_bias)

        # v communication launch
        v_res = all_to_all_tensor(
            v, scatter_dim=head_dimension, gather_dim=seq_dimension, group=sp_group, async_op=True
        )

        # q communication collect
        q = q_res()
        q = unpadding_tensor_for_seqeunce_parallel(q, seq_dimension, unpadded_dim_size)
        q = q.reshape(list(q.shape[:-1]) + [-1, head_dim]).contiguous()

        # k communication collect
        k = k_res()
        k = unpadding_tensor_for_seqeunce_parallel(k, seq_dimension, unpadded_dim_size)
        k = k.reshape(list(k.shape[:-1]) + [-1, head_dim]).contiguous()

        # qk normalization (if needed)
        if norm_type is not None:
            if isinstance(normalized_shape, numbers.Integral):
                normalized_shape = (normalized_shape,)
            normalized_shape = torch.Size(normalized_shape)
            global fused_layer_norm_cuda
            if fused_layer_norm_cuda is None:
                fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
            norm_q_weight = norm_q_weight.contiguous()
            norm_k_weight = norm_k_weight.contiguous()
            output_q, mean_q, invvar_q = None, None, None
            output_k, mean_k, invvar_k = None, None, None
            if norm_type == "rmsnorm":
                output_q, invvar_q = fused_layer_norm_cuda.rms_forward_affine(q, normalized_shape, norm_q_weight, eps)
                output_k, invvar_k = fused_layer_norm_cuda.rms_forward_affine(k, normalized_shape, norm_k_weight, eps)
            elif norm_type == "layernorm":
                output_q, mean_q, invvar_q = fused_layer_norm_cuda.forward_affine(
                    q, normalized_shape, norm_q_weight, norm_q_bias, eps
                )
                output_k, mean_k, invvar_k = fused_layer_norm_cuda.forward_affine(
                    k, normalized_shape, norm_k_weight, norm_k_bias, eps
                )
            else:
                raise NotImplementedError(f"{norm_type} is not supported in async-ulysses now!")
        else:
            output_q = q
            output_k = k
            mean_q = None
            mean_k = None
            invvar_q = None
            invvar_k = None

        # v communication collect
        v = v_res()
        v = unpadding_tensor_for_seqeunce_parallel(v, seq_dimension, unpadded_dim_size)
        v = v.reshape(list(v.shape[:-1]) + [-1, head_dim]).contiguous()

        # save ctx for backward
        ctx.sp_group = sp_group
        ctx.head_dimension = head_dimension
        ctx.seq_dimension = seq_dimension
        ctx.norm_type = norm_type
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        ctx.save_for_backward(
            hidden_states,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            q,
            norm_q_weight,
            norm_q_bias,
            mean_q,
            invvar_q,
            k,
            norm_k_weight,
            norm_k_bias,
            mean_k,
            invvar_k,
        )

        return output_q, output_k, v

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor):
        # get ctx for backward
        sp_group = ctx.sp_group
        seq_dimension = ctx.seq_dimension
        head_dimension = ctx.head_dimension
        norm_type = ctx.norm_type
        normalized_shape = ctx.normalized_shape
        eps = ctx.eps
        (
            hidden_states,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            q,
            norm_q_weight,
            norm_q_bias,
            mean_q,
            invvar_q,
            k,
            norm_k_weight,
            norm_k_bias,
            mean_k,
            invvar_k,
        ) = ctx.saved_tensors

        # initialize grads
        grad_hidden_states = None
        grad_q_weight = None
        grad_q_bias = None
        grad_k_weight = None
        grad_k_bias = None
        grad_v_weight = None
        grad_v_bias = None
        grad_norm_q_weight = None
        grad_norm_q_bias = None
        grad_norm_k_weight = None
        grad_norm_k_bias = None

        # v grad communication launch
        grad_v = grad_output[2].contiguous()
        grad_v = grad_v.reshape(list(grad_v.shape[:-2]) + [-1]).contiguous()
        grad_v = padding_tensor_for_seqeunce_parallel(grad_v, dim=seq_dimension)
        grad_v_res = all_to_all_tensor(
            grad_v,
            scatter_dim=seq_dimension,
            gather_dim=head_dimension,
            group=sp_group,
            async_op=True,
        )

        # qk normalization backward (if needed)
        if norm_type is not None:
            if norm_type == "rmsnorm":
                grad_k, grad_norm_k_weight = fused_layer_norm_cuda.rms_backward_affine(
                    grad_output[1].contiguous(),
                    invvar_k,
                    k,
                    normalized_shape,
                    norm_k_weight,
                    eps,
                    False,
                )
                grad_q, grad_norm_q_weight = fused_layer_norm_cuda.rms_backward_affine(
                    grad_output[0].contiguous(),
                    invvar_q,
                    q,
                    normalized_shape,
                    norm_q_weight,
                    eps,
                    False,
                )
            elif norm_type == "layernorm":
                grad_k, grad_norm_k_weight, grad_norm_k_bias = fused_layer_norm_cuda.backward_affine(
                    grad_output[1].contiguous(),
                    mean_k,
                    invvar_k,
                    k,
                    normalized_shape,
                    norm_k_weight,
                    norm_k_bias,
                    eps,
                    False,
                )
                grad_q, grad_norm_q_weight, grad_norm_q_bias = fused_layer_norm_cuda.backward_affine(
                    grad_output[0].contiguous(),
                    mean_q,
                    invvar_q,
                    q,
                    normalized_shape,
                    norm_q_weight,
                    norm_q_bias,
                    eps,
                    False,
                )
            else:
                raise NotImplementedError(f"{norm_type} is not supported in async-ulysses now!")
        else:
            grad_k = grad_output[1].contiguous()
            grad_q = grad_output[0].contiguous()
            grad_norm_k_weight = None
            grad_norm_q_weight = None

        # v grad communication collect
        grad_v = grad_v_res()

        # k grad communication launch
        grad_k = grad_k.reshape(list(grad_k.shape[:-2]) + [-1]).contiguous()
        grad_k = padding_tensor_for_seqeunce_parallel(grad_k, dim=seq_dimension)
        grad_k_res = all_to_all_tensor(
            grad_k,
            scatter_dim=seq_dimension,
            gather_dim=head_dimension,
            group=sp_group,
            async_op=True,
        )

        # v projection grad
        grad_v_input = grad_v @ v_weight
        grad_v_weight = grad_v.transpose(-1, -2) @ hidden_states
        if v_bias is not None and ctx.needs_input_grad[7]:
            grad_v_bias = grad_v.sum(0)

        # k grad communication collect
        grad_k = grad_k_res()

        # q grad communication launch
        grad_q = grad_q.reshape(list(grad_q.shape[:-2]) + [-1]).contiguous()
        grad_q = padding_tensor_for_seqeunce_parallel(grad_q, dim=seq_dimension)
        grad_q_res = all_to_all_tensor(
            grad_q,
            scatter_dim=seq_dimension,
            gather_dim=head_dimension,
            group=sp_group,
            async_op=True,
        )

        # k projection grad
        grad_k_input = grad_k @ k_weight
        grad_k_weight = grad_k.transpose(-1, -2) @ hidden_states
        if k_bias is not None and ctx.needs_input_grad[5]:
            grad_k_bias = grad_k.sum(0)

        # q grad communication collect
        grad_q = grad_q_res()

        # q projection grad
        grad_q_input = grad_q @ q_weight
        grad_q_weight = grad_q.transpose(-1, -2) @ hidden_states
        if q_bias is not None and ctx.needs_input_grad[3]:
            grad_q_bias = grad_q.sum(0)

        # grad
        grad_hidden_states = grad_q_input + grad_k_input + grad_v_input

        return (
            grad_hidden_states,
            None,
            None,
            grad_q_weight,
            grad_q_bias,
            grad_k_weight,
            grad_k_bias,
            grad_v_weight,
            grad_v_bias,
            None,
            grad_norm_q_weight,
            grad_norm_q_bias,
            grad_norm_k_weight,
            grad_norm_k_bias,
            None,
            None,
            None,
            None,
            None,
        )


class AsyncUlyssesOutputProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: Tensor,
        seq_dimension: int,
        head_dimension: int,
        proj_weight: Tensor,
        proj_bias: Tensor,
        unpadded_dim_size: int,
        group: ProcessGroup,
    ):
        sp_group = get_ulysses_sequence_parallel_group() if group is None else group

        # out projection
        hidden_states = padding_tensor_for_seqeunce_parallel(hidden_states, seq_dimension)
        hidden_states = all_to_all_tensor(
            hidden_states, scatter_dim=seq_dimension, gather_dim=head_dimension, group=sp_group
        )
        o = F.linear(hidden_states, proj_weight, proj_bias)

        # save ctx for backward
        ctx.sp_group = sp_group
        ctx.head_dimension = head_dimension
        ctx.seq_dimension = seq_dimension
        ctx.unpadded_dim_size = unpadded_dim_size

        ctx.save_for_backward(
            hidden_states,
            proj_weight,
            proj_bias,
        )

        return o

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor):
        # get ctx for backward
        sp_group = ctx.sp_group
        head_dimension = ctx.head_dimension
        seq_dimension = ctx.seq_dimension
        unpadded_dim_size = ctx.unpadded_dim_size
        (
            hidden_states,
            proj_weight,
            proj_bias,
        ) = ctx.saved_tensors

        # initialize grads
        grad_o = None
        grad_proj_weight = None
        grad_proj_bias = None

        # output grad
        grad_o = grad_output[0] @ (proj_weight)

        # output grad communication launch
        grad_out_res = all_to_all_tensor(
            grad_o, scatter_dim=head_dimension, gather_dim=seq_dimension, group=sp_group, async_op=True
        )

        grad_proj_weight = grad_output[0].transpose(-1, -2) @ (hidden_states)
        if proj_bias is not None and ctx.needs_input_grad[3]:
            grad_proj_bias = grad_output[0].sum(0)

        # output grad communication collect
        grad_o = grad_out_res()
        grad_o = unpadding_tensor_for_seqeunce_parallel(grad_o, seq_dimension, unpadded_dim_size)

        return (
            grad_o,
            None,
            None,
            grad_proj_weight,
            grad_proj_bias,
            None,
            None,
        )


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
):
    return AsyncUlyssesQKVProjection.apply(
        hidden_states,
        seq_dimension,
        head_dimension,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        norm_type,
        norm_q_weight,
        norm_q_bias,
        norm_k_weight,
        norm_k_bias,
        normalized_shape,
        eps,
        unpadded_dim_size,
        head_dim,
        group,
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
    return AsyncUlyssesOutputProjection.apply(
        hidden_states,
        seq_dimension,
        head_dimension,
        proj_weight,
        proj_bias,
        unpadded_dim_size,
        group,
    )
