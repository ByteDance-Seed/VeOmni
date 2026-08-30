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

"""DiT async Ulysses QKV: project, optional QK norm, then A2A."""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import Tensor

from ......distributed.sequence_parallel.comm import get_ulysses_sequence_parallel_group
from ......distributed.sequence_parallel.ulysses import all_to_all_tensor
from ......distributed.sequence_parallel.utils import (
    padding_tensor_for_seqeunce_parallel,
    unpadding_tensor_for_seqeunce_parallel,
)
from .....compound import InnerHandle, append_inner, resolve_inner_kernel, take_inner
from .....registry import SavedState
from ...backward import layer_norm_backward, linear_backward
from ...norm import layernorm_forward, normalize_shape
from ...qkv_state import QKVMeta, qkv_grads, unpack_qkv


def forward(
    hidden_states: Tensor,
    q_weight: Tensor,
    q_bias: Tensor | None,
    k_weight: Tensor,
    k_bias: Tensor | None,
    v_weight: Tensor,
    v_bias: Tensor | None,
    norm_q_weight: Tensor | None,
    norm_q_bias: Tensor | None,
    norm_k_weight: Tensor | None,
    norm_k_bias: Tensor | None,
    *,
    seq_dimension: int,
    head_dimension: int,
    unpadded_dim_size: int,
    head_dim: int | None = None,
    group: Any = None,
    norm_type: str | None = None,
    normalized_shape: int | tuple[int, ...] | None = None,
    eps: float | None = None,
    rms_norm: InnerHandle = None,
) -> tuple[tuple[Tensor, Tensor, Tensor], SavedState]:
    """Project QKV, optional QK norm, then all-to-all.

    Unlike the dense path, there is no head view and no KV repeat. Norm runs
    on the pre-collect projection so each all-to-all can cover the next one.
    """
    sp_group = get_ulysses_sequence_parallel_group() if group is None else group
    shape = normalize_shape(normalized_shape)
    rms = None
    saved_rms_q = None
    saved_rms_k = None
    mean_q = mean_k = invvar_q = invvar_k = None

    q = F.linear(hidden_states, q_weight, q_bias)
    if norm_type == "rmsnorm":
        rms = resolve_inner_kernel(rms_norm, kernel="rms_norm", variant="standard")
        output_q, saved_rms_q = rms.forward(q, norm_q_weight, eps=eps)
    elif norm_type == "layernorm":
        output_q, mean_q, invvar_q = layernorm_forward(q, norm_q_weight, norm_q_bias, shape, eps)
    elif norm_type is None:
        output_q = q
    else:
        raise NotImplementedError(f"{norm_type} is not supported in async-ulysses now!")
    output_q_res = all_to_all_tensor(
        output_q, scatter_dim=head_dimension, gather_dim=seq_dimension, group=sp_group, async_op=True
    )

    k = F.linear(hidden_states, k_weight, k_bias)
    if norm_type == "rmsnorm":
        output_k, saved_rms_k = rms.forward(k, norm_k_weight, eps=eps)
    elif norm_type == "layernorm":
        output_k, mean_k, invvar_k = layernorm_forward(k, norm_k_weight, norm_k_bias, shape, eps)
    elif norm_type is None:
        output_k = k
    else:
        raise NotImplementedError(f"{norm_type} is not supported in async-ulysses now!")
    output_k_res = all_to_all_tensor(
        output_k, scatter_dim=head_dimension, gather_dim=seq_dimension, group=sp_group, async_op=True
    )

    v = F.linear(hidden_states, v_weight, v_bias)
    v_res = all_to_all_tensor(v, scatter_dim=head_dimension, gather_dim=seq_dimension, group=sp_group, async_op=True)

    # Wait after all three launches. Q and K already finished their pre-A2A norms.
    output_q = unpadding_tensor_for_seqeunce_parallel(output_q_res(), seq_dimension, unpadded_dim_size)
    output_k = unpadding_tensor_for_seqeunce_parallel(output_k_res(), seq_dimension, unpadded_dim_size)
    v = unpadding_tensor_for_seqeunce_parallel(v_res(), seq_dimension, unpadded_dim_size)

    saved: list[Tensor | None] = [
        hidden_states,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
    ]
    meta = QKVMeta(
        seq_dimension=seq_dimension,
        head_dimension=head_dimension,
        unpadded_dim_size=unpadded_dim_size,
        head_dim=head_dim,
        group=sp_group,
        norm_type=norm_type,
        normalized_shape=shape,
        eps=eps,
        rms=rms,
    )
    if norm_type == "layernorm":
        saved.extend(
            [q, norm_q_weight, norm_q_bias, mean_q, invvar_q, k, norm_k_weight, norm_k_bias, mean_k, invvar_k]
        )
    elif norm_type == "rmsnorm":
        # Flatten nested RMS SavedState into the outer tensor list.
        meta.rms_q = append_inner(saved, saved_rms_q)
        meta.rms_k = append_inner(saved, saved_rms_k)
    return (output_q, output_k, v), SavedState(tuple(saved), meta)


def backward(grad_output: tuple[Tensor, Tensor, Tensor], saved: SavedState) -> tuple[Tensor | None, ...]:
    """Reverse all-to-all first, then the QK norm, then the linear grads."""
    meta, hidden_states, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, rest = unpack_qkv(saved)
    seq_dimension = meta.seq_dimension
    head_dimension = meta.head_dimension
    sp_group = meta.group

    if meta.norm_type == "layernorm":
        (
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
        ) = rest
    elif meta.norm_type == "rmsnorm":
        saved_rms_q, rest = take_inner(rest, meta.rms_q)
        saved_rms_k, rest = take_inner(rest, meta.rms_k)

    # Reverse all-to-all before the QK-norm backward. Norm ran pre-collect.
    grad_v = padding_tensor_for_seqeunce_parallel(grad_output[2].contiguous(), dim=seq_dimension)
    grad_v_res = all_to_all_tensor(
        grad_v,
        scatter_dim=seq_dimension,
        gather_dim=head_dimension,
        group=sp_group,
        async_op=True,
    )
    grad_v = grad_v_res()

    grad_k = padding_tensor_for_seqeunce_parallel(grad_output[1].contiguous(), dim=seq_dimension)
    grad_k_res = all_to_all_tensor(
        grad_k,
        scatter_dim=seq_dimension,
        gather_dim=head_dimension,
        group=sp_group,
        async_op=True,
    )

    grad_v_input, grad_v_weight, grad_v_bias = linear_backward(
        grad_v,
        hidden_states,
        v_weight,
        has_bias=v_bias is not None,
    )
    grad_k = grad_k_res()

    grad_norm_q_weight = grad_norm_q_bias = grad_norm_k_weight = grad_norm_k_bias = None
    if meta.norm_type == "rmsnorm":
        grad_k, grad_norm_k_weight = meta.rms.backward(grad_k, saved_rms_k)
    elif meta.norm_type == "layernorm":
        grad_k, grad_norm_k_weight, grad_norm_k_bias = layer_norm_backward(
            grad_k, k, mean_k, invvar_k, norm_k_weight, norm_k_bias, meta.normalized_shape, meta.eps
        )
    elif meta.norm_type is not None:
        raise NotImplementedError(f"{meta.norm_type} is not supported in async-ulysses now!")

    grad_q = padding_tensor_for_seqeunce_parallel(grad_output[0].contiguous(), dim=seq_dimension)
    grad_q_res = all_to_all_tensor(
        grad_q,
        scatter_dim=seq_dimension,
        gather_dim=head_dimension,
        group=sp_group,
        async_op=True,
    )

    grad_k_input, grad_k_weight, grad_k_bias = linear_backward(
        grad_k,
        hidden_states,
        k_weight,
        has_bias=k_bias is not None,
    )
    grad_q = grad_q_res()
    if meta.norm_type == "rmsnorm":
        grad_q, grad_norm_q_weight = meta.rms.backward(grad_q, saved_rms_q)
    elif meta.norm_type == "layernorm":
        grad_q, grad_norm_q_weight, grad_norm_q_bias = layer_norm_backward(
            grad_q, q, mean_q, invvar_q, norm_q_weight, norm_q_bias, meta.normalized_shape, meta.eps
        )

    grad_q_input, grad_q_weight, grad_q_bias = linear_backward(
        grad_q,
        hidden_states,
        q_weight,
        has_bias=q_bias is not None,
    )
    return qkv_grads(
        grad_q_input + grad_k_input + grad_v_input,
        grad_q_weight,
        grad_q_bias,
        grad_k_weight,
        grad_k_bias,
        grad_v_weight,
        grad_v_bias,
        grad_norm_q_weight,
        grad_norm_q_bias,
        grad_norm_k_weight,
        grad_norm_k_bias,
    )
