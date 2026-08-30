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

"""Dense async Ulysses QKV: project, A2A, then optional QK norm."""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import Tensor

from ......distributed.sequence_parallel.comm import (
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
)
from ......distributed.sequence_parallel.ulysses import all_to_all_tensor
from ......distributed.sequence_parallel.utils import (
    padding_tensor_for_seqeunce_parallel,
    unpadding_tensor_for_seqeunce_parallel,
)
from .....compound import InnerHandle, append_inner, resolve_inner_kernel, take_inner
from .....registry import SavedState
from ...backward import layer_norm_backward, linear_backward, reduce_repeated_kv_gradient
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
    head_dim: int,
    group: Any = None,
    norm_type: str | None = None,
    normalized_shape: int | tuple[int, ...] | None = None,
    eps: float | None = None,
    rms_norm: InnerHandle = None,
) -> tuple[tuple[Tensor, Tensor, Tensor], SavedState]:
    """Project QKV, all-to-all heads-to-seq, then optional QK norm.

    Each projection launches an async all-to-all before the next linear starts.
    Q and K wait before the post-collect norm. V stays in flight so that
    collective covers the QK norm.
    """
    sp_group = get_ulysses_sequence_parallel_group() if group is None else group
    ulysses_size = get_ulysses_sequence_parallel_world_size()
    num_q_heads = q_weight.shape[0] // head_dim
    num_kv_heads = k_weight.shape[0] // head_dim
    batch_size = hidden_states.shape[0]

    if num_q_heads % ulysses_size != 0:
        raise ValueError(f"num_query_heads ({num_q_heads}) must be divisible by ulysses_size ({ulysses_size})")

    need_repeat_kv = False
    n_repeat = 1
    original_num_kv_heads = num_kv_heads
    if ulysses_size > num_kv_heads:
        if ulysses_size % num_kv_heads != 0:
            raise ValueError(
                f"ulysses_size ({ulysses_size}) must be divisible by num_key_value_heads ({num_kv_heads})"
            )
        need_repeat_kv = True
        n_repeat = ulysses_size // num_kv_heads

    # Launch Q/K/V all-to-all as soon as each projection is ready.
    q = F.linear(hidden_states, q_weight, q_bias)
    q = q.view(batch_size, -1, num_q_heads, head_dim)
    q_res = all_to_all_tensor(q, scatter_dim=head_dimension, gather_dim=seq_dimension, group=sp_group, async_op=True)

    k = F.linear(hidden_states, k_weight, k_bias)
    k = k.view(batch_size, -1, num_kv_heads, head_dim)
    if need_repeat_kv:
        k = k.repeat_interleave(n_repeat, dim=2)
    k_res = all_to_all_tensor(k, scatter_dim=head_dimension, gather_dim=seq_dimension, group=sp_group, async_op=True)

    v = F.linear(hidden_states, v_weight, v_bias)
    v = v.view(batch_size, -1, num_kv_heads, head_dim)
    if need_repeat_kv:
        v = v.repeat_interleave(n_repeat, dim=2)
    v_res = all_to_all_tensor(v, scatter_dim=head_dimension, gather_dim=seq_dimension, group=sp_group, async_op=True)

    # Wait Q/K for the post-collect norm. Leave V in flight.
    q = unpadding_tensor_for_seqeunce_parallel(q_res(), seq_dimension, unpadded_dim_size)
    k = unpadding_tensor_for_seqeunce_parallel(k_res(), seq_dimension, unpadded_dim_size)
    q = q.contiguous()
    k = k.contiguous()

    shape = normalize_shape(normalized_shape)
    rms = None
    saved_rms_q = None
    saved_rms_k = None
    mean_q = mean_k = invvar_q = invvar_k = None
    if norm_type == "rmsnorm":
        rms = resolve_inner_kernel(rms_norm, kernel="rms_norm", variant="standard")
        output_q, saved_rms_q = rms.forward(q, norm_q_weight, eps=eps)
        output_k, saved_rms_k = rms.forward(k, norm_k_weight, eps=eps)
    elif norm_type == "layernorm":
        output_q, mean_q, invvar_q = layernorm_forward(q, norm_q_weight, norm_q_bias, shape, eps)
        output_k, mean_k, invvar_k = layernorm_forward(k, norm_k_weight, norm_k_bias, shape, eps)
    elif norm_type is None:
        output_q = q
        output_k = k
    else:
        raise NotImplementedError(f"{norm_type} is not supported in async-ulysses now!")

    v = unpadding_tensor_for_seqeunce_parallel(v_res(), seq_dimension, unpadded_dim_size)  # wait after QK norm

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
        need_repeat_kv=need_repeat_kv,
        n_repeat=n_repeat,
        original_num_kv_heads=original_num_kv_heads,
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
    """Reverse the QK norm, then overlap reverse all-to-all with linear grads."""
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

    # Start V reverse all-to-all first so QK-norm math overlaps it.
    grad_v = padding_tensor_for_seqeunce_parallel(grad_output[2].contiguous(), dim=seq_dimension)
    grad_v_res = all_to_all_tensor(
        grad_v,
        scatter_dim=seq_dimension,
        gather_dim=head_dimension,
        group=sp_group,
        async_op=True,
    )

    grad_norm_q_weight = grad_norm_q_bias = grad_norm_k_weight = grad_norm_k_bias = None
    if meta.norm_type == "rmsnorm":
        grad_k, grad_norm_k_weight = meta.rms.backward(grad_output[1], saved_rms_k)
        grad_q, grad_norm_q_weight = meta.rms.backward(grad_output[0], saved_rms_q)
    elif meta.norm_type == "layernorm":
        grad_k, grad_norm_k_weight, grad_norm_k_bias = layer_norm_backward(
            grad_output[1], k, mean_k, invvar_k, norm_k_weight, norm_k_bias, meta.normalized_shape, meta.eps
        )
        grad_q, grad_norm_q_weight, grad_norm_q_bias = layer_norm_backward(
            grad_output[0], q, mean_q, invvar_q, norm_q_weight, norm_q_bias, meta.normalized_shape, meta.eps
        )
    elif meta.norm_type is None:
        grad_k = grad_output[1].contiguous()
        grad_q = grad_output[0].contiguous()
    else:
        raise NotImplementedError(f"{meta.norm_type} is not supported in async-ulysses now!")

    grad_v = grad_v_res()
    if meta.need_repeat_kv:
        grad_v = reduce_repeated_kv_gradient(
            grad_v,
            meta.original_num_kv_heads,
            meta.n_repeat,
            head_dimension=head_dimension,
        )

    grad_k = padding_tensor_for_seqeunce_parallel(grad_k, dim=seq_dimension)
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
    if meta.need_repeat_kv:
        grad_k = reduce_repeated_kv_gradient(
            grad_k,
            meta.original_num_kv_heads,
            meta.n_repeat,
            head_dimension=head_dimension,
        )

    grad_q = padding_tensor_for_seqeunce_parallel(grad_q, dim=seq_dimension)
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
