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

"""standard chunk_gated_delta_rule NPU vendored-Triton pair."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .....registry import SavedState
from ...optional import optional_tensor, unused_like


@dataclass(frozen=True)
class _Meta:
    """Scale, chunk size, and which optional tensors were real."""

    scale: float
    chunk_size: int
    output_final_state: bool
    has_initial_state: bool
    has_cu_seqlens: bool


def _l2norm(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    """Match the FLA / vendored ``l2norm`` used before the chunk kernel."""
    original_dtype = x.dtype
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return (x * inv_norm).to(original_dtype)


def _chunk_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    scale: float,
    initial_state: Tensor | None,
    output_final_state: bool,
    cu_seqlens: Tensor | None,
    chunk_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    from ...vendor.triton.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from ...vendor.triton.chunk_o import chunk_fwd_o
    from ...vendor.triton.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from ...vendor.triton.cumsum import chunk_local_cumsum
    from ...vendor.triton.solve_tril import solve_tril
    from ...vendor.triton.wy_fast import recompute_w_u_fwd

    g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens, head_first=False)
    a = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )
    a = solve_tril(A=a, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=a, g=g, cu_seqlens=cu_seqlens)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    output = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    return g, output, a, final_state


def _chunk_bwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    a: Tensor,
    scale: float,
    initial_state: Tensor | None,
    grad_output: Tensor,
    grad_final_state: Tensor | None,
    cu_seqlens: Tensor | None,
    chunk_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    from ...vendor.triton.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
    from ...vendor.triton.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local
    from ...vendor.triton.cumsum import chunk_local_cumsum
    from ...vendor.triton.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd

    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=a, g=g, cu_seqlens=cu_seqlens)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    grad_v = chunk_bwd_dv_local(
        q=q,
        k=k,
        g=g,
        do=grad_output,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    grad_h, grad_h0, grad_v = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g,
        h0=initial_state,
        dht=grad_final_state,
        do=grad_output,
        dv=grad_v,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    grad_q, grad_k, grad_w, grad_g = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=grad_v,
        do=grad_output,
        dh=grad_h,
        chunk_size=chunk_size,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    grad_k2, grad_v, grad_beta, grad_g2 = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=a,
        dw=grad_w,
        du=grad_v,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    grad_k.add_(grad_k2)
    grad_g.add_(grad_g2)
    if grad_g.dtype != torch.float32:
        raise ValueError(f"dg current type is {grad_g.dtype} , should be float32")
    grad_g = chunk_local_cumsum(grad_g, chunk_size=chunk_size, reverse=True, cu_seqlens=cu_seqlens, head_first=False)
    return grad_q, grad_k, grad_v, grad_beta, grad_g, grad_h0


def forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    *,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_size: int = 64,
    cu_seqlens_list: list[int] | None = None,
    chunk_indices: object = None,
    chunk_indices_list: object = None,
    scale: float | None = None,
) -> tuple[tuple[Tensor, Tensor], SavedState]:
    """Vendored NPU Triton chunk gated delta rule. Layout is FLA ``[B, T, H, D]``.

    Empty or omitted *initial_state* / *cu_seqlens* are unused. Unused final
    state is an empty tensor so the registry output stays tensors-only.
    """
    from ...vendor.triton.utils import input_guard

    del cu_seqlens_list, chunk_indices, chunk_indices_list, scale
    if initial_state is None:
        initial_state = unused_like(query)
    if cu_seqlens is None:
        cu_seqlens = unused_like(query, dtype=torch.int32)

    if query.dtype != key.dtype or key.dtype != value.dtype:
        raise ValueError(
            f"q current type is {query.dtype} , k current type is {key.dtype} ,"
            f"v current type is {value.dtype} , they should are equal"
        )
    if query.dtype == torch.float32:
        raise ValueError("chunk_gated_delta_rule npu does not support float32. Please use bfloat16.")
    if beta.ndim != 3:
        raise ValueError(
            f"beta current shape len is {beta.ndim}, beta must be of shape [B, T, H] if head_first=False."
        )

    initial_opt = optional_tensor(initial_state)
    cu_opt = optional_tensor(cu_seqlens)
    if cu_opt is not None:
        if query.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {query.shape[0]} when using `cu_seqlens`."
            )
        if initial_opt is not None and initial_opt.shape[0] != len(cu_opt) - 1:
            raise ValueError(
                "The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_opt) - 1} rather than {initial_opt.shape[0]}."
            )

    scale = key.shape[-1] ** -0.5
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query)
        key = _l2norm(key)

    guarded_fwd = input_guard(_chunk_fwd)
    g_cum, output, a, final_state = guarded_fwd(
        query,
        key,
        value,
        g,
        beta,
        scale,
        initial_opt,
        output_final_state,
        cu_opt,
        chunk_size,
    )
    if final_state is None:
        final_state = output.new_empty(0)
    return (output.to(query.dtype), final_state), SavedState(
        (query, key, value, g_cum, beta, a, initial_state, cu_seqlens),
        _Meta(scale, chunk_size, output_final_state, initial_opt is not None, cu_opt is not None),
    )


def backward(grad_output: tuple[Tensor, Tensor], saved: SavedState) -> tuple[Tensor | None, ...]:
    """Return grads for ``query, key, value, g, beta, initial_state, cu_seqlens``."""
    from ...vendor.triton.utils import input_guard

    meta = saved.metadata
    assert isinstance(meta, _Meta)
    query, key, value, g_cum, beta, a, initial_state, cu_seqlens = saved.tensors
    do, dht = grad_output
    initial_opt = initial_state if meta.has_initial_state else None
    cu_opt = cu_seqlens if meta.has_cu_seqlens else None
    dht_opt = dht if meta.output_final_state else None

    guarded_bwd = input_guard(_chunk_bwd)
    grad_q, grad_k, grad_v, grad_beta, grad_g, grad_h0 = guarded_bwd(
        query,
        key,
        value,
        g_cum,
        beta,
        a,
        meta.scale,
        initial_opt,
        do,
        dht_opt,
        cu_opt,
        meta.chunk_size,
    )
    if not meta.has_initial_state:
        grad_h0 = None
    return (
        grad_q.to(query),
        grad_k.to(key),
        grad_v.to(value),
        grad_g.to(g_cum),
        grad_beta.to(beta),
        grad_h0,
        None,
    )
