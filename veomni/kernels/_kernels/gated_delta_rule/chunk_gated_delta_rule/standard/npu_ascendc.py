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

"""standard chunk_gated_delta_rule AscendC pair.

Heavy compute is ``torch.ops.npu.*`` from ``fla_npu``. Glue around those
ops lives in ``vendor/triton_core`` and ``vendor/triton``.
The kernel contract is FLA ``[B, T, H, D]``; the fused ops want
``[B, H, T, D]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .....registry import SavedState
from ...optional import optional_tensor, unused_like


_DEFAULT_VARLEN_CHUNK_SIZES = (16, 32, 64, 128, 608 * 2)


@dataclass(frozen=True)
class _Meta:
    """Scale, chunk size, optional-tensor flags, and host-side varlen tables."""

    scale: float
    chunk_size: int
    output_final_state: bool
    has_initial_state: bool
    has_cu_seqlens: bool
    cu_seqlens_list: list[int] | None
    chunk_indices: dict[str, Tensor | None] | None
    chunk_indices_list: dict[str, list[int] | None] | None


def _l2norm(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    """Match the FLA / vendored eager ``l2norm`` used before the fused ops."""
    original_dtype = x.dtype
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return (x * inv_norm).to(original_dtype)


def _cdiv(a: Tensor, b: int) -> Tensor:
    return (a + b - 1) // b


def _prepare_lens(cu_seqlens: Tensor) -> Tensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def _prepare_chunk_indices(cu_seqlens: Tensor, chunk_size: int) -> Tensor:
    indices = torch.cat([torch.arange(n) for n in _cdiv(_prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def _prepare_chunk_indices_list(cu_seqlens: list[int] | Tensor, chunk_size: int) -> list[int]:
    if isinstance(cu_seqlens, Tensor):
        cu_seqlens = [int(x) for x in cu_seqlens.detach().cpu().tolist()]

    indices: list[int] = []
    for seq_idx in range(len(cu_seqlens) - 1):
        length = int(cu_seqlens[seq_idx + 1]) - int(cu_seqlens[seq_idx])
        if length <= 0:
            continue
        for chunk_idx in range((length + chunk_size - 1) // chunk_size):
            indices.extend([seq_idx, chunk_idx])
    return indices


def _as_int_list(value: list[int] | Tensor | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, Tensor):
        return [int(x) for x in value.detach().cpu().flatten().tolist()]
    return [int(x) for x in value]


def _as_chunk_dict(
    value: dict[str, Tensor | None] | Tensor | None,
    chunk_size: int,
) -> dict[str, Tensor | None]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return {str(chunk_size): value}


def _as_chunk_list_dict(
    value: dict[str, list[int] | None] | list[int] | Tensor | None,
    chunk_size: int,
) -> dict[str, list[int] | None]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): _as_int_list(v) for k, v in value.items()}
    return {str(chunk_size): _as_int_list(value)}


def _next_power_of_2(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _cumsum_block_t(g: Tensor, chunk_size: int) -> int:
    heads = int(g.shape[-1])
    return _next_power_of_2((1 << 17) // max(1, heads * int(chunk_size)))


def _ensure_varlen_metadata(
    g: Tensor,
    cu_seqlens: Tensor,
    chunk_size: int,
) -> tuple[Tensor, list[int], dict[str, Tensor | None], dict[str, list[int] | None]]:
    cu_seqlens = cu_seqlens.to(device=g.device, dtype=torch.int64)
    cu_seqlens_list = _as_int_list(cu_seqlens)
    assert cu_seqlens_list is not None

    tensor_indices = _as_chunk_dict(None, chunk_size)
    list_indices = _as_chunk_list_dict(None, chunk_size)
    required_sizes = set(_DEFAULT_VARLEN_CHUNK_SIZES)
    required_sizes.add(int(chunk_size))
    required_sizes.add(_cumsum_block_t(g, chunk_size))

    for size in required_sizes:
        key = str(size)
        tensor_indices[key] = _prepare_chunk_indices(cu_seqlens, size)
        list_indices[key] = _prepare_chunk_indices_list(cu_seqlens_list, size)
    return cu_seqlens, cu_seqlens_list, tensor_indices, list_indices


def _chunk_tensor(
    chunk_indices: dict[str, Tensor | None] | None,
    chunk_size: int,
) -> Tensor | None:
    if chunk_indices is None:
        return None
    return chunk_indices.get(str(chunk_size))


def _chunk_list(
    chunk_indices_list: dict[str, list[int] | None] | None,
    chunk_size: int,
) -> list[int] | None:
    if chunk_indices_list is None:
        return None
    return chunk_indices_list.get(str(chunk_size))


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
    cu_seqlens_list: list[int] | None,
    chunk_indices: dict[str, Tensor | None] | None,
    chunk_indices_list: dict[str, list[int] | None] | None,
    chunk_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    import fla_npu  # noqa: F401

    from ...vendor.triton.cumsum import chunk_local_cumsum
    from ...vendor.triton.utils import is_arch35
    from ...vendor.triton_core.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd

    if is_arch35():
        from ...vendor.triton_core.solve_tril import solve_tril
    else:
        from ...vendor.triton.solve_tril import solve_tril

    g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens, head_first=False)
    a = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=_chunk_tensor(chunk_indices, chunk_size),
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )
    if is_arch35():
        a = solve_tril(
            A=a,
            cu_seqlens=cu_seqlens,
            chunk_indices=_chunk_tensor(chunk_indices, chunk_size),
            output_dtype=k.dtype,
        )
    else:
        a = solve_tril(A=a, cu_seqlens=cu_seqlens, output_dtype=k.dtype)

    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous().float()
    a = a.transpose(1, 2).contiguous()
    w, u = torch.ops.npu.npu_recompute_w_u_fwd(
        k,
        v,
        beta,
        a,
        chunk_size,
        g=g,
        gk=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )
    h, v_new, final_state = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g,
        gk=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        use_exp2=False,
        transpose_state_layout=False,
    )
    if not output_final_state:
        final_state = None
    output = torch.ops.npu.npu_chunk_fwd_o(
        q,
        k,
        v_new,
        h,
        scale,
        g=g,
        g_gamma=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        chunk_size=chunk_size,
        transpose_state_layout=False,
    )
    g = g.transpose(1, 2).contiguous()
    output = output.transpose(1, 2).contiguous()
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
    cu_seqlens: Tensor | None,
    cu_seqlens_list: list[int] | None,
    chunk_indices: dict[str, Tensor | None] | None,
    chunk_indices_list: dict[str, list[int] | None] | None,
    chunk_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    import fla_npu  # noqa: F401

    from ...vendor.triton.cumsum import chunk_local_cumsum

    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous().float()
    w, u = torch.ops.npu.npu_recompute_w_u_fwd(
        k,
        v,
        beta,
        a,
        chunk_size,
        g=g,
        gk=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )
    grad_output = grad_output.transpose(1, 2).contiguous()
    h, v_new, _ = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g,
        gk=None,
        initial_state=initial_state,
        output_final_state=False,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        use_exp2=False,
        transpose_state_layout=False,
    )
    grad_v = torch.ops.npu.npu_chunk_bwd_dv_local(
        q,
        k,
        grad_output,
        g,
        scale,
        chunk_size,
        g_gamma=None,
        A=a,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )
    _dh, _dh0, grad_v = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
        q,
        k,
        w,
        grad_output,
        grad_v,
        scale,
        chunk_size,
        g=g,
        gK=None,
        h0=None,
        dht=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        use_exp2=False,
        transpose_state_layout=False,
    )
    grad_q, grad_k, grad_w, grad_g = torch.ops.npu.npu_chunk_bwd_dqkwg(
        q,
        k,
        v_new,
        g,
        h,
        grad_output,
        _dh,
        grad_v,
        chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        w=None,
        g_gamma=None,
        scale=scale,
        use_exp2=False,
        transpose_state_layout=False,
    )
    grad_a = torch.ops.npu.npu_prepare_wy_repr_bwd_da(
        k,
        v,
        beta.float(),
        a,
        grad_w,
        grad_v,
        g.float(),
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )
    grad_k2, grad_v, grad_beta, grad_g2 = torch.ops.npu.npu_prepare_wy_repr_bwd_full(
        k,
        v,
        beta,
        a,
        grad_a,
        grad_w,
        grad_v,
        g,
        chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )
    grad_beta = grad_beta.transpose(1, 2).contiguous()
    grad_g2 = grad_g2.transpose(1, 2).contiguous()
    grad_g = grad_g.transpose(1, 2).contiguous()
    grad_k.add_(grad_k2)
    grad_g.add_(grad_g2)
    if grad_g.dtype != torch.float32:
        raise ValueError(f"dg current type is {grad_g.dtype}, should be float32")
    grad_g = chunk_local_cumsum(
        grad_g,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        head_first=False,
    )
    return grad_q, grad_k, grad_v, grad_beta, grad_g


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
    """AscendC fused chunk gated delta rule. Layout is FLA ``[B, T, H, D]``.

    Empty or omitted *initial_state* / *cu_seqlens* are unused. Unused final
    state is an empty tensor so the registry output stays tensors-only. The
    fused backward does not produce ``dh0``. Extra NPU varlen tables are
    accepted; this path rebuilds them from ``cu_seqlens``.
    """
    from ...vendor.triton.utils import input_guard

    del scale
    if initial_state is None:
        initial_state = unused_like(query)
    if cu_seqlens is None:
        cu_seqlens = unused_like(query, dtype=torch.int32)

    query_h = query.transpose(1, 2).contiguous()
    key_h = key.transpose(1, 2).contiguous()
    value_h = value.transpose(1, 2).contiguous()

    if query_h.dtype != key_h.dtype or key_h.dtype != value_h.dtype:
        raise ValueError(
            f"q current type is {query_h.dtype}, k current type is {key_h.dtype}, "
            f"v current type is {value_h.dtype}; they should be equal"
        )
    if query_h.dtype == torch.float32:
        raise ValueError("chunk_gated_delta_rule npu_ascendc does not support float32. Please use float16/bfloat16.")
    if beta.ndim != 3 or g.ndim != 3:
        raise ValueError("g and beta must be rank-3 tensors with shape [B, T, H].")
    if query_h.ndim != 4 or key_h.ndim != 4 or value_h.ndim != 4:
        raise ValueError("q, k and v must be rank-4 tensors with shape [B, H, T, D] after layout convert.")
    if query_h.shape[:3] != key_h.shape[:3] or query_h.shape[:3] != value_h.shape[:3]:
        raise ValueError(f"q/k/v shape prefixes must match, got {query_h.shape}, {key_h.shape}, {value_h.shape}.")
    if g.shape != beta.shape:
        raise ValueError(f"g and beta shapes must match, got {g.shape} and {beta.shape}.")
    if g.shape[0] != query_h.shape[0] or g.shape[1] != query_h.shape[2] or g.shape[2] != query_h.shape[1]:
        raise ValueError(
            f"Expected q/k/v in [B, T, H, D] and g/beta in [B, T, H]; got q={tuple(query.shape)}, g={tuple(g.shape)}."
        )
    if chunk_size != 2 ** (chunk_size.bit_length() - 1):
        raise ValueError(f"chunk_size must be a power of 2, got {chunk_size}.")

    initial_opt = optional_tensor(initial_state)
    if torch.is_grad_enabled() and initial_opt is not None and initial_opt.requires_grad:
        raise NotImplementedError(
            "npu_ascendc chunk_gated_delta_rule cannot differentiate initial_state "
            "(the AscendC backward returns no dh0, same as MindSpeed-MM). Detach "
            "initial_state, or use the 'npu' (Triton) backend if you need that gradient."
        )

    cu_opt = optional_tensor(cu_seqlens)
    cu_seqlens_list: list[int] | None = None
    chunk_indices: dict[str, Tensor | None] | None = None
    chunk_indices_list: dict[str, list[int] | None] | None = None
    if cu_opt is not None:
        cu_opt, cu_seqlens_list, chunk_indices, chunk_indices_list = _ensure_varlen_metadata(g, cu_opt, chunk_size)
        if query_h.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {query_h.shape[0]} when using cu_seqlens. "
                "Please flatten variable-length inputs before processing."
            )
        if initial_opt is not None and initial_opt.shape[0] != len(cu_seqlens_list) - 1:
            raise ValueError(
                "The number of initial states is expected to match the number of input sequences, "
                f"got initial_state.shape[0]={initial_opt.shape[0]} and sequences={len(cu_seqlens_list) - 1}."
            )

    if use_qk_l2norm_in_kernel:
        query_h = _l2norm(query_h)
        key_h = _l2norm(key_h)

    scale = key_h.shape[-1] ** -0.5
    guarded_fwd = input_guard(_chunk_fwd)
    g_cum, output, a, final_state = guarded_fwd(
        query_h,
        key_h,
        value_h,
        g,
        beta,
        scale,
        initial_opt,
        output_final_state,
        cu_opt,
        cu_seqlens_list,
        chunk_indices,
        chunk_indices_list,
        chunk_size,
    )
    if final_state is None:
        final_state = output.new_empty(0)
    return (output.to(query_h.dtype), final_state), SavedState(
        (query_h, key_h, value_h, g_cum, beta, a, initial_state, cu_seqlens),
        _Meta(
            scale,
            chunk_size,
            output_final_state,
            initial_opt is not None,
            cu_opt is not None,
            cu_seqlens_list,
            chunk_indices,
            chunk_indices_list,
        ),
    )


def backward(grad_output: tuple[Tensor, Tensor], saved: SavedState) -> tuple[Tensor | None, ...]:
    """Return grads for ``query, key, value, g, beta, initial_state, cu_seqlens``.

    ``query`` / ``key`` / ``value`` grads are converted back to FLA ``[B, T, H, D]``.
    ``initial_state`` grad is always ``None`` on this path.
    """
    from ...vendor.triton.utils import input_guard

    meta = saved.metadata
    assert isinstance(meta, _Meta)
    query_h, key_h, value_h, g_cum, beta, a, initial_state, cu_seqlens = saved.tensors
    do, _dht = grad_output
    initial_opt = initial_state if meta.has_initial_state else None
    cu_opt = cu_seqlens if meta.has_cu_seqlens else None

    guarded_bwd = input_guard(_chunk_bwd)
    grad_q_h, grad_k_h, grad_v_h, grad_beta, grad_g = guarded_bwd(
        query_h,
        key_h,
        value_h,
        g_cum,
        beta,
        a,
        meta.scale,
        initial_opt,
        do,
        cu_opt,
        meta.cu_seqlens_list,
        meta.chunk_indices,
        meta.chunk_indices_list,
        meta.chunk_size,
    )
    return (
        grad_q_h.transpose(1, 2).to(query_h),
        grad_k_h.transpose(1, 2).to(key_h),
        grad_v_h.transpose(1, 2).to(value_h),
        grad_g.to(g_cum),
        grad_beta.to(beta),
        None,
        None,
    )
