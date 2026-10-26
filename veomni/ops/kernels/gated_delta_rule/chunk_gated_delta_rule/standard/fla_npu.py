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

Uses the stable ``fla_npu.ops.ascendc`` GDN forward/backward pair on A3/A5.
The library selects arch22/arch35; this adapter uses their common BF16,
K=V=128, chunk_size=64 contract. Forward can use the Phase6 fused kernel;
backward is a composite ACLNN executor, not a single fused kernel.
The kernel contract is FLA ``[B, T, H, D]``; the fused ``fla_npu.ops.ascendc``
path wants ``[B, H, T, D]``.
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
    use_qk_l2norm: bool
    has_initial_state: bool
    has_cu_seqlens: bool
    cu_seqlens_list: list[int] | None
    chunk_indices: dict[str, Tensor | None] | None
    chunk_indices_list: dict[str, list[int] | None] | None


def _l2norm_fwd(x: Tensor, dim: int = -1, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    """Return the FLA-style L2-normalized tensor and reciprocal norm."""
    original_dtype = x.dtype
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return (x * inv_norm).to(original_dtype), inv_norm


def _l2norm_bwd(y: Tensor, inv_norm: Tensor, grad_y: Tensor, dim: int = -1) -> Tensor:
    """Apply the L2-normalization Jacobian to ``grad_y``."""
    projection = (grad_y * y).sum(dim=dim, keepdim=True)
    return ((grad_y - y * projection) * inv_norm).to(y)


def _cdiv(a: Tensor, b: int) -> Tensor:
    """Compute integer ceiling division elementwise."""
    return (a + b - 1) // b


def _prepare_lens(cu_seqlens: Tensor) -> Tensor:
    """Return individual sequence lengths from cumulative boundaries."""
    return cu_seqlens[1:] - cu_seqlens[:-1]


def _prepare_chunk_indices(cu_seqlens: Tensor, chunk_size: int) -> Tensor:
    """Build device-side sequence and chunk indices for variable lengths.

    Empty sequences emit no rows, but later sequences keep their original
    ``cu_seqlens`` index. Do not renumber after a zero-length gap.
    """
    lengths = _prepare_lens(cu_seqlens)
    n_chunks = _cdiv(lengths, chunk_size)
    seq_ids = torch.repeat_interleave(
        torch.arange(n_chunks.numel(), device=cu_seqlens.device, dtype=cu_seqlens.dtype),
        n_chunks,
    )
    chunk_ids = (
        torch.cat([torch.arange(int(n), device=cu_seqlens.device, dtype=cu_seqlens.dtype) for n in n_chunks.tolist()])
        if int(n_chunks.sum().item()) > 0
        else cu_seqlens.new_empty(0)
    )
    if seq_ids.numel() == 0:
        return cu_seqlens.new_empty((0, 2))
    return torch.stack([seq_ids, chunk_ids], 1)


def _prepare_chunk_indices_list(cu_seqlens: list[int] | Tensor, chunk_size: int) -> list[int]:
    """Build the flattened host-side sequence and chunk index table."""
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
    """Normalize an optional integer tensor or sequence to a Python list."""
    if value is None:
        return None
    if isinstance(value, Tensor):
        return [int(x) for x in value.detach().cpu().flatten().tolist()]
    return [int(x) for x in value]


def _as_chunk_dict(
    value: dict[str, Tensor | None] | Tensor | None,
    chunk_size: int,
) -> dict[str, Tensor | None]:
    """Normalize device chunk metadata to a mapping keyed by chunk size."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return {str(chunk_size): value}


def _as_chunk_list_dict(
    value: dict[str, list[int] | None] | list[int] | Tensor | None,
    chunk_size: int,
) -> dict[str, list[int] | None]:
    """Normalize host chunk metadata to a mapping keyed by chunk size."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): _as_int_list(v) for k, v in value.items()}
    return {str(chunk_size): _as_int_list(value)}


def _next_power_of_2(value: int) -> int:
    """Return the smallest power of two greater than or equal to ``value``."""
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _cumsum_block_t(g: Tensor, chunk_size: int) -> int:
    """Choose the sequence tile used by the cumulative-sum kernel."""
    heads = int(g.shape[-1])
    return _next_power_of_2((1 << 17) // max(1, heads * int(chunk_size)))


def _ensure_varlen_metadata(
    g: Tensor,
    cu_seqlens: Tensor,
    chunk_size: int,
    cu_seqlens_list: list[int] | None = None,
    chunk_indices: dict[str, Tensor | None] | Tensor | None = None,
    chunk_indices_list: dict[str, list[int] | None] | list[int] | Tensor | None = None,
) -> tuple[Tensor, list[int], dict[str, Tensor | None], dict[str, list[int] | None]]:
    """Materialize every host and device table needed by variable-length kernels."""
    cu_seqlens = cu_seqlens.to(device=g.device, dtype=torch.int64)
    cu_seqlens_list = _as_int_list(cu_seqlens_list) or _as_int_list(cu_seqlens)
    assert cu_seqlens_list is not None

    tensor_indices = _as_chunk_dict(chunk_indices, chunk_size)
    list_indices = _as_chunk_list_dict(chunk_indices_list, chunk_size)
    required_sizes = set(_DEFAULT_VARLEN_CHUNK_SIZES)
    required_sizes.add(int(chunk_size))
    required_sizes.add(_cumsum_block_t(g, chunk_size))

    for size in required_sizes:
        key = str(size)
        if key not in tensor_indices or tensor_indices[key] is None:
            tensor_indices[key] = _prepare_chunk_indices(cu_seqlens, size)
        if key not in list_indices or list_indices[key] is None:
            list_indices[key] = _prepare_chunk_indices_list(cu_seqlens_list, size)
    return cu_seqlens, cu_seqlens_list, tensor_indices, list_indices


def precompute_varlen_metadata(
    cu_seqlens: Tensor,
    num_heads: int,
    chunk_size: int = 64,
    device: torch.device | str | None = None,
) -> tuple[list[int], dict[str, Tensor | None], dict[str, list[int] | None]]:
    """Build host-side varlen tables once for every GDN layer.

    Hoists ``cu_seqlens.tolist()`` and chunk-index construction so later
    layers can reuse the same tables.
    """
    if cu_seqlens.device.type != "cpu":
        cu_seqlens = cu_seqlens.cpu()
    cu_seqlens_list = _as_int_list(cu_seqlens)
    assert cu_seqlens_list is not None
    cumsum_block_size = _next_power_of_2((1 << 17) // max(1, int(num_heads) * int(chunk_size)))
    required_sizes = set(_DEFAULT_VARLEN_CHUNK_SIZES) | {int(chunk_size), cumsum_block_size}
    chunk_indices: dict[str, Tensor | None] = {}
    chunk_indices_list: dict[str, list[int] | None] = {}
    for size in required_sizes:
        key = str(size)
        chunk_indices[key] = _prepare_chunk_indices(cu_seqlens, size)
        chunk_indices_list[key] = _prepare_chunk_indices_list(cu_seqlens_list, size)
    if device is not None:
        chunk_indices = {k: v.to(device=device) if v is not None else None for k, v in chunk_indices.items()}
    return cu_seqlens_list, chunk_indices, chunk_indices_list


def _chunk_tensor(
    chunk_indices: dict[str, Tensor | None] | None,
    chunk_size: int,
) -> Tensor | None:
    """Select device chunk metadata for one chunk size."""
    if chunk_indices is None:
        return None
    return chunk_indices.get(str(chunk_size))


def _chunk_list(
    chunk_indices_list: dict[str, list[int] | None] | None,
    chunk_size: int,
) -> list[int] | None:
    """Select host chunk metadata for one chunk size."""
    if chunk_indices_list is None:
        return None
    return chunk_indices_list.get(str(chunk_size))


def _get_gdn_ops():
    """Load the stable AscendC API without legacy dispatcher registration."""
    try:
        from fla_npu.ops import ascendc
    except ImportError as error:
        raise RuntimeError(
            "The fla_npu backend requires an installed A3/A5 fla_npu build with "
            "fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd and "
            "chunk_gated_delta_rule_bwd. Check the package and CANN environment."
        ) from error
    for name in ("chunk_gated_delta_rule_fwd", "chunk_gated_delta_rule_bwd"):
        if not callable(getattr(ascendc, name, None)):
            raise RuntimeError(f"Installed fla_npu is missing the stable GDN entry {name}.")
    return ascendc


def _validate_gdn_contract(q: Tensor, k: Tensor, v: Tensor, chunk_size: int, initial_state: Tensor | None):
    """Reject inputs outside the common A3/A5 complete forward/backward domain."""
    if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise ValueError("fla_npu GDN on A3/A5 requires bfloat16 q/k/v.")
    if any(x.shape[-1] != 128 for x in (q, k, v)) or chunk_size != 64:
        raise ValueError("fla_npu GDN on A3/A5 requires K=V=128 and chunk_size=64.")
    if initial_state is not None and initial_state.dtype != torch.bfloat16:
        raise ValueError("fla_npu GDN on A3/A5 requires a bfloat16 initial_state.")


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
    """Run GDN with head-first QKV and sequence-major gates/output."""
    _validate_gdn_contract(q, k, v, chunk_size, initial_state)
    ops = _get_gdn_ops()
    # Keep normalization outside the kernel and natural-exponent gates on both
    # architectures. disable_recompute exports the two backward intermediates.
    output, final_state, g_cum, a, _, _, _, _, _, _ = ops.chunk_gated_delta_rule_fwd(
        q,
        k,
        v,
        g.contiguous(),
        beta.contiguous(),
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        scale=scale,
        layout="BNSD",
        use_exp2=False,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        disable_recompute=True,
        return_intermediate_states=False,
        state_v_first=False,
    )
    if g_cum is None or a is None:
        raise RuntimeError("fla_npu GDN forward did not export g_cumsum/A with disable_recompute=True.")
    # The stable API already returns output and cumulative gates in BSND/BSN.
    return g_cum, output, a, final_state if output_final_state else None


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
    """Consume saved cumulative gates/A and return gradients of raw inputs."""
    ops = _get_gdn_ops()
    grad_q, grad_k, grad_v, grad_beta, grad_g, _, _, _ = ops.chunk_gated_delta_rule_bwd(
        q,
        k,
        v,
        g,
        beta.contiguous(),
        a,
        grad_output.contiguous(),
        scale,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        initial_state=initial_state,
        dht=None,
        layout="BNSD",
        use_exp2=False,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        state_v_first=False,
    )
    # dG is already with respect to the raw gate: do not reverse-cumsum again.
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
    state is an empty tensor so the registry output stays tensors-only. This
    adapter does not expose ``dh0`` or consume ``dht``. Trainable
    ``initial_state`` and ``output_final_state=True`` with trainable inputs
    are rejected. Extra NPU varlen tables are accepted; missing keys are
    filled from ``cu_seqlens``.
    """
    from ...vendor.triton.utils import input_guard

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
        raise ValueError("chunk_gated_delta_rule fla_npu does not support float32. Please use bfloat16.")
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
    if chunk_size <= 0 or chunk_size & (chunk_size - 1):
        raise ValueError(f"chunk_size must be a power of 2, got {chunk_size}.")

    initial_opt = optional_tensor(initial_state)
    if initial_opt is not None and initial_opt.requires_grad:
        raise NotImplementedError(
            "fla_npu chunk_gated_delta_rule cannot differentiate initial_state "
            "(this adapter does not expose state gradients). Detach "
            "initial_state, or use the 'triton' backend if you need that gradient."
        )
    inputs_need_grad = any(tensor.requires_grad for tensor in (query, key, value, g, beta))
    if output_final_state and inputs_need_grad:
        raise NotImplementedError(
            "fla_npu chunk_gated_delta_rule cannot differentiate the final state "
            "(this adapter does not pass dht). Set output_final_state=False, or use "
            "the 'triton' backend if you need a final-state loss."
        )

    cu_opt = optional_tensor(cu_seqlens)
    if cu_opt is not None:
        cu_opt, cu_seqlens_list, chunk_indices, chunk_indices_list = _ensure_varlen_metadata(
            g,
            cu_opt,
            chunk_size,
            cu_seqlens_list=cu_seqlens_list,
            chunk_indices=chunk_indices,
            chunk_indices_list=chunk_indices_list,
        )
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
        query_h, query_inv_norm = _l2norm_fwd(query_h)
        key_h, key_inv_norm = _l2norm_fwd(key_h)
    else:
        query_inv_norm = unused_like(query_h)
        key_inv_norm = unused_like(key_h)

    if scale is None:
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
    saved_cu_seqlens = cu_opt if cu_opt is not None else cu_seqlens
    return (output.to(query_h.dtype), final_state), SavedState(
        (
            query_h,
            key_h,
            value_h,
            g_cum,
            beta,
            a,
            initial_state,
            saved_cu_seqlens,
            query_inv_norm,
            key_inv_norm,
        ),
        _Meta(
            scale,
            chunk_size,
            output_final_state,
            use_qk_l2norm_in_kernel,
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
    (
        query_h,
        key_h,
        value_h,
        g_cum,
        beta,
        a,
        initial_state,
        cu_seqlens,
        query_inv_norm,
        key_inv_norm,
    ) = saved.tensors
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
    if meta.use_qk_l2norm:
        grad_q_h = _l2norm_bwd(query_h, query_inv_norm, grad_q_h)
        grad_k_h = _l2norm_bwd(key_h, key_inv_norm, grad_k_h)
    return (
        grad_q_h.transpose(1, 2).to(query_h),
        grad_k_h.transpose(1, 2).to(key_h),
        grad_v_h.transpose(1, 2).to(value_h),
        grad_g.to(g_cum),
        grad_beta.to(beta),
        None,
        None,
    )
