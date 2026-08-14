"""AscendC port of Qwen3.5's chunk gated delta-rule.

Adapted from MindSpeed-MM's ``mindspeed_mm/fsdp/ops/gdn/flash_gated_delta_rule.py``,
which in turn derives from fla_npu commit c2e3d83f
(https://github.com/flashserve/flash-linear-attention-npu):
the heavy GDN compute is provided by the external ``fla_npu`` package as
``torch.ops.npu.*`` fused ops; the Triton glue lives under ``triton_core``.

VeOmni adaptations vs. upstream (keep this list in sync when re-syncing):
- MM-absolute imports rewritten to package-relative.
- Dropped MM's async-offload / ``skip_recompute`` machinery
  (``TrainingContext`` / ``OffloadManager`` / ``SwapTensor`` / ``get_current_stream``):
  VeOmni has no such framework, so only the plain compute path remains.
"""

import warnings
from typing import Dict, Optional

import torch
import torch_npu

from .triton_core.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from .triton_core.l2norm import l2norm_bwd, l2norm_fwd
from .triton.cumsum import chunk_local_cumsum
from .triton_core.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from ..varlen_metadata import (
    _DEFAULT_VARLEN_CHUNK_SIZES,
    _next_power_of_2,
    precompute_varlen_metadata,
    prepare_chunk_indices,
    prepare_chunk_indices_list,
)
from .triton.utils import is_arch35
from .varlen_active_segments import build_active_varlen_segments, empty_varlen_result

if is_arch35():
    from .triton_core.solve_tril import solve_tril
else:
    from .triton.solve_tril import solve_tril

_disable_compile = getattr(getattr(torch, "compiler", None), "disable", lambda fn: fn)
_FLA_NPU_LOADED = False


def _ensure_fla_npu() -> None:
    """Import ``fla_npu`` immediately before calling its registered ops."""
    global _FLA_NPU_LOADED
    if _FLA_NPU_LOADED:
        return
    try:
        import fla_npu  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "AscendC chunk gated delta rule requires fla_npu to register "
            "torch.ops.npu.* GDN kernels. Install fla_npu on this NPU image, "
            "or select chunk_gated_delta_rule_implementation='npu'."
        ) from error
    _FLA_NPU_LOADED = True


def _as_int_list(value: Optional[list[int] | torch.Tensor]) -> Optional[list[int]]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.detach().cpu().flatten().tolist()]
    return [int(x) for x in value]


def _as_chunk_dict(
    value: Optional[Dict[str, Optional[torch.LongTensor]] | torch.LongTensor],
    chunk_size: int,
) -> Dict[str, Optional[torch.LongTensor]]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return {str(chunk_size): value}


def _as_chunk_list_dict(
    value: Optional[Dict[str, Optional[list[int]]] | list[int] | torch.Tensor],
    chunk_size: int,
) -> Dict[str, Optional[list[int]]]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): _as_int_list(v) for k, v in value.items()}
    return {str(chunk_size): _as_int_list(value)}


def _cumsum_block_t(g: torch.Tensor, chunk_size: int) -> int:
    # Keep this aligned with mindspeed_mm.fsdp.ops.gdn.triton_core.cumsum.chunk_local_cumsum_scalar.
    h = int(g.shape[-1])
    return _next_power_of_2((1 << 17) // max(1, h * int(chunk_size)))


def _ensure_varlen_metadata(
    g: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
    cu_seqlens_list: Optional[list[int]],
    chunk_indices: Optional[Dict[str, Optional[torch.LongTensor]] | torch.LongTensor],
    chunk_indices_list: Optional[Dict[str, Optional[list[int]]] | list[int] | torch.Tensor],
    chunk_size: int,
) -> tuple[
    Optional[torch.LongTensor],
    Optional[list[int]],
    Optional[Dict[str, Optional[torch.LongTensor]]],
    Optional[Dict[str, Optional[list[int]]]],
]:
    if cu_seqlens is None:
        return None, None, None, None

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
            tensor_indices[key] = prepare_chunk_indices(cu_seqlens, size)
        if key not in list_indices or list_indices[key] is None:
            list_indices[key] = prepare_chunk_indices_list(cu_seqlens_list, size)

    return cu_seqlens, cu_seqlens_list, tensor_indices, list_indices


def _chunk_tensor(
    chunk_indices: Optional[Dict[str, Optional[torch.LongTensor]]],
    chunk_size: int,
) -> Optional[torch.LongTensor]:
    if chunk_indices is None:
        return None
    return chunk_indices.get(str(chunk_size))


def _chunk_list(
    chunk_indices_list: Optional[Dict[str, Optional[list[int]]]],
    chunk_size: int,
) -> Optional[list[int]]:
    if chunk_indices_list is None:
        return None
    return chunk_indices_list.get(str(chunk_size))


def flash_chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_list: Optional[list[int]] = None,
    chunk_indices: Optional[Dict[str, Optional[torch.LongTensor]]] = None,
    chunk_indices_list: Optional[Dict[str, Optional[list[int]]]] = None,
    chunk_size: int = 64,
):
    _ensure_fla_npu()
    g = chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        head_first=False,
    )

    # A is the WY lower-triangular representation before inversion.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=_chunk_tensor(chunk_indices, chunk_size),
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )

    if is_arch35():
        A = solve_tril(
            A=A,
            cu_seqlens=cu_seqlens,
            chunk_indices=_chunk_tensor(chunk_indices, chunk_size),
            output_dtype=k.dtype,
        )
    else:
        A = solve_tril(
            A=A,
            cu_seqlens=cu_seqlens,
            output_dtype=k.dtype,
        )

    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous().float()
    A = A.transpose(1, 2).contiguous()

    w, u = torch.ops.npu.npu_recompute_w_u_fwd(
        k,
        v,
        beta,
        A,
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

    o = torch.ops.npu.npu_chunk_fwd_o(
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
    o = o.transpose(1, 2).contiguous()
    return g, o, A, final_state


def flash_chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    do: torch.Tensor,
    dht: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_list: Optional[list[int]] = None,
    chunk_indices: Optional[Dict[str, Optional[torch.LongTensor]]] = None,
    chunk_indices_list: Optional[Dict[str, Optional[list[int]]]] = None,
    chunk_size: int = 64,
):
    _ensure_fla_npu()
    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous().float()

    w, u = torch.ops.npu.npu_recompute_w_u_fwd(
        k,
        v,
        beta,
        A,
        chunk_size,
        g=g,
        gk=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    do = do.transpose(1, 2).contiguous()

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

    dv = torch.ops.npu.npu_chunk_bwd_dv_local(
        q,
        k,
        do,
        g,
        scale,
        chunk_size,
        g_gamma=None,
        A=A,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    dh, dh0, dv = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
        q,
        k,
        w,
        do,
        dv,
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
    dh0 = None

    dq, dk, dw, dg = torch.ops.npu.npu_chunk_bwd_dqkwg(
        q,
        k,
        v_new,
        g,
        h,
        do,
        dh,
        dv,
        chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        w=None,
        g_gamma=None,
        scale=scale,
        use_exp2=False,
        transpose_state_layout=False,
    )

    dA = torch.ops.npu.npu_prepare_wy_repr_bwd_da(
        k,
        v,
        beta.float(),
        A,
        dw,
        dv,
        g.float(),
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    dk2, dv, db, dg2 = torch.ops.npu.npu_prepare_wy_repr_bwd_full(
        k,
        v,
        beta,
        A,
        dA,
        dw,
        dv,
        g,
        chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    db = db.transpose(1, 2).contiguous()
    dg2 = dg2.transpose(1, 2).contiguous()
    dg = dg.transpose(1, 2).contiguous()

    dk.add_(dk2)
    dg.add_(dg2)
    if dg.dtype != torch.float32:
        raise ValueError(f"dg current type is {dg.dtype}, should be float32")

    dg = chunk_local_cumsum(
        dg,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        head_first=False,
    )

    return dq, dk, dv, db, dg, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: Optional[torch.Tensor],
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
        cu_seqlens_list: Optional[list[int]] = None,
        chunk_indices: Optional[Dict[str, Optional[torch.LongTensor]]] = None,
        chunk_indices_list: Optional[Dict[str, Optional[list[int]]]] = None,
        use_qk_l2norm_in_kernel: bool = False,
        chunk_size: int = 64,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        g, o, A, final_state = flash_chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_list=cu_seqlens_list,
            chunk_indices=chunk_indices,
            chunk_indices_list=chunk_indices_list,
            chunk_size=chunk_size,
        )

        ctx.save_for_backward(q, k, v, g, beta, A)
        ctx.q_rstd = q_rstd
        ctx.k_rstd = k_rstd
        ctx.initial_state = initial_state
        ctx.cu_seqlens = cu_seqlens
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.chunk_size = chunk_size
        ctx.cu_seqlens_list = cu_seqlens_list
        ctx.chunk_indices = chunk_indices
        ctx.chunk_indices_list = chunk_indices_list
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: Optional[torch.Tensor]):
        q, k, v, g, beta, A = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = flash_chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=ctx.initial_state,
            do=do,
            dht=dht,
            cu_seqlens=ctx.cu_seqlens,
            cu_seqlens_list=ctx.cu_seqlens_list,
            chunk_indices=ctx.chunk_indices,
            chunk_indices_list=ctx.chunk_indices_list,
            chunk_size=ctx.chunk_size,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, ctx.q_rstd, dq)
            dk = l2norm_bwd(k, ctx.k_rstd, dk)
        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg.to(g),
            db.to(beta),
            None,
            dh0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@_disable_compile
def flash_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_list: Optional[list[int]] = None,
    chunk_indices: Optional[Dict[str, Optional[torch.LongTensor]] | torch.LongTensor] = None,
    chunk_indices_list: Optional[Dict[str, Optional[list[int]]] | list[int] | torch.Tensor] = None,
    chunk_size: int = 64,
    head_first: bool = False,
):
    r"""
    Flash-linear-attention NPU port of xtuner's GDN entry.

    This port keeps the xtuner NPU layout:
        q, k: [B, H, T, K]
        v:    [B, H, T, V]
        g:    [B, T, H]
        beta: [B, T, H]

    It returns:
        o: [B, T, H, V]
        final_state: [N, H, K, V] when output_final_state=True, else None
    """
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError(
            f"q current type is {q.dtype}, k current type is {k.dtype}, "
            f"v current type is {v.dtype}; they should be equal"
        )
    if q.dtype == torch.float32:
        raise ValueError("ChunkGatedDeltaRuleFunction does not support float32. Please use float16/bfloat16.")
    if beta.ndim != 3 or g.ndim != 3:
        raise ValueError("g and beta must be rank-3 tensors with shape [B, T, H].")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must be rank-4 tensors with shape [B, H, T, D].")
    if q.shape[:3] != k.shape[:3] or q.shape[:3] != v.shape[:3]:
        raise ValueError(f"q/k/v shape prefixes must match, got {q.shape}, {k.shape}, {v.shape}.")
    if g.shape != beta.shape:
        raise ValueError(f"g and beta shapes must match, got {g.shape} and {beta.shape}.")
    if g.shape[0] != q.shape[0] or g.shape[1] != q.shape[2] or g.shape[2] != q.shape[1]:
        raise ValueError(
            f"Expected q/k/v in [B, H, T, D] and g/beta in [B, T, H]; got q={tuple(q.shape)}, g={tuple(g.shape)}."
        )

    if head_first:
        warnings.warn(
            "head_first is kept only for API compatibility. This NPU port always expects q/k/v as [B, H, T, D].",
            stacklevel=2,
        )
    if chunk_size != 2 ** (chunk_size.bit_length() - 1):
        raise ValueError(f"chunk_size must be a power of 2, got {chunk_size}.")

    active_plan = None
    original_initial_state = initial_state
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using cu_seqlens. "
                "Please flatten variable-length inputs before processing."
            )
        active_plan = build_active_varlen_segments(
            cu_seqlens,
            cu_seqlens_list=cu_seqlens_list,
            token_count=int(q.shape[2]),
            initial_state=initial_state,
        )
        if active_plan.all_empty:
            dependencies = [q, k, v, g, beta]
            if initial_state is not None:
                dependencies.append(initial_state)
            return empty_varlen_result(
                v.transpose(1, 2).contiguous(),
                dependencies=dependencies,
                plan=active_plan,
                initial_state=initial_state,
                output_final_state=output_final_state,
                state_shape=(int(q.shape[1]), int(q.shape[3]), int(v.shape[3])),
            )

    # The mathematically empty recurrence above has no native backward and can
    # preserve both state VJPs with ordinary PyTorch edges.  The following
    # restrictions apply only to paths that actually launch AscendC kernels.
    # Keep them outside the custom Function because its forward executes with
    # autograd disabled.
    if torch.is_grad_enabled():
        if initial_state is not None and initial_state.requires_grad:
            raise NotImplementedError(
                "npu_ascendc chunk_gated_delta_rule cannot differentiate initial_state "
                "(the AscendC backward returns no dh0). Detach initial_state, or use the "
                "'npu' (Triton) backend if you need that gradient."
            )
        differentiable_input = any(tensor.requires_grad for tensor in (q, k, v, g, beta))
        if output_final_state and differentiable_input:
            raise NotImplementedError(
                "npu_ascendc chunk_gated_delta_rule cannot differentiate final_state "
                "(the AscendC backward ignores dht). Set output_final_state=False, or use "
                "the 'npu' (Triton) backend if you need that gradient."
            )

    if active_plan is not None:
        if active_plan.has_empty:
            cu_seqlens = active_plan.compact_cu_seqlens
            cu_seqlens_list = list(active_plan.compact_cu_seqlens_list)
            initial_state = active_plan.compact_initial_state(initial_state)
            # Caller metadata was compiled against full-N sequence ordinals.
            # Reusing it after compaction would bind chunks to the wrong state.
            chunk_indices = None
            chunk_indices_list = None
        if cu_seqlens_list is not None and chunk_indices is not None and chunk_indices_list is not None:
            cu_seqlens = cu_seqlens.to(device=g.device, dtype=torch.int64)
        else:
            cu_seqlens, cu_seqlens_list, chunk_indices, chunk_indices_list = _ensure_varlen_metadata(
                g=g,
                cu_seqlens=cu_seqlens,
                cu_seqlens_list=cu_seqlens_list,
                chunk_indices=chunk_indices,
                chunk_indices_list=chunk_indices_list,
                chunk_size=chunk_size,
            )
    else:
        cu_seqlens_list = None
        chunk_indices = None
        chunk_indices_list = None

    _ensure_fla_npu()

    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q = l2norm(q, dim=-1, eps=1e-6)
        k = l2norm(k, dim=-1, eps=1e-6)

    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        float(scale),
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_list,
        chunk_indices,
        chunk_indices_list,
        False,
        chunk_size,
    )
    if active_plan is not None:
        final_state = active_plan.restore_final_state(final_state, original_initial_state)
    return o, final_state


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """Eager L2 normalization aligned with the FLA library's ``l2norm`` kernel.

    Computes ``x / sqrt(sum(x^2) + eps)`` in the input dtype.  Unlike the fused
    ``triton_core.l2norm`` path (which accumulates in fp32 inside the kernel),
    this eager implementation runs ``(x*x).sum()`` in the input dtype.
    """
    original_dtype = x.dtype
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    # Restore original dtype — ``torch.rsqrt`` promotes bf16 inputs to fp32.
    return (x * inv_norm).to(original_dtype)


__all__ = [
    "flash_gated_delta_rule",
    "flash_chunk_gated_delta_rule_fwd",
    "flash_chunk_gated_delta_rule_bwd",
    "prepare_chunk_indices",
    "prepare_chunk_indices_list",
    "precompute_varlen_metadata",
]
