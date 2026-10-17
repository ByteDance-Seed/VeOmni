# Copyright © 2026 Huawei Technologies Co., Ltd.
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
#
# This file contains code copied and/or modified from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li


import warnings

import torch
from fla_npu.ops.ascendc import chunk_kda_bwd as ascendc_chunk_kda_bwd
from fla_npu.ops.ascendc import chunk_kda_fwd as ascendc_chunk_kda_fwd
from triton_ascend_kernels.attention.fla.kda.chunk import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    chunk_kda_fwd,
    input_guard,
    l2norm_bwd,
    l2norm_fwd,
    prepare_chunk_indices,
)
from triton_ascend_kernels.attention.fla.kda.chunk import (
    chunk_kda_bwd as triton_chunk_kda_bwd,
)


def _fla_npu_compat_canonical_chunk_indices_none():
    """fla-npu @3f4c016 compat (kept local so site-packages stays pristine).

    chunk_kda_bwd crashes on the padded_tail path when a single packed
    sequence is used (cu=None): after padding it calls
    _canonical_chunk_indices(None, ...) unconditionally. Upstream guards
    every other call site with "cu is not None"; only this one is missed.
    Make the helper tolerate None (return None) so the launch takes the
    non-varlen path. Remove when upstream fixes it.
    """
    try:
        from fla_npu.ops.ascendc import _stable
    except Exception:
        return
    orig = getattr(_stable, "_canonical_chunk_indices", None)
    if orig is None or getattr(orig, "_k3_none_safe", False):
        return

    def _canonical_chunk_indices(cu_seqlens, chunk_size, *args, **kwargs):
        if cu_seqlens is None:
            return None
        return orig(cu_seqlens, chunk_size, *args, **kwargs)

    _canonical_chunk_indices._k3_none_safe = True
    _stable._canonical_chunk_indices = _canonical_chunk_indices


_fla_npu_compat_canonical_chunk_indices_none()


def fused_beta_sigmoid_fwd(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return (torch.sigmoid(x.float()) * scale).to(x.dtype)


def fused_beta_sigmoid_bwd(x: torch.Tensor, dy: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    x = x.float()
    s = torch.sigmoid(x)
    return (dy.float() * scale * s * (1.0 - s)).to(x.dtype)


def fused_beta_sigmoid(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Compute ``scale * sigmoid(x)``.

    Used to map raw beta logits into ``[0, scale]`` before launching the chunk
    kernel when ``use_beta_sigmoid_in_kernel=True``.
    """
    return fused_beta_sigmoid_fwd(x, scale)


def _host_int_tuple(value):
    """Convert tensor metadata to one flat Host tuple with at most one D2H."""

    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
    if hasattr(value, "tolist"):
        value = value.tolist()
    return tuple(int(item) for item in value)


def _bnsd_to_bsnd(tensor):
    """Restore the sequence-major layout consumed by the Triton backward."""

    if tensor is None:
        return None
    if tensor.dim() != 4:
        raise RuntimeError(
            f"AscendC KDA saved tensors must be rank-4 BNSD tensors, but received shape {tuple(tensor.shape)}."
        )
    return tensor.permute(0, 2, 1, 3).contiguous()


def _bsnd_to_bnsd(tensor):
    if tensor is None:
        return None
    if tensor.dim() != 4:
        raise RuntimeError(f"Expected one rank-4 BSND tensor, got shape {tuple(tensor.shape)}.")
    return tensor.permute(0, 2, 1, 3).contiguous()


def _bsh_to_bhs(tensor):
    if tensor is None:
        return None
    if tensor.dim() != 3:
        raise RuntimeError(f"Expected one rank-3 BSH tensor, got shape {tuple(tensor.shape)}.")
    return tensor.permute(0, 2, 1).contiguous()


def _bhs_to_bsh(tensor):
    if tensor is None:
        return None
    if tensor.dim() != 3:
        raise RuntimeError(f"Expected one rank-3 BHS tensor, got shape {tuple(tensor.shape)}.")
    return tensor.permute(0, 2, 1).contiguous()


def _bsnd_to_ntd(tensor):
    if tensor is None:
        return None
    if tensor.dim() != 4 or tensor.shape[0] != 1:
        raise RuntimeError(
            f"Varlen KDA tensors must use packed BSND with B=1, but received shape {tuple(tensor.shape)}."
        )
    return tensor.squeeze(0).permute(1, 0, 2).contiguous()


def _bsh_to_nt(tensor):
    if tensor is None:
        return None
    if tensor.dim() != 3 or tensor.shape[0] != 1:
        raise RuntimeError(
            f"Varlen KDA tensors must use packed BSH with B=1, but received shape {tuple(tensor.shape)}."
        )
    return tensor.squeeze(0).transpose(0, 1).contiguous()


def _ntd_to_bsnd(tensor):
    if tensor is None:
        return None
    if tensor.dim() != 3:
        raise RuntimeError(f"Expected one rank-3 NTD tensor, got shape {tuple(tensor.shape)}.")
    return tensor.permute(1, 0, 2).unsqueeze(0).contiguous()


def _nt_to_bsh(tensor):
    if tensor is None:
        return None
    if tensor.dim() != 2:
        raise RuntimeError(f"Expected one rank-2 NT tensor, got shape {tuple(tensor.shape)}.")
    return tensor.transpose(0, 1).unsqueeze(0).contiguous()


def _varlen_saved_tensor(tensor):
    if tensor is None:
        return None
    if tensor.shape[0] != 1:
        raise RuntimeError(
            "Varlen AscendC intermediates must have a leading singleton batch dimension, "
            f"but received shape {tuple(tensor.shape)}."
        )
    return tensor.squeeze(0)


def _host_chunk_count(cu_seqlens, chunk_size):
    return sum((end - start + chunk_size - 1) // chunk_size for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]))


def _can_use_fused_kda_bwd(
    q,
    k,
    v,
    g_input,
    beta,
    gk,
    Aqk,
    Akk,
    w,
    qg,
    kg,
    v_new,
    h,
    *,
    forward_used_ascendc,
    disable_recompute,
    return_intermediate_states,
    chunk_size,
    initial_state,
    output_final_state,
    state_v_first,
    use_gate_in_kernel,
    A_log,
    dt_bias,
    cu_seqlens_host,
    safe_gate,
    lower_bound,
):
    if (
        ascendc_chunk_kda_bwd is None
        or not forward_used_ascendc
        or return_intermediate_states
        or chunk_size != 64
        or initial_state is not None
        or output_final_state
        or state_v_first
    ):
        return False

    if q.dim() != 4 or k.shape != q.shape or v.dim() != 4 or beta.dim() != 3:
        return False
    batch, tokens, heads, key_dim = q.shape
    if v.shape[:2] != (batch, tokens) or beta.shape != (batch, tokens, v.shape[2]):
        return False
    value_heads, value_dim = v.shape[2:]
    if key_dim != 128 or value_dim not in (128, 256) or heads != value_heads:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if beta.dtype not in (torch.bfloat16, torch.float32):
        return False

    expected_key = (batch, value_heads, tokens, key_dim)
    expected_value = (batch, value_heads, tokens, value_dim)
    expected_chunk = (batch, value_heads, tokens, chunk_size)

    intermediates = (
        (gk, expected_key, torch.float32),
        (Aqk, expected_chunk, q.dtype),
        (Akk, expected_chunk, q.dtype),
        (w, expected_key, q.dtype),
        (qg, expected_key, q.dtype),
        (kg, expected_key, q.dtype),
        (v_new, expected_value, q.dtype),
    )

    if any(
        tensor is None or tuple(tensor.shape) != shape or tensor.dtype != dtype or not tensor.is_contiguous()
        for tensor, shape, dtype in intermediates
    ):
        return False

    if cu_seqlens_host is None:
        chunk_count = (tokens + chunk_size - 1) // chunk_size
        expected_h = (batch, chunk_count, value_heads, key_dim, value_dim)
    else:
        if batch != 1 or len(cu_seqlens_host) < 2:
            return False
        if (
            cu_seqlens_host[0] != 0
            or cu_seqlens_host[-1] != tokens
            or any(start > end for start, end in zip(cu_seqlens_host, cu_seqlens_host[1:]))
        ):
            return False
        chunk_count = _host_chunk_count(cu_seqlens_host, chunk_size)
        expected_h = (1, chunk_count, value_heads, key_dim, value_dim)
    if h is None or tuple(h.shape) != expected_h or h.dtype != q.dtype or not h.is_contiguous():
        return False

    if not use_gate_in_kernel:
        return True
    if g_input.shape != q.shape or g_input.dtype not in (torch.bfloat16, torch.float32):
        return False
    if A_log is None or A_log.dtype != torch.float32 or tuple(A_log.shape) != (heads,):
        return False
    if dt_bias is not None and (dt_bias.dtype != torch.float32 or dt_bias.numel() != heads * key_dim):
        return False
    effective_lower_bound = -5.0 if lower_bound is None else lower_bound
    if safe_gate and not -5.0 <= effective_lower_bound < 0.0:
        return False
    if cu_seqlens_host is not None and chunk_count > 1024:
        return False
    return True


class ChunkKDAFunction(torch.autograd.Function):
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
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        chunk_size: int = 64,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
        transpose_state_layout: bool = False,
    ):
        # Apply l2norm
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        beta_raw = beta
        if use_beta_sigmoid_in_kernel:
            beta = fused_beta_sigmoid(beta_raw, scale=2.0 if allow_neg_eigval else 1.0)

        chunk_indices = None
        g_input = g
        use_fused_kda_bwd = False
        forward_used_ascendc = False
        cu_seqlens_host = None

        use_triton = chunk_size != 64

        if use_triton:
            if cu_seqlens is not None:
                chunk_indices = prepare_chunk_indices(
                    cu_seqlens,
                    chunk_size,
                    cu_seqlens_cpu=cu_seqlens_cpu,
                )
            (o, final_state, g_cumsum, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state) = chunk_kda_fwd(
                q=q,
                k=k,
                v=v,
                g=g_input,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                chunk_indices=chunk_indices,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
                use_gate_in_kernel=use_gate_in_kernel,
                A_log=A_log,
                dt_bias=dt_bias,
                chunk_size=chunk_size,
                disable_recompute=disable_recompute,
                return_intermediate_states=return_intermediate_states,
                # state_v_first=state_v_first,
            )
        else:
            # The AscendC Python entry consumes Host metadata. Prefer the CPU
            # copy already prepared by the model to avoid another D2H.
            cu_seqlens_host = _host_int_tuple(cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens)
            forward_used_ascendc = True

            (
                o,
                final_state,
                g_cumsum,
                Aqk,
                Akk,
                w,
                u,
                qg,
                kg,
                v_new,
                h,
                initial_state,
            ) = ascendc_chunk_kda_fwd(
                q,
                k,
                v,
                g_input,
                beta,
                float(scale),
                int(chunk_size),
                layout="BSND",
                initial_state=initial_state,
                output_final_state=bool(output_final_state),
                cu_seqlens=cu_seqlens_host,
                chunk_indices=None,
                safe_gate=bool(safe_gate),
                lower_bound=lower_bound,
                use_gate_in_kernel=bool(use_gate_in_kernel),
                A_log=A_log if use_gate_in_kernel else None,
                dt_bias=(dt_bias.reshape(-1) if dt_bias is not None else None) if use_gate_in_kernel else None,
                disable_recompute=bool(disable_recompute),
                return_intermediate_states=bool(return_intermediate_states),
                state_v_first=bool(state_v_first),
            )

            use_fused_kda_bwd = _can_use_fused_kda_bwd(
                q,
                k,
                v,
                g_input,
                beta,
                g_cumsum,
                Aqk,
                Akk,
                w,
                qg,
                kg,
                v_new,
                h,
                forward_used_ascendc=forward_used_ascendc,
                disable_recompute=disable_recompute,
                return_intermediate_states=return_intermediate_states,
                chunk_size=chunk_size,
                initial_state=initial_state,
                output_final_state=output_final_state,
                state_v_first=state_v_first,
                use_gate_in_kernel=use_gate_in_kernel,
                A_log=A_log,
                dt_bias=dt_bias,
                cu_seqlens_host=cu_seqlens_host,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
            )

            if use_fused_kda_bwd:
                # The fused backward consumes the AscendC head-major outputs
                # directly and does not use the forward-only u intermediate.
                u = None
            elif not return_intermediate_states:
                # Triton backward consumes BSND intermediates.
                g_cumsum = _bnsd_to_bsnd(g_cumsum)
                Aqk = _bnsd_to_bsnd(Aqk)
                Akk = _bnsd_to_bsnd(Akk)
                w = _bnsd_to_bsnd(w)
                u = _bnsd_to_bsnd(u)
                qg = _bnsd_to_bsnd(qg)
                kg = _bnsd_to_bsnd(kg)
                v_new = _bnsd_to_bsnd(v_new)

        if (
            not use_fused_kda_bwd
            and not return_intermediate_states
            and cu_seqlens is not None
            and chunk_indices is None
        ):
            chunk_indices = prepare_chunk_indices(
                cu_seqlens,
                chunk_size,
                cu_seqlens_cpu=cu_seqlens_cpu,
            )

        if return_intermediate_states:
            assert torch.is_inference_mode_enabled(), "return_intermediate_states is only allowed in inference mode"
            assert disable_recompute is False, "return_intermediate_states must be used with disable_recompute=False"
            return o.type_as(q), final_state, h

        saved_cu_seqlens = None if use_fused_kda_bwd else cu_seqlens
        saved_chunk_indices = None if use_fused_kda_bwd else chunk_indices
        ctx.save_for_backward(
            q,
            q_rstd,
            k,
            k_rstd,
            v,
            g_cumsum,
            g_input,
            beta_raw,
            beta,
            A_log,
            dt_bias,
            Aqk,
            Akk,
            w,
            u,
            qg,
            kg,
            v_new,
            h,
            initial_state,
            saved_cu_seqlens,
            saved_chunk_indices,
        )
        ctx.chunk_size = chunk_size
        ctx.safe_gate = safe_gate
        ctx.scale = scale
        ctx.lower_bound = lower_bound
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
        ctx.allow_neg_eigval = allow_neg_eigval
        ctx.disable_recompute = disable_recompute
        ctx.state_v_first = state_v_first
        ctx.use_fused_kda_bwd = use_fused_kda_bwd
        ctx.cu_seqlens_host = cu_seqlens_host if use_fused_kda_bwd else None
        return o.type_as(q), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor,
    ):
        (
            q,
            q_rstd,
            k,
            k_rstd,
            v,
            g_cumsum,
            g_input,
            beta_raw,
            beta,
            A_log,
            dt_bias,
            Aqk,
            Akk,
            w,
            u,
            qg,
            kg,
            v_new,
            h,
            initial_state,
            cu_seqlens,
            chunk_indices,
        ) = ctx.saved_tensors

        if ctx.use_fused_kda_bwd:
            if dht is not None:
                raise RuntimeError("Fused AscendC KDA backward does not support final-state gradients.")

            is_varlen = ctx.cu_seqlens_host is not None
            if is_varlen:
                q_head = _bsnd_to_ntd(q)
                k_head = _bsnd_to_ntd(k)
                v_head = _bsnd_to_ntd(v)
                beta_head = _bsh_to_nt(beta)
                do_head = _bsnd_to_ntd(do)
                raw_g_head = _bsnd_to_ntd(g_input) if ctx.use_gate_in_kernel else None
                gk_head = _varlen_saved_tensor(g_cumsum)
                Aqk_head = _varlen_saved_tensor(Aqk)
                Akk_head = _varlen_saved_tensor(Akk)
                w_head = _varlen_saved_tensor(w)
                qg_head = _varlen_saved_tensor(qg)
                kg_head = _varlen_saved_tensor(kg)
                v_new_head = _varlen_saved_tensor(v_new)
                h_head = _varlen_saved_tensor(h)
            else:
                q_head = _bsnd_to_bnsd(q)
                k_head = _bsnd_to_bnsd(k)
                v_head = _bsnd_to_bnsd(v)
                beta_head = _bsh_to_bhs(beta)
                do_head = _bsnd_to_bnsd(do)
                raw_g_head = _bsnd_to_bnsd(g_input) if ctx.use_gate_in_kernel else None
                gk_head, Aqk_head, Akk_head = g_cumsum, Aqk, Akk
                w_head, qg_head, kg_head = w, qg, kg
                v_new_head, h_head = v_new, h

            gate_A_log = A_log if ctx.use_gate_in_kernel else None
            gate_dt_bias = None
            if ctx.use_gate_in_kernel and dt_bias is not None:
                gate_dt_bias = dt_bias.reshape(q.shape[2], q.shape[3]).contiguous()

            dq_head, dk_head, dv_head, db_head, dg_head, dh0, dA, dbias = ascendc_chunk_kda_bwd(
                q_head,
                k_head,
                v_head,
                beta_head,
                gk_head,
                Aqk_head,
                Akk_head,
                w_head,
                qg_head,
                kg_head,
                v_new_head,
                h_head,
                do_head,
                float(ctx.scale),
                raw_g=raw_g_head,
                A_log=gate_A_log,
                dt_bias=gate_dt_bias,
                initial_state=None,
                dht=None,
                cu_seqlens=ctx.cu_seqlens_host,
                chunk_indices=None,
                chunk_size=ctx.chunk_size,
                safe_gate=ctx.safe_gate,
                lower_bound=ctx.lower_bound,
                use_gate_in_kernel=ctx.use_gate_in_kernel,
                disable_recompute=True,
                use_exp2=True,
                state_v_first=False,
            )

            if is_varlen:
                dq, dk, dv = map(_ntd_to_bsnd, (dq_head, dk_head, dv_head))
                db, dg = _nt_to_bsh(db_head), _ntd_to_bsnd(dg_head)
            else:
                dq, dk, dv = map(_bnsd_to_bsnd, (dq_head, dk_head, dv_head))
                db, dg = _bhs_to_bsh(db_head), _bnsd_to_bsnd(dg_head)
            if dbias is not None:
                dbias = dbias.reshape(dt_bias.shape)
        else:
            dq, dk, dv, db, dg, dh0, dA, dbias = triton_chunk_kda_bwd(
                q=q,
                k=k,
                v=v,
                beta=beta,
                Aqk=Aqk,
                Akk=Akk,
                scale=ctx.scale,
                initial_state=initial_state,
                do=do,
                dht=dht,
                g=g_cumsum,
                g_org=g_input if ctx.use_gate_in_kernel else None,
                state_v_first=ctx.state_v_first,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_size=ctx.chunk_size,
                safe_gate=ctx.safe_gate,
                lower_bound=ctx.lower_bound,
                use_gate_in_kernel=ctx.use_gate_in_kernel,
                A_log=A_log,
                dt_bias=dt_bias,
                disable_recompute=ctx.disable_recompute,
                w=w,
                u=u,
                qg=qg,
                kg=kg,
                v_new=v_new,
                h=h,
            )
        if dbias is not None and dt_bias is not None and dbias.shape != dt_bias.shape:
            dbias = dbias.reshape(dt_bias.shape)

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        if ctx.use_beta_sigmoid_in_kernel:
            db = fused_beta_sigmoid_bwd(beta_raw, db, scale=2.0 if ctx.allow_neg_eigval else 1.0)

        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg.to(g_input),
            db.to(beta_raw),
            dA,
            dbias,
            None,
            dh0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@torch.compiler.disable
def chunk_kda_ascendc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    **kwargs,
):
    if "transpose_state_layout" in kwargs:
        if state_v_first:
            raise ValueError("Cannot pass both `state_v_first` and the deprecated `transpose_state_layout`.")
        warnings.warn(
            "`transpose_state_layout` is deprecated and renamed to `state_v_first`.",
            DeprecationWarning,
            stacklevel=2,
        )
        state_v_first = kwargs.pop("transpose_state_layout")

    # state_v_first is deprecated in favor of the explicit argument, but we keep
    # the value as False for the AscendC implementation.
    state_v_first = False

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be in float32."

    A_log, dt_bias = None, None
    if use_gate_in_kernel:
        assert "A_log" in kwargs, "A_log must be provided when use_gate_in_kernel=True."
        A_log, dt_bias = kwargs["A_log"], kwargs.get("dt_bias")
        if A_log.dtype != torch.float32:
            A_log = A_log.float()
        if dt_bias is not None and dt_bias.dtype != torch.float32:
            dt_bias = dt_bias.float()

    chunk_size = kwargs.pop("chunk_size", 64)
    if chunk_size not in (32, 64):
        raise ValueError(f"`chunk_size` must be either 32 or 64 for KDA, got {chunk_size}.")

    if safe_gate and use_gate_in_kernel:
        if lower_bound is None:
            raise ValueError("`lower_bound` must be specified when `safe_gate=True` and `use_gate_in_kernel=True`.")
        if not (-5 <= lower_bound < 0):
            raise ValueError(f"`lower_bound` must be in the safe range [-5, 0), got {lower_bound}.")

    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("`allow_neg_eigval=True` requires `use_beta_sigmoid_in_kernel=True`.")

    # Validate head dimensions for GVA
    B, T, H, K, HV = *q.shape, v.shape[2]
    assert q.shape == k.shape, f"q and k must have the same shape, got q={q.shape} vs k={k.shape}"
    assert K <= 256, f"Currently we only support key headdim <=256 for KDA, got {K}."
    assert HV % H == 0, (
        f"For GVA, num_v_heads (HV={HV}) must be evenly divisible by num_qk_heads (H={H}), but got HV % H = {HV % H}"
    )
    assert g.shape == (B, T, HV, K), f"g must have shape [B, T, HV, K]={[B, T, HV, K]}, got {list(g.shape)}"
    assert beta.shape == (B, T, HV), f"beta must have shape [B, T, HV]={[B, T, HV]}, got {list(beta.shape)}"

    if scale is None:
        scale = K**-0.5
    return ChunkKDAFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        state_v_first,
        cu_seqlens,
        cu_seqlens_cpu,
        safe_gate,
        lower_bound,
        chunk_size,
        disable_recompute,
        return_intermediate_states,
    )
