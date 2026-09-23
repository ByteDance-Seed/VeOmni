# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Selector-index QSA Triton implementation for Qwen4-Exp GPU attention."""

import torch
import triton
import triton.language as tl


def qsa_is_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_indices: torch.Tensor,
    selected_counts: torch.Tensor,
) -> bool:
    """Return whether tensors satisfy the CUDA QSA kernel contract."""
    return (
        q.device.type == "cuda"
        and k.device == q.device
        and v.device == q.device
        and q.dtype in (torch.float16, torch.bfloat16)
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and q.ndim == k.ndim == v.ndim == 4
        and q.shape[0] == k.shape[0] == v.shape[0]
        and q.shape[2:] == k.shape[2:] == v.shape[2:]
        and q.shape[1] % k.shape[1] == 0
        and selected_indices.device == q.device
        and selected_counts.device == q.device
        and selected_indices.dtype == torch.int32
        and selected_counts.dtype == torch.int32
        and selected_indices.ndim == 3
        and selected_counts.shape == selected_indices.shape[:2]
        and selected_indices.shape[:2] == q.shape[0:1] + q.shape[2:3]
    )


@triton.jit
def _qsa_fwd_tail_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    lse_ptr,
    idx_ptr,
    cnt_ptr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qt: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kt: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vt: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_ot: tl.constexpr,
    stride_od: tl.constexpr,
    stride_lb: tl.constexpr,
    stride_lh: tl.constexpr,
    stride_lt: tl.constexpr,
    stride_ib: tl.constexpr,
    stride_it: tl.constexpr,
    stride_is: tl.constexpr,
    stride_cb: tl.constexpr,
    stride_ct: tl.constexpr,
    sm_scale,
    T: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D: tl.constexpr,
    FULL_S: tl.constexpr,
    TAIL_N: tl.constexpr,
    S_MAX: tl.constexpr,
    GQA_N_REP: tl.constexpr,
):
    """Streaming-softmax forward over the per-query selector index list.

    Rows whose selector is empty (``cnt == 0``) contribute no keys, so the
    accumulator stays at its initial ``-inf`` running max; the final store maps
    those rows to a zero output instead of the ``-inf - -inf`` NaN.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    q_idx = tl.program_id(2)
    if q_idx >= T:
        return

    off_d = tl.arange(0, BLOCK_D)
    d_mask = off_d < D
    kv_head = pid_h // GQA_N_REP
    q_off = q_ptr + pid_b * stride_qb + pid_h * stride_qh + q_idx * stride_qt
    q = tl.load(q_off + off_d * stride_qd, mask=d_mask, other=0.0)
    q_f32 = q.to(tl.float32)
    cnt = tl.load(cnt_ptr + pid_b * stride_cb + q_idx * stride_ct)
    cnt = tl.minimum(cnt, S_MAX)
    k_base = k_ptr + pid_b * stride_kb + kv_head * stride_kh
    v_base = v_ptr + pid_b * stride_vb + kv_head * stride_vh
    idx_base = idx_ptr + pid_b * stride_ib + q_idx * stride_it

    m_i = tl.full([1], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    off_n = tl.arange(0, BLOCK_N)

    for blk_start in range(0, FULL_S, BLOCK_N):
        n_mask = blk_start + off_n < cnt
        sel_idx = tl.load(idx_base + (blk_start + off_n) * stride_is, mask=n_mask, other=T)
        sel_mask = (sel_idx < T) & n_mask
        k_ptrs = k_base + sel_idx[:, None] * stride_kt + off_d[None, :] * stride_kd
        v_ptrs = v_base + sel_idx[:, None] * stride_vt + off_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=sel_mask[:, None] & d_mask[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=sel_mask[:, None] & d_mask[None, :], other=0.0)
        score = tl.sum(q_f32[None, :] * k.to(tl.float32), axis=1) * sm_scale
        score = tl.where(sel_mask, score, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(score, axis=0, keep_dims=True))
        # Guard the running max against the empty/all-masked case: when m_ij is
        # still -inf, ``score - m_ij`` would be ``-inf - -inf = NaN`` and poison
        # the acc/l_i reductions. Subtracting 0.0 instead keeps masked scores at
        # ``exp(-inf) = 0`` so empty rows accumulate nothing.
        m_ij_safe = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        prob = tl.exp(score - m_ij_safe)
        alpha = tl.exp(m_i - m_ij_safe)
        acc = acc * alpha + tl.sum(prob[:, None] * v.to(tl.float32), axis=0)
        l_i = l_i * alpha + tl.sum(prob, axis=0, keep_dims=True)
        m_i = m_ij

    if TAIL_N > 0:
        off_tail = tl.arange(0, TAIL_N)
        n_mask = FULL_S + off_tail < cnt
        sel_idx = tl.load(idx_base + (FULL_S + off_tail) * stride_is, mask=n_mask, other=T)
        sel_mask = (sel_idx < T) & n_mask
        k_ptrs = k_base + sel_idx[:, None] * stride_kt + off_d[None, :] * stride_kd
        v_ptrs = v_base + sel_idx[:, None] * stride_vt + off_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=sel_mask[:, None] & d_mask[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=sel_mask[:, None] & d_mask[None, :], other=0.0)
        score = tl.sum(q_f32[None, :] * k.to(tl.float32), axis=1) * sm_scale
        score = tl.where(sel_mask, score, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(score, axis=0, keep_dims=True))
        m_ij_safe = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        prob = tl.exp(score - m_ij_safe)
        alpha = tl.exp(m_i - m_ij_safe)
        acc = acc * alpha + tl.sum(prob[:, None] * v.to(tl.float32), axis=0)
        l_i = l_i * alpha + tl.sum(prob, axis=0, keep_dims=True)
        m_i = m_ij

    o_off = out_ptr + pid_b * stride_ob + pid_h * stride_oh + q_idx * stride_ot
    # Empty rows never accumulated (l_i == 0); emit a finite zero instead of 0/0.
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    tl.store(o_off + off_d * stride_od, acc / safe_l, mask=d_mask)
    lse_off = lse_ptr + pid_b * stride_lb + pid_h * stride_lh + q_idx * stride_lt
    lse_val = tl.where(l_i > 0.0, m_i + tl.log(safe_l), float("-inf"))
    tl.store(lse_off + tl.arange(0, 1), lse_val)


@triton.jit
def _qsa_bwd_outdelta_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    do_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    lse_ptr,
    idx_ptr,
    cnt_ptr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qt: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kt: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vt: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_ot: tl.constexpr,
    stride_od: tl.constexpr,
    stride_dob: tl.constexpr,
    stride_doh: tl.constexpr,
    stride_dot: tl.constexpr,
    stride_dod: tl.constexpr,
    stride_dqb: tl.constexpr,
    stride_dqh: tl.constexpr,
    stride_dqt: tl.constexpr,
    stride_dqd: tl.constexpr,
    stride_dkb: tl.constexpr,
    stride_dkh: tl.constexpr,
    stride_dkt: tl.constexpr,
    stride_dkd: tl.constexpr,
    stride_dvb: tl.constexpr,
    stride_dvh: tl.constexpr,
    stride_dvt: tl.constexpr,
    stride_dvd: tl.constexpr,
    stride_lb: tl.constexpr,
    stride_lh: tl.constexpr,
    stride_lt: tl.constexpr,
    stride_ib: tl.constexpr,
    stride_it: tl.constexpr,
    stride_is: tl.constexpr,
    stride_cb: tl.constexpr,
    stride_ct: tl.constexpr,
    sm_scale,
    T: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D: tl.constexpr,
    FULL_S: tl.constexpr,
    TAIL_N: tl.constexpr,
    S_MAX: tl.constexpr,
    GQA_N_REP: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    q_idx = tl.program_id(2)
    if q_idx >= T:
        return

    off_d = tl.arange(0, BLOCK_D)
    d_mask = off_d < D
    kv_head = pid_h // GQA_N_REP
    q_off = q_ptr + pid_b * stride_qb + pid_h * stride_qh + q_idx * stride_qt
    o_off = out_ptr + pid_b * stride_ob + pid_h * stride_oh + q_idx * stride_ot
    do_off = do_ptr + pid_b * stride_dob + pid_h * stride_doh + q_idx * stride_dot
    dq_off = dq_ptr + pid_b * stride_dqb + pid_h * stride_dqh + q_idx * stride_dqt
    q = tl.load(q_off + off_d * stride_qd, mask=d_mask, other=0.0)
    q_f32 = q.to(tl.float32)
    out_f32 = tl.load(o_off + off_d * stride_od, mask=d_mask, other=0.0).to(tl.float32)
    do_f32 = tl.load(do_off + off_d * stride_dod, mask=d_mask, other=0.0).to(tl.float32)
    delta = tl.sum(do_f32 * out_f32, axis=0, keep_dims=True)
    lse = tl.load(lse_ptr + pid_b * stride_lb + pid_h * stride_lh + q_idx * stride_lt)
    cnt = tl.load(cnt_ptr + pid_b * stride_cb + q_idx * stride_ct)
    cnt = tl.minimum(cnt, S_MAX)
    k_base = k_ptr + pid_b * stride_kb + kv_head * stride_kh
    v_base = v_ptr + pid_b * stride_vb + kv_head * stride_vh
    dk_base = dk_ptr + pid_b * stride_dkb + kv_head * stride_dkh
    dv_base = dv_ptr + pid_b * stride_dvb + kv_head * stride_dvh
    idx_base = idx_ptr + pid_b * stride_ib + q_idx * stride_it
    off_n = tl.arange(0, BLOCK_N)
    dq_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for blk_start in range(0, FULL_S, BLOCK_N):
        n_mask = blk_start + off_n < cnt
        sel_idx = tl.load(idx_base + (blk_start + off_n) * stride_is, mask=n_mask, other=T)
        sel_mask = (sel_idx < T) & n_mask
        k_ptrs = k_base + sel_idx[:, None] * stride_kt + off_d[None, :] * stride_kd
        v_ptrs = v_base + sel_idx[:, None] * stride_vt + off_d[None, :] * stride_vd
        key = tl.load(k_ptrs, mask=sel_mask[:, None] & d_mask[None, :], other=0.0)
        value = tl.load(v_ptrs, mask=sel_mask[:, None] & d_mask[None, :], other=0.0)
        score = tl.sum(q_f32[None, :] * key.to(tl.float32), axis=1) * sm_scale
        prob = tl.where(sel_mask, tl.exp(score - lse), 0.0)
        dprob = tl.sum(do_f32[None, :] * value.to(tl.float32), axis=1)
        dscore = tl.where(sel_mask, (dprob - delta) * prob * sm_scale, 0.0)
        dq_acc += tl.sum(dscore[:, None] * key.to(tl.float32), axis=0)
        dk_ptrs = dk_base + sel_idx[:, None] * stride_dkt + off_d[None, :] * stride_dkd
        dv_ptrs = dv_base + sel_idx[:, None] * stride_dvt + off_d[None, :] * stride_dvd
        write_mask = sel_mask[:, None] & d_mask[None, :]
        tl.atomic_add(dk_ptrs, dscore[:, None] * q_f32[None, :], mask=write_mask)
        tl.atomic_add(dv_ptrs, prob[:, None] * do_f32[None, :], mask=write_mask)

    if TAIL_N > 0:
        off_tail = tl.arange(0, TAIL_N)
        tail_mask = FULL_S + off_tail < cnt
        tail_idx = tl.load(idx_base + (FULL_S + off_tail) * stride_is, mask=tail_mask, other=T)
        tail_sel = (tail_idx < T) & tail_mask
        tail_k_ptrs = k_base + tail_idx[:, None] * stride_kt + off_d[None, :] * stride_kd
        tail_v_ptrs = v_base + tail_idx[:, None] * stride_vt + off_d[None, :] * stride_vd
        tail_key = tl.load(tail_k_ptrs, mask=tail_sel[:, None] & d_mask[None, :], other=0.0)
        tail_value = tl.load(tail_v_ptrs, mask=tail_sel[:, None] & d_mask[None, :], other=0.0)
        tail_score = tl.sum(q_f32[None, :] * tail_key.to(tl.float32), axis=1) * sm_scale
        tail_prob = tl.where(tail_sel, tl.exp(tail_score - lse), 0.0)
        tail_dprob = tl.sum(do_f32[None, :] * tail_value.to(tl.float32), axis=1)
        tail_dscore = tl.where(tail_sel, (tail_dprob - delta) * tail_prob * sm_scale, 0.0)
        dq_acc += tl.sum(tail_dscore[:, None] * tail_key.to(tl.float32), axis=0)
        tail_dk_ptrs = dk_base + tail_idx[:, None] * stride_dkt + off_d[None, :] * stride_dkd
        tail_dv_ptrs = dv_base + tail_idx[:, None] * stride_dvt + off_d[None, :] * stride_dvd
        tail_write_mask = tail_sel[:, None] & d_mask[None, :]
        tl.atomic_add(tail_dk_ptrs, tail_dscore[:, None] * q_f32[None, :], mask=tail_write_mask)
        tl.atomic_add(tail_dv_ptrs, tail_prob[:, None] * do_f32[None, :], mask=tail_write_mask)

    tl.store(dq_off + off_d * stride_dqd, dq_acc.to(q.dtype), mask=d_mask)


class _QSASparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, selected_indices, selected_counts, sm_scale=None):
        batch, heads, seqlen, dim = q.shape
        kv_heads = k.shape[1]
        gqa_n_rep = heads // kv_heads
        if sm_scale is None:
            sm_scale = dim**-0.5
        s_max = selected_indices.shape[-1]
        full_s = (s_max // 64) * 64
        tail_n = triton.next_power_of_2(s_max - full_s) if s_max != full_s else 0
        out = torch.empty_like(q)
        lse = torch.empty(batch, heads, seqlen, dtype=torch.float32, device=q.device)
        _qsa_fwd_tail_kernel[(batch, heads, seqlen)](
            q,
            k,
            v,
            out,
            lse,
            selected_indices,
            selected_counts,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            selected_indices.stride(0),
            selected_indices.stride(1),
            selected_indices.stride(2),
            selected_counts.stride(0),
            selected_counts.stride(1),
            sm_scale,
            seqlen,
            BLOCK_N=64,
            BLOCK_D=triton.next_power_of_2(dim),
            D=dim,
            FULL_S=full_s,
            TAIL_N=tail_n,
            S_MAX=s_max,
            GQA_N_REP=gqa_n_rep,
        )
        ctx.save_for_backward(q, k, v, out, lse, selected_indices, selected_counts)
        ctx.sm_scale = sm_scale
        return out

    @staticmethod
    def backward(ctx, d_out):
        q, k, v, out, lse, selected_indices, selected_counts = ctx.saved_tensors
        batch, heads, seqlen, dim = q.shape
        kv_heads = k.shape[1]
        gqa_n_rep = heads // kv_heads
        d_q = torch.zeros_like(q)
        d_k = torch.zeros(batch, kv_heads, seqlen, dim, dtype=torch.float32, device=k.device)
        d_v = torch.zeros(batch, kv_heads, seqlen, dim, dtype=torch.float32, device=v.device)
        s_max = selected_indices.shape[-1]
        full_s = (s_max // 64) * 64
        tail_n = triton.next_power_of_2(s_max - full_s) if s_max != full_s else 0
        _qsa_bwd_outdelta_kernel[(batch, heads, seqlen)](
            q,
            k,
            v,
            out,
            d_out,
            d_q,
            d_k,
            d_v,
            lse,
            selected_indices,
            selected_counts,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            d_out.stride(0),
            d_out.stride(1),
            d_out.stride(2),
            d_out.stride(3),
            d_q.stride(0),
            d_q.stride(1),
            d_q.stride(2),
            d_q.stride(3),
            d_k.stride(0),
            d_k.stride(1),
            d_k.stride(2),
            d_k.stride(3),
            d_v.stride(0),
            d_v.stride(1),
            d_v.stride(2),
            d_v.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            selected_indices.stride(0),
            selected_indices.stride(1),
            selected_indices.stride(2),
            selected_counts.stride(0),
            selected_counts.stride(1),
            ctx.sm_scale,
            seqlen,
            BLOCK_N=64,
            BLOCK_D=triton.next_power_of_2(dim),
            D=dim,
            FULL_S=full_s,
            TAIL_N=tail_n,
            S_MAX=s_max,
            GQA_N_REP=gqa_n_rep,
            maxnreg=128,
            num_warps=8,
        )
        return d_q.to(q.dtype), d_k.to(k.dtype), d_v.to(v.dtype), None, None, None


def qsa_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_indices: torch.Tensor,
    selected_counts: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Compute Qwen4-Exp selector-index sparse attention on CUDA."""
    if not qsa_is_supported(q, k, v, selected_indices, selected_counts):
        raise ValueError("QSA Triton received unsupported tensor shapes, dtypes, or devices.")
    return _QSASparseAttention.apply(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        selected_indices.contiguous(),
        selected_counts.contiguous(),
        sm_scale,
    )
