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

"""dLLM DG_PREFIX Triton attention: each packed segment is ``H_prefix || current_canvas``.

The public entry is ``dllm_attention``. Visibility is decided by the caller
before the launch; inside a segment every query attends every key.
"""

import torch
import triton
import triton.language as tl


def _dllm_attention_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor = None,
    cu_seqlens_k: torch.Tensor = None,
    max_seqlen_q: int = None,
    max_seqlen_k: int = None,
    sm_scale: float = None,
):
    """Low-level packed DG_PREFIX entry point.

    Each packed segment must use the physical layout::

        Q  = noisy current canvas
        KV = clean H_prefix || noisy current canvas

    ``H_prefix`` must already be truncated before calling this function. The
    kernel intentionally applies no causal or block mask inside a segment:
    every current-canvas query attends bidirectionally to that full segment.
    Different segments may have different prefix and canvas lengths.
    """
    return DLMAttentionDraft.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        sm_scale,
    )


def pack_dg_prefix_kv(
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    canvas_k: torch.Tensor,
    canvas_v: torch.Tensor,
    cu_seqlens_prefix: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
):
    """Interleave authoritative ``prefix || current_canvas`` KV segments."""
    if prefix_k.shape != prefix_v.shape:
        raise ValueError("DG_PREFIX prefix K/V shapes must match")
    if canvas_k.shape != canvas_v.shape:
        raise ValueError("DG_PREFIX canvas K/V shapes must match")
    if prefix_k.ndim != 4 or canvas_k.ndim != 4:
        raise ValueError("DG_PREFIX projected K/V tensors must have shape (1, heads, tokens, dim)")
    if prefix_k.shape[:2] != canvas_k.shape[:2] or prefix_k.shape[-1] != canvas_k.shape[-1]:
        raise ValueError("DG_PREFIX prefix and canvas K/V head layouts must match")
    if prefix_k.shape[0] != 1 or canvas_k.shape[0] != 1:
        raise ValueError("DG_PREFIX projected K/V must already use the packed batch=1 ABI")
    cu_seqlens_prefix = cu_seqlens_prefix.to(device=prefix_k.device, dtype=torch.int32)
    cu_seqlens_q = cu_seqlens_q.to(device=canvas_k.device, dtype=torch.int32)
    if cu_seqlens_prefix.ndim != 1 or cu_seqlens_prefix.shape != cu_seqlens_q.shape:
        raise ValueError("DG_PREFIX prefix and canvas cu_seqlens must have equal one-dimensional shapes")
    if int(cu_seqlens_prefix[0].item()) != 0 or int(cu_seqlens_q[0].item()) != 0:
        raise ValueError("DG_PREFIX cu_seqlens must start at zero")
    if int(cu_seqlens_prefix[-1].item()) != prefix_k.shape[2]:
        raise ValueError("cu_seqlens_prefix does not cover the packed prefix tensor")
    if int(cu_seqlens_q[-1].item()) != canvas_k.shape[2]:
        raise ValueError("cu_seqlens_q does not cover the packed canvas tensor")
    if bool(torch.any(cu_seqlens_q.diff() <= 0).item()):
        raise ValueError("Every DG_PREFIX canvas segment must be non-empty")

    k_parts = []
    v_parts = []
    k_lens = []
    for segment in range(cu_seqlens_q.numel() - 1):
        ps, pe = int(cu_seqlens_prefix[segment].item()), int(cu_seqlens_prefix[segment + 1].item())
        qs, qe = int(cu_seqlens_q[segment].item()), int(cu_seqlens_q[segment + 1].item())
        k_parts.extend([prefix_k[:, :, ps:pe, :], canvas_k[:, :, qs:qe, :]])
        v_parts.extend([prefix_v[:, :, ps:pe, :], canvas_v[:, :, qs:qe, :]])
        k_lens.append((pe - ps) + (qe - qs))
    cu_seqlens_k = torch.tensor(
        [0] + list(torch.tensor(k_lens, dtype=torch.long).cumsum(0).tolist()),
        device=canvas_k.device,
        dtype=torch.int32,
    )
    return torch.cat(k_parts, dim=2), torch.cat(v_parts, dim=2), cu_seqlens_k


def dllm_attention(
    q: torch.Tensor,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    canvas_k: torch.Tensor,
    canvas_v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_prefix: torch.Tensor,
    max_seqlen_q: int = None,
    max_seqlen_k: int = None,
    sm_scale: float = None,
):
    """Safe DiffusionGemma single-canvas attention from separate KV parts.

    This is the public DG_PREFIX API. It constructs every KV segment itself,
    so a canvas is necessarily at the tail and an accidental extra tensor
    cannot silently become visible future context.
    """
    if sm_scale is None:
        raise ValueError("DG_PREFIX requires explicit sm_scale; pass 1.0 after Gemma q_norm")
    if q.shape[0] != 1 or q.shape[2] != canvas_k.shape[2]:
        raise ValueError("DG_PREFIX Q must use packed batch=1 and align with current-canvas K/V")
    k, v, cu_seqlens_k = pack_dg_prefix_kv(
        prefix_k,
        prefix_v,
        canvas_k,
        canvas_v,
        cu_seqlens_prefix,
        cu_seqlens_q,
    )
    if max_seqlen_q is None:
        max_seqlen_q = int(cu_seqlens_q.diff().max().item())
    if max_seqlen_k is None:
        max_seqlen_k = int(cu_seqlens_k.diff().max().item())
    return _dllm_attention_packed(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        sm_scale=sm_scale,
    )


@triton.jit
def draft_fwd_kernel(
    Q,
    K,
    V,
    qk_scale,
    cu_seqlens_q,
    cu_seqlens_k,
    L,
    Out,
    stride_qh,
    stride_qt,
    stride_kh,
    stride_kt,
    stride_vh,
    stride_vt,
    Q_TOTAL_TOKENS,
    HEAD_Q: tl.constexpr,
    HEAD_KV: tl.constexpr,
    KV_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """Triton forward kernel for packed DG_PREFIX attention."""
    DOT_DTYPE = tl.bfloat16
    off_z = tl.program_id(0)
    off_h = tl.program_id(1)
    start_m = tl.program_id(2)

    off_kvh = off_h // KV_GROUP

    q_start = tl.load(cu_seqlens_q + off_z)
    q_len = tl.load(cu_seqlens_q + off_z + 1) - q_start

    k_start = tl.load(cu_seqlens_k + off_z)
    k_len = tl.load(cu_seqlens_k + off_z + 1) - k_start

    q_block_start = BLOCK_Q * start_m
    if q_block_start >= q_len:
        return

    q_base = Q + off_h * stride_qh + q_start * stride_qt
    q_ptr = tl.make_block_ptr(
        base=q_base,
        shape=(q_len, HEAD_DIM),
        strides=(stride_qt, 1),
        offsets=(q_block_start, 0),
        block_shape=(BLOCK_Q, HEAD_DIM),
        order=(1, 0),
    )

    o_base = Out + q_start * HEAD_Q * HEAD_DIM + off_h * HEAD_DIM
    o_ptr = tl.make_block_ptr(
        base=o_base,
        shape=(q_len, HEAD_DIM),
        strides=(HEAD_Q * HEAD_DIM, 1),
        offsets=(q_block_start, 0),
        block_shape=(BLOCK_Q, HEAD_DIM),
        order=(1, 0),
    )

    k_base = K + off_kvh * stride_kh + k_start * stride_kt
    k_ptr = tl.make_block_ptr(
        base=k_base,
        shape=(HEAD_DIM, k_len),
        strides=(1, stride_kt),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_KV),
        order=(0, 1),
    )

    v_base = V + off_kvh * stride_vh + k_start * stride_vt
    v_ptr = tl.make_block_ptr(
        base=v_base,
        shape=(k_len, HEAD_DIM),
        strides=(stride_vt, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_KV, HEAD_DIM),
        order=(1, 0),
    )

    q = tl.load(q_ptr, boundary_check=(0,), padding_option="zero")
    q_dot = q.to(DOT_DTYPE)

    offs_q = q_block_start + tl.arange(0, BLOCK_Q)
    mask_q = offs_q < q_len

    m_i = tl.full([BLOCK_Q], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    hi = k_len

    for kv_block_start in range(0, hi, BLOCK_KV):
        offs_k = kv_block_start + tl.arange(0, BLOCK_KV)
        mask_k = offs_k < k_len

        k = tl.load(k_ptr, boundary_check=(1,), padding_option="zero")
        v = tl.load(v_ptr, boundary_check=(0,), padding_option="zero")

        k_dot = k.to(DOT_DTYPE)
        v_dot = v.to(DOT_DTYPE)

        # The caller has physically truncated each KV segment to
        # H_prefix || current_noisy_canvas. Every decoder query may attend
        # to every key in that segment, including the whole current canvas.
        mask_final = mask_q[:, None] & mask_k[None, :]

        qk = tl.dot(q_dot, k_dot) * qk_scale
        qk = tl.where(mask_final, qk, -float("inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        is_all_masked = m_i_new == -float("inf")
        m_i_new_safe = tl.where(is_all_masked, 0.0, m_i_new)

        alpha = tl.math.exp2(m_i - m_i_new_safe)
        p = tl.math.exp2(qk - m_i_new_safe[:, None])

        alpha = tl.where(is_all_masked, 1.0, alpha)
        p = tl.where(is_all_masked[:, None], 0.0, p)

        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

        acc *= alpha[:, None]
        acc += tl.dot(p.to(DOT_DTYPE), v_dot)

        k_ptr = tl.advance(k_ptr, (0, BLOCK_KV))
        v_ptr = tl.advance(v_ptr, (BLOCK_KV, 0))

    acc = acc / (l_i[:, None] + 1e-20)
    l_ptrs = L + off_h * Q_TOTAL_TOKENS + q_start + offs_q
    is_all_masked = m_i == -float("inf")
    l_i_safe = tl.where(is_all_masked, 1.0, l_i)
    m_i_safe = tl.where(is_all_masked, 0.0, m_i)
    tl.store(l_ptrs, m_i_safe + tl.math.log2(l_i_safe), mask=mask_q)
    tl.store(o_ptr, acc.to(Out.dtype.element_ty), boundary_check=(0,))


@triton.jit
def _draft_bwd_preprocess(
    O,
    DO,
    Delta,
    BLOCK_Q: tl.constexpr,
    Q_TOTAL_TOKENS,
    HEAD_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    block_id = tl.program_id(0)
    head_id = tl.program_id(1)
    block_begin = block_id * BLOCK_Q
    pos_off = block_begin + tl.arange(0, BLOCK_Q)
    head_dim_off = tl.arange(0, HEAD_DIM)

    o_off = pos_off[:, None] * HEAD_Q * HEAD_DIM + head_id * HEAD_DIM + head_dim_off[None, :]
    delta_off = head_id * Q_TOTAL_TOKENS + pos_off

    o = tl.load(O + o_off, mask=pos_off[:, None] < Q_TOTAL_TOKENS).to(tl.float32)
    do = tl.load(DO + o_off, mask=pos_off[:, None] < Q_TOTAL_TOKENS).to(tl.float32)

    delta = tl.sum(o * do, axis=1)

    tl.store(Delta + delta_off, delta, mask=pos_off < Q_TOTAL_TOKENS)


@triton.jit
def _draft_bwd_dkv_kernel(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DK,
    DV,
    L,
    D,
    cu_seqlens_q,
    cu_seqlens_k,
    stride_qh,
    stride_qt,
    stride_kh,
    stride_kt,
    stride_vh,
    stride_vt,
    stride_dkh,
    stride_dkt,
    stride_dvh,
    stride_dvt,
    Q_TOTAL_TOKENS,
    HEAD_Q: tl.constexpr,
    HEAD_KV: tl.constexpr,
    KV_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    DOT_DTYPE = tl.bfloat16
    qk_scale = sm_scale * 1.44269504
    off_z = tl.program_id(0)
    off_kvh = tl.program_id(1)
    start_n = tl.program_id(2)

    q_start = tl.load(cu_seqlens_q + off_z)
    q_len = tl.load(cu_seqlens_q + off_z + 1) - q_start

    k_start = tl.load(cu_seqlens_k + off_z)
    k_len = tl.load(cu_seqlens_k + off_z + 1) - k_start

    k_block_start = BLOCK_KV * start_n
    if k_block_start >= k_len:
        return

    K_block_ptr = tl.make_block_ptr(
        base=K + off_kvh * stride_kh + k_start * stride_kt,
        shape=(k_len, HEAD_DIM),
        strides=(stride_kt, 1),
        offsets=(k_block_start, 0),
        block_shape=(BLOCK_KV, HEAD_DIM),
        order=(0, 1),
    )
    DK_block_ptr = tl.make_block_ptr(
        base=DK + off_kvh * stride_dkh + k_start * stride_dkt,
        shape=(k_len, HEAD_DIM),
        strides=(stride_dkt, 1),
        offsets=(k_block_start, 0),
        block_shape=(BLOCK_KV, HEAD_DIM),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + off_kvh * stride_vh + k_start * stride_vt,
        shape=(k_len, HEAD_DIM),
        strides=(stride_vt, 1),
        offsets=(k_block_start, 0),
        block_shape=(BLOCK_KV, HEAD_DIM),
        order=(0, 1),
    )
    DV_block_ptr = tl.make_block_ptr(
        base=DV + off_kvh * stride_dvh + k_start * stride_dvt,
        shape=(k_len, HEAD_DIM),
        strides=(stride_dvt, 1),
        offsets=(k_block_start, 0),
        block_shape=(BLOCK_KV, HEAD_DIM),
        order=(0, 1),
    )

    k = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
    v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

    k_dot = k.to(DOT_DTYPE)
    v_dot = v.to(DOT_DTYPE)

    dk = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    offs_k = k_block_start + tl.arange(0, BLOCK_KV)
    mask_k = offs_k < k_len

    qlo = 0

    for i in range(KV_GROUP):
        off_h = off_kvh * KV_GROUP + i
        D_ptrs = D + off_h * Q_TOTAL_TOKENS + q_start
        l_ptrs = L + off_h * Q_TOTAL_TOKENS + q_start

        DO_block_ptr = tl.make_block_ptr(
            base=DO + q_start * HEAD_Q * HEAD_DIM + off_h * HEAD_DIM,
            shape=(q_len, HEAD_DIM),
            strides=(HEAD_Q * HEAD_DIM, 1),
            offsets=(qlo, 0),
            block_shape=(BLOCK_Q, HEAD_DIM),
            order=(0, 1),
        )
        Q_block_ptr = tl.make_block_ptr(
            base=Q + off_h * stride_qh + q_start * stride_qt,
            shape=(q_len, HEAD_DIM),
            strides=(stride_qt, 1),
            offsets=(qlo, 0),
            block_shape=(BLOCK_Q, HEAD_DIM),
            order=(0, 1),
        )

        for q_block_start in range(qlo, q_len, BLOCK_Q):
            offs_q = q_block_start + tl.arange(0, BLOCK_Q)
            mask_q = offs_q < q_len

            q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
            q_dot = q.to(DOT_DTYPE)

            mask_final = mask_k[:, None] & mask_q[None, :]

            qk = tl.dot(k_dot, tl.trans(q_dot))
            qk = tl.where(mask_final, qk, -float("inf"))

            l_i = tl.load(l_ptrs + offs_q, mask=mask_q, other=0.0)
            p = tl.math.exp2(qk * qk_scale - l_i[None, :])

            do = tl.load(DO_block_ptr, boundary_check=(0,), padding_option="zero")
            do_dot = do.to(DOT_DTYPE)
            Di = tl.load(D_ptrs + offs_q, mask=mask_q, other=0.0)
            dp = tl.dot(v_dot, tl.trans(do_dot))

            ds = sm_scale * p * (dp - Di[None, :])

            dv += tl.dot(p.to(DOT_DTYPE), do_dot)
            dk += tl.dot(ds.to(DOT_DTYPE), q_dot)

            Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_Q, 0))
            DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_Q, 0))

    tl.store(DK_block_ptr, dk.to(K.dtype.element_ty), boundary_check=(0,))
    tl.store(DV_block_ptr, dv.to(V.dtype.element_ty), boundary_check=(0,))


@triton.jit
def _draft_bwd_dq_kernel(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DQ,
    L,
    D,
    cu_seqlens_q,
    cu_seqlens_k,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_dqt,
    stride_dqh,
    Q_TOTAL_TOKENS,
    HEAD_Q: tl.constexpr,
    HEAD_KV: tl.constexpr,
    KV_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    DOT_DTYPE = tl.bfloat16
    qk_scale = sm_scale * 1.44269504
    off_z = tl.program_id(0)
    off_h = tl.program_id(1)
    start_m = tl.program_id(2)

    off_kvh = off_h // KV_GROUP

    q_start = tl.load(cu_seqlens_q + off_z)
    q_len = tl.load(cu_seqlens_q + off_z + 1) - q_start

    k_start = tl.load(cu_seqlens_k + off_z)
    k_len = tl.load(cu_seqlens_k + off_z + 1) - k_start

    q_block_start = start_m * BLOCK_Q
    if q_block_start >= q_len:
        return

    L_base = L + off_h * Q_TOTAL_TOKENS + q_start
    D_base = D + off_h * Q_TOTAL_TOKENS + q_start

    DO_block_ptr = tl.make_block_ptr(
        base=DO + q_start * HEAD_Q * HEAD_DIM + off_h * HEAD_DIM,
        shape=(q_len, HEAD_DIM),
        strides=(HEAD_Q * HEAD_DIM, 1),
        offsets=(q_block_start, 0),
        block_shape=(BLOCK_Q, HEAD_DIM),
        order=(0, 1),
    )
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_start * stride_qt + off_h * stride_qh,
        shape=(q_len, HEAD_DIM),
        strides=(stride_qt, 1),
        offsets=(q_block_start, 0),
        block_shape=(BLOCK_Q, HEAD_DIM),
        order=(0, 1),
    )
    DQ_block_ptr = tl.make_block_ptr(
        base=DQ + q_start * stride_dqt + off_h * stride_dqh,
        shape=(q_len, HEAD_DIM),
        strides=(stride_dqt, 1),
        offsets=(q_block_start, 0),
        block_shape=(BLOCK_Q, HEAD_DIM),
        order=(0, 1),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_start * stride_kt + off_kvh * stride_kh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_kt, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_KV, HEAD_DIM),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + k_start * stride_vt + off_kvh * stride_vh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_vt, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_KV, HEAD_DIM),
        order=(0, 1),
    )

    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    do = tl.load(DO_block_ptr, boundary_check=(0,), padding_option="zero")

    q_dot = q.to(DOT_DTYPE)
    do_dot = do.to(DOT_DTYPE)

    offs_q = q_block_start + tl.arange(0, BLOCK_Q)
    mask_q = offs_q < q_len

    l_i = tl.load(L_base + offs_q, mask=mask_q, other=0.0)
    d_i = tl.load(D_base + offs_q, mask=mask_q, other=0.0)

    dq = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    hi = k_len

    for k_block_start in range(0, hi, BLOCK_KV):
        offs_k = k_block_start + tl.arange(0, BLOCK_KV)
        mask_k = offs_k < k_len

        k = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

        k_dot = k.to(DOT_DTYPE)
        v_dot = v.to(DOT_DTYPE)

        mask_final = mask_q[:, None] & mask_k[None, :]

        qk = tl.dot(q_dot, tl.trans(k_dot))
        qk = tl.where(mask_final, qk, -float("inf"))

        p = tl.math.exp2(qk * qk_scale - l_i[:, None])

        dp = tl.dot(do_dot, tl.trans(v_dot))
        ds = (p * (dp - d_i[:, None]) * sm_scale).to(Q.dtype.element_ty)

        dq += tl.dot(ds.to(DOT_DTYPE), k_dot)

        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_KV, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_KV, 0))

    tl.store(DQ_block_ptr, dq.to(Q.dtype.element_ty), boundary_check=(0,))


class DLMAttentionDraft(torch.autograd.Function):
    """Packed DG_PREFIX attention. Q is the noisy canvas; K/V are H_prefix || canvas.

    Tensors are ``(1, heads, tokens, dim)`` going in and ``(1, tokens, heads, dim)`` out.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        sm_scale=None,
    ):
        """Run packed DG_PREFIX attention forward pass."""
        if sm_scale is None:
            raise ValueError("DG_PREFIX requires explicit sm_scale=1.0 after Gemma q_norm")
        qk_scale = sm_scale * 1.44269504

        bsz, HEAD_Q, q_len, HEAD_DIM = q.shape
        _, HEAD_KV, kv_len, _ = k.shape
        assert HEAD_Q % HEAD_KV == 0, f"query heads must be divisible by KV heads, got {HEAD_Q=} {HEAD_KV=}"
        KV_GROUP = HEAD_Q // HEAD_KV
        assert bsz == 1, "only support batch=1 for now"
        Q_TOTAL_TOKENS = bsz * q_len
        assert k.shape[-1] == HEAD_DIM
        assert v.shape[-1] == HEAD_DIM
        assert q.stride(-1) == 1
        assert k.stride(-1) == 1
        assert v.stride(-1) == 1
        assert kv_len >= q_len, f"DG_PREFIX requires kv_len >= q_len, got {kv_len=} and {q_len=}"

        if cu_seqlens_q is None:
            cu_seqlens_q = torch.tensor([0, q_len], device=q.device, dtype=torch.int32)

        if cu_seqlens_k is None:
            cu_seqlens_k = torch.tensor([0, kv_len], device=q.device, dtype=torch.int32)

        num_segments = cu_seqlens_q.numel() - 1
        assert cu_seqlens_k.numel() == cu_seqlens_q.numel(), (
            f"cu_seqlens_q and cu_seqlens_k must have the same number of segments, "
            f"got {cu_seqlens_q.numel()=} vs {cu_seqlens_k.numel()=}"
        )
        assert cu_seqlens_q[-1].item() == q_len, (
            f"cu_seqlens_q[-1] must equal packed q_len, got {cu_seqlens_q[-1].item()} vs {q_len}"
        )
        assert cu_seqlens_k[-1].item() == kv_len, (
            f"cu_seqlens_k[-1] must equal packed kv_len, got {cu_seqlens_k[-1].item()} vs {kv_len}"
        )
        q_segment_lens = cu_seqlens_q.diff()
        k_segment_lens = cu_seqlens_k.diff()
        assert bool(torch.all(k_segment_lens >= q_segment_lens).item()), (
            "Each packed DG_PREFIX segment must contain H_prefix plus its current canvas"
        )
        assert bool(torch.all(q_segment_lens > 0).item()), "DG_PREFIX canvas lengths must be positive"

        if max_seqlen_q is None:
            max_seqlen_q = q_len
        if max_seqlen_k is None:
            max_seqlen_k = kv_len

        o = torch.empty((bsz, q_len, HEAD_Q, HEAD_DIM), device=q.device, dtype=q.dtype)
        L = torch.empty((HEAD_Q, Q_TOTAL_TOKENS), device=q.device, dtype=torch.float32)

        if HEAD_DIM <= 128:
            fwd_block_q, fwd_block_kv, num_warps, maxnreg = 128, 64, 8, 128
        elif HEAD_DIM <= 256:
            fwd_block_q, fwd_block_kv, num_warps, maxnreg = 32, 32, 8, 255
        elif HEAD_DIM <= 512:
            fwd_block_q, fwd_block_kv, num_warps, maxnreg = 16, 16, 8, 255
        else:
            raise NotImplementedError(f"DG_PREFIX attention head_dim {HEAD_DIM} is not supported")
        if HEAD_DIM <= 64:
            num_warps = 4

        grid = (num_segments, HEAD_Q, triton.cdiv(max_seqlen_q, fwd_block_q))

        draft_fwd_kernel[grid](
            q,
            k,
            v,
            qk_scale,
            cu_seqlens_q,
            cu_seqlens_k,
            L,
            o,
            stride_qh=q.stride(1),
            stride_qt=q.stride(2),
            stride_kh=k.stride(1),
            stride_kt=k.stride(2),
            stride_vh=v.stride(1),
            stride_vt=v.stride(2),
            Q_TOTAL_TOKENS=Q_TOTAL_TOKENS,
            HEAD_Q=HEAD_Q,
            HEAD_KV=HEAD_KV,
            KV_GROUP=KV_GROUP,
            HEAD_DIM=HEAD_DIM,
            BLOCK_Q=fwd_block_q,
            BLOCK_KV=fwd_block_kv,
            num_stages=3,
            num_warps=num_warps,
            maxnreg=maxnreg,
        )

        ctx.save_for_backward(q, k, v, o, L, cu_seqlens_q, cu_seqlens_k)
        ctx.sm_scale = sm_scale
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.Q_TOTAL_TOKENS = Q_TOTAL_TOKENS
        ctx.num_segments = num_segments
        ctx.HEAD_Q = HEAD_Q
        ctx.HEAD_KV = HEAD_KV
        ctx.KV_GROUP = KV_GROUP
        ctx.HEAD_DIM = HEAD_DIM
        return o

    @staticmethod
    def backward(ctx, do, *args):
        """Run packed DG_PREFIX attention backward pass."""
        q, k, v, o, L, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors

        Q_TOTAL_TOKENS = ctx.Q_TOTAL_TOKENS
        num_segments = ctx.num_segments
        HEAD_Q = ctx.HEAD_Q
        HEAD_KV = ctx.HEAD_KV
        KV_GROUP = ctx.KV_GROUP
        HEAD_DIM = ctx.HEAD_DIM

        do = do.contiguous().view(Q_TOTAL_TOKENS, HEAD_Q, HEAD_DIM)
        o_flat = o.view(Q_TOTAL_TOKENS, HEAD_Q, HEAD_DIM)
        dq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
        dk = torch.empty(k.shape, device=k.device, dtype=k.dtype)
        dv = torch.empty(v.shape, device=v.device, dtype=v.dtype)

        delta = torch.empty([HEAD_Q, Q_TOTAL_TOKENS], device=o.device, dtype=torch.float32)
        preprocess_block_q = 256 if HEAD_DIM <= 128 else (64 if HEAD_DIM <= 256 else 32)
        grid = (triton.cdiv(Q_TOTAL_TOKENS, preprocess_block_q), HEAD_Q)
        _draft_bwd_preprocess[grid](
            o_flat,
            do,
            delta,
            BLOCK_Q=preprocess_block_q,
            Q_TOTAL_TOKENS=Q_TOTAL_TOKENS,
            HEAD_Q=HEAD_Q,
            HEAD_DIM=HEAD_DIM,
        )

        if HEAD_DIM <= 128:
            dkv_block_q, dkv_block_kv = 32, 128
            dq_block_q, dq_block_kv = 128, 32
            bwd_maxnreg = 128
        elif HEAD_DIM <= 256:
            dkv_block_q, dkv_block_kv = 16, 32
            dq_block_q, dq_block_kv = 32, 16
            bwd_maxnreg = 255
        else:
            dkv_block_q, dkv_block_kv = 16, 16
            dq_block_q, dq_block_kv = 16, 16
            bwd_maxnreg = 255

        grid = (num_segments, HEAD_KV, triton.cdiv(ctx.max_seqlen_k, dkv_block_kv))
        _draft_bwd_dkv_kernel[grid](
            q,
            k,
            v,
            ctx.sm_scale,
            do,
            dk,
            dv,
            L,
            delta,
            cu_seqlens_q,
            cu_seqlens_k,
            stride_qt=q.stride(2),
            stride_qh=q.stride(1),
            stride_kt=k.stride(2),
            stride_kh=k.stride(1),
            stride_vt=v.stride(2),
            stride_vh=v.stride(1),
            stride_dkt=dk.stride(2),
            stride_dkh=dk.stride(1),
            stride_dvt=dv.stride(2),
            stride_dvh=dv.stride(1),
            Q_TOTAL_TOKENS=Q_TOTAL_TOKENS,
            HEAD_Q=HEAD_Q,
            HEAD_KV=HEAD_KV,
            KV_GROUP=KV_GROUP,
            HEAD_DIM=HEAD_DIM,
            BLOCK_Q=dkv_block_q,
            BLOCK_KV=dkv_block_kv,
            num_stages=3,
            num_warps=8,
        )

        grid = (num_segments, HEAD_Q, triton.cdiv(ctx.max_seqlen_q, dq_block_q))
        _draft_bwd_dq_kernel[grid](
            q,
            k,
            v,
            ctx.sm_scale,
            do,
            dq,
            L,
            delta,
            cu_seqlens_q,
            cu_seqlens_k,
            stride_qt=q.stride(2),
            stride_qh=q.stride(1),
            stride_kt=k.stride(2),
            stride_kh=k.stride(1),
            stride_vt=v.stride(2),
            stride_vh=v.stride(1),
            stride_dqt=dq.stride(2),
            stride_dqh=dq.stride(1),
            Q_TOTAL_TOKENS=Q_TOTAL_TOKENS,
            HEAD_Q=HEAD_Q,
            HEAD_KV=HEAD_KV,
            KV_GROUP=KV_GROUP,
            HEAD_DIM=HEAD_DIM,
            BLOCK_Q=dq_block_q,
            BLOCK_KV=dq_block_kv,
            num_stages=3,
            num_warps=8,
            maxnreg=bwd_maxnreg,
        )
        return dq, dk, dv, None, None, None, None, None
