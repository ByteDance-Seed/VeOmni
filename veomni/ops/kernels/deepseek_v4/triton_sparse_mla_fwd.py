# Copyright 2026 Advanced Micro Devices, Inc. and ByteDance Ltd. and/or its affiliates
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
#
# Portions of this file are adapted from AMD-AGI/Primus under the following license:
#
# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Adapted from AMD-AGI/Primus v26.7.0 (v4_attention_kernels/_triton_v2/dsa_fwd_v4_triton.py
# and _amd_knobs.py); modified for VeOmni.

"""DeepSeek-V4 sparse-MLA forward in plain Triton.

Single MQA latent (K == V == the first ``kv_lora_rank`` channels), per-token
absolute top-k indices and an optional per-head softmax sink. The QK / PV GEMMs
lower to MFMA through ``tl.dot``. One program handles one query token and a block
of heads; it gathers the selected KV latent rows tile by tile and runs an online
softmax with a sink-augmented denominator.
"""

import contextlib

import torch
import triton
import triton.language as tl


@contextlib.contextmanager
def amd_pingpong_disabled():
    """Compile inside this scope with AMD's block ping-pong / async-copy knobs off.

    Those knobs double-buffer LDS operand tiles, which slows these kernels down and
    overflows LDS for the backward's wide dKV tile. They are read at compile time
    and are not part of Triton's cache key, so kernels compiled here keep the
    non-ping-pong schedule while every other kernel keeps the ambient setting.
    """
    amd = getattr(getattr(triton, "knobs", None), "amd", None)
    if amd is None:
        yield
        return
    prev_pp = amd.use_block_pingpong
    prev_ac = amd.use_async_copy
    try:
        amd.use_block_pingpong = False
        amd.use_async_copy = False
        yield
    finally:
        amd.use_block_pingpong = prev_pp
        amd.use_async_copy = prev_ac


def _get_fwd_configs():
    return [
        triton.Config({"BLOCK_H": bh, "TILE_K": tk, "waves_per_eu": wpe}, num_warps=4, num_stages=ns)
        for bh in (32, 64)
        for tk in (16, 32)
        for ns in (2, 3, 4)
        for wpe in (0, 1)
    ]


@triton.autotune(configs=_get_fwd_configs(), key=["num_heads", "TOPK", "D_V", "D_ROPE", "HAS_ROPE"])
@triton.jit
def _sparse_mla_fwd_tr_kernel(
    Q_ptr,  # [total_tokens, num_heads, D_QK] bf16
    KV_ptr,  # [num_kv, 1, D_QK]               bf16
    TopK_ptr,  # [total_tokens, TOPK]            int32
    Sink_ptr,  # [num_heads]                     fp32 (guarded by HAS_SINK)
    O_ptr,  # [total_tokens, num_heads, D_V]  bf16
    LSE_ptr,  # [total_tokens, num_heads]       fp32 (sink-inclusive)
    stride_q_t,
    stride_q_h,
    stride_kv_t,
    stride_o_t,
    stride_o_h,
    stride_topk_t,
    scale,
    num_heads,
    num_kv,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TILE_K: tl.constexpr,
    D_V: tl.constexpr,
    D_ROPE: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    tok = tl.program_id(0)
    hg = tl.program_id(1)

    offs_h = hg * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < num_heads
    offs_v = tl.arange(0, D_V)
    offs_r = tl.arange(0, D_ROPE)

    q_base = tok.to(tl.int64) * stride_q_t + offs_h.to(tl.int64)[:, None] * stride_q_h
    q_lora = tl.load(Q_ptr + q_base + offs_v[None, :], mask=mask_h[:, None], other=0.0)
    if HAS_ROPE:
        q_rope = tl.load(Q_ptr + q_base + (D_V + offs_r)[None, :], mask=mask_h[:, None], other=0.0)

    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D_V], dtype=tl.float32)

    topk_base = tok.to(tl.int64) * stride_topk_t
    for kt in range(0, TOPK, TILE_K):
        offs_k = kt + tl.arange(0, TILE_K)
        idx = tl.load(TopK_ptr + topk_base + offs_k, mask=offs_k < TOPK, other=-1)
        # Bound on both sides: an index >= num_kv would otherwise be masked in
        # as valid and read past the end of KV_ptr ([num_kv, 1, D_QK]).
        valid = (idx >= 0) & (idx < num_kv)
        safe = tl.where(valid, idx, 0).to(tl.int64)

        kv_base = safe[:, None] * stride_kv_t
        k_lora = tl.load(KV_ptr + kv_base + offs_v[None, :], mask=valid[:, None], other=0.0)

        # S = q @ k^T over [lora ++ rope]  -> [BLOCK_H, TILE_K]
        s = tl.dot(q_lora, tl.trans(k_lora))
        if HAS_ROPE:
            k_rope = tl.load(KV_ptr + kv_base + (D_V + offs_r)[None, :], mask=valid[:, None], other=0.0)
            s += tl.dot(q_rope, tl.trans(k_rope))
        s = s * scale
        s = tl.where(valid[None, :] & mask_h[:, None], s, float("-inf"))

        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        m_new = tl.where(m_new > float("-inf"), m_new, 0.0)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(k_lora.dtype), k_lora)
        m_i = m_new

    if HAS_SINK:
        sink = tl.load(Sink_ptr + offs_h, mask=mask_h, other=float("-inf"))
        m_f = tl.maximum(m_i, sink)
        af = tl.exp(m_i - m_f)
        l_t = l_i * af + tl.exp(sink - m_f)
        acc = acc * af[:, None]
        acc = acc / l_t[:, None]
        lse = m_f + tl.log(l_t)
    else:
        acc = acc / l_i[:, None]
        lse = m_i + tl.log(l_i)

    o_base = tok.to(tl.int64) * stride_o_t + offs_h.to(tl.int64)[:, None] * stride_o_h
    tl.store(O_ptr + o_base + offs_v[None, :], acc.to(O_ptr.dtype.element_ty), mask=mask_h[:, None])
    tl.store(LSE_ptr + tok.to(tl.int64) * num_heads + offs_h, lse, mask=mask_h)


def sparse_mla_fwd_v4_triton(q, kv, topk_indices, attn_sink=None, kv_lora_rank=512, scale=None):
    """DeepSeek-V4 sparse-MLA forward.

    Args:
        q:            [total_tokens, num_heads, d_qk] bf16
        kv:           [num_kv, 1, d_qk] bf16 (or [num_kv, d_qk]); single MQA latent
        topk_indices: [total_tokens, topk] int32 (SWA + sparse, -1 = invalid)
        attn_sink:    [num_heads] fp32 optional per-head learnable sink
        kv_lora_rank: int, default 512
        scale:        float, default 1/sqrt(d_qk)

    Returns:
        o:   [total_tokens, num_heads, kv_lora_rank] (q.dtype)
        lse: [total_tokens, num_heads] fp32 (sink-inclusive when attn_sink given)
    """
    assert q.is_contiguous() and topk_indices.is_contiguous()
    total_tokens, num_heads, d_qk = q.shape
    rope_rank = d_qk - kv_lora_rank
    topk = topk_indices.shape[1]
    if scale is None:
        scale = 1.0 / (d_qk**0.5)
    if kv.dim() == 2:
        kv = kv.unsqueeze(1)
    assert kv.is_contiguous()
    assert kv.shape[0] >= total_tokens and kv.shape[-1] == d_qk

    has_sink = attn_sink is not None
    if has_sink:
        assert attn_sink.is_contiguous() and attn_sink.dtype == torch.float32
        assert attn_sink.shape == (num_heads,)
        sink_ptr = attn_sink
    else:
        sink_ptr = torch.empty(1, dtype=torch.float32, device=q.device)

    o = torch.empty(total_tokens, num_heads, kv_lora_rank, dtype=q.dtype, device=q.device)
    lse = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=q.device)

    # V4 single-latent form: the D_ROPE block of q/kv is a zero pad (RoPE baked
    # in-place over the 512 latent), so the rope QK term is provably zero — skip it.
    has_rope = False

    def grid(meta):
        return (total_tokens, triton.cdiv(num_heads, meta["BLOCK_H"]))

    with amd_pingpong_disabled():
        _sparse_mla_fwd_tr_kernel[grid](
            Q_ptr=q,
            KV_ptr=kv,
            TopK_ptr=topk_indices,
            Sink_ptr=sink_ptr,
            O_ptr=o,
            LSE_ptr=lse,
            stride_q_t=q.stride(0),
            stride_q_h=q.stride(1),
            stride_kv_t=kv.stride(0),
            stride_o_t=o.stride(0),
            stride_o_h=o.stride(1),
            stride_topk_t=topk_indices.stride(0),
            scale=scale,
            num_heads=num_heads,
            num_kv=kv.shape[0],
            TOPK=topk,
            D_V=kv_lora_rank,
            D_ROPE=rope_rank,
            HAS_ROPE=has_rope,
            HAS_SINK=has_sink,
        )
    return o, lse
