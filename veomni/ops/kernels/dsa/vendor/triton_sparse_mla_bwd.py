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
# Adapted from AMD-AGI/Primus v26.7.0 (v4_attention_kernels/_triton_v2/dsa_bwd_v4_triton.py,
# _triton_v2/dsa_bwd_kernels.py and _gluon_dsa/_dsa_bwd_gather.py);
# modified for VeOmni.

"""DeepSeek-V4 sparse-MLA backward with non-atomic chunked dQ/dKV gather."""

import torch
import triton
import triton.language as tl

from .triton_sparse_mla_fwd import amd_pingpong_disabled


_MAX_RANK_CHUNK_WORKSPACE_BYTES = 4 * 1024**3


def _rank_chunk_workspace_bytes(total_tokens: int, num_heads: int, d_qk: int, rank_chunk: int) -> int:
    """Bytes for the two bf16 score buffers and the bf16 dKV intermediate."""
    return total_tokens * rank_chunk * (4 * num_heads + 2 * d_qk)


def _select_rank_chunk(total_tokens: int, num_heads: int, d_qk: int, topk: int) -> int:
    """Keep Primus' short-sequence choice while bounding rank-dependent workspace."""
    default = min(topk, 1536 if num_heads >= 128 else 256)
    bytes_per_rank = _rank_chunk_workspace_bytes(total_tokens, num_heads, d_qk, 1)
    budgeted = max(1, _MAX_RANK_CHUNK_WORKSPACE_BYTES // bytes_per_rank)
    if budgeted >= default:
        return default
    return 1 << (budgeted.bit_length() - 1)


def _select_dq_launch(small_lds: bool) -> tuple[int, int, int | None]:
    """Return the dQ head tile, K tile and stage count for the device LDS."""
    return (32, 16, 1) if small_lds else (64, 16, None)


def _build_inverted_topk_slice(topk_indices_slice, R_CHUNK, num_kv=None):
    """Build a CSR inverted index for a topk slice, excluding invalid (-1) entries.

    Args:
        topk_indices_slice: [T, R_CHUNK] int32; -1 marks padding.
        R_CHUNK:  int — number of ranks in this slice (constexpr width)
        num_kv:   int or None — number of KV tokens. When the KV buffer holds
          more rows than query tokens (V4 [local ++ pool], num_kv = S + P), pass
          it so ``inv_ptr`` has length ``num_kv + 1`` even if the last KV tokens
          are referenced by no query. Defaults to ``T`` (query tokens).

    Returns:
        inv_ptr:  [num_kv+1] int32 — row pointers (kv_token -> range in inv_data)
        inv_data: [valid_entries] int32 — flat indices q*R_CHUNK+local_r, sorted by KV token
    """
    T, RC = topk_indices_slice.shape
    n_kv = T if num_kv is None else int(num_kv)
    flat_kv = topk_indices_slice.reshape(-1).long()  # [T*R_CHUNK]; -1 marks invalid

    # Sort all entries by KV token. argsort returns the flat positions (q*RC+r) in
    # KV-token order; invalid (-1) entries sort to the front and are never referenced
    # because inv_ptr[0] starts past them.
    inv_data = torch.argsort(flat_kv, stable=True).to(torch.int32)  # [T*R_CHUNK]
    # bincount of (kv+1): bin 0 = #invalid, bins 1..num_kv = per-KV counts.
    counts = torch.bincount(flat_kv + 1, minlength=n_kv + 1)  # [num_kv+1]
    inv_ptr = torch.cumsum(counts, dim=0).to(torch.int32)  # [num_kv+1]; inv_ptr[0]=#invalid

    return inv_ptr, inv_data


@triton.jit
def _bwd_chunk_dq_store_ds(
    Q_ptr,  # [T, H, D] bf16
    KV_ptr,  # [T, 1, D] bf16
    dO_ptr,  # [T, H, D_V] bf16
    TopK_ptr,  # [T, TOPK] int32
    LSE_ptr,  # [T, H] fp32
    Delta_ptr,  # [T, H] fp32 (computed here on the first chunk, read after)
    O_ptr,  # [T, H, D_V] bf16 (fwd output, for Delta = rowsum(O*dO))
    dQ_ptr,  # [T, H, D] fp32 — read-modify-write across chunks, cast once on return
    dS_ptr,  # [T, H, R_CHUNK] bf16 — output chunk dS
    P_ptr,  # [T, H, R_CHUNK] bf16 — output chunk P
    stride_q_t: tl.int64,
    stride_q_h: tl.int64,
    stride_kv_t: tl.int64,
    stride_do_t: tl.int64,
    stride_do_h: tl.int64,
    stride_o_t: tl.int64,
    stride_o_h: tl.int64,
    stride_dq_t: tl.int64,
    stride_dq_h: tl.int64,
    stride_topk_t: tl.int64,
    stride_ds_t: tl.int64,
    stride_ds_h: tl.int64,
    scale: tl.float32,
    num_heads: tl.int32,
    R_START: tl.int32,
    R_CHUNK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TILE_K: tl.constexpr,
    D_V: tl.constexpr,
    D_ROPE: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    IS_FIRST_CHUNK: tl.constexpr,
):
    """dQ accumulation for rank chunk [R_START, R_START+R_CHUNK), plus stores
    chunk dS and P to [T, H, R_CHUNK] buffers for _bwd_compute_dkv_intermediate.
    Grid: (total_tokens, num_hg).

    Delta = rowsum(O*dO) is fused here (computed and stored on the first chunk,
    reloaded on later chunks) instead of a separate preprocess kernel.

    HAS_ROPE=False (V4 single-latent): skips all rope MMAs/loads; dQ has no rope columns."""
    token_idx = tl.program_id(0)
    hg_idx = tl.program_id(1)
    offs_h = hg_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < num_heads
    offs_v = tl.arange(0, D_V)

    q_base = token_idx * stride_q_t
    Q_lora = tl.load(Q_ptr + q_base + offs_h[:, None] * stride_q_h + offs_v[None, :], mask=mask_h[:, None], other=0.0)
    if HAS_ROPE:
        offs_r = tl.arange(0, D_ROPE)
        Q_rope = tl.load(
            Q_ptr + q_base + offs_h[:, None] * stride_q_h + (D_V + offs_r[None, :]),
            mask=mask_h[:, None],
            other=0.0,
        )
    do_base = token_idx * stride_do_t
    dO_val = tl.load(
        dO_ptr + do_base + offs_h[:, None] * stride_do_h + offs_v[None, :], mask=mask_h[:, None], other=0.0
    )
    lse = tl.load(LSE_ptr + token_idx * num_heads + offs_h, mask=mask_h, other=0.0)
    if IS_FIRST_CHUNK:
        # Delta = rowsum(O * dO): fold the preprocess in (dO already loaded).
        O_val = tl.load(
            O_ptr + token_idx * stride_o_t + offs_h[:, None] * stride_o_h + offs_v[None, :],
            mask=mask_h[:, None],
            other=0.0,
        )
        delta = tl.sum(O_val.to(tl.float32) * dO_val.to(tl.float32), axis=1)
        tl.store(Delta_ptr + token_idx * num_heads + offs_h, delta, mask=mask_h)
    else:
        delta = tl.load(Delta_ptr + token_idx * num_heads + offs_h, mask=mask_h, other=0.0)

    dq_base = token_idx * stride_dq_t
    if IS_FIRST_CHUNK:
        dQ_lora = tl.zeros([BLOCK_H, D_V], dtype=tl.float32)
    else:
        dQ_lora = tl.load(
            dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + offs_v[None, :],
            mask=mask_h[:, None],
            other=0.0,
        ).to(tl.float32)
    if HAS_ROPE:
        if IS_FIRST_CHUNK:
            dQ_rope = tl.zeros([BLOCK_H, D_ROPE], dtype=tl.float32)
        else:
            dQ_rope = tl.load(
                dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + (D_V + offs_r[None, :]),
                mask=mask_h[:, None],
                other=0.0,
            ).to(tl.float32)

    NUM_TILES: tl.constexpr = (R_CHUNK + TILE_K - 1) // TILE_K
    topk_base = token_idx * stride_topk_t + R_START
    offs_tile = tl.arange(0, TILE_K)
    ds_base = token_idx * stride_ds_t + hg_idx * BLOCK_H * stride_ds_h

    for t in range(NUM_TILES):
        tile_start = t * TILE_K
        tile_offs = tile_start + offs_tile
        valid = tile_offs < R_CHUNK
        topk_pos = tl.load(TopK_ptr + topk_base + tile_offs, mask=valid, other=-1)
        valid = valid & (topk_pos != -1)
        safe_pos = tl.where(valid, topk_pos, 0)

        K_lora_T = tl.load(KV_ptr + safe_pos[None, :] * stride_kv_t + offs_v[:, None], mask=valid[None, :], other=0.0)

        S = tl.dot(Q_lora, K_lora_T)
        if HAS_ROPE:
            K_rope_T = tl.load(
                KV_ptr + safe_pos[None, :] * stride_kv_t + (D_V + offs_r[:, None]),
                mask=valid[None, :],
                other=0.0,
            )
            S += tl.dot(Q_rope, K_rope_T)
        S = tl.where(valid[None, :] & mask_h[:, None], S * scale, float("-inf"))
        P = tl.exp(S - lse[:, None])
        P = tl.where(valid[None, :] & mask_h[:, None], P, 0.0)
        dP = tl.dot(dO_val, K_lora_T)
        dS = P * (dP - delta[:, None]) * scale
        dS = tl.where(valid[None, :] & mask_h[:, None], dS, 0.0)

        dS_bf = dS.to(tl.bfloat16)
        dQ_lora += tl.dot(dS_bf, tl.trans(K_lora_T)).to(tl.float32)
        if HAS_ROPE:
            dQ_rope += tl.dot(dS_bf, tl.trans(K_rope_T)).to(tl.float32)

        local_h = tl.arange(0, BLOCK_H)
        tl.store(
            dS_ptr + ds_base + local_h[:, None] * stride_ds_h + tile_offs[None, :],
            dS_bf,
            mask=mask_h[:, None] & valid[None, :],
        )
        tl.store(
            P_ptr + ds_base + local_h[:, None] * stride_ds_h + tile_offs[None, :],
            P.to(tl.bfloat16),
            mask=mask_h[:, None] & valid[None, :],
        )

    tl.store(
        dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + offs_v[None, :],
        dQ_lora,
        mask=mask_h[:, None],
    )
    if HAS_ROPE:
        tl.store(
            dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + (D_V + offs_r[None, :]),
            dQ_rope,
            mask=mask_h[:, None],
        )


@triton.jit
def _bwd_compute_dkv_intermediate(
    Q_ptr,  # [T, H, D_QK] bf16  (UNtransposed; loaded transposed via strided index)
    dO_ptr,  # [T, H, D_V]  bf16 (UNtransposed)
    dS_ptr,  # [T, H, R_CHUNK] bf16
    P_ptr,  # [T, H, R_CHUNK] bf16
    Interm_ptr,  # [T, R_CHUNK, D_QK] bf16 — output, one writer per (q, rank)
    stride_q_t: tl.int64,
    stride_q_h: tl.int64,
    stride_do_t: tl.int64,
    stride_do_h: tl.int64,
    stride_ds_t: tl.int64,
    stride_ds_h: tl.int64,
    stride_interm_t: tl.int64,  # R_CHUNK * D_QK
    stride_interm_k: tl.int64,  # D_QK
    num_heads: tl.int32,
    R_CHUNK: tl.constexpr,
    TILE_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_HG: tl.constexpr,
    D_V: tl.constexpr,
    D_ROPE: tl.constexpr,
    HAS_ROPE: tl.constexpr,
):
    """dKV intermediate for one chunk, reusing the stored dS/P (no recompute).
    Loads Q/dO untransposed with a [D, BLOCK_H] strided index pattern (contiguous
    along D per head), so no transposed copies are needed. Grid: (total_tokens,).

    HAS_ROPE=False (V4 single-latent): skips Q_rope load, the dKV_rope MMA and the
    interm rope store."""
    token_idx = tl.program_id(0)

    NUM_TILES: tl.constexpr = (R_CHUNK + TILE_K - 1) // TILE_K
    offs_tile = tl.arange(0, TILE_K)
    offs_v = tl.arange(0, D_V)
    if HAS_ROPE:
        offs_r = tl.arange(0, D_ROPE)

    interm_base_t = token_idx * stride_interm_t
    q_base = token_idx * stride_q_t
    do_base = token_idx * stride_do_t
    ds_base = token_idx * stride_ds_t

    for t in range(NUM_TILES):
        tile_start = t * TILE_K
        tile_offs = tile_start + offs_tile
        valid = tile_offs < R_CHUNK

        dKV_lora = tl.zeros([D_V, TILE_K], dtype=tl.float32)
        if HAS_ROPE:
            dKV_rope = tl.zeros([D_ROPE, TILE_K], dtype=tl.float32)

        for hg in range(NUM_HG):
            offs_h = hg * BLOCK_H + tl.arange(0, BLOCK_H)
            mask_h = offs_h < num_heads

            # [D_V, BLOCK_H] loaded transposed: rows=offs_v (stride 1, contiguous),
            # cols=offs_h (stride stride_q_h). No HBM transpose copy needed.
            Q_lora_T = tl.load(
                Q_ptr + q_base + offs_h[None, :] * stride_q_h + offs_v[:, None],
                mask=mask_h[None, :],
                other=0.0,
            )
            dO_T = tl.load(
                dO_ptr + do_base + offs_h[None, :] * stride_do_h + offs_v[:, None],
                mask=mask_h[None, :],
                other=0.0,
            )

            dS_val = tl.load(
                dS_ptr + ds_base + offs_h[:, None] * stride_ds_h + tile_offs[None, :],
                mask=mask_h[:, None] & valid[None, :],
                other=0.0,
            )
            P_val = tl.load(
                P_ptr + ds_base + offs_h[:, None] * stride_ds_h + tile_offs[None, :],
                mask=mask_h[:, None] & valid[None, :],
                other=0.0,
            )

            dKV_lora += tl.dot(Q_lora_T, dS_val.to(Q_lora_T.dtype)).to(tl.float32)
            dKV_lora += tl.dot(dO_T, P_val.to(dO_T.dtype)).to(tl.float32)
            if HAS_ROPE:
                Q_rope_T = tl.load(
                    Q_ptr + q_base + offs_h[None, :] * stride_q_h + (D_V + offs_r[:, None]),
                    mask=mask_h[None, :],
                    other=0.0,
                )
                dKV_rope += tl.dot(Q_rope_T, dS_val.to(Q_rope_T.dtype)).to(tl.float32)

        interm_lora_ptrs = Interm_ptr + interm_base_t + tile_offs[None, :] * stride_interm_k + offs_v[:, None]
        tl.store(interm_lora_ptrs, dKV_lora.to(tl.bfloat16), mask=valid[None, :])

        if HAS_ROPE:
            interm_rope_ptrs = (
                Interm_ptr + interm_base_t + tile_offs[None, :] * stride_interm_k + D_V + offs_r[:, None]
            )
            tl.store(interm_rope_ptrs, dKV_rope.to(tl.bfloat16), mask=valid[None, :])


@triton.jit
def _bwd_dkv_gather_acc(
    Interm_ptr,  # [T, R_CHUNK, D] bf16 — chunk intermediate
    InvPtr_ptr,  # [T+1] int32 — CSR row pointers
    InvData_ptr,  # [T*R_CHUNK] int32 — encoded as q*R_CHUNK + local_r
    dKV_acc_ptr,  # [T, D] fp32 — accumulator (read-modify-write across chunks)
    stride_interm_r: tl.int64,  # D
    stride_acc_t: tl.int64,  # D
    D_V: tl.constexpr,
    D_ROPE: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    BLOCK_K: tl.constexpr = 64,
):
    """Gather one chunk's bf16 intermediate into the fp32 dKV accumulator.
    Grid: (num_kv,) — one CTA per KV token, no atomics. Tiled BLOCK_K rows/iter.

    HAS_ROPE=False (V4 single-latent): interm has no rope columns, so skip
    reading/accumulating them."""
    k = tl.program_id(0)
    offs_v = tl.arange(0, D_V)
    offs_k = tl.arange(0, BLOCK_K)
    if HAS_ROPE:
        offs_r = tl.arange(0, D_ROPE)

    start = tl.load(InvPtr_ptr + k)
    end = tl.load(InvPtr_ptr + k + 1)

    acc_base = k.to(tl.int64) * stride_acc_t
    dkv_acc_lora = tl.load(dKV_acc_ptr + acc_base + offs_v).to(tl.float32)
    if HAS_ROPE:
        dkv_acc_rope = tl.load(dKV_acc_ptr + acc_base + D_V + offs_r).to(tl.float32)

    n_entries = end - start
    for ti in range(0, tl.cdiv(n_entries, BLOCK_K)):
        e_local = ti * BLOCK_K + offs_k
        valid = e_local < n_entries
        entry = tl.load(InvData_ptr + start + e_local, mask=valid, other=0).to(tl.int64)
        rp = entry[:, None] * stride_interm_r
        rows_v = tl.load(Interm_ptr + rp + offs_v[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        dkv_acc_lora += tl.sum(rows_v, axis=0)
        if HAS_ROPE:
            rows_r = tl.load(Interm_ptr + rp + D_V + offs_r[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
            dkv_acc_rope += tl.sum(rows_r, axis=0)

    tl.store(dKV_acc_ptr + acc_base + offs_v, dkv_acc_lora)
    if HAS_ROPE:
        tl.store(dKV_acc_ptr + acc_base + D_V + offs_r, dkv_acc_rope)


def sparse_mla_bwd_v4_triton(
    q, kv, o, do, topk_indices, lse, attn_sink=None, kv_lora_rank=512, scale=None, small_lds=False
):
    """DeepSeek-V4 sparse-MLA backward.

    Returns ``(dq, dkv, d_sink)`` with ``dkv`` shaped like ``kv`` and ``d_sink``
    ``[num_heads]`` fp32 (or ``None`` when ``attn_sink`` is None). ``small_lds``
    selects the schedule that fits 64 KiB of LDS per workgroup (gfx942); the
    default one needs 160 KiB (gfx950).
    """
    assert q.is_contiguous() and o.is_contiguous() and do.is_contiguous()
    assert topk_indices.is_contiguous() and lse.is_contiguous()
    total_tokens, num_heads, d_qk = q.shape
    rope_rank = d_qk - kv_lora_rank
    topk = topk_indices.shape[1]
    if scale is None:
        scale = 1.0 / (d_qk**0.5)
    if kv.dim() == 2:
        kv = kv.unsqueeze(1)
    assert kv.is_contiguous()
    num_kv = kv.shape[0]

    has_sink = attn_sink is not None
    if has_sink:
        assert attn_sink.dtype == torch.float32 and attn_sink.shape == (num_heads,)

    # Delta = rowsum(O*dO) is fused into the first dQ chunk (no preprocess kernel).
    delta = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=q.device)

    # Primus' short-sequence choices minimize dQ reloads. At long context the
    # dS/P/intermediate buffers dominate instead, so reduce the chunk in powers
    # of two to keep their combined allocation at or below 4 GiB.
    R_CHUNK = _select_rank_chunk(total_tokens, num_heads, d_qk, topk)
    # The fp32 cross-chunk accumulator doubles dQ's output storage pressure.
    # gfx942 therefore needs both a 32-head tile and single-stage scheduling;
    # gfx950 keeps Primus' faster 64-head tile.
    BH_DQ, TK_DQ, num_stages = _select_dq_launch(small_lds)
    # A wider dKV tile (BH_DKV=64, TK_DKV=128) over one head group cuts Q/dO
    # re-loads for H<=64 and 128-aligned chunks. It regresses for H>=128, for
    # partial tiles and on 64 KiB-LDS devices.
    use_wide_dkv = not small_lds and num_heads <= 64 and R_CHUNK % 128 == 0
    BH_DKV, TK_DKV = (64, 128) if use_wide_dkv else (32, 64)
    num_hg_dq = triton.cdiv(num_heads, BH_DQ)
    num_hg_dkv = triton.cdiv(num_heads, BH_DKV)

    # V4 single-latent form: RoPE is baked into the 512-dim latent, so skip all
    # rope MMAs, K_rope loads and interm-rope traffic. Callers pass
    # d_qk == kv_lora_rank (no width pad).
    HAS_ROPE = False

    # Keep dQ in fp32 across rank chunks. Casting after every chunk loses small
    # later contributions when the adaptive schedule splits a long sequence
    # into many chunks.
    dq_acc = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    chunk_dS = torch.empty(total_tokens, num_heads, R_CHUNK, dtype=torch.bfloat16, device=q.device)
    chunk_P = torch.empty(total_tokens, num_heads, R_CHUNK, dtype=torch.bfloat16, device=q.device)
    dkv_acc = torch.zeros(num_kv, d_qk, dtype=torch.float32, device=q.device)
    interm = torch.empty(total_tokens, R_CHUNK, d_qk, dtype=torch.bfloat16, device=q.device)

    # pad topk to an R_CHUNK multiple (-1 = invalid).
    topk_padded_len = ((topk + R_CHUNK - 1) // R_CHUNK) * R_CHUNK
    if topk_padded_len != topk:
        pad = torch.full((total_tokens, topk_padded_len - topk), -1, dtype=torch.int32, device=q.device)
        topk_padded = torch.cat([topk_indices, pad], dim=1).contiguous()
    else:
        topk_padded = topk_indices

    all_csr = [
        _build_inverted_topk_slice(topk_padded[:, rs : rs + R_CHUNK], R_CHUNK, num_kv=num_kv)
        for rs in range(0, topk, R_CHUNK)
    ]

    with amd_pingpong_disabled():
        for chunk_idx, r_start in enumerate(range(0, topk, R_CHUNK)):
            is_first = r_start == 0

            _bwd_chunk_dq_store_ds[(total_tokens, num_hg_dq)](
                q,
                kv,
                do,
                topk_padded,
                lse,
                delta,
                o,
                dq_acc,
                chunk_dS,
                chunk_P,
                q.stride(0),
                q.stride(1),
                kv.stride(0),
                do.stride(0),
                do.stride(1),
                o.stride(0),
                o.stride(1),
                dq_acc.stride(0),
                dq_acc.stride(1),
                topk_padded.stride(0),
                chunk_dS.stride(0),
                chunk_dS.stride(1),
                scale,
                num_heads,
                r_start,
                R_CHUNK=R_CHUNK,
                BLOCK_H=BH_DQ,
                TILE_K=TK_DQ,
                D_V=kv_lora_rank,
                D_ROPE=rope_rank,
                HAS_ROPE=HAS_ROPE,
                IS_FIRST_CHUNK=is_first,
                num_warps=4,
                waves_per_eu=1,
                num_stages=num_stages,
            )

            _bwd_compute_dkv_intermediate[(total_tokens,)](
                q,
                do,
                chunk_dS,
                chunk_P,
                interm,
                q.stride(0),
                q.stride(1),
                do.stride(0),
                do.stride(1),
                chunk_dS.stride(0),
                chunk_dS.stride(1),
                interm.stride(0),
                interm.stride(1),
                num_heads,
                R_CHUNK=R_CHUNK,
                TILE_K=TK_DKV,
                BLOCK_H=BH_DKV,
                NUM_HG=num_hg_dkv,
                D_V=kv_lora_rank,
                D_ROPE=rope_rank,
                HAS_ROPE=HAS_ROPE,
                num_warps=4,
                num_stages=num_stages,
            )

            inv_ptr, inv_data = all_csr[chunk_idx]
            _bwd_dkv_gather_acc[(num_kv,)](
                interm,
                inv_ptr,
                inv_data,
                dkv_acc,
                interm.stride(1),
                dkv_acc.stride(0),
                D_V=kv_lora_rank,
                D_ROPE=rope_rank,
                HAS_ROPE=HAS_ROPE,
                num_warps=4,
                num_stages=num_stages,
            )

    d_sink = None
    if has_sink:
        d_sink = -(torch.exp(attn_sink.unsqueeze(0) - lse) * delta).sum(0)

    dq = dq_acc.to(q.dtype)
    dkv = dkv_acc.to(kv.dtype).unsqueeze(1)
    return dq, dkv, d_sink
