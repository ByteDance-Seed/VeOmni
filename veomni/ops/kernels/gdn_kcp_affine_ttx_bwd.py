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

"""Ascend TTX BC8/M1 analytical backward for the KCP affine summary.

The public dispatcher exposes only the persistent column-tiled forward/replay
plus column-tiled VJP path. Reference kernels remain private implementation
details and cannot be selected by production configuration.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor


try:
    import triton
    import triton.language as tl

    _TRITON_OK = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    _TRITON_OK = False


# M1: cache readonly eye[H,K,K] per (device,H,K); reused across layers/calls.
_EYE_CACHE: dict[tuple, Tensor] = {}


def _npu_ready() -> bool:
    return bool(getattr(torch, "npu", None)) and torch.npu.is_available()


def _cached_eye(device: torch.device, H: int, K: int) -> Tensor:
    key = (device.type, device.index, int(H), int(K))
    eye = _EYE_CACHE.get(key)
    if eye is None or eye.device != device or eye.shape != (H, K, K):
        eye = torch.eye(K, device=device, dtype=torch.float32).expand(H, K, K).contiguous()
        _EYE_CACHE[key] = eye
    return eye


def _resolve_segment_bounds(
    *,
    cu_seqlens: Optional[Tensor],
    cu_pts: Optional[Sequence[int]],
    bsz: int,
    t_total: int,
) -> Tuple[List[int], List[int]]:
    """Host segment [start,end) lists. Prefer precomputed ``cu_pts`` (no D2H)."""
    if cu_seqlens is None and cu_pts is None:
        starts = list(range(0, bsz * t_total, t_total))
        ends = [s + t_total for s in starts]
        return starts, ends
    if bsz != 1:
        raise ValueError("varlen affine bwd expects batch=1")
    if cu_pts is not None:
        pts = [int(x) for x in cu_pts]
    else:
        assert cu_seqlens is not None
        pts = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    if not pts or pts[0] != 0:
        raise ValueError(f"cu_seqlens must start at 0, got {pts[:3]}")
    if pts[-1] != t_total:
        raise ValueError(f"cu_seqlens must end at T={t_total}, got {pts[-3:]}")
    if any(right < left for left, right in zip(pts, pts[1:])):
        raise ValueError(f"cu_seqlens must be nondecreasing, got {pts}")
    return pts[:-1], pts[1:]


def _bwd_chunk() -> int:
    raw = os.environ.get("VEOMNI_GDN_AFFINE_BWD_CHUNK", "128").strip()
    try:
        chunk = int(raw)
    except ValueError as exc:
        raise ValueError(f"VEOMNI_GDN_AFFINE_BWD_CHUNK must be 128 or 256, got {raw!r}") from exc
    if chunk not in (128, 256):
        raise ValueError(f"VEOMNI_GDN_AFFINE_BWD_CHUNK must be 128 or 256, got {chunk}")
    return chunk


def _fwd_coltile_bc() -> int:
    raw = os.environ.get("VEOMNI_GDN_AFFINE_REPLAY_COLUMN_TILE", "8").strip()
    try:
        column_tile = int(raw)
    except ValueError as exc:
        raise ValueError(f"VEOMNI_GDN_AFFINE_REPLAY_COLUMN_TILE must be 8 or 32, got {raw!r}") from exc
    if column_tile not in (8, 32):
        raise ValueError(f"VEOMNI_GDN_AFFINE_REPLAY_COLUMN_TILE must be 8 or 32, got {column_tile}")
    return column_tile


if _TRITON_OK:

    @triton.jit
    def _affine_fwd_token_kernel(
        k_ptr,
        v_ptr,
        g_ptr,
        beta_ptr,
        he_ptr,
        M_ptr,
        t_abs,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
    ):
        """One token, all heads (grid=H). HBM he/M; UB holds BK×BV tiles."""
        i_h = tl.program_id(0)
        NK = (K + BK - 1) // BK
        NV = (V + BV - 1) // BV
        p_k = k_ptr + t_abs * (H * K) + i_h * K
        p_v = v_ptr + t_abs * (H * V) + i_h * V
        b_eg = tl.exp(tl.load(g_ptr + t_abs * H + i_h).to(tl.float32))
        b_beta = tl.load(beta_ptr + t_abs * H + i_h).to(tl.float32)

        for i_v in range(0, NV):
            offs_v = i_v * BV + tl.arange(0, BV)
            mask_v = offs_v < V
            b_khe = tl.zeros([BV], dtype=tl.float32)
            for i_k in range(0, NK):
                offs_k = i_k * BK + tl.arange(0, BK)
                mask_k = offs_k < K
                b_k = tl.load(p_k + offs_k, mask=mask_k, other=0).to(tl.float32)
                b_he = tl.load(
                    he_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0.0,
                )
                b_khe += tl.sum(b_k[:, None] * b_he, 0)
            b_v = tl.load(p_v + offs_v, mask=mask_v, other=0).to(tl.float32)
            for i_k in range(0, NK):
                offs_k = i_k * BK + tl.arange(0, BK)
                mask_k = offs_k < K
                b_k = tl.load(p_k + offs_k, mask=mask_k, other=0).to(tl.float32)
                b_he = tl.load(
                    he_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0.0,
                )
                b_he = b_eg * (b_he - b_beta * b_k[:, None] * b_khe[None, :]) + (b_beta * b_k[:, None] * b_v[None, :])
                tl.store(
                    he_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                    b_he,
                    mask=mask_k[:, None] & mask_v[None, :],
                )

        for i_c in range(0, NK):
            offs_c = i_c * BK + tl.arange(0, BK)
            mask_c = offs_c < K
            b_kM = tl.zeros([BK], dtype=tl.float32)
            for i_r in range(0, NK):
                offs_r = i_r * BK + tl.arange(0, BK)
                mask_r = offs_r < K
                b_k = tl.load(p_k + offs_r, mask=mask_r, other=0).to(tl.float32)
                b_M = tl.load(
                    M_ptr + i_h * K * K + offs_r[:, None] * K + offs_c[None, :],
                    mask=mask_r[:, None] & mask_c[None, :],
                    other=0.0,
                )
                b_kM += tl.sum(b_k[:, None] * b_M, 0)
            for i_r in range(0, NK):
                offs_r = i_r * BK + tl.arange(0, BK)
                mask_r = offs_r < K
                b_k = tl.load(p_k + offs_r, mask=mask_r, other=0).to(tl.float32)
                b_M = tl.load(
                    M_ptr + i_h * K * K + offs_r[:, None] * K + offs_c[None, :],
                    mask=mask_r[:, None] & mask_c[None, :],
                    other=0.0,
                )
                b_M = b_eg * (b_M - b_beta * b_k[:, None] * b_kM[None, :])
                tl.store(
                    M_ptr + i_h * K * K + offs_r[:, None] * K + offs_c[None, :],
                    b_M,
                    mask=mask_r[:, None] & mask_c[None, :],
                )

    @triton.jit
    def _affine_bwd_token_kernel(
        k_ptr,
        v_ptr,
        g_ptr,
        beta_ptr,
        he_ptr,
        M_ptr,
        dhe_ptr,
        dM_ptr,
        gk_ptr,
        gv_ptr,
        gg_ptr,
        gb_ptr,
        t_abs,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
    ):
        """One reverse token VJP + push Aᵀ. he/M are pre-token states."""
        i_h = tl.program_id(0)
        NK = (K + BK - 1) // BK
        NV = (V + BV - 1) // BV
        p_k = k_ptr + t_abs * (H * K) + i_h * K
        p_v = v_ptr + t_abs * (H * V) + i_h * V
        b_eg = tl.exp(tl.load(g_ptr + t_abs * H + i_h).to(tl.float32))
        b_beta = tl.load(beta_ptr + t_abs * H + i_h).to(tl.float32)

        tr = tl.zeros([1], dtype=tl.float32)
        k_dA_k = tl.zeros([1], dtype=tl.float32)
        for i_v in range(0, NV):
            offs_v = i_v * BV + tl.arange(0, BV)
            mask_v = offs_v < V
            k_he = tl.zeros([BV], dtype=tl.float32)
            k_dhe = tl.zeros([BV], dtype=tl.float32)
            for i_k in range(0, NK):
                offs_k = i_k * BK + tl.arange(0, BK)
                mask_k = offs_k < K
                b_k = tl.load(p_k + offs_k, mask=mask_k, other=0).to(tl.float32)
                he_t = tl.load(
                    he_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0.0,
                )
                dhe = tl.load(
                    dhe_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0.0,
                )
                tr += tl.sum(he_t * dhe)
                k_he += tl.sum(b_k[:, None] * he_t, 0)
                k_dhe += tl.sum(b_k[:, None] * dhe, 0)
            k_dA_k += tl.sum(k_he * k_dhe)
        for i_c in range(0, NK):
            offs_c = i_c * BK + tl.arange(0, BK)
            mask_c = offs_c < K
            k_Mt = tl.zeros([BK], dtype=tl.float32)
            k_dM = tl.zeros([BK], dtype=tl.float32)
            for i_r in range(0, NK):
                offs_r = i_r * BK + tl.arange(0, BK)
                mask_r = offs_r < K
                b_k = tl.load(p_k + offs_r, mask=mask_r, other=0).to(tl.float32)
                M_t = tl.load(
                    M_ptr + i_h * K * K + offs_r[:, None] * K + offs_c[None, :],
                    mask=mask_r[:, None] & mask_c[None, :],
                    other=0.0,
                )
                dM = tl.load(
                    dM_ptr + i_h * K * K + offs_r[:, None] * K + offs_c[None, :],
                    mask=mask_r[:, None] & mask_c[None, :],
                    other=0.0,
                )
                tr += tl.sum(M_t * dM)
                k_Mt += tl.sum(b_k[:, None] * M_t, 0)
                k_dM += tl.sum(b_k[:, None] * dM, 0)
            k_dA_k += tl.sum(k_Mt * k_dM)

        tr_s = tl.sum(tr)
        kdak_s = tl.sum(k_dA_k)
        deg = tr_s - b_beta * kdak_s

        dbeta_b = tl.zeros([1], dtype=tl.float32)
        for i_v in range(0, NV):
            offs_v = i_v * BV + tl.arange(0, BV)
            mask_v = offs_v < V
            b_v = tl.load(p_v + offs_v, mask=mask_v, other=0).to(tl.float32)
            acc = tl.zeros([BV], dtype=tl.float32)
            for i_k in range(0, NK):
                offs_k = i_k * BK + tl.arange(0, BK)
                mask_k = offs_k < K
                b_k = tl.load(p_k + offs_k, mask=mask_k, other=0).to(tl.float32)
                dhe = tl.load(
                    dhe_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0.0,
                )
                acc += tl.sum(b_k[:, None] * dhe, 0)
            dbeta_b += tl.sum(acc * b_v)
        dbeta = -b_eg * kdak_s + tl.sum(dbeta_b)
        tl.store(gg_ptr + t_abs * H + i_h, tl.load(gg_ptr + t_abs * H + i_h) + deg * b_eg)
        tl.store(gb_ptr + t_abs * H + i_h, tl.load(gb_ptr + t_abs * H + i_h) + dbeta)

        for i_k in range(0, NK):
            offs_k = i_k * BK + tl.arange(0, BK)
            mask_k = offs_k < K
            dk = tl.zeros([BK], dtype=tl.float32)
            for i_v in range(0, NV):
                offs_v = i_v * BV + tl.arange(0, BV)
                mask_v = offs_v < V
                k_he = tl.zeros([BV], dtype=tl.float32)
                k_dhe = tl.zeros([BV], dtype=tl.float32)
                for i_kk in range(0, NK):
                    offs_kk = i_kk * BK + tl.arange(0, BK)
                    mask_kk = offs_kk < K
                    b_kk = tl.load(p_k + offs_kk, mask=mask_kk, other=0).to(tl.float32)
                    he_t = tl.load(
                        he_ptr + i_h * K * V + offs_kk[:, None] * V + offs_v[None, :],
                        mask=mask_kk[:, None] & mask_v[None, :],
                        other=0.0,
                    )
                    dhe = tl.load(
                        dhe_ptr + i_h * K * V + offs_kk[:, None] * V + offs_v[None, :],
                        mask=mask_kk[:, None] & mask_v[None, :],
                        other=0.0,
                    )
                    k_he += tl.sum(b_kk[:, None] * he_t, 0)
                    k_dhe += tl.sum(b_kk[:, None] * dhe, 0)
                dhe_tile = tl.load(
                    dhe_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0.0,
                )
                he_tile = tl.load(
                    he_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0.0,
                )
                b_v = tl.load(p_v + offs_v, mask=mask_v, other=0).to(tl.float32)
                dk += -b_eg * b_beta * (tl.sum(dhe_tile * k_he[None, :], 1) + tl.sum(he_tile * k_dhe[None, :], 1))
                dk += b_beta * tl.sum(dhe_tile * b_v[None, :], 1)
            for i_c in range(0, NK):
                offs_c = i_c * BK + tl.arange(0, BK)
                mask_c = offs_c < K
                k_Mt = tl.zeros([BK], dtype=tl.float32)
                k_dM = tl.zeros([BK], dtype=tl.float32)
                for i_r in range(0, NK):
                    offs_r = i_r * BK + tl.arange(0, BK)
                    mask_r = offs_r < K
                    b_kr = tl.load(p_k + offs_r, mask=mask_r, other=0).to(tl.float32)
                    M_t = tl.load(
                        M_ptr + i_h * K * K + offs_r[:, None] * K + offs_c[None, :],
                        mask=mask_r[:, None] & mask_c[None, :],
                        other=0.0,
                    )
                    dM = tl.load(
                        dM_ptr + i_h * K * K + offs_r[:, None] * K + offs_c[None, :],
                        mask=mask_r[:, None] & mask_c[None, :],
                        other=0.0,
                    )
                    k_Mt += tl.sum(b_kr[:, None] * M_t, 0)
                    k_dM += tl.sum(b_kr[:, None] * dM, 0)
                dM_tile = tl.load(
                    dM_ptr + i_h * K * K + offs_k[:, None] * K + offs_c[None, :],
                    mask=mask_k[:, None] & mask_c[None, :],
                    other=0.0,
                )
                M_tile = tl.load(
                    M_ptr + i_h * K * K + offs_k[:, None] * K + offs_c[None, :],
                    mask=mask_k[:, None] & mask_c[None, :],
                    other=0.0,
                )
                dk += -b_eg * b_beta * (tl.sum(dM_tile * k_Mt[None, :], 1) + tl.sum(M_tile * k_dM[None, :], 1))
            gk_ptrs = gk_ptr + t_abs * (H * K) + i_h * K + offs_k
            tl.store(gk_ptrs, tl.load(gk_ptrs, mask=mask_k, other=0.0) + dk, mask=mask_k)

        for i_v in range(0, NV):
            offs_v = i_v * BV + tl.arange(0, BV)
            mask_v = offs_v < V
            dv = tl.zeros([BV], dtype=tl.float32)
            for i_k in range(0, NK):
                offs_k = i_k * BK + tl.arange(0, BK)
                mask_k = offs_k < K
                b_k = tl.load(p_k + offs_k, mask=mask_k, other=0).to(tl.float32)
                dhe = tl.load(
                    dhe_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0.0,
                )
                dv += b_beta * tl.sum(dhe * b_k[:, None], 0)
            gv_ptrs = gv_ptr + t_abs * (H * V) + i_h * V + offs_v
            tl.store(gv_ptrs, tl.load(gv_ptrs, mask=mask_v, other=0.0) + dv, mask=mask_v)

        # push A^T through dhe / dM
        for i_v in range(0, NV):
            offs_v = i_v * BV + tl.arange(0, BV)
            mask_v = offs_v < V
            k_dhe = tl.zeros([BV], dtype=tl.float32)
            for i_k in range(0, NK):
                offs_k = i_k * BK + tl.arange(0, BK)
                mask_k = offs_k < K
                b_k = tl.load(p_k + offs_k, mask=mask_k, other=0).to(tl.float32)
                dhe = tl.load(
                    dhe_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0.0,
                )
                k_dhe += tl.sum(b_k[:, None] * dhe, 0)
            for i_k in range(0, NK):
                offs_k = i_k * BK + tl.arange(0, BK)
                mask_k = offs_k < K
                b_k = tl.load(p_k + offs_k, mask=mask_k, other=0).to(tl.float32)
                dhe = tl.load(
                    dhe_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0.0,
                )
                dhe = b_eg * (dhe - b_beta * b_k[:, None] * k_dhe[None, :])
                tl.store(
                    dhe_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                    dhe,
                    mask=mask_k[:, None] & mask_v[None, :],
                )
        for i_c in range(0, NK):
            offs_c = i_c * BK + tl.arange(0, BK)
            mask_c = offs_c < K
            k_dM = tl.zeros([BK], dtype=tl.float32)
            for i_r in range(0, NK):
                offs_r = i_r * BK + tl.arange(0, BK)
                mask_r = offs_r < K
                b_k = tl.load(p_k + offs_r, mask=mask_r, other=0).to(tl.float32)
                dM = tl.load(
                    dM_ptr + i_h * K * K + offs_r[:, None] * K + offs_c[None, :],
                    mask=mask_r[:, None] & mask_c[None, :],
                    other=0.0,
                )
                k_dM += tl.sum(b_k[:, None] * dM, 0)
            for i_r in range(0, NK):
                offs_r = i_r * BK + tl.arange(0, BK)
                mask_r = offs_r < K
                b_k = tl.load(p_k + offs_r, mask=mask_r, other=0).to(tl.float32)
                dM = tl.load(
                    dM_ptr + i_h * K * K + offs_r[:, None] * K + offs_c[None, :],
                    mask=mask_r[:, None] & mask_c[None, :],
                    other=0.0,
                )
                dM = b_eg * (dM - b_beta * b_k[:, None] * k_dM[None, :])
                tl.store(
                    dM_ptr + i_h * K * K + offs_r[:, None] * K + offs_c[None, :],
                    dM,
                    mask=mask_r[:, None] & mask_c[None, :],
                )

    @triton.jit(do_not_specialize=["t_start", "t_len"])
    def _affine_fwd_chunk_kernel(
        k_ptr,
        v_ptr,
        g_ptr,
        beta_ptr,
        he_ptr,
        M_ptr,
        he_buf_ptr,
        M_buf_ptr,
        t_start,
        t_len,
        H: tl.constexpr,
        STORE_STATES: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
    ):
        """Chunk of tokens (grid=H). Optional pre-token he/M dump to he_buf/M_buf[ti]."""
        i_h = tl.program_id(0)
        NK = (K + BK - 1) // BK
        NV = (V + BV - 1) // BV
        for ti in range(0, t_len):
            if STORE_STATES:
                for i_v in range(0, NV):
                    offs_v = i_v * BV + tl.arange(0, BV)
                    mask_v = offs_v < V
                    for i_k in range(0, NK):
                        offs_k = i_k * BK + tl.arange(0, BK)
                        mask_k = offs_k < K
                        b_he = tl.load(
                            he_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                            mask=mask_k[:, None] & mask_v[None, :],
                            other=0.0,
                        )
                        tl.store(
                            he_buf_ptr + ti * (H * K * V) + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                            b_he,
                            mask=mask_k[:, None] & mask_v[None, :],
                        )
                for i_c in range(0, NK):
                    offs_c = i_c * BK + tl.arange(0, BK)
                    mask_c = offs_c < K
                    for i_r in range(0, NK):
                        offs_r = i_r * BK + tl.arange(0, BK)
                        mask_r = offs_r < K
                        b_M = tl.load(
                            M_ptr + i_h * K * K + offs_r[:, None] * K + offs_c[None, :],
                            mask=mask_r[:, None] & mask_c[None, :],
                            other=0.0,
                        )
                        tl.store(
                            M_buf_ptr + ti * (H * K * K) + i_h * K * K + offs_r[:, None] * K + offs_c[None, :],
                            b_M,
                            mask=mask_r[:, None] & mask_c[None, :],
                        )
            t_abs = t_start + ti
            p_k = k_ptr + t_abs * (H * K) + i_h * K
            p_v = v_ptr + t_abs * (H * V) + i_h * V
            b_eg = tl.exp(tl.load(g_ptr + t_abs * H + i_h).to(tl.float32))
            b_beta = tl.load(beta_ptr + t_abs * H + i_h).to(tl.float32)
            for i_v in range(0, NV):
                offs_v = i_v * BV + tl.arange(0, BV)
                mask_v = offs_v < V
                b_khe = tl.zeros([BV], dtype=tl.float32)
                for i_k in range(0, NK):
                    offs_k = i_k * BK + tl.arange(0, BK)
                    mask_k = offs_k < K
                    b_k = tl.load(p_k + offs_k, mask=mask_k, other=0).to(tl.float32)
                    b_he = tl.load(
                        he_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                        mask=mask_k[:, None] & mask_v[None, :],
                        other=0.0,
                    )
                    b_khe += tl.sum(b_k[:, None] * b_he, 0)
                b_v = tl.load(p_v + offs_v, mask=mask_v, other=0).to(tl.float32)
                for i_k in range(0, NK):
                    offs_k = i_k * BK + tl.arange(0, BK)
                    mask_k = offs_k < K
                    b_k = tl.load(p_k + offs_k, mask=mask_k, other=0).to(tl.float32)
                    b_he = tl.load(
                        he_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                        mask=mask_k[:, None] & mask_v[None, :],
                        other=0.0,
                    )
                    b_he = b_eg * (b_he - b_beta * b_k[:, None] * b_khe[None, :]) + (
                        b_beta * b_k[:, None] * b_v[None, :]
                    )
                    tl.store(
                        he_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                        b_he,
                        mask=mask_k[:, None] & mask_v[None, :],
                    )

            for i_c in range(0, NK):
                offs_c = i_c * BK + tl.arange(0, BK)
                mask_c = offs_c < K
                b_kM = tl.zeros([BK], dtype=tl.float32)
                for i_r in range(0, NK):
                    offs_r = i_r * BK + tl.arange(0, BK)
                    mask_r = offs_r < K
                    b_k = tl.load(p_k + offs_r, mask=mask_r, other=0).to(tl.float32)
                    b_M = tl.load(
                        M_ptr + i_h * K * K + offs_r[:, None] * K + offs_c[None, :],
                        mask=mask_r[:, None] & mask_c[None, :],
                        other=0.0,
                    )
                    b_kM += tl.sum(b_k[:, None] * b_M, 0)
                for i_r in range(0, NK):
                    offs_r = i_r * BK + tl.arange(0, BK)
                    mask_r = offs_r < K
                    b_k = tl.load(p_k + offs_r, mask=mask_r, other=0).to(tl.float32)
                    b_M = tl.load(
                        M_ptr + i_h * K * K + offs_r[:, None] * K + offs_c[None, :],
                        mask=mask_r[:, None] & mask_c[None, :],
                        other=0.0,
                    )
                    b_M = b_eg * (b_M - b_beta * b_k[:, None] * b_kM[None, :])
                    tl.store(
                        M_ptr + i_h * K * K + offs_r[:, None] * K + offs_c[None, :],
                        b_M,
                        mask=mask_r[:, None] & mask_c[None, :],
                    )

    @triton.jit(do_not_specialize=["t_start", "t_len"])
    def _affine_fwd_chunk_coltile_kernel(
        k_ptr,
        v_ptr,
        g_ptr,
        beta_ptr,
        he_ptr,
        M_ptr,
        he_buf_ptr,
        M_buf_ptr,
        t_start,
        t_len,
        H: tl.constexpr,
        STORE_STATES: tl.constexpr,
        IS_HE: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BC: tl.constexpr,
    ):
        """Persist one output-column tile in UB across a forward/replay chunk."""
        i_tile = tl.program_id(0)
        i_h = tl.program_id(1)
        offs_c = i_tile * BC + tl.arange(0, BC)
        offs_k = tl.arange(0, K)
        if IS_HE:
            mask_c = offs_c < V
            state = tl.load(
                he_ptr + i_h * K * V + offs_k[:, None] * V + offs_c[None, :],
                mask=mask_c[None, :],
                other=0.0,
            )
        else:
            mask_c = offs_c < K
            state = tl.load(
                M_ptr + i_h * K * K + offs_k[:, None] * K + offs_c[None, :],
                mask=mask_c[None, :],
                other=0.0,
            )

        for ti in range(0, t_len):
            if STORE_STATES:
                if IS_HE:
                    tl.store(
                        he_buf_ptr + ti * (H * K * V) + i_h * K * V + offs_k[:, None] * V + offs_c[None, :],
                        state,
                        mask=mask_c[None, :],
                    )
                else:
                    tl.store(
                        M_buf_ptr + ti * (H * K * K) + i_h * K * K + offs_k[:, None] * K + offs_c[None, :],
                        state,
                        mask=mask_c[None, :],
                    )

            t_abs = t_start + ti
            p_k = k_ptr + t_abs * (H * K) + i_h * K
            b_k = tl.load(p_k + offs_k).to(tl.float32)
            b_eg = tl.exp(tl.load(g_ptr + t_abs * H + i_h).to(tl.float32))
            b_beta = tl.load(beta_ptr + t_abs * H + i_h).to(tl.float32)
            k_state = tl.sum(b_k[:, None] * state, 0)
            if IS_HE:
                p_v = v_ptr + t_abs * (H * V) + i_h * V
                b_v = tl.load(p_v + offs_c, mask=mask_c, other=0.0).to(tl.float32)
                state = b_eg * (state - b_beta * b_k[:, None] * k_state[None, :]) + (
                    b_beta * b_k[:, None] * b_v[None, :]
                )
            else:
                state = b_eg * (state - b_beta * b_k[:, None] * k_state[None, :])

        if IS_HE:
            tl.store(
                he_ptr + i_h * K * V + offs_k[:, None] * V + offs_c[None, :],
                state,
                mask=mask_c[None, :],
            )
        else:
            tl.store(
                M_ptr + i_h * K * K + offs_k[:, None] * K + offs_c[None, :],
                state,
                mask=mask_c[None, :],
            )

    @triton.jit(do_not_specialize=["t_start", "t_len"])
    def _affine_bwd_chunk_kernel(
        k_ptr,
        v_ptr,
        g_ptr,
        beta_ptr,
        he_buf_ptr,
        M_buf_ptr,
        dhe_ptr,
        dM_ptr,
        gk_ptr,
        gv_ptr,
        gg_ptr,
        gb_ptr,
        t_start,
        t_len,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
    ):
        """Reverse VJP over a chunk; he/M pre-token states from he_buf/M_buf[ti]."""
        i_h = tl.program_id(0)
        NK = (K + BK - 1) // BK
        NV = (V + BV - 1) // BV
        for ti_rev in range(0, t_len):
            ti = t_len - 1 - ti_rev
            # materialize pre-token state into registers via HBM scratch on dhe? use he from buf directly
            # Load he_buf[ti] / M_buf[ti] into temporary he/M working copies via local ptrs:
            # We treat he_buf[ti] as he_ptr and M_buf[ti] as M_ptr for this token.
            he_ptr = he_buf_ptr + ti * (H * K * V)
            M_ptr = M_buf_ptr + ti * (H * K * K)
            t_abs = t_start + ti
            p_k = k_ptr + t_abs * (H * K) + i_h * K
            p_v = v_ptr + t_abs * (H * V) + i_h * V
            b_eg = tl.exp(tl.load(g_ptr + t_abs * H + i_h).to(tl.float32))
            b_beta = tl.load(beta_ptr + t_abs * H + i_h).to(tl.float32)
            tr = tl.zeros([1], dtype=tl.float32)
            k_dA_k = tl.zeros([1], dtype=tl.float32)
            for i_v in range(0, NV):
                offs_v = i_v * BV + tl.arange(0, BV)
                mask_v = offs_v < V
                k_he = tl.zeros([BV], dtype=tl.float32)
                k_dhe = tl.zeros([BV], dtype=tl.float32)
                for i_k in range(0, NK):
                    offs_k = i_k * BK + tl.arange(0, BK)
                    mask_k = offs_k < K
                    b_k = tl.load(p_k + offs_k, mask=mask_k, other=0).to(tl.float32)
                    he_t = tl.load(
                        he_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                        mask=mask_k[:, None] & mask_v[None, :],
                        other=0.0,
                    )
                    dhe = tl.load(
                        dhe_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                        mask=mask_k[:, None] & mask_v[None, :],
                        other=0.0,
                    )
                    tr += tl.sum(he_t * dhe)
                    k_he += tl.sum(b_k[:, None] * he_t, 0)
                    k_dhe += tl.sum(b_k[:, None] * dhe, 0)
                k_dA_k += tl.sum(k_he * k_dhe)
            for i_c in range(0, NK):
                offs_c = i_c * BK + tl.arange(0, BK)
                mask_c = offs_c < K
                k_Mt = tl.zeros([BK], dtype=tl.float32)
                k_dM = tl.zeros([BK], dtype=tl.float32)
                for i_r in range(0, NK):
                    offs_r = i_r * BK + tl.arange(0, BK)
                    mask_r = offs_r < K
                    b_k = tl.load(p_k + offs_r, mask=mask_r, other=0).to(tl.float32)
                    M_t = tl.load(
                        M_ptr + i_h * K * K + offs_r[:, None] * K + offs_c[None, :],
                        mask=mask_r[:, None] & mask_c[None, :],
                        other=0.0,
                    )
                    dM = tl.load(
                        dM_ptr + i_h * K * K + offs_r[:, None] * K + offs_c[None, :],
                        mask=mask_r[:, None] & mask_c[None, :],
                        other=0.0,
                    )
                    tr += tl.sum(M_t * dM)
                    k_Mt += tl.sum(b_k[:, None] * M_t, 0)
                    k_dM += tl.sum(b_k[:, None] * dM, 0)
                k_dA_k += tl.sum(k_Mt * k_dM)

            tr_s = tl.sum(tr)
            kdak_s = tl.sum(k_dA_k)
            deg = tr_s - b_beta * kdak_s

            dbeta_b = tl.zeros([1], dtype=tl.float32)
            for i_v in range(0, NV):
                offs_v = i_v * BV + tl.arange(0, BV)
                mask_v = offs_v < V
                b_v = tl.load(p_v + offs_v, mask=mask_v, other=0).to(tl.float32)
                acc = tl.zeros([BV], dtype=tl.float32)
                for i_k in range(0, NK):
                    offs_k = i_k * BK + tl.arange(0, BK)
                    mask_k = offs_k < K
                    b_k = tl.load(p_k + offs_k, mask=mask_k, other=0).to(tl.float32)
                    dhe = tl.load(
                        dhe_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                        mask=mask_k[:, None] & mask_v[None, :],
                        other=0.0,
                    )
                    acc += tl.sum(b_k[:, None] * dhe, 0)
                dbeta_b += tl.sum(acc * b_v)
            dbeta = -b_eg * kdak_s + tl.sum(dbeta_b)
            tl.store(gg_ptr + t_abs * H + i_h, tl.load(gg_ptr + t_abs * H + i_h) + deg * b_eg)
            tl.store(gb_ptr + t_abs * H + i_h, tl.load(gb_ptr + t_abs * H + i_h) + dbeta)

            for i_k in range(0, NK):
                offs_k = i_k * BK + tl.arange(0, BK)
                mask_k = offs_k < K
                dk = tl.zeros([BK], dtype=tl.float32)
                for i_v in range(0, NV):
                    offs_v = i_v * BV + tl.arange(0, BV)
                    mask_v = offs_v < V
                    k_he = tl.zeros([BV], dtype=tl.float32)
                    k_dhe = tl.zeros([BV], dtype=tl.float32)
                    for i_kk in range(0, NK):
                        offs_kk = i_kk * BK + tl.arange(0, BK)
                        mask_kk = offs_kk < K
                        b_kk = tl.load(p_k + offs_kk, mask=mask_kk, other=0).to(tl.float32)
                        he_t = tl.load(
                            he_ptr + i_h * K * V + offs_kk[:, None] * V + offs_v[None, :],
                            mask=mask_kk[:, None] & mask_v[None, :],
                            other=0.0,
                        )
                        dhe = tl.load(
                            dhe_ptr + i_h * K * V + offs_kk[:, None] * V + offs_v[None, :],
                            mask=mask_kk[:, None] & mask_v[None, :],
                            other=0.0,
                        )
                        k_he += tl.sum(b_kk[:, None] * he_t, 0)
                        k_dhe += tl.sum(b_kk[:, None] * dhe, 0)
                    dhe_tile = tl.load(
                        dhe_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                        mask=mask_k[:, None] & mask_v[None, :],
                        other=0.0,
                    )
                    he_tile = tl.load(
                        he_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                        mask=mask_k[:, None] & mask_v[None, :],
                        other=0.0,
                    )
                    b_v = tl.load(p_v + offs_v, mask=mask_v, other=0).to(tl.float32)
                    dk += -b_eg * b_beta * (tl.sum(dhe_tile * k_he[None, :], 1) + tl.sum(he_tile * k_dhe[None, :], 1))
                    dk += b_beta * tl.sum(dhe_tile * b_v[None, :], 1)
                for i_c in range(0, NK):
                    offs_c = i_c * BK + tl.arange(0, BK)
                    mask_c = offs_c < K
                    k_Mt = tl.zeros([BK], dtype=tl.float32)
                    k_dM = tl.zeros([BK], dtype=tl.float32)
                    for i_r in range(0, NK):
                        offs_r = i_r * BK + tl.arange(0, BK)
                        mask_r = offs_r < K
                        b_kr = tl.load(p_k + offs_r, mask=mask_r, other=0).to(tl.float32)
                        M_t = tl.load(
                            M_ptr + i_h * K * K + offs_r[:, None] * K + offs_c[None, :],
                            mask=mask_r[:, None] & mask_c[None, :],
                            other=0.0,
                        )
                        dM = tl.load(
                            dM_ptr + i_h * K * K + offs_r[:, None] * K + offs_c[None, :],
                            mask=mask_r[:, None] & mask_c[None, :],
                            other=0.0,
                        )
                        k_Mt += tl.sum(b_kr[:, None] * M_t, 0)
                        k_dM += tl.sum(b_kr[:, None] * dM, 0)
                    dM_tile = tl.load(
                        dM_ptr + i_h * K * K + offs_k[:, None] * K + offs_c[None, :],
                        mask=mask_k[:, None] & mask_c[None, :],
                        other=0.0,
                    )
                    M_tile = tl.load(
                        M_ptr + i_h * K * K + offs_k[:, None] * K + offs_c[None, :],
                        mask=mask_k[:, None] & mask_c[None, :],
                        other=0.0,
                    )
                    dk += -b_eg * b_beta * (tl.sum(dM_tile * k_Mt[None, :], 1) + tl.sum(M_tile * k_dM[None, :], 1))
                gk_ptrs = gk_ptr + t_abs * (H * K) + i_h * K + offs_k
                tl.store(gk_ptrs, tl.load(gk_ptrs, mask=mask_k, other=0.0) + dk, mask=mask_k)

            for i_v in range(0, NV):
                offs_v = i_v * BV + tl.arange(0, BV)
                mask_v = offs_v < V
                dv = tl.zeros([BV], dtype=tl.float32)
                for i_k in range(0, NK):
                    offs_k = i_k * BK + tl.arange(0, BK)
                    mask_k = offs_k < K
                    b_k = tl.load(p_k + offs_k, mask=mask_k, other=0).to(tl.float32)
                    dhe = tl.load(
                        dhe_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                        mask=mask_k[:, None] & mask_v[None, :],
                        other=0.0,
                    )
                    dv += b_beta * tl.sum(dhe * b_k[:, None], 0)
                gv_ptrs = gv_ptr + t_abs * (H * V) + i_h * V + offs_v
                tl.store(gv_ptrs, tl.load(gv_ptrs, mask=mask_v, other=0.0) + dv, mask=mask_v)

            # push A^T through dhe / dM
            for i_v in range(0, NV):
                offs_v = i_v * BV + tl.arange(0, BV)
                mask_v = offs_v < V
                k_dhe = tl.zeros([BV], dtype=tl.float32)
                for i_k in range(0, NK):
                    offs_k = i_k * BK + tl.arange(0, BK)
                    mask_k = offs_k < K
                    b_k = tl.load(p_k + offs_k, mask=mask_k, other=0).to(tl.float32)
                    dhe = tl.load(
                        dhe_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                        mask=mask_k[:, None] & mask_v[None, :],
                        other=0.0,
                    )
                    k_dhe += tl.sum(b_k[:, None] * dhe, 0)
                for i_k in range(0, NK):
                    offs_k = i_k * BK + tl.arange(0, BK)
                    mask_k = offs_k < K
                    b_k = tl.load(p_k + offs_k, mask=mask_k, other=0).to(tl.float32)
                    dhe = tl.load(
                        dhe_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                        mask=mask_k[:, None] & mask_v[None, :],
                        other=0.0,
                    )
                    dhe = b_eg * (dhe - b_beta * b_k[:, None] * k_dhe[None, :])
                    tl.store(
                        dhe_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :],
                        dhe,
                        mask=mask_k[:, None] & mask_v[None, :],
                    )
            for i_c in range(0, NK):
                offs_c = i_c * BK + tl.arange(0, BK)
                mask_c = offs_c < K
                k_dM = tl.zeros([BK], dtype=tl.float32)
                for i_r in range(0, NK):
                    offs_r = i_r * BK + tl.arange(0, BK)
                    mask_r = offs_r < K
                    b_k = tl.load(p_k + offs_r, mask=mask_r, other=0).to(tl.float32)
                    dM = tl.load(
                        dM_ptr + i_h * K * K + offs_r[:, None] * K + offs_c[None, :],
                        mask=mask_r[:, None] & mask_c[None, :],
                        other=0.0,
                    )
                    k_dM += tl.sum(b_k[:, None] * dM, 0)
                for i_r in range(0, NK):
                    offs_r = i_r * BK + tl.arange(0, BK)
                    mask_r = offs_r < K
                    b_k = tl.load(p_k + offs_r, mask=mask_r, other=0).to(tl.float32)
                    dM = tl.load(
                        dM_ptr + i_h * K * K + offs_r[:, None] * K + offs_c[None, :],
                        mask=mask_r[:, None] & mask_c[None, :],
                        other=0.0,
                    )
                    dM = b_eg * (dM - b_beta * b_k[:, None] * k_dM[None, :])
                    tl.store(
                        dM_ptr + i_h * K * K + offs_r[:, None] * K + offs_c[None, :],
                        dM,
                        mask=mask_r[:, None] & mask_c[None, :],
                    )

    @triton.jit(do_not_specialize=["t_start", "t_len"])
    def _affine_bwd_chunk_he_coltile_kernel(
        k_ptr,
        v_ptr,
        g_ptr,
        beta_ptr,
        he_buf_ptr,
        dhe_ptr,
        gv_ptr,
        dk_partial_ptr,
        dg_partial_ptr,
        db_partial_ptr,
        t_start,
        t_len,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BC: tl.constexpr,
        BT: tl.constexpr,
    ):
        """Reverse one he column tile while keeping its adjoint in UB."""
        i_v = tl.program_id(0)
        i_h = tl.program_id(1)
        offs_k = tl.arange(0, K)
        offs_v = i_v * BC + tl.arange(0, BC)
        mask_v = offs_v < V

        dhe_ptrs = dhe_ptr + i_h * K * V + offs_k[:, None] * V + offs_v[None, :]
        dhe_ub = tl.load(dhe_ptrs, mask=mask_v[None, :], other=0.0).to(tl.float32)

        for ti_rev in range(0, t_len):
            ti = t_len - 1 - ti_rev
            t_abs = t_start + ti
            p_k = k_ptr + t_abs * (H * K) + i_h * K
            p_v = v_ptr + t_abs * (H * V) + i_h * V
            b_k = tl.load(p_k + offs_k).to(tl.float32)
            b_v = tl.load(p_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            b_eg = tl.exp(tl.load(g_ptr + t_abs * H + i_h).to(tl.float32))
            b_beta = tl.load(beta_ptr + t_abs * H + i_h).to(tl.float32)

            he_ptrs = he_buf_ptr + ti * (H * K * V) + i_h * K * V + offs_k[:, None] * V + offs_v[None, :]
            he_ub = tl.load(he_ptrs, mask=mask_v[None, :], other=0.0).to(tl.float32)
            k_he = tl.sum(b_k[:, None] * he_ub, 0)
            k_dhe = tl.sum(b_k[:, None] * dhe_ub, 0)
            kdak = tl.sum(k_he * k_dhe)
            tr = tl.sum(he_ub * dhe_ub)

            dk = -b_eg * b_beta * (tl.sum(dhe_ub * k_he[None, :], 1) + tl.sum(he_ub * k_dhe[None, :], 1))
            dk += b_beta * tl.sum(dhe_ub * b_v[None, :], 1)
            dg = b_eg * (tr - b_beta * kdak)
            db = -b_eg * kdak + tl.sum(k_dhe * b_v)
            dv = b_beta * k_dhe

            dk_ptrs = dk_partial_ptr + (((i_v * BT + ti) * H + i_h) * K) + offs_k
            scalar_offset = (i_v * BT + ti) * H + i_h
            tl.store(dk_ptrs, dk)
            tl.store(dg_partial_ptr + scalar_offset, dg)
            tl.store(db_partial_ptr + scalar_offset, db)
            tl.store(gv_ptr + t_abs * (H * V) + i_h * V + offs_v, dv, mask=mask_v)

            dhe_ub = b_eg * (dhe_ub - b_beta * b_k[:, None] * k_dhe[None, :])

        tl.store(dhe_ptrs, dhe_ub, mask=mask_v[None, :])

    @triton.jit(do_not_specialize=["t_start", "t_len"])
    def _affine_bwd_chunk_M_coltile_kernel(
        k_ptr,
        g_ptr,
        beta_ptr,
        M_buf_ptr,
        dM_ptr,
        dk_partial_ptr,
        dg_partial_ptr,
        db_partial_ptr,
        t_start,
        t_len,
        H: tl.constexpr,
        K: tl.constexpr,
        BC: tl.constexpr,
        BT: tl.constexpr,
        TILE_OFFSET: tl.constexpr,
    ):
        """Reverse one M column tile while keeping its adjoint in UB."""
        i_c = tl.program_id(0)
        i_h = tl.program_id(1)
        i_tile = TILE_OFFSET + i_c
        offs_k = tl.arange(0, K)
        offs_c = i_c * BC + tl.arange(0, BC)
        mask_c = offs_c < K

        dM_ptrs = dM_ptr + i_h * K * K + offs_k[:, None] * K + offs_c[None, :]
        dM_ub = tl.load(dM_ptrs, mask=mask_c[None, :], other=0.0).to(tl.float32)

        for ti_rev in range(0, t_len):
            ti = t_len - 1 - ti_rev
            t_abs = t_start + ti
            p_k = k_ptr + t_abs * (H * K) + i_h * K
            b_k = tl.load(p_k + offs_k).to(tl.float32)
            b_eg = tl.exp(tl.load(g_ptr + t_abs * H + i_h).to(tl.float32))
            b_beta = tl.load(beta_ptr + t_abs * H + i_h).to(tl.float32)

            M_ptrs = M_buf_ptr + ti * (H * K * K) + i_h * K * K + offs_k[:, None] * K + offs_c[None, :]
            M_ub = tl.load(M_ptrs, mask=mask_c[None, :], other=0.0).to(tl.float32)
            k_M = tl.sum(b_k[:, None] * M_ub, 0)
            k_dM = tl.sum(b_k[:, None] * dM_ub, 0)
            kdak = tl.sum(k_M * k_dM)
            tr = tl.sum(M_ub * dM_ub)

            dk = -b_eg * b_beta * (tl.sum(dM_ub * k_M[None, :], 1) + tl.sum(M_ub * k_dM[None, :], 1))
            dg = b_eg * (tr - b_beta * kdak)
            db = -b_eg * kdak

            dk_ptrs = dk_partial_ptr + (((i_tile * BT + ti) * H + i_h) * K) + offs_k
            scalar_offset = (i_tile * BT + ti) * H + i_h
            tl.store(dk_ptrs, dk)
            tl.store(dg_partial_ptr + scalar_offset, dg)
            tl.store(db_partial_ptr + scalar_offset, db)

            dM_ub = b_eg * (dM_ub - b_beta * b_k[:, None] * k_dM[None, :])

        tl.store(dM_ptrs, dM_ub, mask=mask_c[None, :])


def _triton_analytical_bwd(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    grad_hm: Tensor,
    *,
    cu_seqlens: Optional[Tensor] = None,
    cu_pts: Optional[Sequence[int]] = None,
    use_qk_l2norm: bool = True,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    from veomni.distributed.context_parallel.gdn_kcp import _l2norm_vjp, unpack_affine_hm

    key_in = key
    if use_qk_l2norm:
        key = key * torch.rsqrt(key.pow(2).sum(dim=-1, keepdim=True) + eps)

    k_c = key.contiguous().float()
    v_c = value.contiguous().float()
    g_c = g.contiguous().float()
    b_c = beta.contiguous().float()

    bsz, t_total, H, K = int(k_c.shape[0]), int(k_c.shape[1]), int(k_c.shape[2]), int(k_c.shape[3])
    V = int(v_c.shape[-1])
    device = k_c.device
    bt = _bwd_chunk()
    bk = bv = 32
    if K > 128 or V > 128 or K % bk != 0 or V % bv != 0:
        raise ValueError(f"triton bwd shape unsupported K={K} V={V}")

    starts, ends = _resolve_segment_bounds(cu_seqlens=cu_seqlens, cu_pts=cu_pts, bsz=bsz, t_total=t_total)
    if cu_seqlens is None and cu_pts is None:
        k_flat = k_c.reshape(bsz * t_total, H, K)
        v_flat = v_c.reshape(bsz * t_total, H, V)
        g_flat = g_c.reshape(bsz * t_total, H)
        b_flat = b_c.reshape(bsz * t_total, H)
    else:
        k_flat, v_flat, g_flat, b_flat = k_c[0], v_c[0], g_c[0], b_c[0]

    grad_k = torch.zeros_like(k_flat)
    grad_v = torch.zeros_like(v_flat)
    grad_g = torch.zeros_like(g_flat)
    grad_b = torch.zeros_like(b_flat)

    max_t = max((e - s) for s, e in zip(starts, ends)) if starts else 0
    n_ck_max = (max_t + bt - 1) // bt if max_t else 0
    he_ckpt = torch.empty(n_ck_max + 1, H, K, V, device=device, dtype=torch.float32)
    M_ckpt = torch.empty(n_ck_max + 1, H, K, K, device=device, dtype=torch.float32)
    he_buf = torch.empty(bt, H, K, V, device=device, dtype=torch.float32)
    M_buf = torch.empty(bt, H, K, K, device=device, dtype=torch.float32)
    he = torch.empty(H, K, V, device=device, dtype=torch.float32)
    M = torch.empty(H, K, K, device=device, dtype=torch.float32)
    eye = _cached_eye(device, H, K)

    for seg_i, (seg_start, seg_end) in enumerate(zip(starts, ends)):
        t_len = int(seg_end - seg_start)
        if t_len <= 0:
            continue
        n_ck = (t_len + bt - 1) // bt
        dhe, dM = unpack_affine_hm(grad_hm[seg_i].float().contiguous(), v_dim=V)
        dhe = dhe.contiguous()
        dM = dM.contiguous()

        he.zero_()
        M.copy_(eye)
        he_ckpt[0].copy_(he)
        M_ckpt[0].copy_(M)
        ck = 1
        for t in range(t_len):
            _affine_fwd_token_kernel[(H,)](
                k_flat,
                v_flat,
                g_flat,
                b_flat,
                he,
                M,
                int(seg_start + t),
                H=H,
                K=K,
                V=V,
                BK=bk,
                BV=bv,
            )
            if (t + 1) % bt == 0 or (t + 1) == t_len:
                he_ckpt[ck].copy_(he)
                M_ckpt[ck].copy_(M)
                ck += 1

        for c in range(n_ck - 1, -1, -1):
            t0 = c * bt
            t1 = min(t_len, (c + 1) * bt)
            clen = t1 - t0
            he.copy_(he_ckpt[c])
            M.copy_(M_ckpt[c])
            for i in range(clen):
                he_buf[i].copy_(he)
                M_buf[i].copy_(M)
                _affine_fwd_token_kernel[(H,)](
                    k_flat,
                    v_flat,
                    g_flat,
                    b_flat,
                    he,
                    M,
                    int(seg_start + t0 + i),
                    H=H,
                    K=K,
                    V=V,
                    BK=bk,
                    BV=bv,
                )
            for i in range(clen - 1, -1, -1):
                t = t0 + i
                he.copy_(he_buf[i])
                M.copy_(M_buf[i])
                _affine_bwd_token_kernel[(H,)](
                    k_flat,
                    v_flat,
                    g_flat,
                    b_flat,
                    he,
                    M,
                    dhe,
                    dM,
                    grad_k,
                    grad_v,
                    grad_g,
                    grad_b,
                    int(seg_start + t),
                    H=H,
                    K=K,
                    V=V,
                    BK=bk,
                    BV=bv,
                )

    if cu_seqlens is None and cu_pts is None:
        grad_k = grad_k.view(bsz, t_total, H, K)
        grad_v = grad_v.view(bsz, t_total, H, V)
        grad_g = grad_g.view(bsz, t_total, H)
        grad_b = grad_b.view(bsz, t_total, H)
    else:
        grad_k = grad_k.view(1, t_total, H, K)
        grad_v = grad_v.view(1, t_total, H, V)
        grad_g = grad_g.view(1, t_total, H)
        grad_b = grad_b.view(1, t_total, H)

    if use_qk_l2norm:
        grad_k = _l2norm_vjp(key_in.float(), grad_k, eps=eps)

    return (
        grad_k.to(dtype=key_in.dtype),
        grad_v.to(dtype=value.dtype),
        grad_g.to(dtype=g.dtype),
        grad_b.to(dtype=beta.dtype),
    )


def _triton_chunk_analytical_bwd(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    grad_hm: Tensor,
    *,
    cu_seqlens: Optional[Tensor] = None,
    cu_pts: Optional[Sequence[int]] = None,
    use_qk_l2norm: bool = True,
    eps: float = 1e-6,
    coltile: bool = False,
    fwd_coltile: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Host launches O(T/chunk); token loops live inside Triton.

    The coltile path is phase A: he/M use two grid=(tile,H) kernels. Replay,
    deterministic partial reduction, and output copies remain separate.
    The fwd-coltile mode also keeps one K-by-BC state tile in UB per program.
    """
    from veomni.distributed.context_parallel.gdn_kcp import _l2norm_vjp, unpack_affine_hm

    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(f"key/value must be 4D [B,T,H,D], got {tuple(key.shape)} / {tuple(value.shape)}")
    expected_prefix = tuple(key.shape[:3])
    if tuple(value.shape[:3]) != expected_prefix:
        raise ValueError(f"key/value [B,T,H] mismatch: key={tuple(key.shape)} value={tuple(value.shape)}")
    if tuple(g.shape) != expected_prefix or tuple(beta.shape) != expected_prefix:
        raise ValueError(
            f"g/beta must match key [B,T,H]={expected_prefix}, got g={tuple(g.shape)} beta={tuple(beta.shape)}"
        )
    if any(x.device != key.device for x in (value, g, beta, grad_hm)):
        raise ValueError("key/value/g/beta/grad_hm must be on the same device")

    key_in = key
    if use_qk_l2norm:
        key = key * torch.rsqrt(key.pow(2).sum(dim=-1, keepdim=True) + eps)

    k_c = key.contiguous().float()
    v_c = value.contiguous().float()
    g_c = g.contiguous().float()
    b_c = beta.contiguous().float()

    bsz, t_total, H, K = int(k_c.shape[0]), int(k_c.shape[1]), int(k_c.shape[2]), int(k_c.shape[3])
    V = int(v_c.shape[-1])
    if min(bsz, t_total, H, K, V) <= 0:
        raise ValueError(f"affine bwd dimensions must be positive, got B={bsz} T={t_total} H={H} K={K} V={V}")
    device = k_c.device
    bt = _bwd_chunk()
    bk = bv = 32
    fwd_bc = _fwd_coltile_bc() if fwd_coltile else 32
    if fwd_coltile and not coltile:
        raise ValueError("fwd_coltile requires coltile VJP")
    if K > 128 or V > 128 or K % bk != 0 or V % bv != 0:
        raise ValueError(f"triton_chunk bwd shape unsupported K={K} V={V}")
    if coltile and (K & (K - 1)) != 0:
        raise ValueError(f"triton_chunk_coltile requires power-of-two K, got K={K}")
    if fwd_coltile and (K % fwd_bc != 0 or V % fwd_bc != 0):
        raise ValueError(f"fwd-coltile requires K/V divisible by BC={fwd_bc}, got K={K} V={V}")

    starts, ends = _resolve_segment_bounds(cu_seqlens=cu_seqlens, cu_pts=cu_pts, bsz=bsz, t_total=t_total)
    varlen = not (cu_seqlens is None and cu_pts is None)
    if not varlen:
        k_flat = k_c.reshape(bsz * t_total, H, K)
        v_flat = v_c.reshape(bsz * t_total, H, V)
        g_flat = g_c.reshape(bsz * t_total, H)
        b_flat = b_c.reshape(bsz * t_total, H)
    else:
        k_flat, v_flat, g_flat, b_flat = k_c[0], v_c[0], g_c[0], b_c[0]

    expected_grad_shape = (len(starts), H, K, V + K)
    if tuple(grad_hm.shape) != expected_grad_shape:
        raise ValueError(f"grad_hm must be [N,H,K,V+K]={expected_grad_shape}, got {tuple(grad_hm.shape)}")

    grad_k = torch.zeros_like(k_flat)
    grad_v = torch.zeros_like(v_flat)
    grad_g = torch.zeros_like(g_flat)
    grad_b = torch.zeros_like(b_flat)

    max_t = max((e - s) for s, e in zip(starts, ends)) if starts else 0
    n_ck_max = (max_t + bt - 1) // bt if max_t else 0
    he_ckpt = torch.empty(n_ck_max + 1, H, K, V, device=device, dtype=torch.float32)
    M_ckpt = torch.empty(n_ck_max + 1, H, K, K, device=device, dtype=torch.float32)
    he_buf = torch.empty(bt, H, K, V, device=device, dtype=torch.float32)
    M_buf = torch.empty(bt, H, K, K, device=device, dtype=torch.float32)
    he = torch.empty(H, K, V, device=device, dtype=torch.float32)
    M = torch.empty(H, K, K, device=device, dtype=torch.float32)
    eye = _cached_eye(device, H, K)
    nt_he = V // bv
    nt_m = K // bk
    nt = nt_he + nt_m
    nt_fwd_he = V // fwd_bc
    nt_fwd_m = K // fwd_bc
    if coltile:
        dk_partial = torch.empty(nt, bt, H, K, device=device, dtype=torch.float32)
        dg_partial = torch.empty(nt, bt, H, device=device, dtype=torch.float32)
        db_partial = torch.empty(nt, bt, H, device=device, dtype=torch.float32)
    else:
        dk_partial = dg_partial = db_partial = None

    def _launch_fwd_chunk(t0_abs: int, clen: int, *, store: bool) -> None:
        if fwd_coltile:
            _affine_fwd_chunk_coltile_kernel[(nt_fwd_he, H)](
                k_flat,
                v_flat,
                g_flat,
                b_flat,
                he,
                M,
                he_buf,
                M_buf,
                int(t0_abs),
                int(clen),
                H=H,
                STORE_STATES=1 if store else 0,
                IS_HE=True,
                K=K,
                V=V,
                BC=fwd_bc,
            )
            _affine_fwd_chunk_coltile_kernel[(nt_fwd_m, H)](
                k_flat,
                v_flat,
                g_flat,
                b_flat,
                he,
                M,
                he_buf,
                M_buf,
                int(t0_abs),
                int(clen),
                H=H,
                STORE_STATES=1 if store else 0,
                IS_HE=False,
                K=K,
                V=V,
                BC=fwd_bc,
            )
        else:
            _affine_fwd_chunk_kernel[(H,)](
                k_flat,
                v_flat,
                g_flat,
                b_flat,
                he,
                M,
                he_buf,
                M_buf,
                int(t0_abs),
                int(clen),
                H=H,
                STORE_STATES=1 if store else 0,
                K=K,
                V=V,
                BK=bk,
                BV=bv,
            )

    def _launch_bwd_chunk(t0_abs: int, clen: int) -> None:
        if coltile:
            assert dk_partial is not None
            assert dg_partial is not None
            assert db_partial is not None
            _affine_bwd_chunk_he_coltile_kernel[(nt_he, H)](
                k_flat,
                v_flat,
                g_flat,
                b_flat,
                he_buf,
                dhe,
                grad_v,
                dk_partial,
                dg_partial,
                db_partial,
                int(t0_abs),
                int(clen),
                H=H,
                K=K,
                V=V,
                BC=bv,
                BT=bt,
            )
            _affine_bwd_chunk_M_coltile_kernel[(nt_m, H)](
                k_flat,
                g_flat,
                b_flat,
                M_buf,
                dM,
                dk_partial,
                dg_partial,
                db_partial,
                int(t0_abs),
                int(clen),
                H=H,
                K=K,
                BC=bk,
                BT=bt,
                TILE_OFFSET=nt_he,
            )
            grad_slice = slice(int(t0_abs), int(t0_abs + clen))
            grad_k[grad_slice].copy_(dk_partial[:, :clen].sum(dim=0))
            grad_g[grad_slice].copy_(dg_partial[:, :clen].sum(dim=0))
            grad_b[grad_slice].copy_(db_partial[:, :clen].sum(dim=0))
        else:
            _affine_bwd_chunk_kernel[(H,)](
                k_flat,
                v_flat,
                g_flat,
                b_flat,
                he_buf,
                M_buf,
                dhe,
                dM,
                grad_k,
                grad_v,
                grad_g,
                grad_b,
                int(t0_abs),
                int(clen),
                H=H,
                K=K,
                V=V,
                BK=bk,
                BV=bv,
            )

    for seg_i, (seg_start, seg_end) in enumerate(zip(starts, ends)):
        t_len = int(seg_end - seg_start)
        if t_len <= 0:
            continue
        n_ck = (t_len + bt - 1) // bt
        dhe, dM = unpack_affine_hm(grad_hm[seg_i].float().contiguous(), v_dim=V)
        dhe = dhe.contiguous()
        dM = dM.contiguous()

        he.zero_()
        M.copy_(eye)
        he_ckpt[0].copy_(he)
        M_ckpt[0].copy_(M)
        for c in range(n_ck):
            t0 = c * bt
            t1 = min(t_len, (c + 1) * bt)
            clen = t1 - t0
            _launch_fwd_chunk(seg_start + t0, clen, store=False)
            he_ckpt[c + 1].copy_(he)
            M_ckpt[c + 1].copy_(M)

        for c in range(n_ck - 1, -1, -1):
            t0 = c * bt
            t1 = min(t_len, (c + 1) * bt)
            clen = t1 - t0
            he.copy_(he_ckpt[c])
            M.copy_(M_ckpt[c])
            _launch_fwd_chunk(seg_start + t0, clen, store=True)
            _launch_bwd_chunk(seg_start + t0, clen)

    if not varlen:
        grad_k = grad_k.view(bsz, t_total, H, K)
        grad_v = grad_v.view(bsz, t_total, H, V)
        grad_g = grad_g.view(bsz, t_total, H)
        grad_b = grad_b.view(bsz, t_total, H)
    else:
        grad_k = grad_k.view(1, t_total, H, K)
        grad_v = grad_v.view(1, t_total, H, V)
        grad_g = grad_g.view(1, t_total, H)
        grad_b = grad_b.view(1, t_total, H)

    if use_qk_l2norm:
        grad_k = _l2norm_vjp(key_in.float(), grad_k, eps=eps)

    return (
        grad_k.to(dtype=key_in.dtype),
        grad_v.to(dtype=value.dtype),
        grad_g.to(dtype=g.dtype),
        grad_b.to(dtype=beta.dtype),
    )


def ttx_local_affine_analytical_bwd(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    grad_hm: Tensor,
    *,
    cu_seqlens: Optional[Tensor] = None,
    cu_pts: Optional[Sequence[int]] = None,
    use_qk_l2norm: bool = True,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run the frozen persistent column-tiled BC8/M1 backward."""
    triton_ready = _TRITON_OK and _npu_ready() and key.device.type == "npu"
    if not triton_ready:
        raise RuntimeError(
            "KCP ttx_bc8_m1 backward requires Triton on an available NPU; "
            f"triton_ok={_TRITON_OK} device={key.device.type}"
        )
    return _triton_chunk_analytical_bwd(
        key,
        value,
        g,
        beta,
        grad_hm,
        cu_seqlens=cu_seqlens,
        cu_pts=cu_pts,
        use_qk_l2norm=use_qk_l2norm,
        eps=eps,
        coltile=True,
        fwd_coltile=True,
    )
