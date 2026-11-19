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
"""Hole attention over K canvases packed onto one decoder sequence.

The decoder denoises K canvases against one clean encoder cache. Canvas ``k``
may read a *prefix* of that cache and its own noisy tokens, nothing else -- the
holes are the other canvases. ``dllm_attention`` serves the same semantics by
materialising them: it gathers a private copy of every canvas's visible prefix
and concatenates ``prefix || canvas`` into one contiguous KV per segment, so
the KV it computes on grows as ``K*P + K^2*C/2`` and is built twice per layer.

Here the cache is read where it already lives. Each query tile carries the
half-open interval ``[lo, hi)`` of clean keys it may see, so the KV is
``P + K*C`` and no copy is made. The interval is why this kernel needs no
notion of prompts, blocks, padding or windows: left padding moves ``lo``, a
sliding layer moves ``lo``, the future canvases move ``hi``, and the kernel
only ever compares an index against two numbers.

A tile never straddles a canvas, so its interval is a scalar pair rather than a
mask. That is what keeps the visibility out of the inner loop: the forward and
``dq`` loops are *ranged* over ``[lo, hi)``, and the ``dkdv`` loop skips whole
canvases whose interval misses the key tile it owns instead of masking them.
"""

from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl


#: ``tl.dot`` needs at least 16 along every axis, so this is also the coarsest
#: granularity a canvas boundary can be resolved at.
MIN_BLOCK = 16
#: Widest head this kernel will accumulate. DiffusionGemma runs 256 on the
#: sliding layers and 512 on the global ones.
MAX_HEAD_DIM = 512
#: Deepest pipeline worth asking for; past this the tiles stop fitting anyway.
MAX_STAGES = 3
#: Shared memory left to whatever Triton stages besides the tiles counted here.
SHARED_MEMORY_RESERVE = 8 * 1024


def _divisor(canvas_length: int, preferred: int) -> int:
    """Largest power-of-two tile up to ``preferred`` that divides the canvas.

    Rounding down to a power of two comes first. ``min`` alone would hand back
    ``canvas_length`` itself whenever it is the smaller of the two, and a canvas
    of 48 -- a legal multiple of 16 -- would reach ``tl.arange`` as 48 and fail
    to compile rather than tile as 16.
    """
    tile = min(preferred, canvas_length)
    tile = 1 << (tile.bit_length() - 1)
    while tile > MIN_BLOCK and canvas_length % tile:
        tile //= 2
    return tile


def _fwd_tiles(headdim: int, canvas_length: int, element_size: int) -> tuple[int, int, int, int]:
    """``(BLOCK_M, BLOCK_N, num_warps, maxnreg)`` for the forward and ``dq``.

    The output accumulator is one ``BLOCK_M x head_dim`` fp32 tile in registers
    and the operands are resident in shared memory for the whole loop, so both
    have to shrink as the head widens -- the same reason ``dllm_attention``
    steps its own tiles down at 256 and 512, and these are its sizes there.
    fp32 halves them again, being the same tile at twice the bytes.

    These are a starting point, not the final choice: `_schedule` narrows them
    further if the device cannot hold them.
    """
    if headdim <= 64:
        block_m, block_n, warps, maxnreg = 64, 64, 4, 128
    elif headdim <= 128:
        block_m, block_n, warps, maxnreg = 128, 64, 8, 128
    elif headdim <= 256:
        block_m, block_n, warps, maxnreg = 32, 32, 8, 255
    else:
        block_m, block_n, warps, maxnreg = 16, 16, 8, 255
    block_m, block_n = _narrow(block_m, block_n, element_size)
    return _divisor(canvas_length, block_m), _divisor(canvas_length, block_n), warps, maxnreg


def _dkdv_tiles(headdim: int, canvas_length: int, element_size: int) -> tuple[int, int, int | None]:
    """``(BLOCK_M, BLOCK_N, maxnreg)`` for ``dkdv``, which holds two accumulators.

    The narrow-head row runs uncapped, as ``dllm_attention`` runs its own dK/dV
    kernel. Two ``128 x 128`` fp32 accumulators are 128 registers per thread on
    their own, so a 128 cap there would spend the whole budget on them and
    spill everything else.
    """
    if headdim <= 128:
        block_m, block_n, maxnreg = 32, 128, None
    elif headdim <= 256:
        block_m, block_n, maxnreg = 16, 32, 255
    else:
        block_m, block_n, maxnreg = 16, 16, 255
    block_m, block_n = _narrow(block_m, block_n, element_size)
    return _divisor(canvas_length, block_m), _divisor(canvas_length, block_n), maxnreg


def _narrow(block_m: int, block_n: int, element_size: int) -> tuple[int, int]:
    """Halve a bf16-sized tile pair for a wider element type."""
    if element_size <= 2:
        return block_m, block_n
    return max(MIN_BLOCK, block_m // 2), max(MIN_BLOCK, block_n // 2)


@triton.jit
def _hole_fwd_kernel(
    Q,
    KC,
    VC,
    KN,
    VN,
    CleanLo,
    CleanHi,
    Out,
    LSE,
    scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kcb,
    stride_kch,
    stride_kcn,
    stride_vcb,
    stride_vch,
    stride_vcn,
    stride_knb,
    stride_knh,
    stride_knn,
    stride_vnb,
    stride_vnh,
    stride_vnn,
    stride_ob,
    stride_oh,
    stride_om,
    stride_interval,
    nheads_q: tl.constexpr,
    nheads_kv: tl.constexpr,
    seqlen_q,
    clean_length,
    canvas_length: tl.constexpr,
    headdim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    PV_FP32: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    off_hb = tl.program_id(1)
    off_b = (off_hb // nheads_q).to(tl.int64)
    off_hq = off_hb % nheads_q
    off_hkv = off_hq // (nheads_q // nheads_kv)

    Q += off_b * stride_qb + off_hq * stride_qh
    KC += off_b * stride_kcb + off_hkv * stride_kch
    VC += off_b * stride_vcb + off_hkv * stride_vch
    KN += off_b * stride_knb + off_hkv * stride_knh
    VN += off_b * stride_vnb + off_hkv * stride_vnh
    Out += off_b * stride_ob + off_hq * stride_oh
    LSE += off_hb * seqlen_q

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_m < seqlen_q
    d_mask = offs_d < headdim

    q = tl.load(
        Q + offs_m[:, None] * stride_qm + offs_d[None, :],
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    # One tile lives in one canvas, so its visible interval is a scalar pair.
    canvas = start_m // canvas_length
    lo = tl.maximum(tl.load(CleanLo + off_b * stride_interval + canvas), 0)
    hi = tl.minimum(tl.load(CleanHi + off_b * stride_interval + canvas), clean_length)

    # An empty interval must not enter the loop at all. Its one tile would have
    # nothing visible, and a softmax whose running max is still -inf rescales by
    # exp(-inf - -inf), which is NaN rather than the zero it means. Starting at
    # `hi` instead of below `lo` is what makes the range empty in that case.
    for tile in range(tl.where(hi > lo, (lo // BLOCK_N) * BLOCK_N, hi), hi, BLOCK_N):
        key_offsets = tile + offs_n
        visible = (key_offsets >= lo) & (key_offsets < hi)
        k = tl.load(
            KC + key_offsets[:, None] * stride_kcn + offs_d[None, :],
            mask=visible[:, None] & d_mask[None, :],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k), input_precision=DOT_PRECISION)
        scores = tl.where(q_mask[:, None] & visible[None, :], scores * scale, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        probabilities = tl.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(probabilities, axis=1)
        v = tl.load(
            VC + key_offsets[:, None] * stride_vcn + offs_d[None, :],
            mask=visible[:, None] & d_mask[None, :],
            other=0.0,
        )
        if PV_FP32:
            acc = tl.dot(probabilities, v.to(tl.float32), acc=acc * alpha[:, None], input_precision="ieee")
        else:
            acc = tl.dot(probabilities.to(v.dtype), v, acc=acc * alpha[:, None], input_precision=DOT_PRECISION)
        m_i = m_new

    noise_start = canvas * canvas_length
    for tile in range(noise_start, noise_start + canvas_length, BLOCK_N):
        key_offsets = tile + offs_n
        visible = key_offsets < seqlen_q
        k = tl.load(
            KN + key_offsets[:, None] * stride_knn + offs_d[None, :],
            mask=visible[:, None] & d_mask[None, :],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k), input_precision=DOT_PRECISION)
        scores = tl.where(q_mask[:, None] & visible[None, :], scores * scale, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        probabilities = tl.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(probabilities, axis=1)
        v = tl.load(
            VN + key_offsets[:, None] * stride_vnn + offs_d[None, :],
            mask=visible[:, None] & d_mask[None, :],
            other=0.0,
        )
        if PV_FP32:
            acc = tl.dot(probabilities, v.to(tl.float32), acc=acc * alpha[:, None], input_precision="ieee")
        else:
            acc = tl.dot(probabilities.to(v.dtype), v, acc=acc * alpha[:, None], input_precision=DOT_PRECISION)
        m_i = m_new

    tl.store(
        Out + offs_m[:, None] * stride_om + offs_d[None, :],
        (acc / l_i[:, None]).to(Out.dtype.element_ty),
        mask=q_mask[:, None] & d_mask[None, :],
    )
    tl.store(LSE + offs_m, m_i + tl.log(l_i), mask=q_mask)


@triton.jit
def _hole_bwd_dq_kernel(
    Q,
    KC,
    VC,
    KN,
    VN,
    O,
    DO,
    CleanLo,
    CleanHi,
    DQ,
    LSE,
    Delta,
    scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kcb,
    stride_kch,
    stride_kcn,
    stride_vcb,
    stride_vch,
    stride_vcn,
    stride_knb,
    stride_knh,
    stride_knn,
    stride_vnb,
    stride_vnh,
    stride_vnn,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_interval,
    nheads_q: tl.constexpr,
    nheads_kv: tl.constexpr,
    seqlen_q,
    clean_length,
    canvas_length: tl.constexpr,
    headdim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    PV_FP32: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    off_hb = tl.program_id(1)
    off_b = (off_hb // nheads_q).to(tl.int64)
    off_hq = off_hb % nheads_q
    off_hkv = off_hq // (nheads_q // nheads_kv)

    Q += off_b * stride_qb + off_hq * stride_qh
    KC += off_b * stride_kcb + off_hkv * stride_kch
    VC += off_b * stride_vcb + off_hkv * stride_vch
    KN += off_b * stride_knb + off_hkv * stride_knh
    VN += off_b * stride_vnb + off_hkv * stride_vnh
    O += off_b * stride_ob + off_hq * stride_oh
    DO += off_b * stride_ob + off_hq * stride_oh
    DQ += off_b * stride_dqb + off_hq * stride_dqh
    LSE += off_hb * seqlen_q
    Delta += off_hb * seqlen_q

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_m < seqlen_q
    d_mask = offs_d < headdim
    qd_mask = q_mask[:, None] & d_mask[None, :]

    # Delta is needed by every score tile here and by both dK/dV kernels. The
    # query owns it, so compute it once and publish it before going on. Reading
    # `o` before `q` keeps the two from being live at once, which is what the
    # loop's register and shared-memory budget assumes.
    do = tl.load(DO + offs_m[:, None] * stride_om + offs_d[None, :], mask=qd_mask, other=0.0)
    o = tl.load(O + offs_m[:, None] * stride_om + offs_d[None, :], mask=qd_mask, other=0.0)
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    tl.store(Delta + offs_m, delta, mask=q_mask)

    q = tl.load(Q + offs_m[:, None] * stride_qm + offs_d[None, :], mask=qd_mask, other=0.0)
    lse = tl.load(LSE + offs_m, mask=q_mask, other=0.0)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    canvas = start_m // canvas_length
    lo = tl.maximum(tl.load(CleanLo + off_b * stride_interval + canvas), 0)
    hi = tl.minimum(tl.load(CleanHi + off_b * stride_interval + canvas), clean_length)

    # Empty stays empty here for the same reason as the forward: see there.
    for tile in range(tl.where(hi > lo, (lo // BLOCK_N) * BLOCK_N, hi), hi, BLOCK_N):
        key_offsets = tile + offs_n
        visible = (key_offsets >= lo) & (key_offsets < hi)
        kd_mask = visible[:, None] & d_mask[None, :]
        k = tl.load(KC + key_offsets[:, None] * stride_kcn + offs_d[None, :], mask=kd_mask, other=0.0)
        v = tl.load(VC + key_offsets[:, None] * stride_vcn + offs_d[None, :], mask=kd_mask, other=0.0)

        scores = tl.dot(q, tl.trans(k), input_precision=DOT_PRECISION)
        scores = tl.where(q_mask[:, None] & visible[None, :], scores * scale, -float("inf"))
        dp = tl.dot(do, tl.trans(v), input_precision=DOT_PRECISION)
        probabilities = tl.exp(scores - lse[:, None])
        ds = probabilities * (dp - delta[:, None]) * scale
        if PV_FP32:
            acc = tl.dot(ds, k.to(tl.float32), acc=acc, input_precision="ieee")
        else:
            acc = tl.dot(ds.to(k.dtype), k, acc=acc, input_precision=DOT_PRECISION)

    noise_start = canvas * canvas_length
    for tile in range(noise_start, noise_start + canvas_length, BLOCK_N):
        key_offsets = tile + offs_n
        visible = key_offsets < seqlen_q
        kd_mask = visible[:, None] & d_mask[None, :]
        k = tl.load(KN + key_offsets[:, None] * stride_knn + offs_d[None, :], mask=kd_mask, other=0.0)
        v = tl.load(VN + key_offsets[:, None] * stride_vnn + offs_d[None, :], mask=kd_mask, other=0.0)

        scores = tl.dot(q, tl.trans(k), input_precision=DOT_PRECISION)
        scores = tl.where(q_mask[:, None] & visible[None, :], scores * scale, -float("inf"))
        dp = tl.dot(do, tl.trans(v), input_precision=DOT_PRECISION)
        probabilities = tl.exp(scores - lse[:, None])
        ds = probabilities * (dp - delta[:, None]) * scale
        if PV_FP32:
            acc = tl.dot(ds, k.to(tl.float32), acc=acc, input_precision="ieee")
        else:
            acc = tl.dot(ds.to(k.dtype), k, acc=acc, input_precision=DOT_PRECISION)

    tl.store(DQ + offs_m[:, None] * stride_dqm + offs_d[None, :], acc.to(DQ.dtype.element_ty), mask=qd_mask)


@triton.jit
def _hole_bwd_dkdv_kernel(
    Q,
    K,
    V,
    DO,
    CleanLo,
    CleanHi,
    DK,
    DV,
    LSE,
    Delta,
    scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_interval,
    nheads_q: tl.constexpr,
    nheads_kv: tl.constexpr,
    seqlen_q,
    key_length,
    num_canvases,
    canvas_length: tl.constexpr,
    headdim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    PV_FP32: tl.constexpr,
    NOISE_REGION: tl.constexpr,
):
    """One key tile's dK/dV, summed over every query that can see it.

    The key tile owns its gradient: no other program writes it, so the sum runs
    in registers and lands in one store. That is only true because the clean
    cache is per sample here -- a cache shared across the batch would need a
    per-sample buffer, or atomics, to combine what different programs computed.

    A clean tile is read by whichever canvases' intervals cover it, which under
    a sliding window is one or two of them. Walking the canvases and testing the
    interval first is what keeps the other K-2 off this tile's bill; masking
    them inside the query loop instead would cost the full ``K*C x P`` product.
    """
    start_n = tl.program_id(0) * BLOCK_N
    off_hb = tl.program_id(1)
    off_b = (off_hb // nheads_kv).to(tl.int64)
    off_hkv = off_hb % nheads_kv

    Q += off_b * stride_qb
    K += off_b * stride_kb + off_hkv * stride_kh
    V += off_b * stride_vb + off_hkv * stride_vh
    DO += off_b * stride_dob
    DK += off_b * stride_dkb + off_hkv * stride_dkh
    DV += off_b * stride_dvb + off_hkv * stride_dvh

    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    n_mask = offs_n < key_length
    d_mask = offs_d < headdim
    nd_mask = n_mask[:, None] & d_mask[None, :]

    k = tl.load(K + offs_n[:, None] * stride_kn + offs_d[None, :], mask=nd_mask, other=0.0)
    v = tl.load(V + offs_n[:, None] * stride_vn + offs_d[None, :], mask=nd_mask, other=0.0)
    dk = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)

    if NOISE_REGION:
        # A noise tile sits inside one canvas, and only that canvas reads it.
        first_canvas = start_n // canvas_length
        last_canvas = first_canvas + 1
    else:
        first_canvas = 0
        last_canvas = num_canvases

    for group_index in range(0, GROUP_SIZE):
        off_hq = off_hkv * GROUP_SIZE + group_index
        q_head = Q + off_hq * stride_qh
        do_head = DO + off_hq * stride_doh
        off_hqb = off_b * nheads_q + off_hq
        lse_head = LSE + off_hqb * seqlen_q
        delta_head = Delta + off_hqb * seqlen_q

        for canvas in range(first_canvas, last_canvas):
            if NOISE_REGION:
                lo = start_n
                hi = start_n + BLOCK_N
            else:
                lo = tl.maximum(tl.load(CleanLo + off_b * stride_interval + canvas), 0)
                hi = tl.minimum(tl.load(CleanHi + off_b * stride_interval + canvas), key_length)

            if (lo < start_n + BLOCK_N) & (hi > start_n) & (hi > lo):
                visible = n_mask & (offs_n >= lo) & (offs_n < hi)
                canvas_start = canvas * canvas_length
                for start_m in range(canvas_start, canvas_start + canvas_length, BLOCK_M):
                    query_offsets = start_m + offs_m
                    q_mask = query_offsets < seqlen_q
                    qd_mask = q_mask[:, None] & d_mask[None, :]
                    q = tl.load(q_head + query_offsets[:, None] * stride_qm + offs_d[None, :], mask=qd_mask, other=0.0)
                    do = tl.load(
                        do_head + query_offsets[:, None] * stride_dom + offs_d[None, :], mask=qd_mask, other=0.0
                    )

                    scores = tl.dot(q, tl.trans(k), input_precision=DOT_PRECISION)
                    scores = tl.where(q_mask[:, None] & visible[None, :], scores * scale, -float("inf"))
                    dp = tl.dot(do, tl.trans(v), input_precision=DOT_PRECISION)
                    lse = tl.load(lse_head + query_offsets, mask=q_mask, other=0.0)
                    delta = tl.load(delta_head + query_offsets, mask=q_mask, other=0.0)
                    probabilities = tl.exp(scores - lse[:, None])
                    ds = probabilities * (dp - delta[:, None]) * scale

                    if PV_FP32:
                        dk = tl.dot(tl.trans(ds), q.to(tl.float32), acc=dk, input_precision="ieee")
                        dv = tl.dot(tl.trans(probabilities), do.to(tl.float32), acc=dv, input_precision="ieee")
                    else:
                        dk = tl.dot(tl.trans(ds.to(k.dtype)), q, acc=dk, input_precision=DOT_PRECISION)
                        dv = tl.dot(tl.trans(probabilities.to(k.dtype)), do, acc=dv, input_precision=DOT_PRECISION)

    tl.store(DK + offs_n[:, None] * stride_dkn + offs_d[None, :], dk.to(DK.dtype.element_ty), mask=nd_mask)
    tl.store(DV + offs_n[:, None] * stride_dvn + offs_d[None, :], dv.to(DV.dtype.element_ty), mask=nd_mask)


def _dot_precision(dtype: torch.dtype) -> str:
    """``tl.dot`` only honours this for fp32 inputs; keep those exact."""
    return "ieee" if dtype == torch.float32 else "tf32"


def _pv_in_fp32() -> bool:
    """Debug knob: carry the probabilities into every dot in fp32.

    Softmax already runs in fp32 registers; what gets rounded is the operand
    handed to the dot -- ``P`` against ``V`` in the forward, ``dS`` against
    ``K`` and ``P`` against ``dO`` in the backward. At the shipped
    ``sm_scale=1.0`` the top weight is 0.88-0.92, so bf16's 8 mantissa bits
    land that rounding on the one term that carries the row, which is the
    suspected origin of the kernel arm's flatter weights (constraint 41).

    Off by default and read per call: this trades several times the runtime for
    accuracy and exists to attribute a training difference, not to ship.
    Enable with ``VEOMNI_DG_HOLE_PV_FP32=1``.
    """
    return os.environ.get("VEOMNI_DG_HOLE_PV_FP32", "0") == "1"


@functools.lru_cache(maxsize=None)
def _shared_memory_limit(device_index: int) -> int:
    """Opt-in shared memory per block, which a launch may not exceed."""
    return torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin


def _schedule(
    block_m: int,
    block_n: int,
    block_d: int,
    element_size: int,
    device: torch.device,
    resident: tuple[int, int],
    streamed: tuple[int, int],
) -> tuple[int, int, int]:
    """Tile sizes and pipeline depth that fit the device's shared memory.

    A loop holds two kinds of tile, counted here in rows of ``block_d`` as
    multiples of ``(BLOCK_M, BLOCK_N)``. The resident ones are loaded once and
    live for the whole loop -- the query and its gradient, or the key tile a
    ``dkdv`` program owns -- and no pipeline depth makes them cheaper. The
    streamed ones are the per-iteration operands, and those are what a stage
    buys another copy of.

    Triton's own guess counts neither and asks for more than the device has.
    Depth alone cannot save a loop whose resident half already overruns, and
    the sizes in ``_fwd_tiles`` are chosen for an 160KB Ampere block, so on a
    smaller one the tiles have to come down too. Halving keeps them powers of
    two at or above ``MIN_BLOCK``, which still divide the canvas.
    """
    limit = _shared_memory_limit(device.index or 0) - SHARED_MEMORY_RESERVE
    row = block_d * element_size
    while True:
        resident_bytes = (resident[0] * block_m + resident[1] * block_n) * row
        streamed_bytes = (streamed[0] * block_m + streamed[1] * block_n) * row
        if resident_bytes + streamed_bytes <= limit:
            return block_m, block_n, max(1, min(MAX_STAGES, (limit - resident_bytes) // streamed_bytes))
        if block_m == MIN_BLOCK and block_n == MIN_BLOCK:
            raise ValueError(
                f"dllm_hole_attention cannot fit head_dim {block_d} at {element_size} bytes into this "
                f"device's {limit + SHARED_MEMORY_RESERVE} bytes of shared memory, even at its narrowest tile"
            )
        block_m = max(MIN_BLOCK, block_m // 2)
        block_n = max(MIN_BLOCK, block_n // 2)


class _HoleAttention(torch.autograd.Function):
    """Autograd function for DiffusionGemma hole attention over clean cache and noisy canvas."""

    @staticmethod
    def forward(ctx, query, clean_key, clean_value, noise_key, noise_value, clean_lo, clean_hi, sm_scale):
        """Run hole attention forward pass."""
        batch, nheads_q, seqlen_q, headdim = query.shape
        nheads_kv = clean_key.shape[1]
        clean_length = clean_key.shape[2]
        canvas_length = seqlen_q // clean_lo.shape[1]

        block_m, block_n, warps, maxnreg = _fwd_tiles(headdim, canvas_length, query.element_size())
        block_d = triton.next_power_of_2(headdim)
        precision = _dot_precision(query.dtype)
        pv_fp32 = _pv_in_fp32()
        # Resident: the query tile. Streamed: this iteration's K and V.
        block_m, block_n, stages = _schedule(
            block_m, block_n, block_d, query.element_size(), query.device, resident=(1, 0), streamed=(0, 2)
        )

        out = torch.empty_like(query)
        lse = torch.empty((batch * nheads_q, seqlen_q), dtype=torch.float32, device=query.device)

        _hole_fwd_kernel[(triton.cdiv(seqlen_q, block_m), batch * nheads_q)](
            query,
            clean_key,
            clean_value,
            noise_key,
            noise_value,
            clean_lo,
            clean_hi,
            out,
            lse,
            sm_scale,
            *query.stride()[:3],
            *clean_key.stride()[:3],
            *clean_value.stride()[:3],
            *noise_key.stride()[:3],
            *noise_value.stride()[:3],
            *out.stride()[:3],
            clean_lo.stride(0),
            nheads_q=nheads_q,
            nheads_kv=nheads_kv,
            seqlen_q=seqlen_q,
            clean_length=clean_length,
            canvas_length=canvas_length,
            headdim=headdim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            DOT_PRECISION=precision,
            PV_FP32=pv_fp32,
            num_warps=warps,
            num_stages=stages,
            maxnreg=maxnreg,
        )

        ctx.save_for_backward(query, clean_key, clean_value, noise_key, noise_value, clean_lo, clean_hi, out, lse)
        ctx.sm_scale = sm_scale
        ctx.canvas_length = canvas_length
        ctx.precision = precision
        ctx.pv_fp32 = pv_fp32
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """Run hole attention backward pass."""
        query, clean_key, clean_value, noise_key, noise_value, clean_lo, clean_hi, out, lse = ctx.saved_tensors
        batch, nheads_q, seqlen_q, headdim = query.shape
        nheads_kv = clean_key.shape[1]
        clean_length = clean_key.shape[2]
        canvas_length = ctx.canvas_length
        canvases = clean_lo.shape[1]
        scale = ctx.sm_scale
        precision = ctx.precision
        pv_fp32 = ctx.pv_fp32
        block_d = triton.next_power_of_2(headdim)

        grad_output = grad_output.contiguous()
        # Every gradient element is summed in an fp32 register and stored once --
        # no atomics, no partial buffers -- so the input dtype is rounded to
        # exactly as late as an fp32 staging buffer would round it, without
        # costing a second copy of all five gradients.
        grad_query = torch.empty_like(query)
        grad_clean_key = torch.empty_like(clean_key)
        grad_clean_value = torch.empty_like(clean_value)
        grad_noise_key = torch.empty_like(noise_key)
        grad_noise_value = torch.empty_like(noise_value)
        delta = torch.empty((batch * nheads_q, seqlen_q), dtype=torch.float32, device=query.device)

        block_m, block_n, warps, maxnreg = _fwd_tiles(headdim, canvas_length, query.element_size())
        # Resident: the query tile and its grad, which outlive `o`. Streamed: K and V.
        block_m, block_n, stages = _schedule(
            block_m, block_n, block_d, query.element_size(), query.device, resident=(2, 0), streamed=(0, 2)
        )
        _hole_bwd_dq_kernel[(triton.cdiv(seqlen_q, block_m), batch * nheads_q)](
            query,
            clean_key,
            clean_value,
            noise_key,
            noise_value,
            out,
            grad_output,
            clean_lo,
            clean_hi,
            grad_query,
            lse,
            delta,
            scale,
            *query.stride()[:3],
            *clean_key.stride()[:3],
            *clean_value.stride()[:3],
            *noise_key.stride()[:3],
            *noise_value.stride()[:3],
            *out.stride()[:3],
            *grad_query.stride()[:3],
            clean_lo.stride(0),
            nheads_q=nheads_q,
            nheads_kv=nheads_kv,
            seqlen_q=seqlen_q,
            clean_length=clean_length,
            canvas_length=canvas_length,
            headdim=headdim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            DOT_PRECISION=precision,
            PV_FP32=pv_fp32,
            num_warps=warps,
            num_stages=stages,
            maxnreg=maxnreg,
        )

        block_m, block_n, maxnreg = _dkdv_tiles(headdim, canvas_length, query.element_size())
        # Resident: the key tile this program owns, and its value.
        # Streamed: each query tile and its grad.
        block_m, block_n, stages = _schedule(
            block_m, block_n, block_d, query.element_size(), query.device, resident=(0, 2), streamed=(2, 0)
        )
        group_size = nheads_q // nheads_kv
        for keys, values, grad_keys, grad_values, length, noise in (
            (clean_key, clean_value, grad_clean_key, grad_clean_value, clean_length, False),
            (noise_key, noise_value, grad_noise_key, grad_noise_value, seqlen_q, True),
        ):
            _hole_bwd_dkdv_kernel[(triton.cdiv(length, block_n), batch * nheads_kv)](
                query,
                keys,
                values,
                grad_output,
                clean_lo,
                clean_hi,
                grad_keys,
                grad_values,
                lse,
                delta,
                scale,
                *query.stride()[:3],
                *keys.stride()[:3],
                *values.stride()[:3],
                *grad_output.stride()[:3],
                *grad_keys.stride()[:3],
                *grad_values.stride()[:3],
                clean_lo.stride(0),
                nheads_q=nheads_q,
                nheads_kv=nheads_kv,
                seqlen_q=seqlen_q,
                key_length=length,
                num_canvases=canvases,
                canvas_length=canvas_length,
                headdim=headdim,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                GROUP_SIZE=group_size,
                DOT_PRECISION=precision,
                PV_FP32=pv_fp32,
                NOISE_REGION=noise,
                num_warps=8,
                num_stages=stages,
                maxnreg=maxnreg,
            )

        return grad_query, grad_clean_key, grad_clean_value, grad_noise_key, grad_noise_value, None, None, None


def _validate(query, clean_key, clean_value, noise_key, noise_value, clean_lo, clean_hi, sm_scale):
    if not query.is_cuda:
        raise ValueError("dllm_hole_attention is a Triton kernel and needs CUDA tensors")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"dllm_hole_attention supports fp16/bf16/fp32, got {query.dtype}")
    for name, tensor in (
        ("clean_key", clean_key),
        ("clean_value", clean_value),
        ("noise_key", noise_key),
        ("noise_value", noise_value),
    ):
        if tensor.dtype != query.dtype:
            raise ValueError(f"{name} dtype {tensor.dtype} does not match the query's {query.dtype}")
        if tensor.dim() != 4:
            raise ValueError(f"{name} must be (batch, kv_heads, tokens, head_dim), got {tuple(tensor.shape)}")
    if query.dim() != 4:
        raise ValueError(f"query must be (batch, heads, tokens, head_dim), got {tuple(query.shape)}")

    batch, nheads_q, seqlen_q, headdim = query.shape
    nheads_kv = clean_key.shape[1]
    if nheads_q % nheads_kv:
        raise ValueError(f"query heads {nheads_q} must be a multiple of kv heads {nheads_kv}")
    if clean_value.shape != clean_key.shape:
        raise ValueError("clean K/V shapes must match")
    if noise_value.shape != noise_key.shape:
        raise ValueError("noise K/V shapes must match")
    if noise_key.shape != (batch, nheads_kv, seqlen_q, headdim):
        raise ValueError(
            "noise K/V must carry one key per query token: "
            f"got {tuple(noise_key.shape)} against query {tuple(query.shape)}"
        )
    if clean_key.shape[0] != batch or clean_key.shape[3] != headdim:
        raise ValueError(
            f"clean K/V must be (batch={batch}, kv_heads, clean_tokens, head_dim={headdim}), "
            f"got {tuple(clean_key.shape)}"
        )
    if headdim > MAX_HEAD_DIM:
        raise ValueError(f"dllm_hole_attention supports head_dim up to {MAX_HEAD_DIM}, got {headdim}")

    if clean_lo.shape != clean_hi.shape:
        raise ValueError("clean_lo and clean_hi must have the same shape")
    if clean_lo.dim() != 2 or clean_lo.shape[0] != batch:
        raise ValueError(f"clean_lo/clean_hi must be (batch={batch}, canvases), got {tuple(clean_lo.shape)}")
    if clean_lo.dtype != torch.int32 or clean_hi.dtype != torch.int32:
        raise ValueError("clean_lo/clean_hi must be int32")

    canvases = clean_lo.shape[1]
    if canvases == 0 or seqlen_q == 0 or seqlen_q % canvases:
        raise ValueError(f"query length {seqlen_q} does not split into {canvases} non-empty equal canvases")
    canvas_length = seqlen_q // canvases
    if canvas_length % MIN_BLOCK:
        raise ValueError(
            f"canvas_length {canvas_length} must be a multiple of {MIN_BLOCK}: the query tile is sized to "
            "divide the canvas so it cannot straddle two of them, which carry different visible intervals, "
            f"and {MIN_BLOCK} is the narrowest tile a Triton dot accepts"
        )
    if sm_scale is None:
        raise ValueError("dllm_hole_attention requires an explicit sm_scale")


def dllm_hole_attention(
    query: torch.Tensor,
    clean_key: torch.Tensor,
    clean_value: torch.Tensor,
    noise_key: torch.Tensor,
    noise_value: torch.Tensor,
    clean_lo: torch.Tensor,
    clean_hi: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Attention for K canvases packed onto one decoder sequence.

    ``query`` is ``(batch, heads, K * canvas_length, head_dim)``: canvas ``k``
    occupies ``[k * canvas_length, (k + 1) * canvas_length)``. It attends to
    ``clean_key[b, :, clean_lo[b, k]:clean_hi[b, k]]`` and to its own slice of
    ``noise_key`` -- never to another canvas.

    The interval is half-open and given in *cache* coordinates, so it carries
    left padding, the visible-prefix boundary and any sliding window at once.
    It is clamped to the cache -- read as ``[max(lo, 0), min(hi, clean_tokens))``
    -- because the alternative to clamping an interval the caller put past the
    end is reading past the end. An empty or inverted one is allowed and reads
    nothing; a canvas always sees itself, so no row reaches softmax without a
    key.
    """
    _validate(query, clean_key, clean_value, noise_key, noise_value, clean_lo, clean_hi, sm_scale)
    return _HoleAttention.apply(
        query.contiguous(),
        clean_key.contiguous(),
        clean_value.contiguous(),
        noise_key.contiguous(),
        noise_value.contiguous(),
        clean_lo.contiguous(),
        clean_hi.contiguous(),
        sm_scale,
    )


def hole_attention_reference(
    query: torch.Tensor,
    clean_key: torch.Tensor,
    clean_value: torch.Tensor,
    noise_key: torch.Tensor,
    noise_value: torch.Tensor,
    clean_lo: torch.Tensor,
    clean_hi: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Dense oracle for :func:`dllm_hole_attention`, one canvas at a time.

    Runs anywhere and in any dtype, builds no mask, and shares no code with the
    kernel: it slices the interval the kernel is told to respect and hands the
    result to a plain softmax. Any disagreement is the kernel's.
    """
    batch, nheads_q, seqlen_q, headdim = query.shape
    nheads_kv = clean_key.shape[1]
    clean_length = clean_key.shape[2]
    canvases = clean_lo.shape[1]
    canvas_length = seqlen_q // canvases
    repeat = nheads_q // nheads_kv

    outputs = torch.empty_like(query)
    for b in range(batch):
        for k in range(canvases):
            span = slice(k * canvas_length, (k + 1) * canvas_length)
            # Clamped as the kernel clamps it, at both ends. A bare slice would
            # read a negative bound from the far end of the cache instead of
            # reading nothing -- so an interval that sees no clean key at all
            # would come back here having seen almost all of them.
            lo = min(max(int(clean_lo[b, k]), 0), clean_length)
            hi = min(max(int(clean_hi[b, k]), 0), clean_length)
            keys = torch.cat([clean_key[b, :, lo:hi], noise_key[b, :, span]], dim=1)
            values = torch.cat([clean_value[b, :, lo:hi], noise_value[b, :, span]], dim=1)
            keys = keys.repeat_interleave(repeat, dim=0)
            values = values.repeat_interleave(repeat, dim=0)
            scores = torch.einsum("hqd,hkd->hqk", query[b, :, span], keys) * sm_scale
            outputs[b, :, span] = torch.einsum("hqk,hkd->hqd", scores.softmax(dim=-1).to(values.dtype), values)
    return outputs
