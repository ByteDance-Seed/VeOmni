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
"""Causal self-attention over packed variable-length segments, for wide heads.

``flash_attn_varlen_func`` already does this and does it better -- but only up
to ``head_dim`` 256, which it refuses above with *FlashAttention forward only
supports head dimension at most 256*. DiffusionGemma runs 25 sliding layers at
256, which flash serves, and 5 full-attention layers at ``global_head_dim=512``,
which nothing fused does. This kernel is for those five.

The reason to want it is what the alternative costs. Packing S samples onto one
row and handing the layer a block-diagonal mask does not make a dense backend
skip the off-diagonal blocks: sdpa and FFPA both form ``QK^T`` first and apply
the mask to it afterwards, so the arithmetic is ``(sum P)^2`` where only
``sum P^2`` is wanted -- a factor of S at equal segment lengths, and length
bucketing pushes it *towards* that bound rather than away from it. The mask is
the same factor again in bytes, per layer, held for the backward. Reading the
segmentation out of ``cu_seqlens`` instead spends S+1 int32 and computes only
the diagonal blocks.

Two things follow from segments being ragged, and both differ from
``dllm_hole_attention``, whose canvases are all one length:

* The grid is *per segment*, so a query tile lives in exactly one of them and
  causality is a comparison against two scalars rather than a lookup. The price
  is a launch per ``(segment, tile)`` pair that returns immediately when the
  segment is shorter than the tile it was given -- cheaper than the masking a
  straddling tile would need on every iteration.
* Tiles cannot be chosen to divide a segment, so the tail of each one is masked.
  That is the only padding left in the pass, and it is bounded by ``BLOCK_M``
  per segment rather than by the longest sample in the batch.

No sliding window, deliberately: the layers that need one are 256 wide and
belong to flash, which takes ``window_size`` and is faster than this will be.
Adding an untested window here would invite the 25 layers onto a kernel that
has no measurement behind it.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl


#: ``tl.dot`` needs at least 16 along every axis.
MIN_BLOCK = 16
#: Widest head this kernel will accumulate, matching ``dllm_hole_attention``.
#: DiffusionGemma's global layers are 512; flash covers everything below 257.
MAX_HEAD_DIM = 512

# CUDA's limit on the second grid dimension, which carries one segment each.
MAX_SEGMENTS = 65535
#: Deepest pipeline worth asking for; past this the tiles stop fitting anyway.
MAX_STAGES = 3
#: Shared memory left to whatever Triton stages besides the tiles counted here.
SHARED_MEMORY_RESERVE = 8 * 1024


def _fwd_tiles(headdim: int, element_size: int) -> tuple[int, int, int, int]:
    """``(BLOCK_M, BLOCK_N, num_warps, maxnreg)`` for the forward and ``dq``.

    The output accumulator is one ``BLOCK_M x head_dim`` fp32 tile in registers
    and the operands stay resident for the loop, so both shrink as the head
    widens -- the reasoning ``dllm_hole_attention`` sets out, and its sizes up
    to 256. Unlike there, nothing has to divide a canvas: segments are ragged,
    so the tail is masked either way and the tile stays a power of two.

    The wide row is this kernel's own, and it is *not* the hole kernel's 16x16.
    ``scripts/bench_wide_varlen_attention.py`` at the global-layer geometry
    (16 heads over 2, head_dim 512, bf16, forward+backward, four 2048-token
    segments): 16x16 69.30ms, 32x16 66.51ms, 32x32 53.73ms, 64x32 59.93ms,
    16x32 62.89ms. Note what that costs and buys -- ``_schedule`` gives 16x16
    three pipeline stages and 32x32 only one, and the wider tile still wins by
    1.29x. Do not restore the deeper pipeline on the argument that stages are
    free; here they lose. That measurement is at 512, and 256 shares the row
    only because it already held the same tiles, not because it was measured;
    the layers that are 256 wide go to ``flash_attn_varlen_func``, not here.
    """
    if headdim <= 64:
        block_m, block_n, warps, maxnreg = 64, 64, 4, 128
    elif headdim <= 128:
        block_m, block_n, warps, maxnreg = 128, 64, 8, 128
    else:
        block_m, block_n, warps, maxnreg = 32, 32, 8, 255
    block_m, block_n = _narrow(block_m, block_n, element_size)
    return block_m, block_n, warps, maxnreg


def _dkdv_tiles(headdim: int, element_size: int) -> tuple[int, int, int | None]:
    """``(BLOCK_M, BLOCK_N, maxnreg)`` for ``dkdv``, which holds two accumulators.

    The narrow-head row runs uncapped: two ``128 x 128`` fp32 accumulators are
    128 registers per thread on their own, so a 128 cap would spend the whole
    budget on them and spill everything else.

    This kernel's backward lives here rather than in ``dq``, so the 512 row is
    where the tuning paid. With the forward fixed at 32x32, the same bench over
    four 2048-token segments: 16x16 53.93ms, 32x16 54.60ms, 16x32 44.09ms,
    32x32 43.75ms. Widening ``BLOCK_N`` is what matters -- it is the axis this
    program owns a tile of, so it decides how many times the query loop is
    re-walked -- and ``BLOCK_M`` past that is worth almost nothing.
    """
    if headdim <= 128:
        block_m, block_n, maxnreg = 32, 128, None
    elif headdim <= 256:
        block_m, block_n, maxnreg = 16, 32, 255
    else:
        block_m, block_n, maxnreg = 32, 32, 255
    block_m, block_n = _narrow(block_m, block_n, element_size)
    return block_m, block_n, maxnreg


def _narrow(block_m: int, block_n: int, element_size: int) -> tuple[int, int]:
    """Halve a bf16-sized tile pair for a wider element type."""
    if element_size <= 2:
        return block_m, block_n
    return max(MIN_BLOCK, block_m // 2), max(MIN_BLOCK, block_n // 2)


def _dot_precision(dtype: torch.dtype) -> str:
    """``tl.dot`` only honours this for fp32 inputs; keep those exact."""
    return "ieee" if dtype == torch.float32 else "tf32"


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

    Resident tiles are loaded once and live for the whole loop; streamed ones
    are the per-iteration operands, and only those get another copy per stage.
    Triton's own guess counts neither and asks for more than the device has.
    ``dllm_hole_attention`` carries the long form of this argument; this is the
    same accounting against this kernel's own operands.
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
                f"wide_varlen_attention cannot fit head_dim {block_d} at {element_size} bytes into this "
                f"device's {limit + SHARED_MEMORY_RESERVE} bytes of shared memory, even at its narrowest tile"
            )
        block_m = max(MIN_BLOCK, block_m // 2)
        block_n = max(MIN_BLOCK, block_n // 2)


@triton.jit
def _segment_bounds(CuSeqlens, off_seg, total_tokens, cap):
    """This segment's ``[start, length)``, clamped into the row and to ``cap``.

    Clamped into the row rather than trusted because the public API does not
    read ``cu_seqlens`` on the host -- see :func:`wide_varlen_attention` on why
    it must not -- so a malformed one arrives here unexamined. Clamping turns it
    into a wrong answer that the equivalence test can see instead of an
    out-of-bounds read whose symptom depends on what is allocated next.

    ``cap`` is ``max_seqlen``, and clamping to it is what keeps the three
    kernels computing the *same* problem. The forward and ``dq`` grids are sized
    from ``max_seqlen``, so a caller who understates it leaves the rows past it
    unwritten -- including their ``lse`` and ``delta``, which are allocated
    empty. ``dkdv`` walks the queries by segment length instead, so without this
    it would read that uninitialised ``lse``, and ``exp(scores - garbage)`` is an
    inf or a NaN accumulated into the dK/dV of a *valid* key tile. Truncating
    all three alike turns an understated ``max_seqlen`` into the same class of
    failure as a malformed prefix sum: a wrong answer, bounded and visible.
    """
    start = tl.load(CuSeqlens + off_seg).to(tl.int64)
    end = tl.load(CuSeqlens + off_seg + 1).to(tl.int64)
    start = tl.maximum(tl.minimum(start, total_tokens), 0)
    end = tl.maximum(tl.minimum(end, total_tokens), start)
    return start, tl.minimum(end - start, cap)


@triton.jit
def _varlen_fwd_kernel(
    Q,
    K,
    V,
    CuSeqlens,
    Out,
    LSE,
    scale,
    stride_qh,
    stride_qm,
    stride_kh,
    stride_kn,
    stride_vh,
    stride_vn,
    stride_oh,
    stride_om,
    nheads_q: tl.constexpr,
    nheads_kv: tl.constexpr,
    total_tokens,
    max_seqlen,
    headdim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    off_seg = tl.program_id(1)
    # int64 before it multiplies a stride: one head of a long packed row is
    # already ~5e7 elements, and int32 would wrap a few heads into it.
    off_hq = tl.program_id(2).to(tl.int64)

    seg_start, seg_len = _segment_bounds(CuSeqlens, off_seg, total_tokens, max_seqlen)
    # Segments are ragged, so most of them are shorter than the longest one the
    # grid was sized for. Those tiles have no rows and must not divide by the
    # zero their softmax would sum to.
    if start_m >= seg_len:
        return

    off_hkv = off_hq // (nheads_q // nheads_kv)
    Q += off_hq * stride_qh + seg_start * stride_qm
    K += off_hkv * stride_kh + seg_start * stride_kn
    V += off_hkv * stride_vh + seg_start * stride_vn
    Out += off_hq * stride_oh + seg_start * stride_om
    LSE += off_hq * total_tokens + seg_start

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_m < seg_len
    d_mask = offs_d < headdim

    q = tl.load(
        Q + offs_m[:, None] * stride_qm + offs_d[None, :],
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    # Causality bounds the loop rather than only masking it: the last key this
    # tile can see is its own last query, so tiles beyond the diagonal are never
    # walked. Every valid row still reaches softmax with at least its own key,
    # which is why no row here can sum to zero.
    key_end = tl.minimum(start_m + BLOCK_M, seg_len)
    for tile in range(0, key_end, BLOCK_N):
        key_offsets = tile + offs_n
        n_mask = key_offsets < seg_len
        kd_mask = n_mask[:, None] & d_mask[None, :]
        k = tl.load(K + key_offsets[:, None] * stride_kn + offs_d[None, :], mask=kd_mask, other=0.0)

        scores = tl.dot(q, tl.trans(k), input_precision=DOT_PRECISION)
        causal = key_offsets[None, :] <= offs_m[:, None]
        # `n_mask` is redundant while the mask is causal -- a valid row has
        # `offs_m < seg_len`, so `key_offsets <= offs_m` already implies
        # `key_offsets < seg_len`. It is written anyway because that implication
        # is the only thing keeping the next segment's tokens out, and it holds
        # for exactly one mask shape. Give a query anything the diagonal does not
        # bound -- a window, a bidirectional block, a cache longer than the
        # queries -- and the zeros a masked load returns become `score = 0`,
        # whose `exp(0 - m)` is not zero and lands in the softmax denominator.
        # `dkdv` already masks all three axes; matching it costs nothing, since
        # every operand is in registers.
        scores = tl.where(q_mask[:, None] & n_mask[None, :] & causal, scores * scale, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        probabilities = tl.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(probabilities, axis=1)
        v = tl.load(V + key_offsets[:, None] * stride_vn + offs_d[None, :], mask=kd_mask, other=0.0)
        acc = tl.dot(probabilities.to(v.dtype), v, acc=acc * alpha[:, None], input_precision=DOT_PRECISION)
        m_i = m_new

    tl.store(
        Out + offs_m[:, None] * stride_om + offs_d[None, :],
        (acc / l_i[:, None]).to(Out.dtype.element_ty),
        mask=q_mask[:, None] & d_mask[None, :],
    )
    tl.store(LSE + offs_m, m_i + tl.log(l_i), mask=q_mask)


@triton.jit
def _varlen_bwd_dq_kernel(
    Q,
    K,
    V,
    O,
    DO,
    CuSeqlens,
    DQ,
    LSE,
    Delta,
    scale,
    stride_qh,
    stride_qm,
    stride_kh,
    stride_kn,
    stride_vh,
    stride_vn,
    stride_oh,
    stride_om,
    stride_dqh,
    stride_dqm,
    nheads_q: tl.constexpr,
    nheads_kv: tl.constexpr,
    total_tokens,
    max_seqlen,
    headdim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    off_seg = tl.program_id(1)
    off_hq = tl.program_id(2).to(tl.int64)

    seg_start, seg_len = _segment_bounds(CuSeqlens, off_seg, total_tokens, max_seqlen)
    if start_m >= seg_len:
        return

    off_hkv = off_hq // (nheads_q // nheads_kv)
    Q += off_hq * stride_qh + seg_start * stride_qm
    K += off_hkv * stride_kh + seg_start * stride_kn
    V += off_hkv * stride_vh + seg_start * stride_vn
    O += off_hq * stride_oh + seg_start * stride_om
    DO += off_hq * stride_oh + seg_start * stride_om
    DQ += off_hq * stride_dqh + seg_start * stride_dqm
    LSE += off_hq * total_tokens + seg_start
    Delta += off_hq * total_tokens + seg_start

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_m < seg_len
    d_mask = offs_d < headdim
    qd_mask = q_mask[:, None] & d_mask[None, :]

    # Delta is needed by every score tile here and by `dkdv`. The query owns it,
    # so compute it once and publish it before going on; reading `o` before `q`
    # keeps the two from being live at once.
    do = tl.load(DO + offs_m[:, None] * stride_om + offs_d[None, :], mask=qd_mask, other=0.0)
    o = tl.load(O + offs_m[:, None] * stride_om + offs_d[None, :], mask=qd_mask, other=0.0)
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    tl.store(Delta + offs_m, delta, mask=q_mask)

    q = tl.load(Q + offs_m[:, None] * stride_qm + offs_d[None, :], mask=qd_mask, other=0.0)
    lse = tl.load(LSE + offs_m, mask=q_mask, other=0.0)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    key_end = tl.minimum(start_m + BLOCK_M, seg_len)
    for tile in range(0, key_end, BLOCK_N):
        key_offsets = tile + offs_n
        n_mask = key_offsets < seg_len
        kd_mask = n_mask[:, None] & d_mask[None, :]
        k = tl.load(K + key_offsets[:, None] * stride_kn + offs_d[None, :], mask=kd_mask, other=0.0)
        v = tl.load(V + key_offsets[:, None] * stride_vn + offs_d[None, :], mask=kd_mask, other=0.0)

        scores = tl.dot(q, tl.trans(k), input_precision=DOT_PRECISION)
        causal = key_offsets[None, :] <= offs_m[:, None]
        # All three axes, for the reason the forward gives.
        scores = tl.where(q_mask[:, None] & n_mask[None, :] & causal, scores * scale, -float("inf"))
        dp = tl.dot(do, tl.trans(v), input_precision=DOT_PRECISION)
        probabilities = tl.exp(scores - lse[:, None])
        ds = (probabilities * (dp - delta[:, None]) * scale).to(k.dtype)
        acc = tl.dot(ds, k, acc=acc, input_precision=DOT_PRECISION)

    tl.store(DQ + offs_m[:, None] * stride_dqm + offs_d[None, :], acc.to(DQ.dtype.element_ty), mask=qd_mask)


@triton.jit
def _varlen_bwd_dkdv_kernel(
    Q,
    K,
    V,
    DO,
    CuSeqlens,
    DK,
    DV,
    LSE,
    Delta,
    scale,
    stride_qh,
    stride_qm,
    stride_kh,
    stride_kn,
    stride_vh,
    stride_vn,
    stride_doh,
    stride_dom,
    stride_dkh,
    stride_dkn,
    stride_dvh,
    stride_dvn,
    nheads_q: tl.constexpr,
    nheads_kv: tl.constexpr,
    total_tokens,
    max_seqlen,
    headdim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """One key tile's dK/dV, summed over every query in its segment that sees it.

    The key tile owns its gradient -- one ``(segment, tile, kv head)`` triple is
    one program and no other writes it -- so the sum runs in registers and lands
    in a single store, with no atomics and no partial buffer. Packing is what
    makes that hold: a key belongs to exactly one segment, so the queries that
    can reach it are the ones above it in that same segment and nowhere else.

    That ownership rests on ``cu_seqlens`` being non-decreasing, which the
    trusted path cannot check. ``_segment_bounds`` clamps each span into the row,
    which is enough to keep every access in bounds, but it cannot make two spans
    disjoint: ``[0, 100, 80, 200]`` clamps to ``[0, 100)`` and ``[80, 200)``, and
    the two programs then both store rows 80..99. So a decreasing prefix sum on
    the trusted path gives a *nondeterministic* wrong answer rather than the
    reproducible one a clamped-but-ordered prefix sum gives. Both are wrong
    answers to a caller error and neither reads out of bounds, which is the
    guarantee this path makes; determinism is not part of it.
    """
    start_n = tl.program_id(0) * BLOCK_N
    off_seg = tl.program_id(1)
    off_hkv = tl.program_id(2).to(tl.int64)

    seg_start, seg_len = _segment_bounds(CuSeqlens, off_seg, total_tokens, max_seqlen)
    if start_n >= seg_len:
        return

    K += off_hkv * stride_kh + seg_start * stride_kn
    V += off_hkv * stride_vh + seg_start * stride_vn
    DK += off_hkv * stride_dkh + seg_start * stride_dkn
    DV += off_hkv * stride_dvh + seg_start * stride_dvn

    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    n_mask = offs_n < seg_len
    d_mask = offs_d < headdim
    nd_mask = n_mask[:, None] & d_mask[None, :]

    k = tl.load(K + offs_n[:, None] * stride_kn + offs_d[None, :], mask=nd_mask, other=0.0)
    v = tl.load(V + offs_n[:, None] * stride_vn + offs_d[None, :], mask=nd_mask, other=0.0)
    dk = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)

    # Causality again bounds rather than masks: the first query that can see
    # this tile is the one at its own first key, so start at that query tile.
    first_m = (start_n // BLOCK_M) * BLOCK_M
    for group_index in range(0, GROUP_SIZE):
        off_hq = off_hkv * GROUP_SIZE + group_index
        q_head = Q + off_hq * stride_qh + seg_start * stride_qm
        do_head = DO + off_hq * stride_doh + seg_start * stride_dom
        lse_head = LSE + off_hq.to(tl.int64) * total_tokens + seg_start
        delta_head = Delta + off_hq.to(tl.int64) * total_tokens + seg_start

        for start_m in range(first_m, seg_len, BLOCK_M):
            query_offsets = start_m + offs_m
            q_mask = query_offsets < seg_len
            qd_mask = q_mask[:, None] & d_mask[None, :]
            q = tl.load(q_head + query_offsets[:, None] * stride_qm + offs_d[None, :], mask=qd_mask, other=0.0)
            do = tl.load(do_head + query_offsets[:, None] * stride_dom + offs_d[None, :], mask=qd_mask, other=0.0)

            scores = tl.dot(q, tl.trans(k), input_precision=DOT_PRECISION)
            causal = offs_n[None, :] <= query_offsets[:, None]
            scores = tl.where(q_mask[:, None] & n_mask[None, :] & causal, scores * scale, -float("inf"))
            dp = tl.dot(do, tl.trans(v), input_precision=DOT_PRECISION)
            lse = tl.load(lse_head + query_offsets, mask=q_mask, other=0.0)
            delta = tl.load(delta_head + query_offsets, mask=q_mask, other=0.0)
            probabilities = tl.exp(scores - lse[:, None])
            ds = (probabilities * (dp - delta[:, None]) * scale).to(k.dtype)

            dk = tl.dot(tl.trans(ds), q, acc=dk, input_precision=DOT_PRECISION)
            dv = tl.dot(tl.trans(probabilities.to(k.dtype)), do, acc=dv, input_precision=DOT_PRECISION)

    tl.store(DK + offs_n[:, None] * stride_dkn + offs_d[None, :], dk.to(DK.dtype.element_ty), mask=nd_mask)
    tl.store(DV + offs_n[:, None] * stride_dvn + offs_d[None, :], dv.to(DV.dtype.element_ty), mask=nd_mask)


class _WideVarlenAttention(torch.autograd.Function):
    """Autograd function for wide head-dimension variable-length causal attention."""

    @staticmethod
    def forward(ctx, query, key, value, cu_seqlens, max_seqlen, sm_scale):
        """Run wide varlen attention forward pass."""
        nheads_q, total_tokens, headdim = query.shape[1], query.shape[2], query.shape[3]
        nheads_kv = key.shape[1]
        segments = cu_seqlens.numel() - 1

        block_m, block_n, warps, maxnreg = _fwd_tiles(headdim, query.element_size())
        block_d = max(MIN_BLOCK, triton.next_power_of_2(headdim))
        precision = _dot_precision(query.dtype)
        # Resident: the query tile. Streamed: this iteration's K and V.
        block_m, block_n, stages = _schedule(
            block_m, block_n, block_d, query.element_size(), query.device, resident=(1, 0), streamed=(0, 2)
        )

        # Zeroed, not empty: a token no segment covers is written by no program,
        # and `cu_seqlens` is taken on trust here, so the alternative to a
        # defined zero is whatever the allocator last left there travelling into
        # the loss. One memset against an attention call is under a percent.
        out = torch.zeros_like(query)
        # Zero-filled, not empty. `_segment_bounds` clamps the query range to
        # `max_seqlen` so no program should ever read a row it did not write --
        # but that is an invariant spanning three kernels, and if it ever breaks
        # again the difference between these two allocations is the difference
        # between a deterministic wrong answer and a NaN that appears only when
        # the allocator hands back a dirty page. Half a megabyte of memset
        # against a multi-millisecond attention call buys that determinism.
        lse = torch.zeros((nheads_q, total_tokens), dtype=torch.float32, device=query.device)

        _varlen_fwd_kernel[(triton.cdiv(max_seqlen, block_m), segments, nheads_q)](
            query,
            key,
            value,
            cu_seqlens,
            out,
            lse,
            sm_scale,
            query.stride(1),
            query.stride(2),
            key.stride(1),
            key.stride(2),
            value.stride(1),
            value.stride(2),
            out.stride(1),
            out.stride(2),
            nheads_q=nheads_q,
            nheads_kv=nheads_kv,
            total_tokens=total_tokens,
            max_seqlen=max_seqlen,
            headdim=headdim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            DOT_PRECISION=precision,
            num_warps=warps,
            num_stages=stages,
            maxnreg=maxnreg,
        )

        ctx.save_for_backward(query, key, value, cu_seqlens, out, lse)
        ctx.sm_scale = sm_scale
        ctx.max_seqlen = max_seqlen
        ctx.precision = precision
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """Run wide variable-length attention backward pass."""
        query, key, value, cu_seqlens, out, lse = ctx.saved_tensors
        nheads_q, total_tokens, headdim = query.shape[1], query.shape[2], query.shape[3]
        nheads_kv = key.shape[1]
        segments = cu_seqlens.numel() - 1
        max_seqlen = ctx.max_seqlen
        scale = ctx.sm_scale
        precision = ctx.precision
        block_d = max(MIN_BLOCK, triton.next_power_of_2(headdim))

        grad_output = grad_output.contiguous()
        # Every gradient element is summed in an fp32 register and stored once --
        # no atomics, no partial buffers -- so the input dtype is rounded exactly
        # as late as an fp32 staging buffer would round it, without a second copy.
        # Zeroed for the same reason as `out` in the forward.
        grad_query = torch.zeros_like(query)
        grad_key = torch.zeros_like(key)
        grad_value = torch.zeros_like(value)
        delta = torch.zeros((nheads_q, total_tokens), dtype=torch.float32, device=query.device)

        block_m, block_n, warps, maxnreg = _fwd_tiles(headdim, query.element_size())
        # Resident: the query tile and its grad, which outlive `o`. Streamed: K and V.
        block_m, block_n, stages = _schedule(
            block_m, block_n, block_d, query.element_size(), query.device, resident=(2, 0), streamed=(0, 2)
        )
        _varlen_bwd_dq_kernel[(triton.cdiv(max_seqlen, block_m), segments, nheads_q)](
            query,
            key,
            value,
            out,
            grad_output,
            cu_seqlens,
            grad_query,
            lse,
            delta,
            scale,
            query.stride(1),
            query.stride(2),
            key.stride(1),
            key.stride(2),
            value.stride(1),
            value.stride(2),
            out.stride(1),
            out.stride(2),
            grad_query.stride(1),
            grad_query.stride(2),
            nheads_q=nheads_q,
            nheads_kv=nheads_kv,
            total_tokens=total_tokens,
            max_seqlen=max_seqlen,
            headdim=headdim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            DOT_PRECISION=precision,
            num_warps=warps,
            num_stages=stages,
            maxnreg=maxnreg,
        )

        block_m, block_n, maxnreg = _dkdv_tiles(headdim, query.element_size())
        # Resident: the key tile this program owns, and its value.
        # Streamed: each query tile and its grad.
        block_m, block_n, stages = _schedule(
            block_m, block_n, block_d, query.element_size(), query.device, resident=(0, 2), streamed=(2, 0)
        )
        _varlen_bwd_dkdv_kernel[(triton.cdiv(max_seqlen, block_n), segments, nheads_kv)](
            query,
            key,
            value,
            grad_output,
            cu_seqlens,
            grad_key,
            grad_value,
            lse,
            delta,
            scale,
            query.stride(1),
            query.stride(2),
            key.stride(1),
            key.stride(2),
            value.stride(1),
            value.stride(2),
            grad_output.stride(1),
            grad_output.stride(2),
            grad_key.stride(1),
            grad_key.stride(2),
            grad_value.stride(1),
            grad_value.stride(2),
            nheads_q=nheads_q,
            nheads_kv=nheads_kv,
            total_tokens=total_tokens,
            max_seqlen=max_seqlen,
            headdim=headdim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            GROUP_SIZE=nheads_q // nheads_kv,
            DOT_PRECISION=precision,
            num_warps=8,
            num_stages=stages,
            maxnreg=maxnreg,
        )

        return grad_query, grad_key, grad_value, None, None, None


def _validate(query, key, value, cu_seqlens, sm_scale):
    """Everything checkable without reading a device tensor on the host.

    The values in ``cu_seqlens`` are deliberately *not* among them; see
    :func:`wide_varlen_attention`.
    """
    if not query.is_cuda:
        raise ValueError("wide_varlen_attention is a Triton kernel and needs CUDA tensors")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"wide_varlen_attention supports fp16/bf16/fp32, got {query.dtype}")
    if query.dim() != 4:
        raise ValueError(f"query must be (1, heads, tokens, head_dim), got {tuple(query.shape)}")
    for name, tensor in (("key", key), ("value", value)):
        if tensor.dtype != query.dtype:
            raise ValueError(f"{name} dtype {tensor.dtype} does not match the query's {query.dtype}")
        if tensor.dim() != 4:
            raise ValueError(f"{name} must be (1, kv_heads, tokens, head_dim), got {tuple(tensor.shape)}")

    batch, nheads_q, total_tokens, headdim = query.shape
    if batch != 1:
        raise ValueError(
            f"wide_varlen_attention takes one packed row, got batch {batch}. A second row would need its own "
            "`cu_seqlens`, and two packed rows are just one longer pack -- concatenate them and extend the "
            "prefix sum instead."
        )
    if headdim > MAX_HEAD_DIM:
        raise ValueError(f"wide_varlen_attention supports head_dim up to {MAX_HEAD_DIM}, got {headdim}")
    if key.shape != (1, key.shape[1], total_tokens, headdim):
        raise ValueError(
            "key must carry one entry per query token: "
            f"got {tuple(key.shape)} against query {tuple(query.shape)}. This kernel is self-attention; "
            "a shorter cache is a different mask than the one it implements."
        )
    if value.shape != key.shape:
        raise ValueError("key and value shapes must match")

    nheads_kv = key.shape[1]
    if nheads_kv == 0 or nheads_q % nheads_kv:
        raise ValueError(f"query heads {nheads_q} must be a multiple of kv heads {nheads_kv}")

    if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        raise ValueError(
            f"cu_seqlens must be a 1-D prefix sum with at least two entries, got {tuple(cu_seqlens.shape)}"
        )
    if cu_seqlens.dtype != torch.int32:
        raise ValueError(f"cu_seqlens must be int32, got {cu_seqlens.dtype}")

    # Every device compared against the query's, not merely asked whether it is
    # a GPU. A key on another device, or a prefix sum on another device, indexes
    # memory this launch does not own: an illegal access somewhere inside the
    # kernel rather than a message naming the argument.
    for name, tensor in (("key", key), ("value", value), ("cu_seqlens", cu_seqlens)):
        if tensor.device != query.device:
            raise ValueError(
                f"{name} is on {tensor.device} but the query is on {query.device}; "
                "every argument must share one device"
            )

    # `segments` becomes the grid's y extent, which CUDA caps at 65535. Reachable
    # -- 65536 single-token segments is only 64K tokens -- and worth naming here,
    # because the launch failure it otherwise raises points at no argument.
    segments = cu_seqlens.numel() - 1
    if segments > MAX_SEGMENTS:
        raise ValueError(
            f"wide_varlen_attention packs up to {MAX_SEGMENTS} segments per row, got {segments}; split the row"
        )
    if sm_scale is None:
        raise ValueError("wide_varlen_attention requires an explicit sm_scale")


def wide_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int | None = None,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Causal self-attention within each segment of one packed row.

    ``query`` is ``(1, heads, total_tokens, head_dim)`` and ``key``/``value``
    are ``(1, kv_heads, total_tokens, head_dim)``: one row holding every
    segment end to end. ``cu_seqlens`` is the int32 prefix sum of the segment
    lengths, so segment ``s`` spans ``[cu_seqlens[s], cu_seqlens[s + 1])``.
    Token ``i`` of a segment attends to tokens ``[0, i]`` of *that* segment and
    to nothing outside it. Zero-length segments are allowed and read nothing.

    ``sm_scale`` is required rather than defaulted to ``head_dim ** -0.5``:
    Gemma folds the attention scale into ``q_norm``, so the caller that needs
    this kernel passes 1.0, and a helpful default would silently scale twice.

    Passing ``max_seqlen`` avoids a host synchronisation. Leave it ``None`` and
    it is read off ``cu_seqlens``, which also buys a full check of the prefix
    sum -- the sync is already paid, so the check is free there.

    On the production path, where ``max_seqlen`` is known, ``cu_seqlens`` is
    taken on trust and the kernel clamps every bound into the row instead. That
    is deliberate: ``dllm_attention``'s per-segment ``.item()`` loop is why
    ``flat_seq`` beats ``flat_batch`` on the sliding layers by 3x, and a
    validating read here would put the same synchronisation back into five
    layers of every step. A malformed prefix sum therefore gives a wrong answer
    rather than an out-of-bounds read, and the equivalence test is what stands
    between the two.
    """
    _validate(query, key, value, cu_seqlens, sm_scale)
    if max_seqlen is None:
        lengths = cu_seqlens.diff()
        if int(lengths.min().item()) < 0 or int(cu_seqlens[0].item()) != 0:
            raise ValueError("cu_seqlens must be a non-decreasing prefix sum starting at 0")
        if int(cu_seqlens[-1].item()) != query.shape[2]:
            raise ValueError(
                f"cu_seqlens must end at the packed length {query.shape[2]}, got {int(cu_seqlens[-1].item())}"
            )
        max_seqlen = int(lengths.max().item())
    if max_seqlen < 0:
        raise ValueError(f"max_seqlen must be non-negative, got {max_seqlen}")
    # A row of nothing but empty segments is not special-cased. `cdiv(0, tile)`
    # is 0, a zero-extent grid launches no program, and the outputs are already
    # zero-filled -- so it falls out of the ordinary path with the graph intact.
    # Returning `zeros_like` from here instead would hand back a tensor with no
    # `grad_fn`, and this layer's q/k/v would then get no gradient at all while
    # the surrounding layers kept theirs: exactly the silent downgrade that
    # constraint 28 exists to forbid.
    return _WideVarlenAttention.apply(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        cu_seqlens.contiguous(),
        int(max_seqlen),
        sm_scale,
    )


def wide_varlen_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Oracle for :func:`wide_varlen_attention`, one query position at a time.

    Runs anywhere and in any dtype, and shares no arithmetic with the kernel:
    it builds no mask at all, but slices the keys each query is entitled to --
    ``[segment start, that query]`` -- and calls a plain softmax on them. So a
    tile boundary the kernel got wrong, a segment it read from the wrong offset,
    or a query it let see the token after it all surface as a numerical
    disagreement rather than as the same mistake made twice.

    Pass upcast copies when the run under test is not fp32. This computes the
    logits in the dtype it is handed, so a bf16 input rounds them to roughly
    three decimal digits *before* the softmax -- an error the size of a
    low-weight token being wrongly included or excluded, which is precisely the
    class of mistake this exists to catch. The kernel keeps its scores in fp32
    throughout, so comparing it against a bf16 oracle measures the oracle.

    Quadratic in Python and meant for test shapes, not for a training step.
    """
    nheads_q = query.shape[1]
    nheads_kv = key.shape[1]
    repeat = nheads_q // nheads_kv
    total_tokens = query.shape[2]

    keys = key[0].repeat_interleave(repeat, dim=0)
    values = value[0].repeat_interleave(repeat, dim=0)
    outputs = torch.zeros_like(query)
    bounds = [int(x) for x in cu_seqlens.tolist()]
    for start, end in zip(bounds[:-1], bounds[1:]):
        start = min(max(start, 0), total_tokens)
        end = min(max(end, start), total_tokens)
        for position in range(start, end):
            scores = torch.einsum("hd,hkd->hk", query[0, :, position], keys[:, start : position + 1]) * sm_scale
            weights = scores.softmax(dim=-1).to(values.dtype)
            outputs[0, :, position] = torch.einsum("hk,hkd->hd", weights, values[:, start : position + 1])
    return outputs
