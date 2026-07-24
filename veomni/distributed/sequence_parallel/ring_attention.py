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

"""Ring-Attention (context-parallel) primitives for USP.

Implements the Ring-Attention half of Unified Sequence Parallelism
(USP, https://arxiv.org/abs/2405.07719). The sequence is split across the
``cp`` mesh dimension; each rank keeps a local Q/K/V shard. K/V are rotated
around the ``cp`` process group in a ring while an online (FlashAttention-style)
softmax accumulates partial outputs, so full attention is computed without any
rank ever holding the whole sequence.

Causal load balancing uses the zig-zag layout: the global sequence is split
into ``2 * cp_size`` blocks and rank ``r`` owns blocks ``r`` and
``2*cp_size-1-r``. Every ring step then does a constant amount of work. The
matching data reordering lives in ``.data.zigzag_reorder`` / ``.data.zigzag_
undo`` and is applied by the collator before slicing.

The Ulysses half (all-to-all head/seq exchange) is orthogonal: USP first does
the Ulysses all-to-all over the ``ulysses`` group, then runs ring attention over
the ``cp`` group on the resulting shard (see
``veomni/ops/kernels/attention/__init__.py``).
"""

from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from .comm import get_context_parallel_group


try:
    from flash_attn.flash_attn_interface import (
        _flash_attn_backward,
        _flash_attn_forward,
        _flash_attn_varlen_backward,
        _flash_attn_varlen_forward,
    )

    _FA_AVAILABLE = True
except ImportError:  # pragma: no cover - only hit without flash-attn
    _FA_AVAILABLE = False


__all__ = [
    "ring_flash_attn_func",
    "zigzag_ring_flash_attn_func",
    "zigzag_ring_flash_attn_varlen_func",
    "update_out_and_lse",
    "RingComm",
]


def _fa_forward(q, k, v, softmax_scale, causal, dropout_p=0.0):
    out, lse, _, _ = _flash_attn_forward(
        q=q,
        k=k,
        v=v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
    )
    return out, lse


def _fa_backward(dout, q, k, v, out, lse, softmax_scale, causal, dropout_p=0.0):
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    _flash_attn_backward(
        dout=dout,
        q=q,
        k=k,
        v=v,
        out=out,
        softmax_lse=lse,
        dq=dq,
        dk=dk,
        dv=dv,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        rng_state=None,
    )
    return dq, dk, dv


def _update_out_and_lse(out: Tensor, lse: Tensor, block_out: Tensor, block_lse: Tensor):
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)  # (b, h, s) -> (b, s, h, 1)
    # Numerically-stable online softmax merge (sigmoid form). Equivalent to
    #   new_lse = lse + log(1 + exp(block_lse - lse))
    #   out = exp(lse-new_lse)*out + exp(block_lse-new_lse)*block_out
    out = out - torch.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - torch.nn.functional.logsigmoid(lse - block_lse)
    return out, lse


def update_out_and_lse(
    out: Optional[Tensor],
    lse: Optional[Tensor],
    block_out: Tensor,
    block_lse: Tensor,
    slice_=None,
) -> Tuple[Tensor, Tensor]:
    """Merge one FlashAttention block into the running online-softmax state.

    ``out`` is kept fp32 as ``(b, s, h, d)`` and ``lse`` as ``(b, s, h, 1)``.
    ``block_lse`` arrives as ``(b, h, s)`` from FlashAttention. ``slice_`` merges
    ``block_out`` into a sub-slice of the running state (used by the zig-zag path
    to update only the second Q half).
    """
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse must not pass slice_")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
        return out, lse
    if slice_ is not None:
        slice_out, slice_lse = _update_out_and_lse(out[slice_], lse[slice_], block_out, block_lse)
        out[slice_], lse[slice_] = slice_out, slice_lse
        return out, lse
    return _update_out_and_lse(out, lse, block_out, block_lse)


class RingComm:
    """P2P ring communicator over a context-parallel process group.

    ``send_recv`` posts an ``isend`` to the next rank and an ``irecv`` from the
    previous rank; ``commit``/``wait`` drain the batch so a ring step can overlap
    the K/V transfer with the local FlashAttention compute.
    """

    def __init__(self, group: ProcessGroup):
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.send_rank = dist.get_global_rank(group, (self.rank + 1) % self.world_size)
        self.recv_rank = dist.get_global_rank(group, (self.rank - 1) % self.world_size)
        self._ops = []
        self._reqs = None

    def send_recv(self, to_send: Tensor, recv_tensor: Optional[Tensor] = None) -> Tensor:
        res = torch.empty_like(to_send) if recv_tensor is None else recv_tensor
        self._ops.append(dist.P2POp(dist.isend, to_send.contiguous(), self.send_rank, group=self.group))
        self._ops.append(dist.P2POp(dist.irecv, res, self.recv_rank, group=self.group))
        return res

    def commit(self):
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is not None:
            for req in self._reqs:
                req.wait()
        self._ops = []
        self._reqs = None


# --------------------------------------------------------------------------- #
# Plain ring attention (Q fixed, K/V rotate). Correct for full (non-causal)    #
# attention and for causal, but causal work is load-imbalanced across ranks.   #
# --------------------------------------------------------------------------- #
def _ring_forward(group, q, k, v, softmax_scale, causal):
    comm = RingComm(group)
    out = None
    lse = None
    next_kv = None
    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            kv = torch.stack([k, v]).contiguous()
            next_kv = comm.send_recv(kv)
            comm.commit()

        if not causal or step <= comm.rank:
            block_causal = causal and step == 0
            block_out, block_lse = _fa_forward(q, k, v, softmax_scale, block_causal)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_kv[0], next_kv[1]

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2).contiguous()  # -> (b, h, s)
    return out, lse


def _ring_backward(group, dout, q, k, v, out, lse, softmax_scale, causal):
    kv_comm = RingComm(group)
    d_kv_comm = RingComm(group)
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = None
    dv = None
    next_dk = None
    next_dv = None
    next_kv = None

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            kv = torch.stack([k, v]).contiguous()
            next_kv = kv_comm.send_recv(kv)
            kv_comm.commit()

        if not causal or step <= kv_comm.rank:
            block_causal = causal and step == 0
            bdq, bdk, bdv = _fa_backward(dout, q, k, v, out, lse, softmax_scale, block_causal)
            dq += bdq
            if dk is None:
                dk = bdk.to(torch.float32)
                dv = bdv.to(torch.float32)
            else:
                d_kv_comm.wait()
                dk = next_dk + bdk
                dv = next_dv + bdv
        elif step != 0:
            d_kv_comm.wait()
            dk = next_dk
            dv = next_dv

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_kv[0], next_kv[1]

        d_kv = torch.stack([dk, dv]).contiguous()
        recv = d_kv_comm.send_recv(d_kv)
        d_kv_comm.commit()
        next_dk, next_dv = recv[0], recv[1]

    d_kv_comm.wait()
    return dq.to(q.dtype), next_dk.to(k.dtype), next_dv.to(v.dtype)


# --------------------------------------------------------------------------- #
# Zig-zag ring attention: balanced causal attention.                           #
#                                                                              #
# The global sequence is split into ``2 * world_size`` blocks; rank ``r`` holds #
# block ``r`` (first local half) and block ``2*world-1-r`` (second local half), #
# concatenated along the sequence dim. This makes every ring step's causal      #
# workload constant. ``q``/``k``/``v`` are ``(b, 2*local, h, d)``.              #
# --------------------------------------------------------------------------- #
def _zigzag_ring_forward(group, q, k, v, softmax_scale):
    comm = RingComm(group)
    world = comm.world_size
    rank = comm.rank

    block = q.shape[1] // 2
    q1 = q[:, block:]

    out = None
    lse = None
    next_kv = None

    for step in range(world):
        if step + 1 != world:
            kv = torch.stack([k, v]).contiguous()
            next_kv = comm.send_recv(kv)
            comm.commit()

        if step == 0:
            # local KV vs local Q: causal over the full 2-block local sequence.
            block_out, block_lse = _fa_forward(q, k, v, softmax_scale, True)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        elif step <= rank:
            # incoming KV is entirely earlier than both local Q blocks; only the
            # FIRST KV half is needed (the second KV half is later than q_low).
            k0 = k[:, :block]
            v0 = v[:, :block]
            block_out, block_lse = _fa_forward(q, k0, v0, softmax_scale, False)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            # incoming KV is later than q_low but earlier than q_high; only the
            # SECOND Q half attends, to the full incoming KV.
            block_out, block_lse = _fa_forward(q1, k, v, softmax_scale, False)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse, slice_=(slice(None), slice(block, None)))

        if step + 1 != world:
            comm.wait()
            k, v = next_kv[0], next_kv[1]

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2).contiguous()  # -> (b, h, 2*local)
    return out, lse


def _zigzag_ring_backward(group, dout, q, k, v, out, lse, softmax_scale):
    kv_comm = RingComm(group)
    d_kv_comm = RingComm(group)
    world = kv_comm.world_size
    rank = kv_comm.rank
    block = q.shape[1] // 2

    dout1 = dout[:, block:]
    q1 = q[:, block:]
    out1 = out[:, block:]
    lse1 = lse[:, :, block:].contiguous()  # lse layout (b, h, s)

    dq = None
    dk = None
    dv = None
    next_dk = None
    next_dv = None
    next_kv = None
    dk_buffer = None
    dv_buffer = None

    dq_b = torch.empty_like(q)
    dk_b = torch.empty_like(k)
    dv_b = torch.empty_like(v)

    def bwd(dout_, q_, k_, v_, out_, lse_, causal):
        sq = q_.shape[1]
        skv = k_.shape[1]
        _flash_attn_backward(
            dout=dout_,
            q=q_,
            k=k_,
            v=v_,
            out=out_,
            softmax_lse=lse_,
            dq=dq_b[:, :sq],
            dk=dk_b[:, :skv],
            dv=dv_b[:, :skv],
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            deterministic=False,
            rng_state=None,
        )

    for step in range(world):
        if step + 1 != world:
            kv = torch.stack([k, v]).contiguous()
            next_kv = kv_comm.send_recv(kv)
            kv_comm.commit()

        if step == 0:
            bwd(dout, q, k, v, out, lse, causal=True)
            dq = dq_b.to(torch.float32)
            dk = dk_b.to(torch.float32)
            dv = dv_b.to(torch.float32)
        else:
            if step <= rank:
                k0 = k[:, :block]
                v0 = v[:, :block]
                bwd(dout, q, k0, v0, out, lse, causal=False)
                dq += dq_b
            else:
                bwd(dout1, q1, k, v, out1, lse1, causal=False)
                # q1 grad lands in the first half of dq_b; add to dq second half.
                dq[:, block:] += dq_b[:, :block]

            d_kv_comm.wait()
            dk_buffer, dv_buffer = dk, dv
            dk, dv = next_dk, next_dv

            if step <= rank:
                dk[:, :block] += dk_b[:, :block]
                dv[:, :block] += dv_b[:, :block]
            else:
                dk += dk_b
                dv += dv_b

        if step + 1 != world:
            kv_comm.wait()
            k, v = next_kv[0], next_kv[1]

        d_kv = torch.stack([dk, dv]).contiguous()
        recv_buf = None
        if dk_buffer is not None:
            recv_buf = torch.stack([dk_buffer, dv_buffer]).contiguous()
        recv = d_kv_comm.send_recv(d_kv, recv_buf)
        d_kv_comm.commit()
        next_dk, next_dv = recv[0], recv[1]

    d_kv_comm.wait()
    return dq.to(q.dtype), next_dk.to(k.dtype), next_dv.to(v.dtype)


class _RingFlashAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, q, k, v, softmax_scale, causal, zigzag):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        if zigzag:
            out, lse = _zigzag_ring_forward(group, q, k, v, softmax_scale)
        else:
            out, lse = _ring_forward(group, q, k, v, softmax_scale, causal)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.group = group
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.zigzag = zigzag
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        if ctx.zigzag:
            dq, dk, dv = _zigzag_ring_backward(ctx.group, dout, q, k, v, out, lse, ctx.softmax_scale)
        else:
            dq, dk, dv = _ring_backward(ctx.group, dout, q, k, v, out, lse, ctx.softmax_scale, ctx.causal)
        return None, dq, dk, dv, None, None, None


def ring_flash_attn_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    group: Optional[ProcessGroup] = None,
) -> Tensor:
    """Plain ring attention. ``q/k/v`` are ``(b, s_local, h, d)``."""
    if not _FA_AVAILABLE:
        raise RuntimeError("ring_flash_attn_func requires flash-attn (FA2) to be installed.")
    group = get_context_parallel_group() if group is None else group
    return _RingFlashAttn.apply(group, q, k, v, softmax_scale, causal, False)


def zigzag_ring_flash_attn_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    group: Optional[ProcessGroup] = None,
) -> Tensor:
    """Balanced (zig-zag) causal ring attention.

    ``q/k/v`` are ``(b, 2*local, h, d)`` laid out so that dim-1 concatenates the
    two zig-zag blocks the rank owns (see ``.data.zigzag_reorder``).
    """
    if not _FA_AVAILABLE:
        raise RuntimeError("zigzag_ring_flash_attn_func requires flash-attn (FA2) to be installed.")
    assert causal, "zigzag ring attention is only defined for causal attention"
    group = get_context_parallel_group() if group is None else group
    return _RingFlashAttn.apply(group, q, k, v, softmax_scale, causal, True)


# --------------------------------------------------------------------------- #
# Varlen (packed) zig-zag ring attention.                                      #
#                                                                              #
# Packed sequences hold several documents back-to-back. Under USP each document #
# is INDEPENDENTLY zig-zag split across the ``cp`` group (see                  #
# ``.data.zigzag_reorder_varlen``): rank ``r`` holds, for every document, that  #
# document's blocks ``r`` and ``2*cp-1-r`` concatenated. The local packed shard #
# therefore has, per document, a ``front half`` (block r) and a ``back half``  #
# (block 2*cp-1-r). ``cu_seqlens`` here are the LOCAL per-rank document offsets #
# (each document's local length == doc_len / cp); every document length must be #
# divisible by ``2 * cp``.                                                      #
# --------------------------------------------------------------------------- #
def _fa_varlen_forward(q, k, v, cu_q, cu_k, max_q, max_k, softmax_scale, causal):
    out, lse, _, _ = _flash_attn_varlen_forward(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
    )
    return out, lse


def _fa_varlen_backward(dout, q, k, v, out, lse, dq, dk, dv, cu_q, cu_k, max_q, max_k, softmax_scale, causal):
    _flash_attn_varlen_backward(
        dout=dout,
        q=q,
        k=k,
        v=v,
        out=out,
        softmax_lse=lse,
        dq=dq,
        dk=dk,
        dv=dv,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        rng_state=None,
    )


def _varlen_half_index(cu_seqlens: Tensor, front: bool) -> Tensor:
    """Boolean mask selecting each document's front / back half.

    ``cu_seqlens`` are the LOCAL packed offsets; each document occupies
    ``[cu[i], cu[i+1])`` and is split at its midpoint.
    """
    total = int(cu_seqlens[-1].item())
    index = torch.zeros((total,), dtype=torch.bool, device=cu_seqlens.device)
    for i in range(len(cu_seqlens) - 1):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i + 1].item())
        mid = (start + end) // 2
        if front:
            index[start:mid] = True
        else:
            index[mid:end] = True
    return index


def _varlen_half_lse(lse: Tensor, cu_seqlens: Tensor, front: bool) -> Tensor:
    """Take each document's front / back half of a ``(h, total)`` LSE."""
    h = lse.shape[0]
    total_half = int(cu_seqlens[-1].item()) // 2
    new_lse = torch.empty((h, total_half), dtype=lse.dtype, device=lse.device)
    for i in range(len(cu_seqlens) - 1):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i + 1].item())
        seqlen = end - start
        new_start, new_end = start // 2, end // 2
        if front:
            src_start, src_end = start, end - seqlen // 2
        else:
            src_start, src_end = start + seqlen // 2, end
        new_lse[:, new_start:new_end] = lse[:, src_start:src_end]
    return new_lse


def _update_out_and_lse_varlen(out, lse, block_out, block_lse, index=None):
    """Online-softmax merge for varlen tensors.

    ``out`` is ``(total, h, d)`` fp32, ``lse`` is ``(total, h, 1)`` fp32.
    ``block_lse`` arrives as ``(h, block_total)`` and is reshaped to
    ``(block_total, h, 1)``. When ``index`` is given the merge writes into the
    masked rows of ``out``/``lse`` (used for the back-half-only step).
    """
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)  # (h, s) -> (s, h, 1)
    if index is None:
        new_lse = lse - torch.nn.functional.logsigmoid(lse - block_lse)
        out = out - torch.sigmoid(block_lse - lse) * (out - block_out)
        return out, new_lse
    cur_out = out[index]
    cur_lse = lse[index]
    new_lse = cur_lse - torch.nn.functional.logsigmoid(cur_lse - block_lse)
    out[index] = cur_out - torch.sigmoid(block_lse - cur_lse) * (cur_out - block_out)
    lse[index] = new_lse
    return out, lse


def _zigzag_ring_varlen_forward(group, q, k, v, cu_seqlens, max_seqlen, half0, half1, softmax_scale):
    comm = RingComm(group)
    world = comm.world_size
    rank = comm.rank

    half_cu = cu_seqlens // 2
    half_max = max_seqlen // 2
    q1 = q[half1]

    out = None
    lse = None
    next_kv = None

    for step in range(world):
        if step + 1 != world:
            kv = torch.stack([k, v]).contiguous()
            next_kv = comm.send_recv(kv)
            comm.commit()

        if step == 0:
            block_out, block_lse = _fa_varlen_forward(
                q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, softmax_scale, True
            )
            block_out = block_out.to(torch.float32)
            lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)  # (total, h, 1)
            out = block_out
        elif step <= rank:
            k0 = k[half0]
            v0 = v[half0]
            block_out, block_lse = _fa_varlen_forward(
                q, k0, v0, cu_seqlens, half_cu, max_seqlen, half_max, softmax_scale, False
            )
            out, lse = _update_out_and_lse_varlen(out, lse, block_out, block_lse)
        else:
            block_out, block_lse = _fa_varlen_forward(
                q1, k, v, half_cu, cu_seqlens, half_max, max_seqlen, softmax_scale, False
            )
            out, lse = _update_out_and_lse_varlen(out, lse, block_out, block_lse, index=half1)

        if step + 1 != world:
            comm.wait()
            k, v = next_kv[0], next_kv[1]

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(0, 1).contiguous()  # (total, h, 1) -> (h, total)
    return out, lse


def _zigzag_ring_varlen_backward(group, dout, q, k, v, out, lse, cu_seqlens, max_seqlen, half0, half1, softmax_scale):
    kv_comm = RingComm(group)
    d_kv_comm = RingComm(group)
    world = kv_comm.world_size
    rank = kv_comm.rank

    half_cu = cu_seqlens // 2
    half_max = max_seqlen // 2
    block = q.shape[0] // 2

    dout1 = dout[half1]
    q1 = q[half1]
    out1 = out[half1]
    lse1 = _varlen_half_lse(lse, cu_seqlens, front=False).contiguous()

    dq = None
    dk = None
    dv = None
    next_dk = None
    next_dv = None
    next_kv = None
    dk_buffer_prev = None
    dv_buffer_prev = None

    dq_b = torch.empty_like(q)
    dk_b = torch.empty_like(k)
    dv_b = torch.empty_like(v)

    for step in range(world):
        if step + 1 != world:
            kv = torch.stack([k, v]).contiguous()
            next_kv = kv_comm.send_recv(kv)
            kv_comm.commit()

        if step == 0:
            _fa_varlen_backward(
                dout,
                q,
                k,
                v,
                out,
                lse,
                dq_b,
                dk_b,
                dv_b,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                softmax_scale,
                True,
            )
            dq = dq_b.to(torch.float32)
            dk = dk_b.to(torch.float32)
            dv = dv_b.to(torch.float32)
        else:
            if step <= rank:
                k0 = k[half0]
                v0 = v[half0]
                _fa_varlen_backward(
                    dout,
                    q,
                    k0,
                    v0,
                    out,
                    lse,
                    dq_b[: q.shape[0]],
                    dk_b[:block],
                    dv_b[:block],
                    cu_seqlens,
                    half_cu,
                    max_seqlen,
                    half_max,
                    softmax_scale,
                    False,
                )
                dq += dq_b
            else:
                _fa_varlen_backward(
                    dout1,
                    q1,
                    k,
                    v,
                    out1,
                    lse1,
                    dq_b[:block],
                    dk_b[: k.shape[0]],
                    dv_b[: v.shape[0]],
                    half_cu,
                    cu_seqlens,
                    half_max,
                    max_seqlen,
                    softmax_scale,
                    False,
                )
                dq[half1] += dq_b[:block]

            d_kv_comm.wait()
            dk_buffer_prev, dv_buffer_prev = dk, dv
            dk, dv = next_dk, next_dv

            if step <= rank:
                dk[half0] += dk_b[:block]
                dv[half0] += dv_b[:block]
            else:
                dk += dk_b
                dv += dv_b

        if step + 1 != world:
            kv_comm.wait()
            k, v = next_kv[0], next_kv[1]

        d_kv = torch.stack([dk, dv]).contiguous()
        recv_buf = None
        if dk_buffer_prev is not None:
            recv_buf = torch.stack([dk_buffer_prev, dv_buffer_prev]).contiguous()
        recv = d_kv_comm.send_recv(d_kv, recv_buf)
        d_kv_comm.commit()
        next_dk, next_dv = recv[0], recv[1]

    d_kv_comm.wait()
    return dq.to(q.dtype), next_dk.to(k.dtype), next_dv.to(v.dtype)


class _ZigzagRingVarlenFlashAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, q, k, v, cu_seqlens, max_seqlen, softmax_scale):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        k = k.contiguous()
        v = v.contiguous()
        half0 = _varlen_half_index(cu_seqlens, front=True)
        half1 = _varlen_half_index(cu_seqlens, front=False)
        out, lse = _zigzag_ring_varlen_forward(group, q, k, v, cu_seqlens, max_seqlen, half0, half1, softmax_scale)
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens, half0, half1)
        ctx.group = group
        ctx.softmax_scale = softmax_scale
        ctx.max_seqlen = max_seqlen
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse, cu_seqlens, half0, half1 = ctx.saved_tensors
        dq, dk, dv = _zigzag_ring_varlen_backward(
            ctx.group, dout, q, k, v, out, lse, cu_seqlens, ctx.max_seqlen, half0, half1, ctx.softmax_scale
        )
        return None, dq, dk, dv, None, None, None


def zigzag_ring_flash_attn_varlen_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    group: Optional[ProcessGroup] = None,
) -> Tensor:
    """Balanced (zig-zag) causal ring attention for packed (varlen) sequences.

    ``q/k/v`` are ``(total_local, h, d)`` where ``total_local`` concatenates, for
    every document, this rank's two zig-zag block halves. ``cu_seqlens`` are the
    LOCAL per-rank document offsets and every document's local length must be
    even (its global length divisible by ``2 * cp_size``). See
    ``.data.zigzag_reorder_varlen``.
    """
    if not _FA_AVAILABLE:
        raise RuntimeError("zigzag_ring_flash_attn_varlen_func requires flash-attn (FA2).")
    assert causal, "zigzag ring attention is only defined for causal attention"
    group = get_context_parallel_group() if group is None else group
    return _ZigzagRingVarlenFlashAttn.apply(group, q, k, v, cu_seqlens, max_seqlen, softmax_scale)
