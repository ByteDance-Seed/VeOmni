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

"""Ascend NPU zig-zag Ring Attention primitives for USP.

The sequence and ring schedule match :mod:`ring_attention`. The local
attention backend is ``torch_npu.npu_fusion_attention`` instead of a CUDA
FlashAttention low-level kernel. NPU fusion attention exposes softmax maximum
and sum tensors rather than log-sum-exp, so partial ring outputs are merged
with those statistics and saved for the explicit backward operator.
"""

from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.distributed import ProcessGroup

from ....utils.import_utils import is_torch_npu_available
from ..comm import get_context_parallel_group
from .comm import RingComm


if is_torch_npu_available():
    import torch_npu
else:  # pragma: no cover - exercised only when an NPU function is called without torch_npu
    torch_npu = None


ActualSeqLen = Union[Tensor, Sequence[int]]
HalfIndex = Union[slice, Tensor]
RNGState = Tuple[int, int, int]

_CAUSAL_MASK_CACHE: dict[tuple[str, Optional[int]], Tensor] = {}

__all__ = [
    "update_npu_out_and_softmax_stats",
    "forward",
    "packed_forward",
    "zigzag_ring_npu_flash_attn_func",
    "zigzag_ring_npu_flash_attn_varlen_func",
]


def _require_torch_npu() -> None:
    if torch_npu is None:
        raise RuntimeError("NPU ring attention requires torch_npu to be installed.")


def _normalize_cu_seqlens(cu_seqlens: ActualSeqLen) -> Tensor:
    """Normalize packed sequence offsets while preserving the leading zero."""
    if isinstance(cu_seqlens, Tensor):
        offsets = cu_seqlens.detach().to(device="cpu", dtype=torch.long)
    else:
        offsets = torch.as_tensor(cu_seqlens, dtype=torch.long, device="cpu")
    if offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError("cu_seqlens must be a 1-D tensor or sequence with at least two offsets")
    if offsets[0].item() != 0:
        raise ValueError("cu_seqlens must start with zero")
    if (offsets[1:] <= 0).any() or (offsets[1:] < offsets[:-1]).any():
        raise ValueError("cu_seqlens must contain positive, non-decreasing cumulative offsets")
    return offsets.contiguous()


def _to_npu_actual_seq_len(actual_seq_len: Optional[ActualSeqLen]) -> Optional[list[int]]:
    """Convert cumulative lengths to the endpoint list expected by torch_npu."""
    if actual_seq_len is None:
        return None
    if isinstance(actual_seq_len, Tensor):
        values = actual_seq_len.detach().to(device="cpu", dtype=torch.long)
    else:
        values = torch.as_tensor(actual_seq_len, dtype=torch.long, device="cpu")
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("actual_seq_len must be a non-empty 1-D sequence")
    if values[0].item() == 0:
        values = values[1:]
    if values.numel() == 0 or (values <= 0).any() or (values[1:] < values[:-1]).any():
        raise ValueError("actual_seq_len must contain positive, non-decreasing endpoints")
    return [int(value) for value in values.tolist()]


def _causal_mask(device: torch.device) -> Tensor:
    key = (device.type, device.index)
    mask = _CAUSAL_MASK_CACHE.get(key)
    if mask is None:
        # sparse_mode=2 accepts the fixed 2048x2048 compressed causal mask.
        mask = torch.triu(torch.ones((2048, 2048), dtype=torch.bool, device=device), diagonal=1)
        _CAUSAL_MASK_CACHE[key] = mask
    return mask


def _head_num(q: Tensor, input_layout: str) -> int:
    if input_layout == "TND":
        return q.shape[1]
    if input_layout == "BSND":
        return q.shape[2]
    if input_layout == "BNSD":
        return q.shape[1]
    raise ValueError(f"Unsupported NPU attention input layout: {input_layout!r}")


def _npu_fa_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    input_layout: str,
    softmax_scale: float,
    dropout_p: float,
    causal: bool,
    actual_seq_qlen: Optional[ActualSeqLen] = None,
    actual_seq_kvlen: Optional[ActualSeqLen] = None,
    softmax_layout: str = "",
) -> Tuple[Tensor, Tensor, Tensor, RNGState]:
    _require_torch_npu()
    attention_mask = _causal_mask(q.device) if causal else None
    sparse_mode = 2 if causal else 0
    actual_seq_qlen = _to_npu_actual_seq_len(actual_seq_qlen)
    actual_seq_kvlen = _to_npu_actual_seq_len(actual_seq_kvlen)
    block_out, block_max, block_sum, _, seed, offset, numels = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num=_head_num(q, input_layout),
        input_layout=input_layout,
        softmax_layout=softmax_layout,
        atten_mask=attention_mask,
        scale=softmax_scale,
        keep_prob=1.0 - dropout_p,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        sparse_mode=sparse_mode,
    )
    return block_out, block_max, block_sum, (seed, offset, numels)


def _npu_fa_backward(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_max: Tensor,
    softmax_sum: Tensor,
    *,
    input_layout: str,
    softmax_scale: float,
    dropout_p: float,
    causal: bool,
    rng_state: RNGState,
    actual_seq_qlen: Optional[ActualSeqLen] = None,
    actual_seq_kvlen: Optional[ActualSeqLen] = None,
    softmax_layout: str = "",
) -> Tuple[Tensor, Tensor, Tensor]:
    _require_torch_npu()
    attention_mask = _causal_mask(q.device) if causal else None
    actual_seq_qlen = _to_npu_actual_seq_len(actual_seq_qlen)
    actual_seq_kvlen = _to_npu_actual_seq_len(actual_seq_kvlen)
    seed, offset, numels = rng_state
    if softmax_max.ndim == q.ndim - 1:
        softmax_max = softmax_max.unsqueeze(-1).expand(*softmax_max.shape, 8).contiguous()
        softmax_sum = softmax_sum.unsqueeze(-1).expand(*softmax_sum.shape, 8).contiguous()
    dq, dk, dv, *_ = torch_npu.npu_fusion_attention_grad(
        q,
        k,
        v,
        dout,
        head_num=_head_num(q, input_layout),
        input_layout=input_layout,
        atten_mask=attention_mask,
        softmax_max=softmax_max,
        softmax_sum=softmax_sum,
        attention_in=out,
        scale_value=softmax_scale,
        keep_prob=1.0 - dropout_p,
        seed=seed,
        offset=offset,
        numels=numels,
        softmax_layout=softmax_layout,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        sparse_mode=2 if causal else 0,
    )
    return dq, dk, dv


def update_npu_out_and_softmax_stats(
    out: Optional[Tensor],
    softmax_max: Optional[Tensor],
    softmax_sum: Optional[Tensor],
    block_out: Tensor,
    block_max: Tensor,
    block_sum: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Merge an NPU fusion-attention block into the running softmax state."""
    current_max = block_max[..., 0]
    current_sum = block_sum[..., 0]
    if out is None:
        return block_out.to(torch.float32), current_max, current_sum
    if softmax_max is None or softmax_sum is None:
        raise RuntimeError("Previous NPU softmax statistics are missing.")

    merged_max = torch.maximum(softmax_max, current_max)
    previous_sum = softmax_sum * torch.exp(softmax_max - merged_max)
    current_sum = current_sum * torch.exp(current_max - merged_max)
    merged_sum = previous_sum + current_sum

    previous_weight = previous_sum / merged_sum
    current_weight = current_sum / merged_sum
    if block_out.ndim == 4:
        # BSND output pairs with BNS softmax statistics.
        previous_weight = previous_weight.transpose(1, 2)
        current_weight = current_weight.transpose(1, 2)
    merged_out = out * previous_weight.unsqueeze(-1) + block_out * current_weight.unsqueeze(-1)
    return merged_out, merged_max, merged_sum


def _expand_npu_softmax_stats(stats: Tensor) -> Tensor:
    return stats.unsqueeze(-1).expand(*stats.shape, 8).contiguous()


def _zigzag_npu_forward(
    group: ProcessGroup,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmax_scale: float,
    dropout_p: float,
) -> Tuple[Tensor, Tensor, Tensor, list[RNGState]]:
    comm = RingComm(group)
    block = q.shape[1] // 2
    q1 = q[:, block:].contiguous()
    out = softmax_max = softmax_sum = None
    rng_states: list[RNGState] = [(0, 0, 0) for _ in range(comm.world_size)]

    k = k.contiguous()
    v = v.contiguous()
    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k = comm.send_recv(k)
            next_v = comm.send_recv(v)
            comm.commit()

        if step == 0:
            block_out, block_max, block_sum, rng_state = _npu_fa_forward(
                q,
                k,
                v,
                input_layout="BSND",
                softmax_scale=softmax_scale,
                dropout_p=dropout_p,
                causal=True,
            )
            out, softmax_max, softmax_sum = update_npu_out_and_softmax_stats(
                out, softmax_max, softmax_sum, block_out, block_max, block_sum
            )
        elif step <= comm.rank:
            block_out, block_max, block_sum, rng_state = _npu_fa_forward(
                q,
                k[:, :block],
                v[:, :block],
                input_layout="BSND",
                softmax_scale=softmax_scale,
                dropout_p=dropout_p,
                causal=False,
            )
            out, softmax_max, softmax_sum = update_npu_out_and_softmax_stats(
                out, softmax_max, softmax_sum, block_out, block_max, block_sum
            )
        else:
            block_out, block_max, block_sum, rng_state = _npu_fa_forward(
                q1,
                k,
                v,
                input_layout="BSND",
                softmax_scale=softmax_scale,
                dropout_p=dropout_p,
                causal=False,
            )
            out[:, block:], softmax_max[:, :, block:], softmax_sum[:, :, block:] = update_npu_out_and_softmax_stats(
                out[:, block:],
                softmax_max[:, :, block:],
                softmax_sum[:, :, block:],
                block_out,
                block_max,
                block_sum,
            )
        rng_states[step] = rng_state

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    return out.to(q.dtype), softmax_max, softmax_sum, rng_states


def _zigzag_npu_backward(
    group: ProcessGroup,
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_max: Tensor,
    softmax_sum: Tensor,
    rng_states: Sequence[RNGState],
    softmax_scale: float,
    dropout_p: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    kv_comm = RingComm(group)
    d_kv_comm = RingComm(group)
    block = q.shape[1] // 2
    q1 = q[:, block:].contiguous()
    dout1 = dout[:, block:].contiguous()
    out1 = out[:, block:].contiguous()
    softmax_max1 = _expand_npu_softmax_stats(softmax_max[:, :, block:])
    softmax_sum1 = _expand_npu_softmax_stats(softmax_sum[:, :, block:])
    softmax_max = _expand_npu_softmax_stats(softmax_max)
    softmax_sum = _expand_npu_softmax_stats(softmax_sum)

    dq = dk = dv = None
    next_dk = next_dv = None
    dk_buffer = dv_buffer = None
    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k = kv_comm.send_recv(k)
            next_v = kv_comm.send_recv(v)
            kv_comm.commit()

        if step == 0:
            block_dq, block_dk, block_dv = _npu_fa_backward(
                dout.contiguous(),
                q,
                k,
                v,
                out,
                softmax_max,
                softmax_sum,
                input_layout="BSND",
                softmax_scale=softmax_scale,
                dropout_p=dropout_p,
                causal=True,
                rng_state=rng_states[step],
            )
            dq = block_dq.float().clone()
            dk = block_dk.float().clone()
            dv = block_dv.float().clone()
        else:
            if step <= kv_comm.rank:
                block_dq, block_dk, block_dv = _npu_fa_backward(
                    dout.contiguous(),
                    q,
                    k[:, :block],
                    v[:, :block],
                    out,
                    softmax_max,
                    softmax_sum,
                    input_layout="BSND",
                    softmax_scale=softmax_scale,
                    dropout_p=dropout_p,
                    causal=False,
                    rng_state=rng_states[step],
                )
                dq += block_dq.float()
            else:
                block_dq, block_dk, block_dv = _npu_fa_backward(
                    dout1,
                    q1,
                    k,
                    v,
                    out1,
                    softmax_max1,
                    softmax_sum1,
                    input_layout="BSND",
                    softmax_scale=softmax_scale,
                    dropout_p=dropout_p,
                    causal=False,
                    rng_state=rng_states[step],
                )
                dq[:, block:] += block_dq.float()

            d_kv_comm.wait()
            dk_buffer, dv_buffer = dk, dv
            dk, dv = next_dk, next_dv
            if step <= kv_comm.rank:
                dk[:, :block] += block_dk.float()
                dv[:, :block] += block_dv.float()
            else:
                dk += block_dk.float()
                dv += block_dv.float()

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v

        next_dk = d_kv_comm.send_recv(dk, dk_buffer)
        next_dv = d_kv_comm.send_recv(dv, dv_buffer)
        d_kv_comm.commit()

    d_kv_comm.wait()
    return dq.to(q.dtype), next_dk.to(k.dtype), next_dv.to(v.dtype)


class _ZigzagRingNPUFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, q, k, v, dropout_p, softmax_scale, causal):
        if not causal:
            raise ValueError("NPU zig-zag ring attention requires causal=True")
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        out, softmax_max, softmax_sum, rng_states = _zigzag_npu_forward(group, q, k, v, softmax_scale, dropout_p)
        ctx.save_for_backward(q, k, v, out, softmax_max, softmax_sum)
        ctx.group = group
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.rng_states = rng_states
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, softmax_max, softmax_sum = ctx.saved_tensors
        dq, dk, dv = _zigzag_npu_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_max,
            softmax_sum,
            ctx.rng_states,
            ctx.softmax_scale,
            ctx.dropout_p,
        )
        return None, dq, dk, dv, None, None, None


def zigzag_ring_npu_flash_attn_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    group: Optional[ProcessGroup] = None,
    dropout_p: float = 0.0,
) -> Tensor:
    """Balanced causal Ring Attention for dense NPU ``(B, S, N, D)`` tensors."""
    _require_torch_npu()
    group = get_context_parallel_group() if group is None else group
    return _ZigzagRingNPUFlashAttention.apply(group, q, k, v, dropout_p, softmax_scale, causal)


def _varlen_half_index(cu_seqlens: Tensor, *, front: bool) -> HalfIndex:
    if cu_seqlens.numel() == 2:
        midpoint = int(cu_seqlens[-1].item()) // 2
        return slice(None, midpoint) if front else slice(midpoint, None)

    total = int(cu_seqlens[-1].item())
    index = torch.zeros((total,), dtype=torch.bool, device=cu_seqlens.device)
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        start_int, end_int = int(start.item()), int(end.item())
        midpoint = (start_int + end_int) // 2
        if front:
            index[start_int:midpoint] = True
        else:
            index[midpoint:end_int] = True
    return index


def _zigzag_npu_varlen_forward(
    group: ProcessGroup,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    half0: HalfIndex,
    half1: HalfIndex,
    softmax_scale: float,
    dropout_p: float,
) -> Tuple[Tensor, Tensor, Tensor, list[RNGState]]:
    comm = RingComm(group)
    half_cu_seqlens = (cu_seqlens // 2).contiguous()
    block = q.shape[0] // 2
    q1 = q[half1].contiguous()
    out = softmax_max = softmax_sum = None
    rng_states: list[RNGState] = [(0, 0, 0) for _ in range(comm.world_size)]

    def forward_block(block_q: Tensor, block_k: Tensor, block_v: Tensor, causal: bool):
        q_cu_seqlens = half_cu_seqlens if block_q.shape[0] == block else cu_seqlens
        kv_cu_seqlens = half_cu_seqlens if block_k.shape[0] == block else cu_seqlens
        return _npu_fa_forward(
            block_q,
            block_k,
            block_v,
            input_layout="TND",
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            actual_seq_qlen=q_cu_seqlens,
            actual_seq_kvlen=kv_cu_seqlens,
            softmax_layout="TND",
        )

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k = comm.send_recv(k)
            next_v = comm.send_recv(v)
            comm.commit()

        if step == 0:
            block_out, block_max, block_sum, rng_state = forward_block(q, k, v, True)
            out, softmax_max, softmax_sum = update_npu_out_and_softmax_stats(
                out, softmax_max, softmax_sum, block_out, block_max, block_sum
            )
        elif step <= comm.rank:
            block_out, block_max, block_sum, rng_state = forward_block(q, k[half0], v[half0], False)
            out, softmax_max, softmax_sum = update_npu_out_and_softmax_stats(
                out, softmax_max, softmax_sum, block_out, block_max, block_sum
            )
        else:
            block_out, block_max, block_sum, rng_state = forward_block(q1, k, v, False)
            out[half1], softmax_max[half1], softmax_sum[half1] = update_npu_out_and_softmax_stats(
                out[half1], softmax_max[half1], softmax_sum[half1], block_out, block_max, block_sum
            )
        rng_states[step] = rng_state

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    return out.to(q.dtype), softmax_max, softmax_sum, rng_states


def _zigzag_npu_varlen_backward(
    group: ProcessGroup,
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_max: Tensor,
    softmax_sum: Tensor,
    cu_seqlens: Tensor,
    half0: HalfIndex,
    half1: HalfIndex,
    rng_states: Sequence[RNGState],
    softmax_scale: float,
    dropout_p: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    kv_comm = RingComm(group)
    d_kv_comm = RingComm(group)
    half_cu_seqlens = (cu_seqlens // 2).contiguous()
    block = q.shape[0] // 2
    q1 = q[half1].contiguous()
    dout1 = dout[half1].contiguous()
    out1 = out[half1].contiguous()
    softmax_max1 = _expand_npu_softmax_stats(softmax_max[half1])
    softmax_sum1 = _expand_npu_softmax_stats(softmax_sum[half1])
    softmax_max = _expand_npu_softmax_stats(softmax_max)
    softmax_sum = _expand_npu_softmax_stats(softmax_sum)

    def backward_block(
        block_dout: Tensor,
        block_q: Tensor,
        block_k: Tensor,
        block_v: Tensor,
        block_out: Tensor,
        block_max: Tensor,
        block_sum: Tensor,
        causal: bool,
        rng_state: RNGState,
    ):
        q_cu_seqlens = half_cu_seqlens if block_q.shape[0] == block else cu_seqlens
        kv_cu_seqlens = half_cu_seqlens if block_k.shape[0] == block else cu_seqlens
        return _npu_fa_backward(
            block_dout,
            block_q,
            block_k,
            block_v,
            block_out,
            block_max,
            block_sum,
            input_layout="TND",
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            rng_state=rng_state,
            actual_seq_qlen=q_cu_seqlens,
            actual_seq_kvlen=kv_cu_seqlens,
            softmax_layout="TND",
        )

    dq = dk = dv = None
    next_dk = next_dv = None
    dk_buffer = dv_buffer = None
    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k = kv_comm.send_recv(k)
            next_v = kv_comm.send_recv(v)
            kv_comm.commit()

        if step == 0:
            block_dq, block_dk, block_dv = backward_block(
                dout.contiguous(), q, k, v, out, softmax_max, softmax_sum, True, rng_states[step]
            )
            dq = block_dq.float().clone()
            dk = block_dk.float().clone()
            dv = block_dv.float().clone()
        else:
            if step <= kv_comm.rank:
                block_dq, block_dk, block_dv = backward_block(
                    dout.contiguous(),
                    q,
                    k[half0],
                    v[half0],
                    out,
                    softmax_max,
                    softmax_sum,
                    False,
                    rng_states[step],
                )
                dq += block_dq.float()
            else:
                block_dq, block_dk, block_dv = backward_block(
                    dout1,
                    q1,
                    k,
                    v,
                    out1,
                    softmax_max1,
                    softmax_sum1,
                    False,
                    rng_states[step],
                )
                dq[half1] += block_dq.float()

            d_kv_comm.wait()
            dk_buffer, dv_buffer = dk, dv
            dk, dv = next_dk, next_dv
            if step <= kv_comm.rank:
                dk[half0] += block_dk.float()
                dv[half0] += block_dv.float()
            else:
                dk += block_dk.float()
                dv += block_dv.float()

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v

        next_dk = d_kv_comm.send_recv(dk, dk_buffer)
        next_dv = d_kv_comm.send_recv(dv, dv_buffer)
        d_kv_comm.commit()

    d_kv_comm.wait()
    return dq.to(q.dtype), next_dk.to(k.dtype), next_dv.to(v.dtype)


class _ZigzagRingNPUFlashAttentionVarlen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, q, k, v, cu_seqlens, dropout_p, softmax_scale, causal):
        if not causal:
            raise ValueError("NPU zig-zag ring attention requires causal=True")
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        cu_seqlens = _normalize_cu_seqlens(cu_seqlens)
        half0 = _varlen_half_index(cu_seqlens, front=True)
        half1 = _varlen_half_index(cu_seqlens, front=False)
        if isinstance(half0, Tensor):
            half0 = half0.to(q.device)
            half1 = half1.to(q.device)

        out, softmax_max, softmax_sum, rng_states = _zigzag_npu_varlen_forward(
            group, q, k, v, cu_seqlens, half0, half1, softmax_scale, dropout_p
        )
        ctx.half_indices_are_tensors = isinstance(half0, Tensor)
        tensors = [q, k, v, out, softmax_max, softmax_sum, cu_seqlens]
        if ctx.half_indices_are_tensors:
            tensors.extend((half0, half1))
        else:
            ctx.half0, ctx.half1 = half0, half1
        ctx.save_for_backward(*tensors)
        ctx.group = group
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.rng_states = rng_states
        return out

    @staticmethod
    def backward(ctx, dout):
        saved = ctx.saved_tensors
        q, k, v, out, softmax_max, softmax_sum, cu_seqlens = saved[:7]
        if ctx.half_indices_are_tensors:
            half0, half1 = saved[7:]
        else:
            half0, half1 = ctx.half0, ctx.half1
        dq, dk, dv = _zigzag_npu_varlen_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_max,
            softmax_sum,
            cu_seqlens,
            half0,
            half1,
            ctx.rng_states,
            ctx.softmax_scale,
            ctx.dropout_p,
        )
        return None, dq, dk, dv, None, None, None, None


def zigzag_ring_npu_flash_attn_varlen_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: ActualSeqLen,
    max_seqlen: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    group: Optional[ProcessGroup] = None,
    dropout_p: float = 0.0,
) -> Tensor:
    """Balanced causal Ring Attention for packed NPU ``(T, N, D)`` tensors."""
    _require_torch_npu()
    del max_seqlen  # Compatibility with the shared CUDA/NPU varlen wrapper API.
    group = get_context_parallel_group() if group is None else group
    return _ZigzagRingNPUFlashAttentionVarlen.apply(group, q, k, v, cu_seqlens, dropout_p, softmax_scale, causal)


def forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    group: Optional[ProcessGroup] = None,
    dropout_p: float = 0.0,
) -> Tensor:
    """Run balanced causal Ring Attention on fixed-shape NPU tensors."""
    return zigzag_ring_npu_flash_attn_func(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        group=group,
        dropout_p=dropout_p,
    )


def packed_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: ActualSeqLen,
    max_seqlen: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    group: Optional[ProcessGroup] = None,
    dropout_p: float = 0.0,
) -> Tensor:
    """Run balanced causal Ring Attention on packed NPU tensors."""
    return zigzag_ring_npu_flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        softmax_scale=softmax_scale,
        causal=causal,
        group=group,
        dropout_p=dropout_p,
    )
