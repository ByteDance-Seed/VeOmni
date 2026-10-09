"""Bounded-mask QSA using Ascend fused attention without forward replay.

Ulysses/CP own communication. This local kernel keeps compact global indices,
retains the fused softmax statistics, and rebuilds only one mask in backward.
KV interval pruning removes only tokens absent from the compact selection.
"""

import torch
from torch.autograd.function import once_differentiable


def _blocked_mask(indices, kv_length):
    # An extra column absorbs -1 padding without overwriting valid duplicates.
    blocked = torch.ones((*indices.shape[:-1], kv_length + 1), dtype=torch.int8, device=indices.device)
    slots = torch.where(indices >= 0, indices, kv_length).long()
    blocked.scatter_(-1, slots, 0)
    return blocked[..., :kv_length].unsqueeze(1).bool().contiguous()


def _kv_ranges(indices, kv_length, chunk_size):
    # One host synchronization per attention call, not per query block.
    bounds = []
    for start in range(0, indices.shape[1], chunk_size):
        block = indices[:, start : start + chunk_size].to(torch.int32)
        lo = block.masked_fill(block < 0, kv_length).amin()
        hi = block.amax() + 1
        bounds.append(torch.stack((lo, hi)))
    ranges = torch.stack(bounds).cpu().tolist()
    # Alignment improves native kernel tiling. Never exclude a selected token.
    return [
        (max(0, lo // 128 * 128), min(kv_length, (hi + 127) // 128 * 128)) if hi > 0 else (0, 1) for lo, hi in ranges
    ]


def _forward(q, k, v, blocked, scale):
    import torch_npu

    return torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        q.shape[1],
        "BNSD",
        atten_mask=blocked,
        scale=scale,
        keep_prob=1.0,
        sparse_mode=0,
        inner_precise=2,
    )


def _backward(q, k, v, grad, blocked, output, maximum, denominator, scale):
    import torch_npu

    return torch_npu.npu_fusion_attention_grad(
        q,
        k,
        v,
        grad,
        q.shape[1],
        "BNSD",
        atten_mask=blocked,
        softmax_max=maximum,
        softmax_sum=denominator,
        attention_in=output,
        scale_value=scale,
        keep_prob=1.0,
        sparse_mode=0,
        inner_precise=2,
    )[:3]


class _CompactFusedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, indices, scale, chunk_size):
        key, value = key.contiguous(), value.contiguous()
        output = torch.empty_like(query, memory_format=torch.contiguous_format)
        statistics = []
        ranges = _kv_ranges(indices, key.shape[2], chunk_size)
        for block, start in enumerate(range(0, query.shape[2], chunk_size)):
            section = slice(start, start + chunk_size)
            q = query[:, :, section].contiguous()
            lo, hi = ranges[block]
            local_indices = indices[:, section] - lo
            blocked = _blocked_mask(local_indices, hi - lo)
            result, maximum, denominator, *_ = _forward(
                q, key[:, :, lo:hi].contiguous(), value[:, :, lo:hi].contiguous(), blocked, scale
            )
            output[:, :, section].copy_(result)
            statistics.extend((maximum, denominator))
        ctx.save_for_backward(query, key, value, indices, output, *statistics)
        ctx.ranges = ranges
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        query, key, value, indices, output, *statistics = ctx.saved_tensors
        grad_query = torch.empty_like(query, memory_format=torch.contiguous_format)
        # Accumulate disjoint-query contributions before a single BF16 writeback.
        grad_key = torch.zeros_like(key, dtype=torch.float32)
        grad_value = torch.zeros_like(value, dtype=torch.float32)
        for block, start in enumerate(range(0, query.shape[2], ctx.chunk_size)):
            section = slice(start, start + ctx.chunk_size)
            lo, hi = ctx.ranges[block]
            blocked = _blocked_mask(indices[:, section] - lo, hi - lo)
            dq, dk, dv = _backward(
                query[:, :, section].contiguous(),
                key[:, :, lo:hi].contiguous(),
                value[:, :, lo:hi].contiguous(),
                grad_output[:, :, section].contiguous(),
                blocked,
                output[:, :, section].contiguous(),
                statistics[2 * block],
                statistics[2 * block + 1],
                ctx.scale,
            )
            grad_query[:, :, section].copy_(dq)
            grad_key[:, :, lo:hi].add_(dk)
            grad_value[:, :, lo:hi].add_(dv)
        return grad_query, grad_key.to(key.dtype), grad_value.to(value.dtype), None, None, None


def qsa_attn_npu_fused(query, key, value, selected_indices, sm_scale=None, query_chunk_size=2048):
    """Return [B,S,H,D] QSA output, preserving global selection semantics.

    No dense masks are retained across query blocks or forward/backward. Native
    GQA keeps KV heads unexpanded. This is dense fused attention, not a sparse
    FLOP kernel, and does not change CP/Ulysses communication or checkpoint keys.
    """
    if query.device.type != "npu":
        raise RuntimeError("QSA npu_fused requires Ascend NPU tensors")
    if query.dtype != torch.bfloat16 or key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("QSA npu_fused requires BF16 query/key/value")
    from .validation import _validate_qsa_inputs

    _validate_qsa_inputs(query, key, value, selected_indices)
    if not isinstance(query_chunk_size, int) or query_chunk_size <= 0:
        raise ValueError("QSA query_chunk_size must be a positive integer")
    scale = query.shape[-1] ** -0.5 if sm_scale is None else sm_scale
    result = _CompactFusedAttention.apply(query, key, value, selected_indices, scale, query_chunk_size)
    return result.transpose(1, 2).contiguous()
