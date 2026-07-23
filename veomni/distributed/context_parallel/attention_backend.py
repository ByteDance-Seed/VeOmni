# Copyright (c) 2026 ByteDance AI4SE. MindSpeed-compatible Torch FA backend for Ring CP tests.
"""Attention backends used by Ring context parallel.

Torch backend is the Day-1 correctness path (CPU/NPU). NPU fusion attention is
optional and used when ``torch_npu.npu_fusion_attention`` is available.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def _repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def _expand_softmax_stats(stat: Tensor) -> Tensor:
    """Broadcast trailing singleton softmax stats to NPU-style width-8."""
    if stat.size(-1) == 8:
        return stat
    if stat.size(-1) != 1:
        raise ValueError(f"Expected softmax stats trailing dim 1 or 8, got {stat.shape}.")
    return stat.expand(*stat.shape[:-1], 8).contiguous()


def torch_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    softmax_scale: float,
    causal: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Eager BNSD attention that also returns online-softmax statistics.

    Args:
        query/key/value: [B, H, S, D] (GQA allowed: H_q may exceed H_kv)
        softmax_scale: attention scale
        causal: whether to apply a lower-triangular mask on equal-length Q/K

    Returns:
        output [B, Hq, Sq, D], softmax_max/sum [B, Hq, Sq, 8]
    """
    batch, num_q_heads, q_len, head_dim = query.shape
    num_kv_heads = key.shape[1]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"GQA requires Hq % Hkv == 0, got {num_q_heads} and {num_kv_heads}.")
    n_rep = num_q_heads // num_kv_heads
    key_exp = _repeat_kv(key, n_rep)
    value_exp = _repeat_kv(value, n_rep)

    # FP32 scores for stable online-softmax stats.
    scores = torch.matmul(query.float(), key_exp.float().transpose(-1, -2)) * float(softmax_scale)
    if causal:
        if q_len != key_exp.size(2):
            raise ValueError(
                f"Causal torch attention requires equal Q/K lengths, got {q_len} and {key_exp.size(2)}."
            )
        causal_mask = torch.ones(q_len, q_len, dtype=torch.bool, device=query.device).triu(diagonal=1)
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)

    block_max = scores.amax(dim=-1, keepdim=True)
    exp_scores = torch.exp(scores - block_max)
    block_sum = exp_scores.sum(dim=-1, keepdim=True)
    probs = exp_scores / block_sum.clamp_min(torch.finfo(exp_scores.dtype).tiny)
    output = torch.matmul(probs.to(value_exp.dtype), value_exp)

    # Tokens that attended an empty/fully-masked row get -inf max / 0 sum.
    if causal:
        fully_masked = ~torch.isfinite(block_max)
        block_max = torch.where(fully_masked, torch.full_like(block_max, -torch.inf), block_max)
        block_sum = torch.where(fully_masked, torch.zeros_like(block_sum), block_sum)

    softmax_max = _expand_softmax_stats(block_max.to(query.dtype))
    softmax_sum = _expand_softmax_stats(block_sum.to(query.dtype))
    return output, softmax_max, softmax_sum


def torch_attention_backward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dout: Tensor,
    attn_out: Tensor,
    softmax_max: Tensor,
    softmax_sum: Tensor,
    *,
    softmax_scale: float,
    causal: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Block-wise attention backward using global merged online-softmax stats."""
    batch, num_q_heads, q_len, head_dim = query.shape
    num_kv_heads = key.shape[1]
    n_rep = num_q_heads // num_kv_heads
    key_exp = _repeat_kv(key, n_rep)
    value_exp = _repeat_kv(value, n_rep)
    k_len = key_exp.size(2)

    scores = torch.matmul(query.float(), key_exp.float().transpose(-1, -2)) * float(softmax_scale)
    if causal:
        causal_mask = torch.ones(q_len, k_len, dtype=torch.bool, device=query.device).triu(diagonal=1)
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)

    # Reconstruct P from global LSE pieces: P = exp(S - max) / sum.
    global_max = softmax_max[..., :1].float()
    global_sum = softmax_sum[..., :1].float().clamp_min(torch.finfo(torch.float32).tiny)
    probs = torch.exp(scores - global_max) / global_sum
    if causal:
        probs = probs.masked_fill(causal_mask, 0.0)

    dout_f = dout.float()
    out_f = attn_out.float()
    delta = (dout_f * out_f).sum(dim=-1, keepdim=True)

    dv_exp = torch.matmul(probs.transpose(-1, -2), dout_f)
    dp = torch.matmul(dout_f, value_exp.float().transpose(-1, -2))
    ds = probs * (dp - delta)
    dq = torch.matmul(ds, key_exp.float()) * float(softmax_scale)
    dk_exp = torch.matmul(ds.transpose(-1, -2), query.float()) * float(softmax_scale)

    if n_rep == 1:
        dk, dv = dk_exp, dv_exp
    else:
        dk = dk_exp.view(batch, num_kv_heads, n_rep, k_len, head_dim).sum(dim=2)
        dv = dv_exp.view(batch, num_kv_heads, n_rep, k_len, head_dim).sum(dim=2)

    return dq.to(query.dtype), dk.to(key.dtype), dv.to(value.dtype)


def npu_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    num_heads: int,
    softmax_scale: float,
    causal: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """NPU fusion attention forward (BNSD). Falls back is not available here."""
    import torch_npu

    atten_mask = None
    sparse_mode = 0
    pre_tokens = key.size(2)
    next_tokens = pre_tokens
    if causal:
        atten_mask = torch.ones((2048, 2048), dtype=torch.bool, device=query.device)
        atten_mask = torch.triu(atten_mask, diagonal=1)
        next_tokens = 0
        sparse_mode = 3

    outs = torch_npu.npu_fusion_attention(
        query,
        key,
        value,
        num_heads,
        "BNSD",
        pse=None,
        padding_mask=None,
        atten_mask=atten_mask,
        scale=softmax_scale,
        pre_tockens=pre_tokens,
        next_tockens=next_tokens,
        keep_prob=1.0,
        sparse_mode=sparse_mode,
    )
    return outs[0], outs[1], outs[2]


def npu_attention_backward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dout: Tensor,
    attn_out: Tensor,
    softmax_max: Tensor,
    softmax_sum: Tensor,
    *,
    num_heads: int,
    softmax_scale: float,
    causal: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    import torch_npu

    atten_mask = None
    sparse_mode = 0
    pre_tokens = key.size(2)
    next_tokens = pre_tokens
    if causal:
        atten_mask = torch.ones((2048, 2048), dtype=torch.bool, device=query.device)
        atten_mask = torch.triu(atten_mask, diagonal=1)
        next_tokens = 0
        sparse_mode = 3

    grads = torch_npu.npu_fusion_attention_grad(
        query,
        key,
        value,
        dout,
        num_heads,
        "BNSD",
        pse=None,
        padding_mask=None,
        atten_mask=atten_mask,
        softmax_max=softmax_max,
        softmax_sum=softmax_sum,
        attention_in=attn_out,
        scale_value=softmax_scale,
        pre_tockens=pre_tokens,
        next_tockens=next_tokens,
        sparse_mode=sparse_mode,
        keep_prob=1.0,
    )
    return grads[0], grads[1], grads[2]


def has_npu_fusion_attention() -> bool:
    try:
        import torch_npu

        return hasattr(torch_npu, "npu_fusion_attention") and hasattr(torch_npu, "npu_fusion_attention_grad")
    except Exception:
        return False


def torch_packed_causal_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens: Tensor,
    *,
    softmax_scale: Optional[float] = None,
) -> Tensor:
    """Dense packed causal attention with no cross-sample visibility.

    Inputs are BNSD with batch=1: [1, H, T, D]. ``cu_seqlens`` is int32 [N+1].
    """
    if query.size(0) != 1:
        raise ValueError(f"torch_packed_causal_attention expects batch=1, got {query.shape}.")
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    cu = cu_seqlens.detach().cpu().tolist()
    outs = []
    for start, end in zip(cu[:-1], cu[1:]):
        start, end = int(start), int(end)
        outs.append(
            torch_attention_forward(
                query[:, :, start:end],
                key[:, :, start:end],
                value[:, :, start:end],
                softmax_scale=float(softmax_scale),
                causal=True,
            )[0]
        )
    return torch.cat(outs, dim=2)
