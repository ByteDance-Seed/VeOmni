"""Numerical oracle for the Qwen4-Exp QSA routing operation.

The implementation intentionally follows ``Qwen4ExpTextQSAIndexer.forward``
closely.  It is used as a correctness fallback for unsupported layouts and as
the oracle for the Triton tests.
"""

from __future__ import annotations

import math

import torch


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (output * (1.0 + weight.float())).to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    x_rope, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    rotated = x_rope * cos + _rotate_half(x_rope) * sin
    return torch.cat((rotated, x_pass), dim=-1)


@torch.no_grad()
def qsa_reference_mask(
    index_queries: torch.Tensor,
    raw_keys: torch.Tensor,
    full_cos: torch.Tensor,
    full_sin: torch.Tensor,
    attention_mask: torch.Tensor,
    k_norm_weight: torch.Tensor,
    *,
    rms_norm_eps: float,
    token_budget: int,
    compress_ratio: int,
    return_token_ids: bool = False,
) -> torch.Tensor:
    """Return the exact eager QSA mask or its intermediate token IDs.

    Note on ``return_token_ids``: the mask form is the only consumer that matters
    for training and is bit-exact against the Triton fast path.  The token-ID form
    is diagnostic only — within a row the *set* of selected tokens is identical to
    the Triton path, but the *order* is not guaranteed to match: this reference
    emits score-topk permutation for every row whose visible complete blocks are
    non-empty, while the Triton path fills budget-fitting (dense) rows in
    positional order and only permutes rows that actually go through TopK.  Compare
    sorted IDs (or the mask) when testing.

    Args:
        index_queries: Normalized and RoPE-rotated index queries ``[B,Q,H,D]``.
        raw_keys: Uncompressed index keys, including cache, ``[B,KV,D]``.
        full_cos/full_sin: RoPE tensors ``[B,KV,R]`` where ``R <= D``.
        attention_mask: HF four-dimensional bool or additive mask
            ``[B,1,Q,KV]``.
        k_norm_weight: Qwen4-Exp residual RMSNorm parameter ``[D]``.  The
            effective multiplier is ``1 + weight``.
    """
    if attention_mask.dtype == torch.bool:
        visible = attention_mask
    else:
        visible = attention_mask == 0

    batch_size, query_length, num_heads, head_dim = index_queries.shape
    kv_length = raw_keys.shape[1]
    block_budget = token_budget // compress_ratio
    output_width = token_budget + compress_ratio - 1
    selected = torch.full(
        (batch_size, query_length, output_width),
        -1,
        dtype=torch.int64,
        device=index_queries.device,
    )

    for batch_idx in range(batch_size):
        rope_batch = min(batch_idx, full_cos.shape[0] - 1)
        for query_idx in range(query_length):
            local_visible = torch.nonzero(visible[batch_idx, 0, query_idx], as_tuple=False).flatten()
            num_complete = local_visible.numel() // compress_ratio
            if num_complete:
                block_tokens = local_visible[: num_complete * compress_ratio].view(num_complete, compress_ratio)
                grouped = raw_keys[batch_idx].index_select(0, block_tokens.flatten())
                pooled = grouped.view(num_complete, compress_ratio, head_dim).float().mean(1).to(raw_keys.dtype)
                pooled = _rms_norm(pooled, k_norm_weight, rms_norm_eps)
                starts = block_tokens[:, 0]
                keys = _apply_rope(
                    pooled,
                    full_cos[rope_batch].index_select(0, starts),
                    full_sin[rope_batch].index_select(0, starts),
                )
                scores = torch.matmul(index_queries[batch_idx, query_idx].float(), keys.float().T).T
                scores = torch.relu(scores).sum(-1) / math.sqrt(head_dim)
                top_blocks = scores.topk(min(block_budget, num_complete), dim=0).indices
                tokens = block_tokens.index_select(0, top_blocks).flatten()
            else:
                tokens = local_visible.new_empty((0,))
            tail = local_visible[num_complete * compress_ratio :]
            tokens = torch.cat((tokens, tail))
            selected[batch_idx, query_idx, : tokens.numel()] = tokens

    if return_token_ids:
        return selected.to(torch.int32)

    # Route every -1 to a sentinel column and remove that column afterwards.
    # This also avoids duplicate invalid writes racing with a valid token 0.
    bool_output = torch.zeros(
        (batch_size, 1, query_length, kv_length + 1), dtype=torch.bool, device=attention_mask.device
    )
    scatter_ids = torch.where(selected >= 0, selected, kv_length).unsqueeze(1)
    bool_output.scatter_(-1, scatter_ids, True)
    bool_output = bool_output[..., :kv_length]
    if attention_mask.dtype == torch.bool:
        return bool_output

    min_value = torch.finfo(attention_mask.dtype).min
    return torch.where(bool_output, attention_mask.new_zeros(()), min_value)
