"""Fused Triton kernels for Qwen3.8 / Qwen4-Exp QSA routing.

The per-segment fast path targets standard causal-prefix masks. It removes the
per-query Python loop from the HuggingFace reference implementation while
retaining ``torch.topk``: Ascend's optimized TopK is substantially better than
an iterative scalar selection loop in Triton. VeOmni packed varlen training
splits each sequence at ``cu_seq_lens_q`` boundaries before entering the kernel.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from .common import project_qsa_indexer_inputs
from .reference import qsa_reference_mask


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
_fallback_warned = False

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def triton_is_available() -> bool:
    return _TRITON_AVAILABLE


if _TRITON_AVAILABLE:

    @triton.jit
    def _compress_norm_rope_kernel(
        raw_key,
        norm_weight,
        cos,
        sin,
        compressed_key,
        stride_kb: tl.constexpr,
        stride_ks: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_cb: tl.constexpr,
        stride_cs: tl.constexpr,
        stride_cd: tl.constexpr,
        stride_cos_b: tl.constexpr,
        stride_cos_s: tl.constexpr,
        stride_cos_d: tl.constexpr,
        stride_sin_b: tl.constexpr,
        stride_sin_s: tl.constexpr,
        stride_sin_d: tl.constexpr,
        num_blocks: tl.constexpr,
        head_dim: tl.constexpr,
        rotary_dim: tl.constexpr,
        eps: tl.constexpr,
        compress_ratio: tl.constexpr,
        input_is_bf16: tl.constexpr,
        apply_rope: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        block_idx = tl.program_id(1)
        dims = tl.arange(0, BLOCK_D)
        dim_mask = dims < head_dim
        token_start = block_idx * compress_ratio

        pooled = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for lane in tl.static_range(0, compress_ratio):
            ptr = raw_key + batch_idx * stride_kb + (token_start + lane) * stride_ks + dims * stride_kd
            pooled += tl.load(ptr, mask=dim_mask, other=0.0).to(tl.float32)
        pooled *= 1.0 / compress_ratio
        # Match the reference's ``mean(...).to(raw_keys.dtype)`` before norm.
        if input_is_bf16:
            pooled = pooled.to(tl.bfloat16).to(tl.float32)
        else:
            pooled = pooled.to(tl.float16).to(tl.float32)

        variance = tl.sum(pooled * pooled, axis=0) / head_dim
        inv_rms = tl.rsqrt(variance + eps)
        weight = tl.load(norm_weight + dims, mask=dim_mask, other=0.0).to(tl.float32)
        normalized = pooled * inv_rms * (1.0 + weight)
        # Qwen4-Exp RMSNorm casts back before RoPE.
        if input_is_bf16:
            normalized = normalized.to(tl.bfloat16)
        else:
            normalized = normalized.to(tl.float16)

        output = normalized
        if apply_rope:
            half_rotary = rotary_dim // 2
            rope_mask = dim_mask & (dims < rotary_dim)
            pair_dims = tl.where(dims < half_rotary, dims + half_rotary, dims - half_rotary)
            safe_pair_dims = tl.where(rope_mask, pair_dims, dims)
            paired = tl.gather(normalized, safe_pair_dims, 0)
            rotated_half = tl.where(dims < half_rotary, -paired, paired)
            cos_value = tl.load(
                cos + batch_idx * stride_cos_b + token_start * stride_cos_s + dims * stride_cos_d,
                mask=rope_mask,
                other=1.0,
            )
            sin_value = tl.load(
                sin + batch_idx * stride_sin_b + token_start * stride_sin_s + dims * stride_sin_d,
                mask=rope_mask,
                other=0.0,
            )
            output = tl.where(rope_mask, normalized * cos_value + rotated_half * sin_value, normalized)
        output_ptr = compressed_key + batch_idx * stride_cb + block_idx * stride_cs + dims * stride_cd
        tl.store(output_ptr, output, mask=dim_mask)

    @triton.jit
    def _qsa_score_kernel(
        query,
        compressed_key,
        visible_counts,
        scores,
        stride_qb: tl.constexpr,
        stride_qs: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_ks: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vs: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_os: tl.constexpr,
        stride_op: tl.constexpr,
        query_length: tl.constexpr,
        num_blocks: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        compress_ratio: tl.constexpr,
        score_scale: tl.constexpr,
        query_start,
        rows_in_chunk: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        local_q_tile = tl.program_id(1)
        key_tile = tl.program_id(2)

        local_q = local_q_tile * BLOCK_Q + tl.arange(0, BLOCK_Q)
        query_idx = query_start + local_q
        block_idx = key_tile * BLOCK_P + tl.arange(0, BLOCK_P)
        dims = tl.arange(0, BLOCK_D)
        q_mask = (local_q < rows_in_chunk) & (query_idx < query_length)
        p_mask = block_idx < num_blocks
        d_mask = dims < head_dim
        accumulated = tl.zeros((BLOCK_Q, BLOCK_P), dtype=tl.float32)

        k_ptr = compressed_key + batch_idx * stride_kb + block_idx[:, None] * stride_ks + dims[None, :] * stride_kd
        # Every index query head shares the same compressed key.  Keep one key
        # tile resident in UB across all head-wise dot products.
        k_values = tl.load(k_ptr, mask=p_mask[:, None] & d_mask[None, :], other=0.0)

        for head_idx in tl.static_range(0, num_heads):
            q_ptr = (
                query
                + batch_idx * stride_qb
                + query_idx[:, None] * stride_qs
                + head_idx * stride_qh
                + dims[None, :] * stride_qd
            )
            # Q/K storage is BF16/FP16 and tl.dot accumulates in FP32, matching
            # the useful precision of the gold path without the prohibitive
            # Ascend compile cost of an FP32 cube specialization at D=128.
            q_values = tl.load(q_ptr, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
            head_scores = tl.dot(q_values, tl.trans(k_values))
            accumulated += tl.maximum(head_scores, 0.0)

        counts = tl.load(
            visible_counts + batch_idx * stride_vb + query_idx * stride_vs,
            mask=q_mask,
            other=0,
        )
        visible_blocks = counts // compress_ratio
        valid = q_mask[:, None] & p_mask[None, :] & (block_idx[None, :] < visible_blocks[:, None])
        accumulated *= score_scale
        accumulated = tl.where(valid, accumulated, -float("inf"))
        out_ptr = scores + batch_idx * stride_ob + local_q[:, None] * stride_os + block_idx[None, :] * stride_op
        tl.store(out_ptr, accumulated, mask=(local_q[:, None] < rows_in_chunk) & p_mask[None, :])

    @triton.jit
    def _fill_dense_and_tail_kernel(
        visible_counts,
        selected,
        stride_vb: tl.constexpr,
        stride_vs: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_os: tl.constexpr,
        stride_ow: tl.constexpr,
        query_length: tl.constexpr,
        block_budget: tl.constexpr,
        compress_ratio: tl.constexpr,
        output_width: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        query_idx = tl.program_id(1)
        offsets = tl.arange(0, BLOCK_W)
        count = tl.load(visible_counts + batch_idx * stride_vb + query_idx * stride_vs)
        complete_blocks = count // compress_ratio
        selected_blocks = tl.minimum(complete_blocks, block_budget)
        dense_count = selected_blocks * compress_ratio
        tail_count = count - complete_blocks * compress_ratio
        dense_valid = offsets < dense_count
        tail_offset = offsets - dense_count
        tail_valid = (tail_offset >= 0) & (tail_offset < tail_count)
        values = tl.where(dense_valid, offsets, complete_blocks * compress_ratio + tail_offset)
        out_ptr = selected + batch_idx * stride_ob + query_idx * stride_os + offsets * stride_ow
        tl.store(out_ptr, values, mask=(offsets < output_width) & (dense_valid | tail_valid))

    @triton.jit
    def _expand_sparse_topk_kernel(
        top_blocks,
        visible_counts,
        selected,
        stride_tb: tl.constexpr,
        stride_ts: tl.constexpr,
        stride_tk: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vs: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_os: tl.constexpr,
        stride_ow: tl.constexpr,
        query_start,
        rows_in_chunk: tl.constexpr,
        block_budget: tl.constexpr,
        compress_ratio: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        local_q = tl.program_id(1)
        query_idx = query_start + local_q
        offsets = tl.arange(0, BLOCK_T)
        block_slot = offsets // compress_ratio
        lane = offsets - block_slot * compress_ratio
        count = tl.load(visible_counts + batch_idx * stride_vb + query_idx * stride_vs)
        is_sparse = count // compress_ratio > block_budget
        block_id = tl.load(
            top_blocks + batch_idx * stride_tb + local_q * stride_ts + block_slot * stride_tk,
            mask=offsets < block_budget * compress_ratio,
            other=0,
        ).to(tl.int32)
        token_id = block_id * compress_ratio + lane
        out_ptr = selected + batch_idx * stride_ob + query_idx * stride_os + offsets * stride_ow
        tl.store(out_ptr, token_id, mask=is_sparse & (offsets < block_budget * compress_ratio))


def _apply_query_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    half = rotary_dim // 2
    rope, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    rotated_half = torch.cat((-rope[..., half:], rope[..., :half]), dim=-1)
    rotated = rope * cos.unsqueeze(2) + rotated_half * sin.unsqueeze(2)
    return torch.cat((rotated, passthrough), dim=-1)


def _visible_counts(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    return visible, visible[:, 0].sum(dim=-1).to(torch.int32)


def _is_prefix_mask(visible: torch.Tensor, counts: torch.Tensor) -> bool:
    kv_length = visible.shape[-1]
    expected = torch.arange(kv_length, device=visible.device).view(1, 1, 1, kv_length)
    expected = expected < counts[:, None, :, None]
    return bool(torch.equal(visible, expected))


@torch.no_grad()
def qsa_selected_token_mask(
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
    query_chunk_size: int | None = None,
    validate_prefix_mask: bool = True,
    return_token_ids: bool = False,
) -> torch.Tensor:
    """Select QSA tokens using fused Triton Ascend kernels on NPU.

    The accelerated route requires a causal prefix mask: every row must be
    ``True...True,False...False`` (or ``0...0,-inf...-inf`` for additive
    masks).  Set ``validate_prefix_mask=True`` while bringing up a new data
    pipeline; non-prefix masks then use the exact eager fallback.  Disable the
    check in production to avoid a device synchronization at every QSA layer.
    """
    if index_queries.ndim != 4 or raw_keys.ndim != 3:
        raise ValueError("index_queries and raw_keys must be [B,Q,H,D] and [B,KV,D]")
    batch_size, query_length, num_heads, head_dim = index_queries.shape
    kv_length = raw_keys.shape[1]
    if not batch_size or not query_length or not num_heads or not head_dim:
        raise ValueError("QSA does not support empty batch/query/head dimensions")
    if kv_length < query_length:
        raise ValueError("KV length must be greater than or equal to query length")
    if raw_keys.shape != (batch_size, kv_length, head_dim):
        raise ValueError("raw_keys must match the query batch and head dimension")
    if attention_mask.shape != (batch_size, 1, query_length, kv_length):
        raise ValueError("attention_mask must be [B,1,Q,KV]")
    if full_cos.ndim != 3 or full_sin.shape != full_cos.shape:
        raise ValueError("full_cos/full_sin must have the same [B,KV,R] shape")
    if full_cos.shape[1] != kv_length or full_cos.shape[0] not in (1, batch_size):
        raise ValueError("RoPE batch must be one or B and its sequence axis must equal KV")
    if k_norm_weight.shape != (head_dim,):
        raise ValueError("k_norm_weight must have shape [D]")
    rotary_dim = full_cos.shape[-1]
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("RoPE width must be positive, even, and no greater than the head dimension")
    input_tensors = (raw_keys, full_cos, full_sin, attention_mask, k_norm_weight)
    if any(tensor.device != index_queries.device for tensor in input_tensors):
        raise ValueError("all QSA tensors must be on the same device")
    if token_budget <= 0 or compress_ratio <= 1 or token_budget % compress_ratio:
        raise ValueError("token_budget must be positive and divisible by compress_ratio > 1")
    if query_chunk_size is not None and query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive or None for automatic selection")

    visible, counts = _visible_counts(attention_mask)
    device_type = index_queries.device.type
    supported_dtype = index_queries.dtype in (torch.float16, torch.bfloat16)
    use_triton = (
        _TRITON_AVAILABLE
        and device_type == "npu"
        and supported_dtype
        and raw_keys.dtype == index_queries.dtype
        and head_dim >= 16
        and head_dim % 16 == 0
    )
    if validate_prefix_mask and not _is_prefix_mask(visible, counts):
        use_triton = False
    if not use_triton:
        # Loud once: the fallback is exact but runs a per-query Python loop, which is a
        # severe per-step slowdown when it happens silently inside a training run.
        global _fallback_warned
        if not _fallback_warned:
            _fallback_warned = True
            if not _TRITON_AVAILABLE:
                reason = "triton is not importable"
            elif device_type != "npu":
                reason = f"device '{device_type}' is not supported by the Triton kernels"
            elif not supported_dtype or raw_keys.dtype != index_queries.dtype:
                reason = "qsa fast path requires matching fp16/bf16 index tensors"
            else:
                reason = (
                    "mask is not a plain causal prefix (validate_prefix_mask) or "
                    f"head_dim={head_dim} is not a multiple of 16"
                )
            logger.warning(
                "QSA Triton fast path disabled (%s); falling back to the exact eager reference "
                "with a per-query Python loop. Further fallbacks will not be logged.",
                reason,
            )
        return qsa_reference_mask(
            index_queries,
            raw_keys,
            full_cos,
            full_sin,
            attention_mask,
            k_norm_weight,
            rms_norm_eps=rms_norm_eps,
            token_budget=token_budget,
            compress_ratio=compress_ratio,
            return_token_ids=return_token_ids,
        )

    if full_cos.shape[0] == 1 and batch_size != 1:
        full_cos = full_cos.expand(batch_size, -1, -1)
        full_sin = full_sin.expand(batch_size, -1, -1)
    k_norm_weight = k_norm_weight.contiguous()
    num_blocks = kv_length // compress_ratio
    block_budget = token_budget // compress_ratio
    output_width = token_budget + compress_ratio - 1
    block_w = triton.next_power_of_2(output_width)
    selected = torch.full((batch_size, query_length, block_w), -1, dtype=torch.int32, device=index_queries.device)

    _fill_dense_and_tail_kernel[(batch_size, query_length)](
        counts,
        selected,
        *counts.stride(),
        *selected.stride(),
        query_length=query_length,
        block_budget=block_budget,
        compress_ratio=compress_ratio,
        output_width=output_width,
        BLOCK_W=block_w,
    )

    # For a prefix-causal row, score-ordered TopK starts only after the visible
    # complete blocks exceed the budget (position token_budget + 1).
    query_position_offset = kv_length - query_length
    sparse_start = max(0, token_budget - query_position_offset)
    if num_blocks > block_budget and sparse_start < query_length:
        block_q, block_p, block_d = 64, 128, triton.next_power_of_2(head_dim)
        # Compression and key RoPE are only consumed by sparse TopK rows.  For
        # short/prefill sequences whose visible prefix fits in the token
        # budget, skip this preprocessing entirely.
        compressed = torch.empty((batch_size, num_blocks, head_dim), dtype=raw_keys.dtype, device=raw_keys.device)
        _compress_norm_rope_kernel[(batch_size, num_blocks)](
            raw_keys,
            k_norm_weight,
            full_cos,
            full_sin,
            compressed,
            *raw_keys.stride(),
            *compressed.stride(),
            *full_cos.stride(),
            *full_sin.stride(),
            num_blocks=num_blocks,
            head_dim=head_dim,
            rotary_dim=full_cos.shape[-1],
            eps=rms_norm_eps,
            compress_ratio=compress_ratio,
            input_is_bf16=raw_keys.dtype == torch.bfloat16,
            apply_rope=False,
            BLOCK_D=block_d,
        )
        # Triton-Ascend fuses the BF16 RoPE mul/add into an FMA, while the gold
        # PyTorch path rounds each multiplication before addition.  The tiny
        # difference can reorder a TopK boundary, so keep this vectorized NPU
        # operation outside the compression kernel for exact routing parity.
        block_cos = full_cos[:, : num_blocks * compress_ratio : compress_ratio]
        block_sin = full_sin[:, : num_blocks * compress_ratio : compress_ratio]
        compressed = _apply_query_rope(compressed.unsqueeze(2), block_cos, block_sin).squeeze(2)
        padded_num_blocks = triton.cdiv(num_blocks, block_p) * block_p
        sparse_rows = query_length - sparse_start
        if query_chunk_size is None:
            # Cap launch count without allowing the FP32 score workspace to
            # grow beyond 32 MiB per invocation.  Rows stay aligned to the
            # score kernel's BLOCK_Q tile.
            workspace_rows = (32 * 2**20) // (batch_size * padded_num_blocks * 4)
            workspace_rows = max(block_q, workspace_rows // block_q * block_q)
            query_chunk_size = min(512, workspace_rows)
        workspace_rows = min(query_chunk_size, sparse_rows)
        score_workspace = torch.full(
            (batch_size, workspace_rows, padded_num_blocks),
            -torch.inf,
            dtype=torch.float32,
            device=index_queries.device,
        )
        top_values = torch.empty(
            (batch_size, workspace_rows, block_budget),
            dtype=torch.float32,
            device=index_queries.device,
        )
        top_blocks = torch.empty(
            (batch_size, workspace_rows, block_budget),
            dtype=torch.int64,
            device=index_queries.device,
        )
        for query_start in range(sparse_start, query_length, query_chunk_size):
            rows = min(query_chunk_size, query_length - query_start)
            grid = (batch_size, triton.cdiv(workspace_rows, block_q), triton.cdiv(num_blocks, block_p))
            _qsa_score_kernel[grid](
                index_queries,
                compressed,
                counts,
                score_workspace,
                *index_queries.stride(),
                *compressed.stride(),
                *counts.stride(),
                *score_workspace.stride(),
                query_length=query_length,
                num_blocks=num_blocks,
                num_heads=num_heads,
                head_dim=head_dim,
                compress_ratio=compress_ratio,
                score_scale=head_dim**-0.5,
                query_start=query_start,
                rows_in_chunk=workspace_rows,
                BLOCK_Q=block_q,
                BLOCK_P=block_p,
                BLOCK_D=block_d,
            )
            torch.topk(
                score_workspace,
                k=block_budget,
                dim=-1,
                sorted=True,
                out=(top_values, top_blocks),
            )
            block_t = triton.next_power_of_2(token_budget)
            _expand_sparse_topk_kernel[(batch_size, rows)](
                top_blocks,
                counts,
                selected,
                *top_blocks.stride(),
                *counts.stride(),
                *selected.stride(),
                query_start=query_start,
                rows_in_chunk=rows,
                block_budget=block_budget,
                compress_ratio=compress_ratio,
                BLOCK_T=block_t,
            )

    selected = selected[..., :output_width]
    if return_token_ids:
        return selected

    # Triton-Ascend 3.2 can compile the indirect tl.store used by a sparse
    # scatter, but on 910B it may silently lose non-contiguous writes.  Use the
    # optimized ACL scatter implementation for this final stage.  int64 is
    # intentional: torch.scatter does not accept the int32 routing IDs.
    bool_output = torch.zeros(
        (batch_size, 1, query_length, kv_length + 1), dtype=torch.bool, device=attention_mask.device
    )
    scatter_ids = torch.where(selected >= 0, selected, kv_length).to(torch.int64).unsqueeze(1)
    bool_output.scatter_(-1, scatter_ids, True)
    bool_output = bool_output[..., :kv_length]
    if attention_mask.dtype == torch.bool:
        return bool_output
    return torch.where(bool_output, attention_mask.new_zeros(()), torch.finfo(attention_mask.dtype).min)


@torch.no_grad()
def qsa_indexer_forward_triton(
    indexer: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    past_key_values=None,
    cu_seq_lens_q: torch.Tensor | None = None,
    qsa_packed_seq_lens: Sequence[int] | None = None,
) -> torch.Tensor:
    """Dispatch the Triton QSA indexer, splitting packed varlen by segment."""
    if cu_seq_lens_q is not None and past_key_values is not None:
        raise ValueError("Qwen4-Exp packed-varlen QSA does not support cache state.")
    query, raw_keys, full_cos, full_sin = project_qsa_indexer_inputs(
        indexer,
        hidden_states,
        position_embeddings,
        past_key_values,
    )
    if cu_seq_lens_q is None:
        return qsa_selected_token_mask(
            query,
            raw_keys,
            full_cos,
            full_sin,
            attention_mask,
            indexer.k_layernorm.weight,
            rms_norm_eps=indexer.k_layernorm.eps,
            token_budget=indexer.token_budget,
            compress_ratio=indexer.compress_ratio,
            query_chunk_size=getattr(indexer, "qsa_triton_query_chunk_size", None),
            validate_prefix_mask=getattr(indexer, "qsa_triton_validate_prefix_mask", True),
        )
    return _qsa_packed_selected_token_mask(
        query,
        raw_keys,
        full_cos,
        full_sin,
        attention_mask,
        cu_seq_lens_q,
        qsa_packed_seq_lens,
        indexer.k_layernorm.weight,
        rms_norm_eps=indexer.k_layernorm.eps,
        token_budget=indexer.token_budget,
        compress_ratio=indexer.compress_ratio,
        query_chunk_size=getattr(indexer, "qsa_triton_query_chunk_size", None),
    )


def _causal_prefix_mask(length: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    visible = torch.ones((1, 1, length, length), dtype=torch.bool, device=device).tril()
    if dtype == torch.bool:
        return visible
    if not torch.tensor(0, dtype=dtype).is_floating_point():
        raise ValueError("QSA attention mask must be bool or a floating-point dtype")
    min_value = torch.finfo(dtype).min
    return torch.where(visible, torch.zeros((), dtype=dtype, device=device), min_value)


def _selected_token_ids_to_mask(
    selected: torch.Tensor,
    kv_length: int,
    mask_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    bool_output = torch.zeros(
        (selected.shape[0], 1, selected.shape[1], kv_length + 1), dtype=torch.bool, device=device
    )
    scatter_indices = torch.where(selected >= 0, selected, kv_length).to(torch.int64).unsqueeze(1)
    bool_output.scatter_(-1, scatter_indices, True)
    bool_output = bool_output[..., :kv_length]
    if mask_dtype == torch.bool:
        return bool_output
    return torch.where(bool_output, torch.zeros((), dtype=mask_dtype, device=device), torch.finfo(mask_dtype).min)


def _qsa_packed_selected_token_mask(
    index_queries: torch.Tensor,
    raw_keys: torch.Tensor,
    full_cos: torch.Tensor,
    full_sin: torch.Tensor,
    attention_mask: torch.Tensor,
    cu_seq_lens_q: torch.Tensor | None,
    qsa_packed_seq_lens: Sequence[int] | None,
    k_norm_weight: torch.Tensor,
    *,
    rms_norm_eps: float,
    token_budget: int,
    compress_ratio: int,
    query_chunk_size: int | None,
) -> torch.Tensor:
    if attention_mask.ndim != 4:
        raise ValueError("Packed QSA requires a four-dimensional attention mask")
    batch_size, query_length, _, _ = index_queries.shape
    kv_length = raw_keys.shape[1]
    if batch_size != 1:
        raise ValueError("Packed QSA currently requires batch_size == 1")
    if query_length != kv_length:
        raise ValueError("Packed QSA requires equal query and key lengths")
    if full_cos.shape[0] != 1 or full_sin.shape[0] != 1 or full_cos.shape[1] != kv_length:
        raise ValueError("Packed QSA requires per-token RoPE tensors for the packed batch")

    if qsa_packed_seq_lens is not None:
        boundaries = [0]
        for length in qsa_packed_seq_lens:
            boundaries.append(boundaries[-1] + int(length))
    elif cu_seq_lens_q is not None:
        boundaries = cu_seq_lens_q.detach().cpu().tolist()
    else:
        raise ValueError("Packed QSA requires cu_seq_lens_q or qsa_packed_seq_lens")
    if not boundaries or boundaries[0] != 0 or boundaries[-1] != query_length:
        raise ValueError("Packed QSA cu_seq_lens_q must start at 0 and end at the packed sequence length")
    if any(end - start <= 0 for start, end in zip(boundaries[:-1], boundaries[1:])):
        raise ValueError("Packed QSA cu_seq_lens_q must contain positive sequence lengths")

    output_width = token_budget + compress_ratio - 1
    selected = torch.full((1, query_length, output_width), -1, dtype=torch.int32, device=index_queries.device)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segment_length = end - start
        segment_selected = qsa_selected_token_mask(
            index_queries[:, start:end],
            raw_keys[:, start:end],
            full_cos[:, start:end],
            full_sin[:, start:end],
            _causal_prefix_mask(segment_length, attention_mask.dtype, attention_mask.device),
            k_norm_weight,
            rms_norm_eps=rms_norm_eps,
            token_budget=token_budget,
            compress_ratio=compress_ratio,
            query_chunk_size=query_chunk_size,
            validate_prefix_mask=False,
            return_token_ids=True,
        )
        selected[:, start:end] = torch.where(segment_selected >= 0, segment_selected + start, -1)
    return _selected_token_ids_to_mask(selected, kv_length, attention_mask.dtype, attention_mask.device)


def qwen4_exp_qsa_indexer_forward_triton(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    past_key_values=None,
    cu_seq_lens_q: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compatibility wrapper for callers using the MindSpeed-MM function name."""
    return qsa_indexer_forward_triton(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values,
        cu_seq_lens_q,
    )


def install_transformers_qwen4_exp_patch() -> None:
    """Deprecated compatibility hook; VeOmni binds through OpSlot instead."""
    raise RuntimeError("Use the VeOmni qsa_indexer OpSlot; do not monkey-patch Transformers")
