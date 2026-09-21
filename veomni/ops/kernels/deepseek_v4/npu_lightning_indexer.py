from functools import lru_cache

import torch
from cann_ops_transformer import ops


@lru_cache(maxsize=8)
def get_npu_lightning_indexer_metadata(
    num_heads,
    head_dim,
    top_k,
    residual,
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    device,
    *,
    mask_mode=3,
    cmp_ratio=1,
):
    """Build the metadata tensor required by the sparse + cmp + mask_mode=3 path."""
    cmp_residual_k = torch.full((batch_size,), residual, dtype=torch.int32, device=device)
    return ops.lightning_indexer_metadata(
        num_heads,
        1,
        head_dim,
        top_k,
        cmp_residual_k=cmp_residual_k,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
    )


def npu_lightning_indexer(
    q,  # [B, S, N, D]  (already RoPE'd, no rotate_activation)
    compressed_kv,  # [B, T, D]     indexer compressed key
    weights,  # [B, S, N]     fp32, both scales already folded in
    top_k,
    *,
    compress_rate,
    mask_mode=3,
):
    """BSND lightning indexer: select top-k compressed KV entries per query.

    q / compressed_kv are cast to bf16 and made contiguous for the op; weights
    stays fp32. The op has no internal scale, so `weights` must already carry
    weights_scaling * softmax_scale.

    Returns:
        sparse_indices [B, S, top_k] int, sparse_values [B, S, top_k].
    """
    batch, seq_len = q.shape[0], q.shape[1]
    num_heads, head_dim = q.shape[2], q.shape[3]

    S_K = compressed_kv.shape[1]
    top_k = min(top_k, S_K)
    if compress_rate == 0:
        compress_rate = 1
    residual = S_K % compress_rate
    cmp_residual_k = torch.full((batch,), residual, dtype=torch.int32, device=q.device)

    q_in = q.contiguous().to(torch.bfloat16)
    k_in = compressed_kv.unsqueeze(2).contiguous().to(torch.bfloat16)

    metadata = get_npu_lightning_indexer_metadata(
        num_heads,
        head_dim,
        top_k,
        residual,
        batch,
        seq_len,
        S_K,
        q.device,
        mask_mode=mask_mode,
        cmp_ratio=compress_rate,
    )

    sparse_indices, sparse_values = ops.lightning_indexer(
        q_in,
        k_in,
        weights,
        top_k,
        cmp_residual_k=cmp_residual_k,
        metadata=metadata,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=mask_mode,
        cmp_ratio=compress_rate,
        return_value=1,
    )
    sparse_indices = sparse_indices.view(batch, seq_len, top_k)
    sparse_values = sparse_values.view(batch, seq_len, top_k)
    return sparse_indices, sparse_values
