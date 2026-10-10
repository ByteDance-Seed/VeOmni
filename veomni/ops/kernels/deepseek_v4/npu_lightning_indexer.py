from functools import lru_cache

import torch


def _ops():
    try:
        from cann_ops_transformer import ops
    except ImportError as exc:
        raise ImportError("cann_ops_transformer is required for the NPU Lightning Indexer.") from exc
    return ops


def _causal_compression_residual(original_key_len: int, compress_rate: int) -> int:
    """Return the pre-compression key-length remainder required by CANN."""
    if compress_rate == 0:
        compress_rate = 1
    return original_key_len % compress_rate


def _packed_indexer_metadata(sequence_slices, compress_rate, device):
    lengths = torch.tensor([end - start for start, end in sequence_slices], dtype=torch.int32, device=device)
    compressed_lengths = lengths // compress_rate
    cu_seqlens_q = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
    cu_seqlens_k = torch.cat((compressed_lengths.new_zeros(1), compressed_lengths.cumsum(0, dtype=torch.int32)))
    return cu_seqlens_q, cu_seqlens_k, lengths % compress_rate


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
    """Build metadata for the equal-length BSND path."""
    cmp_residual_k = torch.full((batch_size,), residual, dtype=torch.int32, device=device)
    return _ops().lightning_indexer_metadata(
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
    q,
    compressed_kv,
    weights,
    top_k,
    *,
    compress_rate,
    mask_mode=3,
    packed_sequence_slices=None,
):
    """Select compressed KV entries with BSND or packed TND CANN dispatch."""
    batch, seq_len, num_heads, head_dim = q.shape
    if compress_rate == 0:
        compress_rate = 1

    if packed_sequence_slices is not None and len(packed_sequence_slices) > 1:
        if batch != 1:
            raise ValueError(f"Packed TND Lightning Indexer expects batch size 1, got {batch}")
        cu_seqlens_q, cu_seqlens_k, cmp_residual_k = _packed_indexer_metadata(
            packed_sequence_slices, compress_rate, q.device
        )
        if int(cu_seqlens_q[-1].item()) != seq_len:
            raise ValueError("Packed sequence slices must span the complete query sequence")
        if int(cu_seqlens_k[-1].item()) != compressed_kv.shape[1]:
            raise ValueError("Packed compressed boundaries do not match the compressed KV length")
        compressed_lengths = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        top_k = min(top_k, int(compressed_lengths.max().item()))
        max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
        max_seqlen_k = int(compressed_lengths.max().item())
        q_in = q[0].contiguous().to(torch.bfloat16)
        k_in = compressed_kv[0].unsqueeze(1).contiguous().to(torch.bfloat16)
        # CANN's TND Lightning Indexer contract requires scorer weights in FP32.
        weights_in = weights[0].contiguous().float()
        metadata = _ops().lightning_indexer_metadata(
            num_heads,
            1,
            head_dim,
            top_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cmp_residual_k=cmp_residual_k,
            batch_size=len(packed_sequence_slices),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            layout_q="TND",
            layout_k="TND",
            mask_mode=mask_mode,
            cmp_ratio=compress_rate,
        )
        sparse_indices, sparse_values = _ops().lightning_indexer(
            q_in,
            k_in,
            weights_in,
            top_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cmp_residual_k=cmp_residual_k,
            metadata=metadata,
            layout_q="TND",
            layout_k="TND",
            mask_mode=mask_mode,
            cmp_ratio=compress_rate,
            return_value=1,
        )
        return sparse_indices.squeeze(1).unsqueeze(0), sparse_values.squeeze(1).unsqueeze(0)

    compressed_len = compressed_kv.shape[1]
    top_k = min(top_k, compressed_len)
    residual = _causal_compression_residual(seq_len, compress_rate)
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
        compressed_len,
        q.device,
        mask_mode=mask_mode,
        cmp_ratio=compress_rate,
    )
    sparse_indices, sparse_values = _ops().lightning_indexer(
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
    return (
        sparse_indices.view(batch, seq_len, top_k),
        sparse_values.view(batch, seq_len, top_k),
    )
