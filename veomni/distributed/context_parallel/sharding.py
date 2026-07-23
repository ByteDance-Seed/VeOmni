import torch
from torch import Tensor


def _validate_parallel_coordinate(size: int, rank: int, name: str) -> None:
    if size < 1:
        raise ValueError(f"{name}_size must be positive, got {size}.")
    if not 0 <= rank < size:
        raise ValueError(f"{name}_rank must be in [0, {size}), got {rank}.")


def balanced_cp_slice(tensor: Tensor, cp_size: int, cp_rank: int, dim: int = 0) -> Tensor:
    """Select the mirrored two-chunk causal partition for one CP rank."""
    _validate_parallel_coordinate(cp_size, cp_rank, "cp")
    dim = dim % tensor.ndim
    sequence_length = tensor.size(dim)
    num_chunks = 2 * cp_size
    if sequence_length % num_chunks != 0:
        raise ValueError(
            f"Sequence length ({sequence_length}) must be divisible by 2 * cp_size ({num_chunks}) "
            "for balanced causal CP sharding."
        )

    chunks = tensor.chunk(num_chunks, dim=dim)
    return torch.cat((chunks[cp_rank], chunks[num_chunks - cp_rank - 1]), dim=dim).contiguous()


def balanced_cp_restore(tensor: Tensor, cp_size: int, dim: int = 0) -> Tensor:
    """Restore canonical sequence order from rank-major balanced CP shards."""
    if cp_size < 1:
        raise ValueError(f"cp_size must be positive, got {cp_size}.")
    dim = dim % tensor.ndim
    num_chunks = 2 * cp_size
    if tensor.size(dim) % num_chunks != 0:
        raise ValueError(
            f"Gathered sequence length ({tensor.size(dim)}) must be divisible by 2 * cp_size ({num_chunks})."
        )

    rank_major_chunks = tensor.chunk(num_chunks, dim=dim)
    canonical_chunks = [None] * num_chunks
    for cp_rank in range(cp_size):
        canonical_chunks[cp_rank] = rank_major_chunks[2 * cp_rank]
        canonical_chunks[num_chunks - cp_rank - 1] = rank_major_chunks[2 * cp_rank + 1]
    return torch.cat(canonical_chunks, dim=dim).contiguous()


def balanced_cp_to_rank_major(tensor: Tensor, cp_size: int, dim: int = 0) -> Tensor:
    """Convert canonical sequence order to rank-major balanced CP layout."""
    if cp_size < 1:
        raise ValueError(f"cp_size must be positive, got {cp_size}.")
    dim = dim % tensor.ndim
    num_chunks = 2 * cp_size
    if tensor.size(dim) % num_chunks != 0:
        raise ValueError(
            f"Sequence length ({tensor.size(dim)}) must be divisible by 2 * cp_size ({num_chunks})."
        )

    canonical_chunks = tensor.chunk(num_chunks, dim=dim)
    rank_major_chunks = []
    for cp_rank in range(cp_size):
        rank_major_chunks.append(canonical_chunks[cp_rank])
        rank_major_chunks.append(canonical_chunks[num_chunks - cp_rank - 1])
    return torch.cat(rank_major_chunks, dim=dim).contiguous()


def hybrid_cp_slice(
    tensor: Tensor,
    cp_size: int,
    cp_rank: int,
    ulysses_size: int,
    ulysses_rank: int,
    dim: int = 0,
) -> Tensor:
    """Apply balanced Ring-CP sharding followed by contiguous Ulysses sharding."""
    _validate_parallel_coordinate(ulysses_size, ulysses_rank, "ulysses")
    cp_shard = balanced_cp_slice(tensor, cp_size=cp_size, cp_rank=cp_rank, dim=dim)
    dim = dim % cp_shard.ndim
    if cp_shard.size(dim) % ulysses_size != 0:
        raise ValueError(
            f"CP-local sequence length ({cp_shard.size(dim)}) must be divisible by "
            f"ulysses_size ({ulysses_size})."
        )
    return cp_shard.chunk(ulysses_size, dim=dim)[ulysses_rank].contiguous()
