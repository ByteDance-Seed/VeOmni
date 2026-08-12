import pytest
import torch

from veomni.distributed.context_parallel.packed_sharding import (
    apply_packed_context_parallel_partition,
    build_packed_context_parallel_partition,
    pad_packed_samples,
    reorder_sample_major_to_ulysses_rank_major,
    reorder_ulysses_rank_major_to_sample_major,
    ulysses_local_head_count,
)


def test_per_sample_padding_and_cp2_physical_layout():
    values = torch.arange(65)
    padded, cu = pad_packed_samples(values, torch.tensor([0, 1, 65], dtype=torch.int32), multiple=4)

    assert cu.tolist() == [0, 4, 68]
    assert padded.tolist()[:4] == [0, 0, 0, 0]
    assert padded.tolist()[4:] == list(range(1, 65))

    rank0 = build_packed_context_parallel_partition(cu, cp_size=2, cp_rank=0)
    rank1 = build_packed_context_parallel_partition(cu, cp_size=2, cp_rank=1)
    assert rank0.local_cu_seqlens.tolist() == [0, 2, 34]
    assert rank1.local_cu_seqlens.tolist() == [0, 2, 34]
    assert apply_packed_context_parallel_partition(padded, rank0).numel() == 34
    assert apply_packed_context_parallel_partition(padded, rank1).numel() == 34


def test_hybrid_cp4_u2_partitions_cover_each_padded_sample_once():
    cu = torch.tensor([0, 64, 192], dtype=torch.int32)
    all_indices = []
    for cp_rank in range(4):
        for ulysses_rank in range(2):
            partition = build_packed_context_parallel_partition(
                cu,
                cp_size=4,
                cp_rank=cp_rank,
                ulysses_size=2,
                ulysses_rank=ulysses_rank,
            )
            all_indices.extend(partition.token_indices.tolist())

    assert sorted(all_indices) == list(range(192))


def test_ulysses_rank_major_reorder_is_exactly_reversible():
    local_cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    rank_major = torch.arange(20).reshape(1, 10, 2)
    sample_major = reorder_ulysses_rank_major_to_sample_major(
        rank_major,
        local_cu,
        ulysses_size=2,
        sequence_dim=1,
    )
    restored = reorder_sample_major_to_ulysses_rank_major(
        sample_major,
        local_cu,
        ulysses_size=2,
        sequence_dim=1,
    )

    torch.testing.assert_close(restored, rank_major, rtol=0, atol=0)


def test_hybrid_cp_does_not_divide_gdn_heads_by_cp_size():
    # U4×CP8 is the production geometry: CP shards tokens/state, while only
    # Ulysses shards the eight GDN value heads.
    assert ulysses_local_head_count(8, ulysses_size=4) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cp_size": 2, "cp_rank": 0, "ulysses_size": 2, "ulysses_rank": 0},
        {"cp_size": 4, "cp_rank": 3, "ulysses_size": 1, "ulysses_rank": 0},
    ],
)
def test_partition_rejects_unaligned_sample(kwargs):
    with pytest.raises(ValueError, match="must be divisible"):
        build_packed_context_parallel_partition(torch.tensor([0, 65], dtype=torch.int32), **kwargs)
