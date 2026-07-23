import torch

from veomni.distributed.context_parallel.sharding import hybrid_cp_slice
from veomni.distributed.context_parallel.topology import (
    coords_from_sp_rank,
    cp_global_ranks_for_ulysses_rank,
    sp_rank_from_coords,
    ulysses_global_ranks_for_cp_rank,
)


def test_sp_rank_coords_round_trip_u4_cp2():
    cp_size, ulysses_size = 2, 4
    for sp_rank in range(cp_size * ulysses_size):
        cp_rank, ulysses_rank = coords_from_sp_rank(sp_rank, cp_size, ulysses_size)
        assert sp_rank_from_coords(cp_rank, ulysses_rank, ulysses_size) == sp_rank


def test_group_ranks_match_init_sequence_parallel_layout():
    cp_size, ulysses_size = 2, 4
    # DP0 Ulysses groups for each CP slot.
    assert ulysses_global_ranks_for_cp_rank(
        dp_index=0, cp_rank=0, cp_size=cp_size, ulysses_size=ulysses_size
    ) == [0, 1, 2, 3]
    assert ulysses_global_ranks_for_cp_rank(
        dp_index=0, cp_rank=1, cp_size=cp_size, ulysses_size=ulysses_size
    ) == [4, 5, 6, 7]
    # DP0 CP rings for each Ulysses slot.
    assert cp_global_ranks_for_ulysses_rank(
        dp_index=0, ulysses_rank=0, cp_size=cp_size, ulysses_size=ulysses_size
    ) == [0, 4]
    assert cp_global_ranks_for_ulysses_rank(
        dp_index=0, ulysses_rank=3, cp_size=cp_size, ulysses_size=ulysses_size
    ) == [3, 7]


def test_hybrid_slice_for_all_u4_cp2_ranks():
    sequence = torch.arange(64)
    cp_size, ulysses_size = 2, 4
    shards = []
    for sp_rank in range(cp_size * ulysses_size):
        cp_rank, ulysses_rank = coords_from_sp_rank(sp_rank, cp_size, ulysses_size)
        shards.append(
            hybrid_cp_slice(
                sequence,
                cp_size=cp_size,
                cp_rank=cp_rank,
                ulysses_size=ulysses_size,
                ulysses_rank=ulysses_rank,
            )
        )
    assert all(shard.numel() == 8 for shard in shards)
    # Contiguous SP would give [24:32] for rank3; hybrid CP0/U3 is the late half of rank0.
    torch.testing.assert_close(shards[3], torch.arange(56, 64))
    torch.testing.assert_close(shards[4], torch.arange(16, 24))
