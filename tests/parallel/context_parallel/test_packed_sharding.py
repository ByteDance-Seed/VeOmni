import pytest
import torch

from veomni.distributed.context_parallel.attention_backend import torch_packed_causal_attention
from veomni.distributed.context_parallel.packed_sharding import (
    apply_packed_cp_partition,
    build_packed_cp_partition,
)
from veomni.distributed.context_parallel.ring_attention import (
    dense_causal_attention,
    simulate_packed_ring_causal_attention,
)


def test_packed_cp_partition_keeps_sample_boundaries_cp2():
    # Two samples of length 16 each (divisible by 2*cp=4).
    cu = torch.tensor([0, 16, 32], dtype=torch.int32)
    seq = torch.arange(32)
    p0 = build_packed_cp_partition(cu, cp_size=2, cp_rank=0)
    p1 = build_packed_cp_partition(cu, cp_size=2, cp_rank=1)

    torch.testing.assert_close(p0.local_cu_seqlens, torch.tensor([0, 8, 16], dtype=torch.int32))
    torch.testing.assert_close(p1.local_cu_seqlens, torch.tensor([0, 8, 16], dtype=torch.int32))

    # Rank0 per sample: early chunk0 + late chunk3 inside each sample.
    expected0 = torch.cat(
        (
            torch.arange(0, 4),
            torch.arange(12, 16),
            torch.arange(16, 20),
            torch.arange(28, 32),
        )
    )
    torch.testing.assert_close(p0.token_indices, expected0)
    torch.testing.assert_close(apply_packed_cp_partition(seq, p0, dim=0), expected0)


def test_packed_partition_rejects_non_divisible_sample():
    cu = torch.tensor([0, 15, 32], dtype=torch.int32)
    with pytest.raises(ValueError, match="divisible"):
        build_packed_cp_partition(cu, cp_size=2, cp_rank=0)


def test_hybrid_packed_partition_u4_cp2():
    cu = torch.tensor([0, 64], dtype=torch.int32)  # one sample, multiple=16
    parts = [
        build_packed_cp_partition(cu, cp_size=2, cp_rank=cp, ulysses_size=4, ulysses_rank=u)
        for cp in range(2)
        for u in range(4)
    ]
    assert all(p.token_indices.numel() == 8 for p in parts)
    # All indices unique and cover full sequence.
    covered = torch.cat([p.token_indices for p in parts]).sort().values
    torch.testing.assert_close(covered, torch.arange(64))


def test_simulate_packed_ring_matches_packed_dense_and_blocks_cross_sample():
    torch.manual_seed(0)
    cp_size = 2
    batch, hq, hkv, head_dim = 1, 4, 2, 8
    # Two samples length 32 each.
    cu = torch.tensor([0, 32, 64], dtype=torch.int32)
    query = torch.randn(batch, hq, 64, head_dim, dtype=torch.float32)
    key = torch.randn(batch, hkv, 64, head_dim, dtype=torch.float32)
    value = torch.randn(batch, hkv, 64, head_dim, dtype=torch.float32)
    # Amplify sample-1 keys so cross-sample leakage would show up in sample-0 outputs.
    key[:, :, 32:] = key[:, :, 32:] + 50.0

    packed_dense = torch_packed_causal_attention(query, key, value, cu, softmax_scale=head_dim**-0.5)
    ring = simulate_packed_ring_causal_attention(
        query, key, value, cu, cp_size=cp_size, softmax_scale=head_dim**-0.5
    )
    torch.testing.assert_close(ring, packed_dense, atol=1e-4, rtol=1e-4)

    # Sample-0 ring output must match dense attention on sample-0 alone.
    s0 = dense_causal_attention(
        query[:, :, :32], key[:, :, :32], value[:, :, :32], softmax_scale=head_dim**-0.5
    )
    torch.testing.assert_close(ring[:, :, :32], s0, atol=1e-4, rtol=1e-4)


def test_ulysses_rank_major_reorder_roundtrip():
    from veomni.distributed.context_parallel.packed_sharding import (
        reorder_sample_major_to_ulysses_rank_major,
        reorder_ulysses_rank_major_to_sample_major,
        ulysses_local_cu_to_cp_local_cu,
    )

    # Two samples, ulysses-local lengths 4 and 4; U=2 → gathered seq 16.
    local_cu = torch.tensor([0, 4, 8], dtype=torch.int32)
    # Simulate post-gather rank-major: [u0_s0|u0_s1|u1_s0|u1_s1]
    u0 = torch.arange(0, 8)
    u1 = torch.arange(100, 108)
    rank_major = torch.cat([u0, u1]).view(1, 16, 1, 1)
    sample_major = reorder_ulysses_rank_major_to_sample_major(
        rank_major, local_cu, ulysses_size=2, seq_dim=1
    )
    # Expected: [s0_u0|s0_u1|s1_u0|s1_u1] = [0..3|100..103|4..7|104..107]
    expected = torch.tensor([0, 1, 2, 3, 100, 101, 102, 103, 4, 5, 6, 7, 104, 105, 106, 107])
    torch.testing.assert_close(sample_major.view(-1), expected)
    back = reorder_sample_major_to_ulysses_rank_major(
        sample_major, local_cu, ulysses_size=2, seq_dim=1
    )
    torch.testing.assert_close(back, rank_major)
    cp_cu = ulysses_local_cu_to_cp_local_cu(local_cu, 2)
    torch.testing.assert_close(cp_cu, torch.tensor([0, 8, 16], dtype=torch.int32))
