import pytest
import torch

from veomni.distributed.context_parallel.sharding import (
    balanced_cp_restore,
    balanced_cp_slice,
    balanced_cp_to_rank_major,
    hybrid_cp_slice,
)
from veomni.distributed.context_parallel.softmax_update import merge_attention_blocks


@pytest.mark.parametrize("cp_size", [2, 4])
def test_balanced_cp_slice_round_trip(cp_size: int):
    sequence = torch.arange(64)
    local_shards = [balanced_cp_slice(sequence, cp_size=cp_size, cp_rank=rank) for rank in range(cp_size)]

    restored = balanced_cp_restore(torch.cat(local_shards), cp_size=cp_size)

    torch.testing.assert_close(restored, sequence)


@pytest.mark.parametrize("cp_size", [2, 4])
def test_balanced_cp_to_rank_major_round_trip(cp_size: int):
    sequence = torch.arange(64)
    rank_major = balanced_cp_to_rank_major(sequence, cp_size=cp_size)
    restored = balanced_cp_restore(rank_major, cp_size=cp_size)
    torch.testing.assert_close(restored, sequence)
    torch.testing.assert_close(
        rank_major,
        torch.cat([balanced_cp_slice(sequence, cp_size=cp_size, cp_rank=rank) for rank in range(cp_size)]),
    )


def test_hybrid_cp_slice_matches_ring_then_ulysses_partition():
    sequence = torch.arange(64)
    cp_size = 2
    ulysses_size = 4

    local_shards = [
        hybrid_cp_slice(
            sequence,
            cp_size=cp_size,
            cp_rank=cp_rank,
            ulysses_size=ulysses_size,
            ulysses_rank=ulysses_rank,
        )
        for cp_rank in range(cp_size)
        for ulysses_rank in range(ulysses_size)
    ]

    assert all(shard.numel() == 8 for shard in local_shards)
    torch.testing.assert_close(local_shards[0], torch.arange(0, 8))
    torch.testing.assert_close(local_shards[3], torch.arange(56, 64))
    torch.testing.assert_close(local_shards[4], torch.arange(16, 24))
    torch.testing.assert_close(local_shards[7], torch.arange(40, 48))


def test_balanced_cp_slice_rejects_non_divisible_sequence():
    with pytest.raises(ValueError, match="2 \\* cp_size"):
        balanced_cp_slice(torch.arange(63), cp_size=2, cp_rank=0)


def _block_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    scores = query @ key.transpose(-1, -2)
    block_max = scores.max(dim=-1, keepdim=True).values
    exp_scores = torch.exp(scores - block_max)
    block_sum = exp_scores.sum(dim=-1, keepdim=True)
    output = (exp_scores @ value) / block_sum
    stats_shape = (*block_max.shape[:-1], 8)
    return output, block_max.expand(stats_shape), block_sum.expand(stats_shape)


def test_merge_attention_blocks_matches_full_softmax_and_gradients():
    torch.manual_seed(0)
    query = torch.randn(2, 3, 5, 4, dtype=torch.float64, requires_grad=True)
    key_a = torch.randn(2, 3, 7, 4, dtype=torch.float64, requires_grad=True)
    value_a = torch.randn(2, 3, 7, 6, dtype=torch.float64, requires_grad=True)
    key_b = torch.randn(2, 3, 11, 4, dtype=torch.float64, requires_grad=True)
    value_b = torch.randn(2, 3, 11, 6, dtype=torch.float64, requires_grad=True)

    out_a, max_a, sum_a = _block_attention(query, key_a, value_a)
    out_b, max_b, sum_b = _block_attention(query, key_b, value_b)
    merged, _, _ = merge_attention_blocks(out_a, max_a, sum_a, out_b, max_b, sum_b)

    full_scores = query @ torch.cat((key_a, key_b), dim=-2).transpose(-1, -2)
    full_output = torch.softmax(full_scores, dim=-1) @ torch.cat((value_a, value_b), dim=-2)
    torch.testing.assert_close(merged, full_output, atol=1e-10, rtol=1e-10)

    merged.sum().backward()
    merged_grads = [tensor.grad.detach().clone() for tensor in (query, key_a, value_a, key_b, value_b)]

    for tensor in (query, key_a, value_a, key_b, value_b):
        tensor.grad = None
    full_output.sum().backward()
    full_grads = [tensor.grad for tensor in (query, key_a, value_a, key_b, value_b)]

    for actual, expected in zip(merged_grads, full_grads):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_merge_attention_blocks_handles_empty_previous_statistics():
    current_output = torch.randn(1, 2, 3, 4)
    current_max = torch.randn(1, 2, 3, 8)
    current_sum = torch.rand(1, 2, 3, 8) + 0.1
    previous_output = torch.zeros_like(current_output)
    previous_max = torch.full_like(current_max, -torch.inf)
    previous_sum = torch.zeros_like(current_sum)

    merged, merged_max, merged_sum = merge_attention_blocks(
        previous_output,
        previous_max,
        previous_sum,
        current_output,
        current_max,
        current_sum,
    )

    torch.testing.assert_close(merged, current_output)
    torch.testing.assert_close(merged_max, current_max)
    torch.testing.assert_close(merged_sum, current_sum)
