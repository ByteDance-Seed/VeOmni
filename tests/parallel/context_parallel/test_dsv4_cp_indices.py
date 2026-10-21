"""Sparse index builders must address global KV rows from a sharded query axis."""

from __future__ import annotations

import pytest
import torch

from veomni.ops.kernels.dsa.sparse_indices import CompressedCandidates, build_sparse_attention_indices


SEQ_LEN = 64
CP_SIZE = 4
LOCAL_LEN = SEQ_LEN // CP_SIZE
SLIDING_WINDOW = 8
COMPRESSED_LEN = 15


def _packed_position_ids() -> torch.Tensor:
    """Positions for samples [(0, 38), (38, 64)], resetting at the boundary."""
    return torch.cat([torch.arange(38), torch.arange(SEQ_LEN - 38)]).view(1, SEQ_LEN)


def _topk_candidates(seq_len: int, offset: int) -> CompressedCandidates:
    torch.manual_seed(0)
    full = torch.randint(-1, COMPRESSED_LEN, (1, SEQ_LEN, 4), dtype=torch.int32)
    return CompressedCandidates(topk_indices=full[:, offset : offset + seq_len])


@pytest.mark.parametrize("packed", [True, False], ids=["packed", "single"])
@pytest.mark.parametrize("with_topk", [True, False], ids=["topk", "sliding-only"])
def test_shard_indices_equal_the_matching_slice_of_the_full_build(packed: bool, with_topk: bool):
    position_ids = _packed_position_ids() if packed else torch.arange(SEQ_LEN).view(1, SEQ_LEN)

    def build(begin: int, length: int) -> torch.Tensor:
        return build_sparse_attention_indices(
            position_ids=position_ids[:, begin : begin + length],
            sliding_window=SLIDING_WINDOW,
            kv_len=SEQ_LEN,
            compressed_len=COMPRESSED_LEN,
            candidates=_topk_candidates(length, begin) if with_topk else None,
            query_offset=begin,
        )

    full = build(0, SEQ_LEN)
    for rank in range(CP_SIZE):
        begin = rank * LOCAL_LEN
        torch.testing.assert_close(build(begin, LOCAL_LEN), full[:, begin : begin + LOCAL_LEN])


def test_compressed_slots_are_lifted_by_the_full_kv_length_not_the_query_length():
    # The bug this guards: offsetting by the local query length silently points
    # compressed candidates at full-resolution rows.
    candidates = CompressedCandidates(topk_indices=torch.zeros(1, LOCAL_LEN, 1, dtype=torch.int32))
    shard = build_sparse_attention_indices(
        position_ids=_packed_position_ids()[:, :LOCAL_LEN],
        sliding_window=SLIDING_WINDOW,
        kv_len=SEQ_LEN,
        compressed_len=COMPRESSED_LEN,
        candidates=candidates,
    )
    assert torch.all(shard[..., -1:] == SEQ_LEN)
