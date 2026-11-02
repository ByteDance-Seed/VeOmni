"""Sparse index builders must address global KV rows from a sharded query axis."""

from __future__ import annotations

import torch

from veomni.models.transformers.deepseek_v4.packed_utils import (
    CompressedCandidates,
    build_packed_sparse_attention_indices,
    build_sparse_attention_indices,
)


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


def test_packed_shard_indices_equal_the_matching_slice_of_the_full_build():
    position_ids = _packed_position_ids()
    full = build_packed_sparse_attention_indices(
        position_ids=position_ids,
        sliding_window=SLIDING_WINDOW,
        compressed_len=COMPRESSED_LEN,
        candidates=_topk_candidates(SEQ_LEN, 0),
    )
    for rank in range(CP_SIZE):
        begin = rank * LOCAL_LEN
        shard = build_packed_sparse_attention_indices(
            position_ids=position_ids[:, begin : begin + LOCAL_LEN],
            sliding_window=SLIDING_WINDOW,
            compressed_len=COMPRESSED_LEN,
            candidates=_topk_candidates(LOCAL_LEN, begin),
            query_offset=begin,
            kv_full_len=SEQ_LEN,
        )
        torch.testing.assert_close(shard, full[:, begin : begin + LOCAL_LEN])


def test_packed_defaults_reproduce_the_unsharded_build():
    position_ids = _packed_position_ids()
    kwargs = dict(
        position_ids=position_ids,
        sliding_window=SLIDING_WINDOW,
        compressed_len=COMPRESSED_LEN,
        candidates=_topk_candidates(SEQ_LEN, 0),
    )
    explicit = build_packed_sparse_attention_indices(**kwargs, query_offset=0, kv_full_len=SEQ_LEN)
    torch.testing.assert_close(build_packed_sparse_attention_indices(**kwargs), explicit)


def test_compact_shard_indices_equal_the_matching_slice_of_the_full_build():
    full = build_sparse_attention_indices(
        batch_size=1,
        seq_len=SEQ_LEN,
        sliding_window=SLIDING_WINDOW,
        compressed_len=COMPRESSED_LEN,
        compressed_indices=None,
        device=torch.device("cpu"),
    )
    for rank in range(CP_SIZE):
        begin = rank * LOCAL_LEN
        shard = build_sparse_attention_indices(
            batch_size=1,
            seq_len=LOCAL_LEN,
            sliding_window=SLIDING_WINDOW,
            compressed_len=COMPRESSED_LEN,
            compressed_indices=None,
            device=torch.device("cpu"),
            query_offset=begin,
            kv_full_len=SEQ_LEN,
        )
        torch.testing.assert_close(shard, full[:, begin : begin + LOCAL_LEN])


def test_compressed_slots_are_lifted_by_the_full_kv_length_not_the_query_length():
    # The bug this guards: offsetting by the local query length silently points
    # compressed candidates at full-resolution rows.
    candidates = CompressedCandidates(topk_indices=torch.zeros(1, LOCAL_LEN, 1, dtype=torch.int32))
    shard = build_packed_sparse_attention_indices(
        position_ids=_packed_position_ids()[:, :LOCAL_LEN],
        sliding_window=SLIDING_WINDOW,
        compressed_len=COMPRESSED_LEN,
        candidates=candidates,
        query_offset=0,
        kv_full_len=SEQ_LEN,
    )
    assert torch.all(shard[..., -1:] == SEQ_LEN)
