# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing limitations
# under the License.

"""Packed DeepSeek-V4 index builders and compression-metadata sharding."""

from __future__ import annotations

import torch

from veomni.models_kernel.transformers.deepseek_v4.packed_utils import (
    CompressedCandidates,
    build_packed_compression_metadata,
    build_packed_sparse_attention_indices,
    build_sparse_attention_indices,
    shard_packed_compression_metadata,
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
    kwargs = {
        "position_ids": position_ids,
        "sliding_window": SLIDING_WINDOW,
        "compressed_len": COMPRESSED_LEN,
        "candidates": _topk_candidates(SEQ_LEN, 0),
    }
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


def test_shard_packed_compression_metadata_keeps_global_compressed_slots():
    reference = torch.zeros(1, SEQ_LEN, 4)
    position_ids = _packed_position_ids()
    slices = ((0, 38), (38, 64))
    metadata = build_packed_compression_metadata(
        reference,
        position_ids,
        slices,
        compress_rates=(4,),
        block_bias_rates=(4,),
    )[4]
    rank = 1
    window_starts = metadata["window_starts"]
    owned = (window_starts >= rank * LOCAL_LEN) & (window_starts < (rank + 1) * LOCAL_LEN)
    begin = int(owned.nonzero()[0])
    end = int(owned.nonzero()[-1]) + 1
    local = shard_packed_compression_metadata(
        metadata,
        window_begin=begin,
        window_end=end,
        local_seq_len=LOCAL_LEN,
        cp_rank=rank,
        halo=4,
    )
    assert local["window_starts"].shape[0] == int(owned.sum())
    assert local["range_starts"].shape[0] == LOCAL_LEN
    assert local["range_ends"].shape[0] == LOCAL_LEN
    torch.testing.assert_close(
        local["range_starts"],
        metadata["range_starts"][rank * LOCAL_LEN : (rank + 1) * LOCAL_LEN],
    )
    assert local["block_bias"].shape[-2] == LOCAL_LEN
    assert local["block_bias"].shape[-1] == metadata["block_bias"].shape[-1]
