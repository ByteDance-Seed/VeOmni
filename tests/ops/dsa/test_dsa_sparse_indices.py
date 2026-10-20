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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Expanding the structured ``dsa_attention`` / ``deepseek_v4`` contract into slots."""

from __future__ import annotations

import torch

from veomni.ops.kernels.dsa.sparse_indices import CompressedCandidates, build_sparse_attention_indices


def test_packed_rows_reproduce_each_sample_built_alone():
    # ``position_ids`` restarting at a boundary is the whole packed-isolation
    # mechanism: the second sample's window must stop at its own first token.
    position_ids = torch.cat([torch.arange(5), torch.arange(4)]).view(1, 9)
    packed = build_sparse_attention_indices(
        position_ids=position_ids, sliding_window=3, kv_len=9, compressed_len=0, candidates=None
    )
    alone = build_sparse_attention_indices(
        position_ids=torch.arange(4).view(1, 4), sliding_window=3, kv_len=4, compressed_len=0, candidates=None
    )
    torch.testing.assert_close(packed[:, 5:], torch.where(alone >= 0, alone + 5, -1))


def test_attention_mask_narrows_full_resolution_slots_only():
    position_ids = torch.arange(4).view(1, 4)
    blocks_key_0 = torch.zeros(1, 1, 4, 4)
    blocks_key_0[..., 0] = float("-inf")
    indices = build_sparse_attention_indices(
        position_ids=position_ids,
        sliding_window=2,
        kv_len=4,
        compressed_len=1,
        candidates=CompressedCandidates(
            range_starts=torch.zeros(4, dtype=torch.int32),
            range_ends=torch.ones(4, dtype=torch.int32),
        ),
        attention_mask=blocks_key_0,
    )
    # A compressed row has no column in a per-token mask, so it stays visible.
    assert indices.tolist() == [[[-1, -1, 4], [-1, 1, 4], [1, 2, 4], [2, 3, 4]]]


def test_shared_position_row_broadcasts_to_the_batch():
    indices = build_sparse_attention_indices(
        position_ids=torch.arange(3).view(1, 3),
        sliding_window=2,
        kv_len=3,
        compressed_len=2,
        candidates=CompressedCandidates(topk_indices=torch.tensor([[[-1]] * 3, [[0]] * 3], dtype=torch.int32)),
    )
    assert indices.shape == (2, 3, 3)
    assert indices[:, :, -1].tolist() == [[-1, -1, -1], [3, 3, 3]]
