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

"""DeepSeek-V4 lighting indexer Triton adapter (AMD ROCm)."""

from __future__ import annotations

from torch import Tensor


def wrapper(
    index_q: Tensor,
    index_k: Tensor,
    weights: Tensor,
    compress_ratio: int,
    topk: int,
    topk_indices: Tensor | None = None,
    cu_seqlen_ks: Tensor | None = None,
    cu_seqlen_ke: Tensor | None = None,
    packed_sequence_slices: tuple[tuple[int, int], ...] | None = None,
) -> tuple[Tensor, Tensor]:
    """Run the Triton lighting indexer for compressed-KV selection.

    Same contract as the TileLang row: ``index_q`` is ``[S, B, H, D]``,
    ``index_k`` is ``[S_kv, B, D]``, ``weights`` is ``[S, B, H]`` FP32, and
    scores and indices come back as ``[B, S, K]``. ``cu_seqlen_ks`` /
    ``cu_seqlen_ke`` already encode the packed ranges, so
    ``packed_sequence_slices`` is unused.
    """
    del packed_sequence_slices
    from ...vendor.triton_indexer import triton_lighting_indexer

    return triton_lighting_indexer(
        index_q,
        index_k,
        weights,
        compress_ratio,
        topk,
        topk_indices,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )
