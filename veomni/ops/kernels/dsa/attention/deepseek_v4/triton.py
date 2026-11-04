# Copyright 2026 Advanced Micro Devices, Inc. and ByteDance Ltd. and/or its affiliates
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

"""DeepSeek-V4 sparse MQA Triton adapter for AMD ROCm GPUs."""

from __future__ import annotations

import torch
from torch import Tensor

from ...sparse_indices import CompressedCandidates, build_sparse_attention_indices


def wrapper(
    q: Tensor,
    ori_kv: Tensor,
    cmp_kv: Tensor | None,
    attn_sink: Tensor,
    position_ids: Tensor,
    *,
    sliding_window: int,
    compress_ratio: int,
    candidates: CompressedCandidates | None = None,
    packed_sequence_slices: tuple[tuple[int, int], ...] | None = None,
    query_offset: int = 0,
    attention_mask: Tensor | None = None,
    sm_scale: float | None = None,
    return_lse: bool = False,
    dropout: float = 0.0,
    return_attn_weights: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Run Triton sparse MQA with the TileLang row's candidate contract.

    Requested LSE is detached and returned in base-2 units.
    """
    del compress_ratio, packed_sequence_slices
    if dropout:
        raise ValueError("triton DeepSeek-V4 sparse attention requires dropout=0.")
    if return_attn_weights:
        raise ValueError(
            "triton DeepSeek-V4 sparse attention does not support output_attentions=True; "
            "use the eager implementation."
        )

    from ...vendor.triton_sparse_mla import sparse_attn_triton

    kv = ori_kv if cmp_kv is None else torch.cat((ori_kv, cmp_kv), dim=1)
    topk_idxs = build_sparse_attention_indices(
        position_ids=position_ids,
        sliding_window=sliding_window,
        kv_len=ori_kv.shape[1],
        compressed_len=0 if cmp_kv is None else cmp_kv.shape[1],
        candidates=candidates,
        query_offset=query_offset,
        attention_mask=attention_mask,
    )
    return sparse_attn_triton(
        q.contiguous(),
        kv.contiguous(),
        attn_sink.to(dtype=torch.float32),
        topk_idxs,
        sm_scale,
        return_lse,
    )
