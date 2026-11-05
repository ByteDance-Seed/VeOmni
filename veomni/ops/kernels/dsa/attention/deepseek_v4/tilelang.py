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

"""DeepSeek-V4 sparse MQA TileLang adapter (SM90+)."""

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
    """Run TileLang sparse MQA over the candidate slots the contract implies.

    Each valid candidate slot participates independently, including repeated
    indices. When requested, LSE is detached and returned in base-2 units. The
    candidate width is kept at the full window plus selection rather than
    compacted per call, because the kernel specializes on that last dimension
    and a varying width would recompile on every step.
    """
    del compress_ratio, packed_sequence_slices
    if dropout:
        raise ValueError("tilelang DeepSeek-V4 sparse attention requires dropout=0.")
    if return_attn_weights:
        raise ValueError(
            "tilelang DeepSeek-V4 sparse attention does not support output_attentions=True; "
            "use the eager implementation."
        )

    from ...vendor.tilelang_sparse_mla import sparse_attn_tilelang

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

    # MixedPrecision / ``param_dtype=bf16`` stores the learnable sink as bf16.
    # The kernel accumulates in fp32, and upcasting bf16 is lossless.
    return sparse_attn_tilelang(
        q.contiguous(), kv.contiguous(), attn_sink.to(dtype=torch.float32), topk_idxs, sm_scale, return_lse
    )
