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

"""DeepSeek-V4 sparse MQA TileLang adapter (SM90+)."""

from __future__ import annotations

import torch
from torch import Tensor


def wrapper(
    q: Tensor,
    kv: Tensor,
    attn_sink: Tensor,
    topk_idxs: Tensor,
    sm_scale: float | None = None,
    return_lse: bool = False,
    dropout: float = 0.0,
    return_attn_weights: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Run TileLang sparse MQA over selected KV candidate slots.

    ``q`` is ``[B, S, H, D]``, ``kv`` is ``[B, S_kv, D]``, ``attn_sink`` is
    ``[H]``, and ``topk_idxs`` is ``[B, S, K]``. ``K`` is the selected
    candidate width and need not equal the indexer's requested ``topk``. Each
    valid candidate slot participates independently, including repeated
    indices. When requested, LSE is detached and returned in base-2 units.
    """
    if dropout:
        raise ValueError("tilelang DeepSeek-V4 sparse attention requires dropout=0.")
    if return_attn_weights:
        raise ValueError(
            "tilelang DeepSeek-V4 sparse attention does not support output_attentions=True; "
            "use the eager implementation."
        )

    from ...vendor.tilelang_sparse_mla import sparse_attn_tilelang

    # MixedPrecision / ``param_dtype=bf16`` stores the learnable sink as bf16.
    # The kernel accumulates in fp32, and upcasting bf16 is lossless.
    return sparse_attn_tilelang(q, kv, attn_sink.to(dtype=torch.float32), topk_idxs, sm_scale, return_lse)
