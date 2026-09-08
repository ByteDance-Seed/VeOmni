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

"""GLM-DSA cuDNN indexer adapter."""

from __future__ import annotations

from torch import Tensor


def wrapper(
    q: Tensor,
    k: Tensor,
    w: Tensor,
    top_k: int,
    *,
    ratio: int = 1,
    qhead_per_kv_head: int | None = None,
    sm_scale: float = 1.0,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """cuDNN FE indexer top-k. Same face as the eager row.

    ``q`` is ``[B, S, H, D]``, ``k`` is ``[B, T, D]`` or ``[B, T, 1, D]``,
    ``w`` is ``[B, S, H]``. Returns ``[B, S, top_k]`` long indices.
    ``attention_mask`` is accepted for call-face parity with eager. cuDNN
    applies causality through ``ratio``.
    """
    del attention_mask
    from ....vendor.flashmla_cudnn import indexer_select_topk

    return indexer_select_topk(
        q,
        k,
        w,
        top_k,
        ratio=ratio,
        qhead_per_kv_head=qhead_per_kv_head,
        sm_scale=sm_scale,
    )
