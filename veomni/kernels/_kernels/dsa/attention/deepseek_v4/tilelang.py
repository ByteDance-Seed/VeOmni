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

from torch import Tensor


def wrapper(
    q: Tensor,
    kv: Tensor,
    attn_sink: Tensor,
    topk_idxs: Tensor,
    sm_scale: float | None = None,
    return_lse: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Sparse MQA. Same face as the eager row.

    ``q`` is ``[B, S, H, D]``, ``kv`` is ``[B, S_kv, D]``, ``attn_sink`` is
    ``[H]``, ``topk_idxs`` is ``[B, S, topk]``.
    """
    from ....vendor.tilelang_sparse_mla import sparse_attn_tilelang

    return sparse_attn_tilelang(q, kv, attn_sink, topk_idxs, sm_scale, return_lse)
