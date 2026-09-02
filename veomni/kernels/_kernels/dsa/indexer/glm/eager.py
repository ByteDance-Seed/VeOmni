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

"""GLM-DSA indexer eager.

Adapted from HuggingFace ``GlmMoeDsaIndexer.forward`` score / top-k:

    scores = ReLU(q @ k) * softmax_scale
    index_scores = (scores * weights).sum(heads)
    return index_scores.topk(...)

https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
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
    """Official GLM indexer scores. Same face as cuDNN ``indexer_select_topk``.

    ``q`` is ``[B, S, H, D]``. ``k`` is ``[B, T, D]`` or ``[B, T, 1, D]``.
    ``w`` is ``[B, S, H]``. Returns top-k indices ``[B, S, top_k]`` as long.
    ``ratio`` is accepted for call-face parity with cuDNN. Official eager
    applies causality through ``attention_mask``, not ``ratio``.
    ``qhead_per_kv_head`` is unused.
    """
    del ratio, qhead_per_kv_head
    if k.dim() == 4:
        k = k.squeeze(2)
    # Copied from GlmMoeDsaIndexer.forward (transformers glm_moe_dsa).
    scores = torch.einsum("bshd,btd->bsht", q.float(), k.float()) * sm_scale
    scores = F.relu(scores)
    index_scores = torch.einsum("bsht,bsh->bst", scores, w.float())
    if attention_mask is not None:
        index_scores = index_scores + attention_mask
    top_k = min(int(top_k), index_scores.shape[-1])
    return index_scores.topk(top_k, dim=-1).indices.to(torch.long)
