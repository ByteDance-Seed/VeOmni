# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""mHC pre eager math (norm, mix split, Sinkhorn, collapse)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..helper import POST_MULTIPLIER, unweighted_rms


def wrapper(
    hidden_streams: Tensor,
    fn: Tensor,
    scale: Tensor,
    base: Tensor,
    norm_eps: float,
    hc_mult: int,
    sinkhorn_iters: int,
    hc_eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(post, comb, collapsed)``. Regular autograd.

    Matches DeepSeek-V4 ``DeepseekV4HyperConnection.forward`` when the
    TileLang slot is off. *fn* / *scale* / *base* are the module parameters.
    """
    hc = hc_mult
    flat = unweighted_rms(hidden_streams.flatten(start_dim=2).float(), norm_eps)
    pre_w, post_w, comb_w = F.linear(flat, fn.float()).split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = base.split([hc, hc, hc * hc])
    pre_scale, post_scale, comb_scale = scale.unbind(0)
    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + hc_eps
    post = POST_MULTIPLIER * torch.sigmoid(post_w * post_scale + post_b)
    comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
    comb = torch.softmax(comb_logits, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
    return post, comb, collapsed
