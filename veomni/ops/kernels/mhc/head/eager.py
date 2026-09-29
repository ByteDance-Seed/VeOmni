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

"""mHC head eager math (final residual-stream collapse)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..helper import unweighted_rms


def wrapper(
    hidden_streams: Tensor,
    fn: Tensor,
    scale: Tensor,
    base: Tensor,
    norm_eps: float,
    hc_mult: int,
    hc_eps: float,
) -> Tensor:
    """Collapse the residual streams. Regular autograd.

    Matches DeepSeek-V4 ``DeepseekV4HyperHead.forward``. *hc_mult* is
    accepted for the modeling call face; the mix width comes from *fn*.
    """
    del hc_mult
    flat = unweighted_rms(hidden_streams.flatten(2).float(), norm_eps)
    mixes = F.linear(flat, fn.float())
    pre = torch.sigmoid(mixes * scale.float() + base.float()) + hc_eps
    return (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
