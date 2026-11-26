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

"""mHC post eager math (scale sublayer output and mix residual streams)."""

from __future__ import annotations

import torch
from torch import Tensor


def wrapper(
    output: Tensor,
    residual: Tensor,
    post: Tensor,
    comb: Tensor,
) -> Tensor:
    """Mix *output* back into the residual streams. Regular autograd.

    Matches DeepSeek-V4 decoder-layer residual update. *post* is
    ``[B, S, H]``; *comb* is ``[B, S, H, H]``.
    """
    dtype = residual.dtype
    return post.to(dtype).unsqueeze(-1) * output.unsqueeze(-2) + torch.matmul(
        comb.to(dtype).transpose(-1, -2), residual
    )
