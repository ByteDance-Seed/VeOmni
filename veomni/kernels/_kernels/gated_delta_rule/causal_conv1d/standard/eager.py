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

"""standard causal_conv1d eager math (dense depthwise conv)."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from ...optional import optional_tensor


def wrapper(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    cu_seqlens: Tensor,
    *,
    activation: str | None = "silu",
) -> Tensor:
    """Dense causal depthwise conv1d. Empty *bias* is unused.

    *x* is ``[B, S, D]``. *weight* is ``[D, W]``. Nonempty *cu_seqlens* is
    unsupported on eager.
    """
    if optional_tensor(cu_seqlens) is not None:
        raise ValueError("causal_conv1d eager does not support cu_seqlens")

    channels, kernel_size = weight.shape
    x_t = x.transpose(1, 2)
    out = F.conv1d(
        x_t,
        weight.unsqueeze(1),
        optional_tensor(bias),
        padding=kernel_size - 1,
        groups=channels,
    )
    out = out[..., : x.shape[1]]
    if activation in {"silu", "swish"}:
        out = F.silu(out)
    elif activation is not None:
        raise ValueError(f"unsupported causal_conv1d activation: {activation!r}")
    return out.transpose(1, 2).contiguous().to(dtype=x.dtype)
