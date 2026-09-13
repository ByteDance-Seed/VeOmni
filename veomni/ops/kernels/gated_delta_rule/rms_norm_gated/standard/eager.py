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

"""standard rms_norm_gated eager math (RMSNorm then silu(gate))."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def wrapper(
    x: Tensor,
    gate: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    eps: float = 1e-6,
    activation: str | None = "silu",
) -> Tensor:
    """``weight * rms_norm(x) * silu(gate)``. Regular autograd.

    Matches HuggingFace ``Qwen3_5RMSNormGated``. Functional extra is *weight*.
    FLA's unused norm bias and *activation* are accepted; nonempty bias is
    rejected. ``activation`` must be ``silu`` / ``swish`` / ``None``.
    """
    from ...optional import optional_tensor

    if optional_tensor(bias) is not None:
        raise ValueError("rms_norm_gated does not use a norm bias")
    if activation not in {None, "silu", "swish"}:
        raise ValueError(f"unsupported rms_norm_gated activation: {activation!r}")
    input_dtype = x.dtype
    x_f = x.float()
    rstd = (x_f.square().mean(dim=-1, keepdim=True) + eps).rsqrt()
    normed = weight * (x_f * rstd).to(dtype=input_dtype)
    return (normed * F.silu(gate.float())).to(dtype=input_dtype)
