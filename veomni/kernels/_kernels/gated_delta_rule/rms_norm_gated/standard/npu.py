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

"""standard rms_norm_gated npu adapter."""

from __future__ import annotations

import torch
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
    """NPU ``npu_rms_norm`` plus ``npu_swiglu`` on ``cat(gate, normed)``.

    Same math as ``NPUFusedRMSNormGated``. Lazy-imports ``torch_npu``.
    """
    from ...optional import optional_tensor

    if optional_tensor(bias) is not None:
        raise ValueError("rms_norm_gated does not use a norm bias")
    if activation not in {None, "silu", "swish"}:
        raise ValueError(f"unsupported rms_norm_gated activation: {activation!r}")

    import torch_npu

    normed = torch_npu.npu_rms_norm(x, weight, eps)[0]
    return torch_npu.npu_swiglu(torch.cat([gate, normed], dim=-1), dim=-1)
