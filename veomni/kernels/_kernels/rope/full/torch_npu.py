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

"""full RoPE torch_npu adapter."""

from __future__ import annotations

from torch import Tensor

from ....registry import SavedState
from . import eager as _eager


def forward(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, *, unsqueeze_dim: int = 1
) -> tuple[tuple[Tensor, Tensor], SavedState]:
    """NPU fused full RoPE. Empty inputs and backward reuse the eager pair."""
    if q.numel() == 0 or k.numel() == 0:
        return _eager.forward(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    import torch_npu

    cos_u = cos.unsqueeze(unsqueeze_dim)
    sin_u = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos_u, sin_u).to(q.dtype)
    k_embed = torch_npu.npu_rotary_mul(k, cos_u, sin_u).to(k.dtype)
    return (q_embed, k_embed), SavedState((cos, sin), _eager._Meta(False, unsqueeze_dim))


def backward(grad_output: tuple[Tensor, Tensor], saved: SavedState) -> tuple[Tensor, Tensor, None, None]:
    """Return ``(dq, dk, None, None)`` via the eager inverse rotation."""
    return _eager.backward(grad_output, saved)
