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

"""wan RoPE npu adapter."""

from __future__ import annotations

import torch
from torch import Tensor

from ....registry import SavedState
from . import eager as _eager


def forward(x: Tensor, freqs: Tensor, *, head_dim: int) -> tuple[Tensor, SavedState]:
    """NPU interleaved RoPE. Empty inputs and backward reuse the eager pair."""
    if x.numel() == 0:
        return _eager.forward(x, freqs, head_dim=head_dim)

    import torch_npu

    cos = freqs.real.to(torch.float32).unsqueeze(0).repeat_interleave(2, dim=-1).contiguous()
    sin = freqs.imag.to(torch.float32).unsqueeze(0).repeat_interleave(2, dim=-1).contiguous()
    shaped = x.reshape(*x.shape[:2], -1, head_dim).to(torch.float32)
    output = torch_npu.npu_rotary_mul(shaped, cos, sin, rotary_mode="interleave").flatten(-2)
    return output.to(x.dtype), SavedState((freqs,), _eager._Meta(False, head_dim))


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, None]:
    """Return ``(dx, None)`` via the eager conjugate multiply."""
    return _eager.backward(grad_output, saved)
