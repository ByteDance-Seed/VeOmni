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

"""wan RoPE torch_npu adapter."""

from __future__ import annotations

import torch
from torch import Tensor

from ....registry import SavedState
from . import eager as _eager


def forward(x: Tensor, freqs: Tensor | tuple[Tensor, Tensor], *, head_dim: int) -> tuple[Tensor, SavedState]:
    """Apply NPU interleaved RoPE with fixed, non-differentiable ``freqs``.

    A complex table keeps the rank-3 ``unsqueeze(0)`` packing. A diffusers
    real pair is repeat-interleaved on its own leading dims and is not turned
    into a complex tensor for the kernel. Empty inputs and backward reuse the
    eager pair. Backward returns ``dx`` without a frequency gradient.
    """
    if x.numel() == 0:
        return _eager.forward(x, freqs, head_dim=head_dim)

    import torch_npu

    if isinstance(freqs, Tensor):
        cos = freqs.real.to(torch.float32).unsqueeze(0).repeat_interleave(2, dim=-1).contiguous()
        sin = freqs.imag.to(torch.float32).unsqueeze(0).repeat_interleave(2, dim=-1).contiguous()
        saved_freqs = freqs
    else:
        freqs_cos, freqs_sin = freqs
        cos = freqs_cos[..., 0::2].to(torch.float32).repeat_interleave(2, dim=-1).contiguous()
        sin = freqs_sin[..., 1::2].to(torch.float32).repeat_interleave(2, dim=-1).contiguous()
        # Eager backward still multiplies by a complex table. Keep that table
        # in float32 so it can live on NPU. The kernel itself sees only cos/sin.
        cos_u, sin_u = _eager._real_pair_cos_sin(freqs)
        if cos_u.ndim == 2:
            cos_u = cos_u.unsqueeze(1)
            sin_u = sin_u.unsqueeze(1)
        saved_freqs = torch.complex(cos_u.to(torch.float32), sin_u.to(torch.float32))
    shaped = x.reshape(*x.shape[:2], -1, head_dim).to(torch.float32)
    output = torch_npu.npu_rotary_mul(shaped, cos, sin, rotary_mode="interleave").flatten(-2)
    return output.to(x.dtype), SavedState((saved_freqs,), _eager._Meta(False, head_dim, tuple(x.shape)))


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, None]:
    """Return ``(dx, None)`` via the eager conjugate multiply."""
    return _eager.backward(grad_output, saved)
