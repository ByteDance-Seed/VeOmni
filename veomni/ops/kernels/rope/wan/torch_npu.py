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

from dataclasses import dataclass

import torch
from torch import Tensor

from ....registry import SavedState
from . import eager as _eager


@dataclass(frozen=True)
class _Meta:
    """Real-pair backward state. Complex tables keep the eager metadata."""

    empty: bool
    head_dim: int
    input_shape: tuple[int, ...]


def _saved_real_pair(freqs: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    """Float32 cos/sin ``[S, 1, D/2]`` for backward. No complex storage."""
    cos, sin = _eager._real_pair_cos_sin(freqs)
    if cos.ndim == 2:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return cos.to(torch.float32).contiguous(), sin.to(torch.float32).contiguous()


def _real_pair_dx(
    grad_output: Tensor,
    cos: Tensor,
    sin: Tensor,
    head_dim: int,
    input_shape: tuple[int, ...],
) -> Tensor:
    """Conjugate interleaved rotation using real cos/sin.

    ``(g0 + i g1) * (cos - i sin)`` is ``g0*cos + g1*sin`` on even channels
    and ``g1*cos - g0*sin`` on odd channels. ``cos`` and ``sin`` broadcast
    from ``[S, 1, D/2]`` over ``[B, S, H, D/2]``.
    """
    shaped = grad_output.reshape(*grad_output.shape[:2], -1, head_dim).to(torch.float32)
    pairs = shaped.reshape(*shaped.shape[:3], head_dim // 2, 2)
    g0 = pairs[..., 0]
    g1 = pairs[..., 1]
    even = g0 * cos + g1 * sin
    odd = g1 * cos - g0 * sin
    dx = torch.stack((even, odd), dim=-1).flatten(2).to(grad_output.dtype)
    if input_shape and tuple(dx.shape) != input_shape:
        dx = dx.reshape(input_shape)
    return dx


def forward(x: Tensor, freqs: Tensor | tuple[Tensor, Tensor], *, head_dim: int) -> tuple[Tensor, SavedState]:
    """Apply NPU interleaved RoPE with fixed, non-differentiable ``freqs``.

    A complex table keeps the rank-3 ``unsqueeze(0)`` packing and the eager
    conjugate backward. A diffusers real pair is repeat-interleaved on its
    own leading dims. The kernel and the saved backward state both stay on
    real cos/sin, including the empty-input path. Backward returns ``dx``
    without a frequency gradient.
    """
    if x.numel() == 0:
        if isinstance(freqs, tuple):
            cos, sin = _saved_real_pair(freqs)
            return x, SavedState((cos, sin), _Meta(True, head_dim, tuple(x.shape)))
        return _eager.forward(x, freqs, head_dim=head_dim)

    import torch_npu

    if isinstance(freqs, Tensor):
        cos = freqs.real.to(torch.float32).unsqueeze(0).repeat_interleave(2, dim=-1).contiguous()
        sin = freqs.imag.to(torch.float32).unsqueeze(0).repeat_interleave(2, dim=-1).contiguous()
        saved: SavedState = SavedState((freqs,), _eager._Meta(False, head_dim, tuple(x.shape)))
    else:
        freqs_cos, freqs_sin = freqs
        cos = freqs_cos[..., 0::2].to(torch.float32).repeat_interleave(2, dim=-1).contiguous()
        sin = freqs_sin[..., 1::2].to(torch.float32).repeat_interleave(2, dim=-1).contiguous()
        saved = SavedState(_saved_real_pair(freqs), _Meta(False, head_dim, tuple(x.shape)))
    shaped = x.reshape(*x.shape[:2], -1, head_dim).to(torch.float32)
    output = torch_npu.npu_rotary_mul(shaped, cos, sin, rotary_mode="interleave").flatten(-2)
    return output.to(x.dtype), saved


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, None]:
    """Return ``(dx, None)``. Real pairs rotate in real arithmetic."""
    meta = saved.metadata
    if isinstance(meta, _Meta):
        if meta.empty:
            return grad_output, None
        cos, sin = saved.tensors
        return _real_pair_dx(grad_output, cos, sin, meta.head_dim, meta.input_shape), None
    return _eager.backward(grad_output, saved)
