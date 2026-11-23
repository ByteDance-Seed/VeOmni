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

"""wan RoPE eager math (complex multiply by freqs)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ....registry import SavedState


def _real_pair_cos_sin(freqs: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    """Unique cos/sin from a diffusers real pair.

    ``freqs_cos[..., 0::2]`` and ``freqs_sin[..., 1::2]`` are the interleaved
    values. A packed table ``[1, S, 1, D]`` is reduced to ``[S, D/2]`` without
    squeezing the sequence axis.
    """
    freqs_cos, freqs_sin = freqs
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    if cos.ndim == 4 and cos.shape[0] == 1 and cos.shape[2] == 1:
        return cos[0, :, 0, :], sin[0, :, 0, :]
    return cos, sin


def _as_complex_freqs(freqs: Tensor | tuple[Tensor, Tensor]) -> Tensor:
    """Return complex freqs. An existing complex tensor is unchanged.

    A real pair is packed as ``[S, 1, D/2]``. CPU uses float64 so the multiply
    matches the complex-tensor path. NPU storage uses float32.
    """
    if isinstance(freqs, Tensor):
        return freqs
    cos, sin = _real_pair_cos_sin(freqs)
    if cos.ndim == 2:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    dtype = torch.float32 if cos.device.type == "npu" else torch.float64
    return torch.complex(cos.to(dtype=dtype), sin.to(dtype=dtype))


@dataclass(frozen=True)
class _Meta:
    """Empty flag, ``head_dim``, and the caller input shape.

    Forward always returns rank-3 ``[B, S, H*D]``. When the caller passed
    rank-4 ``[B, S, H, D]``, backward restores that shape.
    """

    empty: bool
    head_dim: int
    input_shape: tuple[int, ...] = ()


def forward(x: Tensor, freqs: Tensor | tuple[Tensor, Tensor], *, head_dim: int) -> tuple[Tensor, SavedState]:
    """Complex-multiply ``x`` by ``freqs`` after viewing each head as complex.

    ``freqs`` is complex ``[S, 1, D/2]`` or a diffusers real pair
    ``(freqs_cos, freqs_sin)``. The pair is packed into that complex layout
    before the multiply. ``head_dim`` is the last-axis size used to unflatten
    heads. Non-empty outputs are rank-3: heads are unflattened, rotated, then
    ``flatten(2)`` to ``[B, S, H*D]``. Empty ``x`` is returned unchanged and
    keeps its input rank. ``freqs`` is treated as fixed: backward conjugates
    it to compute ``dx`` and does not return a frequency gradient.
    """
    freqs = _as_complex_freqs(freqs)
    meta = _Meta(x.numel() == 0, head_dim, tuple(x.shape))
    if meta.empty:
        return x, SavedState((freqs,), meta)

    shaped = x.reshape(*x.shape[:2], -1, head_dim).contiguous()
    rotated = torch.view_as_complex(shaped.to(torch.float64).reshape(*shaped.shape[:3], -1, 2))
    output = torch.view_as_real(rotated * freqs).flatten(2).to(x.dtype)
    return output, SavedState((freqs,), meta)


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, None]:
    """Return ``(dx, None)``. ``freqs`` is not differentiated."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    (freqs,) = saved.tensors
    if meta.empty:
        return grad_output, None

    shaped = grad_output.reshape(*grad_output.shape[:2], -1, meta.head_dim).contiguous()
    rotated = torch.view_as_complex(shaped.to(torch.float64).reshape(*shaped.shape[:3], -1, 2))
    dx = torch.view_as_real(rotated * freqs.conj()).flatten(2).to(grad_output.dtype)
    if meta.input_shape and tuple(dx.shape) != meta.input_shape:
        dx = dx.reshape(meta.input_shape)
    return dx, None
