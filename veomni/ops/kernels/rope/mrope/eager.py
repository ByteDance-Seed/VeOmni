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

"""mrope RoPE eager math (remix 3-axis tables, then rotate-half)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ....registry import SavedState
from ..full.eager import _apply, _grad_x, _rotate_half


@dataclass(frozen=True)
class _Meta:
    """Broadcast metadata, remix sections, and whether tables need gradients."""

    empty: bool
    unsqueeze_dim: int
    table_gradients: bool
    sections: tuple[int, ...]


def _section_sizes(mrope_section: Sequence[int]) -> tuple[int, ...]:
    """HF ``mrope_section * 2``: repeat the three axis widths across ``cat(freqs, freqs)``."""
    return tuple(mrope_section) * 2


def _mix_table(table: Tensor, sections: tuple[int, ...], unsqueeze_dim: int) -> Tensor:
    """Split last dim by ``sections`` and take axis ``i % 3`` from each chunk."""
    return torch.cat([chunk[i % 3] for i, chunk in enumerate(table.split(sections, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )


def _unmix_table_gradient(
    grad_mixed: Tensor,
    original: Tensor,
    sections: tuple[int, ...],
    unsqueeze_dim: int,
) -> Tensor:
    """Scatter mixed last-dim grads back onto the 3-axis source table."""
    mixed = torch.cat([chunk[i % 3] for i, chunk in enumerate(original.split(sections, dim=-1))], dim=-1)
    collapsed = grad_mixed.sum_to_size(mixed.unsqueeze(unsqueeze_dim).shape).squeeze(unsqueeze_dim).to(original.dtype)
    grad = original.new_zeros(original.shape)
    offset = 0
    for i, size in enumerate(sections):
        grad[i % 3, ..., offset : offset + size].add_(collapsed[..., offset : offset + size])
        offset += size
    return grad


def forward(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Tensor | None = None,
    unsqueeze_dim: int = 1,
    *,
    mrope_section: Sequence[int],
) -> tuple[tuple[Tensor, Tensor], SavedState]:
    """Rotate ``q`` / ``k`` after remixing 3-axis multimodal ``cos`` / ``sin``.

    ``mrope_section`` is the HF temporal/height/width channel split. Tables
    are remixed with the same ``section * 2`` / ``i % 3`` gather used by
    ``apply_multimodal_rotary_pos_emb``, then rotate-half is applied.
    ``position_ids`` is retained for compatibility and does not participate.
    Empty inputs are returned unchanged.
    """
    del position_ids
    sections = _section_sizes(mrope_section)
    if q.numel() == 0 or k.numel() == 0:
        return (q, k), SavedState((cos, sin), _Meta(True, unsqueeze_dim, False, sections))

    cos_u = _mix_table(cos, sections, unsqueeze_dim)
    sin_u = _mix_table(sin, sections, unsqueeze_dim)
    table_gradients = cos.requires_grad or sin.requires_grad
    tensors = (q, k, cos, sin) if table_gradients else (cos, sin)
    return (
        _apply(q, cos_u, sin_u),
        _apply(k, cos_u, sin_u),
    ), SavedState(tensors, _Meta(False, unsqueeze_dim, table_gradients, sections))


def backward(
    grad_output: tuple[Tensor, Tensor], saved: SavedState
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, None, None]:
    """Return q/k, optional table, and compatibility-argument gradients."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    grad_q, grad_k = grad_output
    if meta.empty:
        return grad_q, grad_k, None, None, None, None

    if meta.table_gradients:
        q, k, cos, sin = saved.tensors
    else:
        cos, sin = saved.tensors
    cos_u = _mix_table(cos, meta.sections, meta.unsqueeze_dim)
    sin_u = _mix_table(sin, meta.sections, meta.unsqueeze_dim)
    dq = _grad_x(grad_q, cos_u, sin_u)
    dk = _grad_x(grad_k, cos_u, sin_u)
    if not meta.table_gradients:
        return dq, dk, None, None, None, None

    grad_cos = None
    if cos.requires_grad:
        grad_cos = _unmix_table_gradient(grad_q * q, cos, meta.sections, meta.unsqueeze_dim) + _unmix_table_gradient(
            grad_k * k, cos, meta.sections, meta.unsqueeze_dim
        )
    grad_sin = None
    if sin.requires_grad:
        grad_sin = _unmix_table_gradient(
            grad_q * _rotate_half(q), sin, meta.sections, meta.unsqueeze_dim
        ) + _unmix_table_gradient(grad_k * _rotate_half(k), sin, meta.sections, meta.unsqueeze_dim)
    return dq, dk, grad_cos, grad_sin, None, None
