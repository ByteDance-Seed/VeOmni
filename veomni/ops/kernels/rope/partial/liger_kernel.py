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
# See the License for the specific language governing limitations
# under the License.

"""partial RoPE Liger adapter.

Liger rotates every supplied channel. Partial RoPE only rotates the
``cos.shape[-1]`` prefix, so this row slices that prefix, calls
``liger_kernel.ops.rope``, then concatenates the unrotated suffix.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ....registry import SavedState
from . import eager as _eager


@dataclass(frozen=True)
class _Meta:
    """Eager-fallback flag, q/k layout, and the rotated prefix width."""

    use_eager: bool
    unsqueeze_dim: int
    rotary_dim: int
    eager_empty: bool = False
    eager_table_gradients: bool = False


def _to_liger_layout(q: Tensor, k: Tensor, unsqueeze_dim: int) -> tuple[Tensor, Tensor]:
    """Liger ``rope_forward`` expects ``[B, H, S, D]`` (HF ``unsqueeze_dim=1``)."""
    if unsqueeze_dim == 1:
        return q, k
    return q.transpose(1, 2), k.transpose(1, 2)


def _from_liger_layout(q: Tensor, k: Tensor, unsqueeze_dim: int) -> tuple[Tensor, Tensor]:
    """Undo ``_to_liger_layout``."""
    if unsqueeze_dim == 1:
        return q, k
    return q.transpose(1, 2), k.transpose(1, 2)


def forward(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1
) -> tuple[tuple[Tensor, Tensor], SavedState]:
    """Liger fused partial RoPE. Only the ``cos.shape[-1]`` prefix is rotated.

    ``unsqueeze_dim`` is the HF broadcast axis and therefore the q/k layout:
    ``1`` is ``[B, H, S, D]``, ``2`` is ``[B, S, H, D]``. Liger only speaks
    ``[B, H, S, D]``, so ``2`` is transposed in and out. Empty input, any
    other ``unsqueeze_dim``, or trainable rotary tables fall back to eager.
    """
    rotary_dim = cos.shape[-1]
    if q.numel() == 0 or k.numel() == 0 or unsqueeze_dim not in (1, 2) or cos.requires_grad or sin.requires_grad:
        output, saved = _eager.forward(q, k, cos, sin, unsqueeze_dim)
        eager_meta = saved.metadata
        assert isinstance(eager_meta, _eager._Meta)
        return output, SavedState(
            saved.tensors,
            _Meta(
                True,
                eager_meta.unsqueeze_dim,
                rotary_dim,
                eager_meta.empty,
                eager_meta.table_gradients,
            ),
        )

    from liger_kernel.ops.rope import rope_forward

    q_in, k_in = _to_liger_layout(q[..., :rotary_dim].contiguous(), k[..., :rotary_dim].contiguous(), unsqueeze_dim)
    q_out, k_out, saved_cos, saved_sin = rope_forward(q_in, k_in, cos, sin)
    q_out, k_out = _from_liger_layout(q_out, k_out, unsqueeze_dim)
    return (
        torch.cat((q_out, q[..., rotary_dim:]), dim=-1),
        torch.cat((k_out, k[..., rotary_dim:]), dim=-1),
    ), SavedState((saved_cos, saved_sin), _Meta(False, unsqueeze_dim, rotary_dim))


def backward(
    grad_output: tuple[Tensor, Tensor], saved: SavedState
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, None]:
    """Rotate prefix gradients through Liger; copy the unrotated suffix."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    if meta.use_eager:
        return _eager.backward(
            grad_output,
            SavedState(
                saved.tensors,
                _eager._Meta(meta.eager_empty, meta.unsqueeze_dim, meta.eager_table_gradients),
            ),
        )

    from liger_kernel.ops.rope import rope_backward

    rotary_dim = meta.rotary_dim
    grad_q, grad_k = grad_output
    dq, dk = _to_liger_layout(
        grad_q[..., :rotary_dim].contiguous(),
        grad_k[..., :rotary_dim].contiguous(),
        meta.unsqueeze_dim,
    )
    cos, sin = saved.tensors
    dq, dk = rope_backward(dq, dk, cos, sin)
    dq, dk = _from_liger_layout(dq, dk, meta.unsqueeze_dim)
    return (
        torch.cat((dq, grad_q[..., rotary_dim:]), dim=-1),
        torch.cat((dk, grad_k[..., rotary_dim:]), dim=-1),
        None,
        None,
        None,
    )
