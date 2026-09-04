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

"""full RoPE Liger adapter."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ....registry import SavedState
from . import eager as _eager


@dataclass(frozen=True)
class _Meta:
    """Eager-fallback flag plus the ``unsqueeze_dim`` that describes q/k layout."""

    use_eager: bool
    unsqueeze_dim: int
    eager_empty: bool = False


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
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, *, unsqueeze_dim: int = 1
) -> tuple[tuple[Tensor, Tensor], SavedState]:
    """Liger fused full RoPE.

    ``unsqueeze_dim`` is the HF broadcast axis and therefore the q/k layout:
    ``1`` is ``[B, H, S, D]``, ``2`` is ``[B, S, H, D]``. Liger only speaks
    ``[B, H, S, D]``, so ``2`` is transposed in and out. Any other value, or
    empty inputs, falls back to the eager pair.
    """
    if q.numel() == 0 or k.numel() == 0 or unsqueeze_dim not in (1, 2):
        output, saved = _eager.forward(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        eager_meta = saved.metadata
        assert isinstance(eager_meta, _eager._Meta)
        return output, SavedState(saved.tensors, _Meta(True, unsqueeze_dim, eager_meta.empty))

    from liger_kernel.ops.rope import rope_forward

    q_in, k_in = _to_liger_layout(q, k, unsqueeze_dim)
    q_out, k_out, saved_cos, saved_sin = rope_forward(q_in, k_in, cos, sin)
    return _from_liger_layout(q_out, k_out, unsqueeze_dim), SavedState(
        (saved_cos, saved_sin), _Meta(False, unsqueeze_dim)
    )


def backward(grad_output: tuple[Tensor, Tensor], saved: SavedState) -> tuple[Tensor, Tensor, None, None]:
    """Return ``(dq, dk, None, None)``. Eager fallbacks reuse the eager backward."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    if meta.use_eager:
        return _eager.backward(
            grad_output, SavedState(saved.tensors, _eager._Meta(meta.eager_empty, meta.unsqueeze_dim))
        )

    from liger_kernel.ops.rope import rope_backward

    grad_q, grad_k = _to_liger_layout(grad_output[0], grad_output[1], meta.unsqueeze_dim)
    cos, sin = saved.tensors
    dq, dk = rope_backward(grad_q, grad_k, cos, sin)
    dq, dk = _from_liger_layout(dq, dk, meta.unsqueeze_dim)
    return dq, dk, None, None
