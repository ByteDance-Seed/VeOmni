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

"""standard SwiGLU Liger adapter (``LigerSiLUMulFunction`` raw pair)."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ....registry import SavedState
from . import eager as _eager


@dataclass(frozen=True)
class _Meta:
    """Whether the empty-tensor path ran."""

    empty: bool


def forward(gate: Tensor, up: Tensor) -> tuple[Tensor, SavedState]:
    """Liger fused ``silu(gate) * up``.

    Empty inputs fall back to the eager pair. Otherwise saves the 2D views
    that ``swiglu_backward`` overwrites in place.
    """
    if gate.numel() == 0 or up.numel() == 0:
        output, saved = _eager.forward(gate, up)
        return output, SavedState(saved.tensors, _Meta(True))

    from liger_kernel.ops.swiglu import swiglu_forward

    saved_gate, saved_up, output = swiglu_forward(gate.contiguous(), up.contiguous())
    return output, SavedState((saved_gate, saved_up), _Meta(False))


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, Tensor]:
    """Return ``(grad_gate, grad_up)``. Empty inputs reuse the eager backward."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    if meta.empty:
        return _eager.backward(grad_output, SavedState(saved.tensors))

    from liger_kernel.ops.swiglu import swiglu_backward

    saved_gate, saved_up = saved.tensors
    return swiglu_backward(saved_gate, saved_up, grad_output.contiguous())
