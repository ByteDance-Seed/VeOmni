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

from ...registry import SavedState
from . import eager as _eager


@dataclass(frozen=True)
class _Meta:
    empty: bool
    unsqueeze_dim: int


def forward(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, *, unsqueeze_dim: int = 1
) -> tuple[tuple[Tensor, Tensor], SavedState]:
    if q.numel() == 0 or k.numel() == 0:
        output, saved = _eager.forward(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        return output, SavedState(saved.tensors, _Meta(True, unsqueeze_dim))

    from liger_kernel.ops.rope import rope_forward

    q_out, k_out, saved_cos, saved_sin = rope_forward(q, k, cos, sin)
    return (q_out, k_out), SavedState((saved_cos, saved_sin), _Meta(False, unsqueeze_dim))


def backward(grad_output: tuple[Tensor, Tensor], saved: SavedState) -> tuple[Tensor, Tensor, None, None]:
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    if meta.empty:
        return _eager.backward(grad_output, SavedState(saved.tensors, _eager._Meta(True, meta.unsqueeze_dim)))

    from liger_kernel.ops.rope import rope_backward

    grad_q, grad_k = grad_output
    cos, sin = saved.tensors
    dq, dk = rope_backward(grad_q, grad_k, cos, sin)
    return dq, dk, None, None
