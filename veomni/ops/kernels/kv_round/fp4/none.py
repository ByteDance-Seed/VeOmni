# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""The ``fp4`` variant's off row: leave stored tensors unchanged.

Written out rather than reusing ``standard/none.py`` because the two variants do
not take the same call arguments: this one takes the element format's
``scale_fmt`` and not ``standard``'s prefix-only ``quant_features``. A caller
selects the variant its checkpoint stores and then passes that variant's
arguments, so the off row has to accept the same keywords the rounding rows do --
otherwise "off" would be the row that raises.
"""

from __future__ import annotations

from torch import Tensor

from ....registry import SavedState


def forward(x: Tensor, *, block_size: int = 16, scale_fmt: str = "e4m3") -> tuple[Tensor, SavedState]:
    """Return ``x`` unchanged; ``block_size`` and ``scale_fmt`` are accepted and ignored."""
    del block_size, scale_fmt
    return x, SavedState(())


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor]:
    """Pass gradient straight through to the input."""
    del saved
    return (grad_output,)
