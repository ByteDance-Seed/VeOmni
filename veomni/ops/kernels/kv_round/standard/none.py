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

"""The "leave stored tensors alone" row for the model-agnostic variant.

Same role as :mod:`.deepseek_v4.none`: every other family spells "off" as
``eager`` because its reference row rounds nothing, while here ``eager`` is a real
rounding implementation.
"""

from __future__ import annotations

from torch import Tensor

from ....registry import SavedState


def forward(x: Tensor, *, block_size: int = 32, quant_features: int = -1) -> tuple[Tensor, SavedState]:
    """Return ``x`` unchanged.

    ``block_size`` and ``quant_features`` are accepted and ignored so the row is
    call-compatible with the rounding rows.
    """
    del block_size, quant_features
    return x, SavedState(())


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor]:
    """Pass the gradient straight through; there is nothing to undo."""
    del saved
    return (grad_output,)
