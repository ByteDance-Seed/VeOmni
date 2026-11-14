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

"""DeepSeek-V4's storage-rounding recipes, on the Triton round trip.

Same two recipes as :mod:`.eager` with the same geometry; only the round trip's
backend differs. ``standard`` owns both backends and this variant fixes the
numbers, so neither file repeats the other's arithmetic.
"""

from __future__ import annotations

from torch import Tensor

from ....registry import SavedState
from ..standard import triton as _standard_triton
from .eager import INDEXER_BLOCK_SIZE, WINDOW_KV_BLOCK_SIZE


def round_window_kv(kv: Tensor, rope_features: int) -> Tensor:
    """Round the NoPE channels of a KV entry, leaving the RoPE tail untouched."""
    return _standard_triton.triton_fp8_roundtrip(
        kv,
        block_size=WINDOW_KV_BLOCK_SIZE,
        quant_features=kv.shape[-1] - rope_features,
    )


def round_indexer_act(x: Tensor) -> Tensor:
    """Round an indexer activation over whole 128-wide groups."""
    return _standard_triton.triton_fp8_roundtrip(x, block_size=INDEXER_BLOCK_SIZE)


def forward(x: Tensor, *, rope_features: int | None = None) -> tuple[Tensor, SavedState]:
    """Round one stored tensor with the DeepSeek-V4 recipe."""
    output = round_indexer_act(x) if rope_features is None else round_window_kv(x, rope_features)
    return output, SavedState(())


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor]:
    """Pass gradient straight through to the input."""
    del saved
    return (grad_output,)
