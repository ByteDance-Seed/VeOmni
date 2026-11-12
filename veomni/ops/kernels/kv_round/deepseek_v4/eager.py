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

"""DeepSeek-V4's storage-rounding recipes, in PyTorch.

The model stores two things quantized, with different geometry, and this variant
is what makes each call self-describing instead of leaving the numbers at the
call site:

* **Window KV and compressed KV** -- FP8 over 64-wide groups covering the NoPE
  channels only. The trailing RoPE channels stay in the model dtype, because
  rotating a rounded value is not what inference does with them.
* **Indexer Q and K** -- FP8 over whole 128-wide heads. The indexer's logits
  compare a rounded query against a rounded key, so both sides round over the
  full head rather than stopping at a RoPE split it does not have.

Both are the generic round trip with the geometry fixed, so the arithmetic lives
in ``standard`` and only the recipe lives here.
"""

from __future__ import annotations

from torch import Tensor

from ....registry import SavedState
from ..standard import eager as _standard_eager


#: Group width over the NoPE channels of a stored KV entry.
WINDOW_KV_BLOCK_SIZE = 64
#: Group width over a whole indexer head.
INDEXER_BLOCK_SIZE = 128


def round_window_kv(kv: Tensor, rope_features: int) -> Tensor:
    """Round the NoPE channels of a KV entry, leaving the RoPE tail untouched.

    Args:
        kv: KV entry whose last dimension is ``nope + rope`` channels.
        rope_features: Trailing channels to leave in the model dtype.
    """
    return _standard_eager.fp8_roundtrip(
        kv,
        block_size=WINDOW_KV_BLOCK_SIZE,
        quant_features=kv.shape[-1] - rope_features,
    )


def round_indexer_act(x: Tensor) -> Tensor:
    """Round an indexer activation over whole 128-wide groups."""
    return _standard_eager.fp8_roundtrip(x, block_size=INDEXER_BLOCK_SIZE)


def forward(
    x: Tensor,
    *,
    rope_features: int | None = None,
) -> tuple[Tensor, SavedState]:
    """Round one stored tensor with the DeepSeek-V4 recipe.

    ``rope_features`` switches between the two recipes: a KV entry passes the
    trailing channels it keeps in the model dtype, while an indexer activation
    passes ``None`` and rounds the whole last dimension.
    """
    output = round_indexer_act(x) if rope_features is None else round_window_kv(x, rope_features)
    return output, SavedState(())


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor]:
    """Pass gradient straight through to the input."""
    del saved
    return (grad_output,)
