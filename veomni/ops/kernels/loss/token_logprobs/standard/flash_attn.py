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

"""standard token-level log-probability FlashAttention Triton cross-entropy adapter."""

from __future__ import annotations

from torch import Tensor


def wrapper(
    logits: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = -100,
) -> Tensor:
    """Per-token ``log p(y_t)`` via ``flash_attn.ops.triton.cross_entropy.cross_entropy_loss``.

    Matches the kernel called by verl's ``FusedLinearForPPOFunction`` for
    bitwise-identical fp32 per-token log-probabilities on GPU.
    Lazy-imports ``flash_attn``.
    """
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    batch_shape = logits.shape[:-1]
    logits_2d = logits.reshape(-1, logits.shape[-1])
    labels_1d = labels.reshape(-1)
    per_token_nll = cross_entropy_loss(logits_2d, labels_1d, ignore_index=ignore_index)[0]
    return -per_token_nll.view(batch_shape)
