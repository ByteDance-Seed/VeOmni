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

"""Backend-safe block selection helpers for Qwen4-Exp QSA."""

from __future__ import annotations

import torch
from torch import Tensor


def select_visible_topk_blocks(
    scores: Tensor,
    visible: Tensor,
    top_count: int,
) -> tuple[Tensor, Tensor]:
    """Select top-k visible blocks from non-negative (ReLU) block scores.

    Masks invisible blocks with finite ``-1.0`` (strictly below ReLU scores
    ``>= 0.0``) so Ascend ``TopKV2`` never faults on sparse ``-inf`` rows while
    preserving exact selection parity across backends. On NPU, boolean
    visibility gathering routes through ``int8`` because bool ``gather`` is
    unsupported.
    """
    masked_scores = scores.masked_fill(~visible, -1.0)
    top_blocks = masked_scores.topk(top_count, dim=-1).indices
    if visible.device.type == "npu":
        top_valid = visible.to(torch.int8).gather(-1, top_blocks).bool()
    else:
        top_valid = visible.gather(-1, top_blocks)
    return top_blocks, top_valid


__all__ = ["select_visible_topk_blocks"]
