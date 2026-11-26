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

"""Portable Qwen4-Exp QSA block top-k."""

from __future__ import annotations

import torch
from torch import Tensor


def wrapper(scores: Tensor, visible: Tensor, top_count: int) -> tuple[Tensor, Tensor]:
    """Select the top-k visible compressed blocks.

    ``scores`` are ReLU-reduced block scores, so every visible entry is ``>= 0``.
    Invisible blocks are filled with finite ``-1.0``, which stays strictly below
    that range. Ascend ``TopKV2`` faults on sparse ``-inf`` rows, so this row
    does not use an infinity sentinel and does not scan ``scores.min()``.

    Returns block indices and a boolean validity mask of the same shape. On NPU,
    the validity gather goes through ``int8`` because bool ``gather`` is unsupported.
    """
    masked_scores = scores.masked_fill(~visible, -1.0)
    top_blocks = masked_scores.topk(top_count, dim=-1).indices
    if visible.device.type == "npu":
        top_valid = visible.to(torch.int8).gather(-1, top_blocks).bool()
    else:
        top_valid = visible.gather(-1, top_blocks)
    return top_blocks, top_valid
