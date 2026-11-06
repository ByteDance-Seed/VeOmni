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

"""Standard dense linear eager implementation."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def wrapper(x: Tensor, weight: Tensor, bias: Tensor, *, block_size: int = 128) -> Tensor:
    """Run ``F.linear`` with an empty tensor denoting an unused bias.

    ``block_size`` keeps the family-wide keyword so one signature serves every
    row; this reference does not quantize, so the value is unused.
    """
    del block_size
    return F.linear(x, weight, None if bias.numel() == 0 else bias)
