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

from typing import Optional

import torch
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss


liger_kernel_cross_entropy = LigerFusedLinearCrossEntropyLoss(reduction="mean")


def fused_liger_kernel_cross_entropy(
    logits: torch.Tensor = None,
    labels: torch.Tensor = None,
    vocab_size: int = None,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    weights = kwargs.pop("weights")
    hidden_states = kwargs.pop("hidden_states")
    return liger_kernel_cross_entropy(weights, hidden_states, labels), logits
