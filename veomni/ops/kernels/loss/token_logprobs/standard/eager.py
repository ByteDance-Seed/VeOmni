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

"""standard token-level log-probability eager math (``log_softmax + gather``)."""

from __future__ import annotations

import torch
from torch import Tensor


def token_entropy_from_logits(logits: Tensor) -> Tensor:
    """Softmax entropy ``H[p] = logsumexp(logits) - sum(p * logits)``.

    Op order matches verl's ``_fused_linear_for_ppo_fwd`` so the fp32
    rounding boundary is identical: ``softmax`` then
    ``logsumexp(logits) - sum(probs * logits)``.
    """
    probs = logits.softmax(dim=-1)
    return torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)


def wrapper(
    logits: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = -100,
) -> Tensor:
    """Per-token ``log p(y_t)`` via decomposed ``clamp + log_softmax + gather + where``.

    Positions where ``labels == ignore_index`` return ``0.0``.
    """
    mask = labels != ignore_index
    safe_labels = labels.clamp(min=0).unsqueeze(-1)
    log_probs = logits.log_softmax(dim=-1).gather(-1, safe_labels).squeeze(-1)
    return torch.where(mask, log_probs, torch.zeros_like(log_probs))
