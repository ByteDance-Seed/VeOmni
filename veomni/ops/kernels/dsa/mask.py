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
# See the License for the specific language governing limitations
# under the License.

"""Translate HF dense masks before fused GLM/DeepSeek-V4 DSA kernels.

Fused indexer/attention rows bake in standard causality and cannot represent
padding or a custom additive mask. Modeling must drop a pure causal mask to
``None`` and refuse anything else, pointing the caller at eager.
"""

from __future__ import annotations

import torch
from torch import Tensor


def is_standard_causal_mask(attention_mask: Tensor | None, *, q_len: int, kv_len: int) -> bool:
    """Whether ``attention_mask`` is only the lower-triangular causal constraint.

    ``None`` is the ``is_causal`` skip used when there is no padding. Additive
    or boolean ``[B, 1, Q, K]`` / ``[B, Q, K]`` masks must match the
    decode-aware triangle ``k > q + kv_len - q_len`` on every entry.
    ``H != 1``, padding, a custom overlay, or a nonzero additive bias on an
    allowed position is not standard.
    """
    if attention_mask is None:
        return True
    mask = attention_mask
    if mask.dim() == 4:
        if mask.shape[1] != 1:
            return False
        mask = mask[:, 0]
    if mask.dim() != 3 or mask.shape[-2] != q_len or mask.shape[-1] != kv_len:
        return False
    q_positions = torch.arange(q_len, device=mask.device)[:, None]
    k_positions = torch.arange(kv_len, device=mask.device)[None, :]
    expected_blocked = (k_positions > q_positions + kv_len - q_len).unsqueeze(0).expand_as(mask)
    if mask.dtype == torch.bool:
        # Accept either convention only when the full triangle matches.
        return bool(torch.equal(mask, ~expected_blocked) or torch.equal(mask, expected_blocked))
    allowed = mask == 0
    blocked = mask < 0
    return bool(torch.equal(blocked, expected_blocked) and torch.equal(allowed, ~expected_blocked))


def translate_fused_dsa_mask(
    attention_mask: Tensor | None,
    *,
    q_len: int,
    kv_len: int,
    fused: bool,
    what: str,
) -> Tensor | None:
    """Drop a standard causal mask for fused DSA, or keep it for eager.

    Fused rows must not see padding or custom masks. Those cases raise and tell
    the caller to use the eager implementation instead of ignoring the mask.
    """
    if not fused:
        return attention_mask
    if is_standard_causal_mask(attention_mask, q_len=q_len, kv_len=kv_len):
        return None
    raise ValueError(
        f"{what} fused implementation only accepts the standard causal mask; "
        "padding or custom masks require the eager implementation."
    )
