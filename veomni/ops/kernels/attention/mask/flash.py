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

"""Flash / Sage mask builder preserving 2D padding metadata."""

from collections.abc import Callable

import torch
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, causal_mask_function


_hf_flash_attention_mask_builder = ALL_MASK_ATTENTION_FUNCTIONS["flash_attention_2"]


def flash_attention_mask_builder(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor | None:
    """Keep FlashAttention's 2D padding mask while causal stays in kwargs.

    Transformers returns ``None`` for an all-valid sequence, but preserves a
    2D mask containing padding so FlashAttention can derive variable lengths.
    SageAttention receives the same value and rejects unsupported padding in
    its forward adapter instead of silently attending to padded tokens.
    """
    return _hf_flash_attention_mask_builder(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=mask_function,
        attention_mask=attention_mask,
        **kwargs,
    )
