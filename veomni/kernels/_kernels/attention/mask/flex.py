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

"""SP-aware FlexAttention mask builder."""

from typing import Callable

import torch
from torch.nn.attention.flex_attention import BlockMask
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, causal_mask_function

from .....distributed.parallel_state import get_parallel_state
from ..ulysses import should_apply_ulysses


def flex_attention_mask_builder(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: torch.Tensor | None = None,
    skip_ulysses: bool = False,
    **kwargs,
) -> BlockMask:
    """Build a Transformers FlexAttention mask.

    Expand local lengths to the Ulysses-global sequence only when the
    adapter would gather Q/K/V itself: sync Ulysses and not
    ``skip_ulysses``. Async Ulysses keeps local tokens, so the mask stays
    local too.
    """
    if should_apply_ulysses(skip_ulysses=skip_ulysses):
        if q_offset != 0 or kv_offset != 0:
            raise ValueError("FlexAttention with Ulysses does not support cached mask offsets.")
        if attention_mask is None or attention_mask.ndim != 2:
            raise ValueError("FlexAttention with Ulysses requires a full-sequence 2D attention mask.")

        parallel_state = get_parallel_state()
        full_sequence_length = q_length * parallel_state.ulysses_size
        if attention_mask.shape[-1] != full_sequence_length:
            raise ValueError(
                "FlexAttention with Ulysses requires the full attention-mask sequence length to equal "
                f"local q_length * ulysses_size, got attention_mask.shape[-1]={attention_mask.shape[-1]}, "
                f"q_length={q_length}, ulysses_size={parallel_state.ulysses_size}."
            )
        q_length = kv_length = full_sequence_length
        q_offset = kv_offset = 0

    return ALL_MASK_ATTENTION_FUNCTIONS["flex_attention"](
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=mask_function,
        attention_mask=attention_mask,
        **kwargs,
    )
