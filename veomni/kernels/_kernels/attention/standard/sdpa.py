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

"""SDPA backend and SP-aware adapter implementation."""

from typing import Optional

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.integrations.sdpa_attention import sdpa_attention_forward as hf_sdpa_attention_forward

from .....distributed.parallel_state import get_parallel_state
from ..ulysses import (
    prepare_ulysses_qkv,
    restore_ulysses_output,
    should_apply_ulysses,
    slice_ulysses_head_auxiliary,
)


# Flash / cuDNN drop dense masks. This kernel exists for mask + Ulysses, so pin
# memory-efficient first. MATH stays as the CPU / unsupported-shape fallback.
_SDPA_MASK_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run Transformers SDPA with optional Ulysses exchange.

    Visibility lives on ``attention_mask``. ``sliding_window`` / ``softcap`` are
    not SDPA kwargs; drop them if the mask already encodes the pattern.

    Uses memory-efficient SDPA so a dense bool / additive mask stays valid.
    Flash is not tried. Use ``veomni_flash_attention_*`` when the pattern can
    stay in attention kwargs.
    """
    del sliding_window, softcap

    if any(dim == 0 for tensor in (query, key, value) for dim in tensor.shape):
        raise ValueError("SDPA does not support query/key/value tensors with zero dimensions.")

    parallel_state = get_parallel_state()
    ulysses_enabled = should_apply_ulysses()
    if ulysses_enabled:
        query, key, value, query_head_count = prepare_ulysses_qkv(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            group=parallel_state.ulysses_group,
            ulysses_size=parallel_state.ulysses_size,
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if "s_aux" in kwargs:
            kwargs["s_aux"] = slice_ulysses_head_auxiliary(
                kwargs["s_aux"],
                query_head_count=query_head_count,
                local_query_head_count=query.shape[1],
                group=parallel_state.ulysses_group,
            )

    with sdpa_kernel(_SDPA_MASK_BACKENDS):
        output, lse = hf_sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )

    if ulysses_enabled:
        output = restore_ulysses_output(output, group=parallel_state.ulysses_group)

    return output, lse
