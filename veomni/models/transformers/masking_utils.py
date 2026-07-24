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

"""VeOmni wrappers around Transformers attention-mask preparation."""

from collections.abc import Callable

import torch
from torch.nn.attention.flex_attention import BlockMask
from transformers.cache_utils import Cache
from transformers.configuration_utils import PreTrainedConfig
from transformers.masking_utils import (
    and_masks,
    packed_sequence_mask_function,
)
from transformers.masking_utils import (
    create_causal_mask as _hf_create_causal_mask,
)
from transformers.masking_utils import (
    create_sliding_window_causal_mask as _hf_create_sliding_window_causal_mask,
)


_FLEX_ATTENTION_IMPLEMENTATION = "veomni_flex_attention_with_sp"


def _packed_sequence_ids_from_cu_seq_lens(
    cu_seq_lens_q: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert flattened cumulative sequence lengths to per-token sequence IDs."""
    if cu_seq_lens_q.ndim != 1:
        raise ValueError(f"cu_seq_lens_q must be a 1D tensor, got shape {tuple(cu_seq_lens_q.shape)}.")
    if cu_seq_lens_q.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"cu_seq_lens_q must use an integer dtype, got {cu_seq_lens_q.dtype}.")
    if cu_seq_lens_q.numel() < 2:
        raise ValueError("cu_seq_lens_q must contain at least the initial and final offsets.")

    cu_seq_lens_q = cu_seq_lens_q.to(device=device)
    sequence_lengths = cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]
    total_length = batch_size * sequence_length
    sequence_ids = torch.repeat_interleave(
        torch.arange(sequence_lengths.numel(), device=device),
        sequence_lengths,
        output_size=total_length,
    )
    return sequence_ids.view(batch_size, sequence_length)


def _add_packed_sequence_mask(
    *,
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | BlockMask | None,
    past_key_values: Cache | None,
    and_mask_function: Callable | None,
    cu_seq_lens_q: torch.Tensor | None,
) -> Callable | None:
    """Add packed self-attention boundaries to the FlexAttention mask predicate."""
    if (
        cu_seq_lens_q is None
        or config._attn_implementation != _FLEX_ATTENTION_IMPLEMENTATION
        or cu_seq_lens_q.numel() == 2
    ):
        return and_mask_function

    if past_key_values is not None:
        raise ValueError("Packed FlexAttention does not support past_key_values.")
    if attention_mask is not None and attention_mask.ndim != 2:
        raise ValueError("Packed FlexAttention requires a 2D attention mask.")

    sequence_length = attention_mask.shape[-1] if attention_mask is not None else inputs_embeds.shape[1]
    sequence_ids = _packed_sequence_ids_from_cu_seq_lens(
        cu_seq_lens_q,
        batch_size=inputs_embeds.shape[0],
        sequence_length=sequence_length,
        device=inputs_embeds.device,
    )
    packed_mask_function = packed_sequence_mask_function(sequence_ids)
    if and_mask_function is None:
        return packed_mask_function
    return and_masks(and_mask_function, packed_mask_function)


def create_causal_mask(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
    block_sequence_ids: torch.Tensor | None = None,
    *,
    cu_seq_lens_q: torch.Tensor | None = None,
) -> torch.Tensor | BlockMask | None:
    """Create a causal mask with optional VeOmni packed-sequence boundaries."""
    and_mask_function = _add_packed_sequence_mask(
        config=config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        and_mask_function=and_mask_function,
        cu_seq_lens_q=cu_seq_lens_q,
    )
    return _hf_create_causal_mask(
        config=config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
        or_mask_function=or_mask_function,
        and_mask_function=and_mask_function,
        block_sequence_ids=block_sequence_ids,
    )


def create_sliding_window_causal_mask(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
    block_sequence_ids: torch.Tensor | None = None,
    *,
    cu_seq_lens_q: torch.Tensor | None = None,
) -> torch.Tensor | BlockMask | None:
    """Create a sliding-window causal mask with optional packed-sequence boundaries."""
    and_mask_function = _add_packed_sequence_mask(
        config=config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        and_mask_function=and_mask_function,
        cu_seq_lens_q=cu_seq_lens_q,
    )
    return _hf_create_sliding_window_causal_mask(
        config=config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
        or_mask_function=or_mask_function,
        and_mask_function=and_mask_function,
        block_sequence_ids=block_sequence_ids,
    )
