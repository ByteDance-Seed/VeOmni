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

"""FlexAttention backend and SP-aware adapter implementation."""

from typing import Callable, Optional

import torch
from torch.nn.attention.flex_attention import BlockMask
from transformers.integrations.flex_attention import flex_attention_forward as hf_flex_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, causal_mask_function

from ....distributed.parallel_state import get_parallel_state
from .ulysses import (
    prepare_ulysses_qkv,
    restore_ulysses_output,
    slice_ulysses_head_auxiliary,
)


# Module-level patch slot for the underlying Transformers FlexAttention adapter.
_flex_attention_forward: Callable = hf_flex_attention_forward
_flex_attention_mask_builder: Callable = ALL_MASK_ATTENTION_FUNCTIONS["flex_attention"]


def flex_attention_mask_builder(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
) -> BlockMask:
    """Build a Transformers FlexAttention mask with Ulysses-global sequence dimensions."""
    parallel_state = get_parallel_state()
    if not parallel_state.ulysses_enabled:
        return _flex_attention_mask_builder(
            batch_size=batch_size,
            q_length=q_length,
            kv_length=kv_length,
            q_offset=q_offset,
            kv_offset=kv_offset,
            mask_function=mask_function,
            attention_mask=attention_mask,
            **kwargs,
        )

    if q_offset != 0 or kv_offset != 0:
        raise ValueError("FlexAttention with Ulysses does not support cached mask offsets.")
    if attention_mask is None or attention_mask.ndim != 2:
        raise ValueError("FlexAttention with Ulysses requires a full-sequence 2D attention mask.")

    full_sequence_length = q_length * parallel_state.ulysses_size
    if attention_mask.shape[-1] != full_sequence_length:
        raise ValueError(
            "FlexAttention with Ulysses requires the full attention-mask sequence length to equal "
            f"local q_length * ulysses_size, got attention_mask.shape[-1]={attention_mask.shape[-1]}, "
            f"q_length={q_length}, ulysses_size={parallel_state.ulysses_size}."
        )

    return _flex_attention_mask_builder(
        batch_size=batch_size,
        q_length=full_sequence_length,
        kv_length=full_sequence_length,
        q_offset=0,
        kv_offset=0,
        mask_function=mask_function,
        attention_mask=attention_mask,
        **kwargs,
    )


def register_veomni_flex_attention_mask_builder() -> None:
    """Register VeOmni's SP-aware wrapper around Transformers' FlexAttention mask builder."""
    ALL_MASK_ATTENTION_FUNCTIONS.register(
        "veomni_flex_attention_with_sp",
        flex_attention_mask_builder,
    )


def flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    skip_ulysses: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the pinned Transformers FlexAttention adapter with optional Ulysses exchange."""
    if not isinstance(attention_mask, BlockMask):
        raise TypeError(f"FlexAttention requires a BlockMask, got {type(attention_mask).__name__}.")

    if any(dim == 0 for tensor in (query, key, value) for dim in tensor.shape):
        raise ValueError("FlexAttention does not support query/key/value tensors with zero dimensions.")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError(
            f"FlexAttention GQA requires query heads ({query.shape[1]}) to be divisible by "
            f"key/value heads ({key.shape[1]})."
        )

    # Transformers models may pass ``sliding_window`` metadata together with a
    # BlockMask that already encodes the window predicate. The BlockMask remains
    # the sole source of visibility semantics; do not reconstruct or modify it
    # from the integer metadata.
    del sliding_window

    kernel_options = dict(kwargs.pop("kernel_options", {}) or {})
    # PyTorch's AUTO backend may select Flex Decoding for short queries and then
    # fail during Inductor kernel selection. Use the standard Triton FlexAttention
    # kernel by default while preserving an explicit caller override.
    kernel_options.setdefault("BACKEND", "TRITON")

    parallel_state = get_parallel_state()
    ulysses_enabled = parallel_state.ulysses_enabled and not skip_ulysses
    if ulysses_enabled:
        # Local head indices restart at zero on every Ulysses rank, so head-specific
        # masks require rank-aware slicing and rebasing before they can be supported.
        if attention_mask.shape[1] != 1:
            raise ValueError("FlexAttention with Ulysses requires a head-broadcast BlockMask.")

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

    output, lse = _flex_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        softcap=softcap,
        kernel_options=kernel_options,
        **kwargs,
    )

    if ulysses_enabled:
        output = restore_ulysses_output(output, group=parallel_state.ulysses_group)
        if lse is not None:
            lse = restore_ulysses_output(
                lse.transpose(1, 2).unsqueeze(-1),
                group=parallel_state.ulysses_group,
            ).squeeze(-1)
            lse = lse.transpose(1, 2).contiguous()

    return output, lse
