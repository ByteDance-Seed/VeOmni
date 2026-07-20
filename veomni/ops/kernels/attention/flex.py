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

"""FlexAttention backend and non-distributed adapter implementation."""

from typing import Callable, Optional

import torch
from torch.nn.attention.flex_attention import BlockMask
from transformers.integrations.flex_attention import flex_attention_forward as hf_flex_attention_forward

from ....distributed.parallel_state import get_parallel_state


# Module-level patch slot for the underlying Transformers FlexAttention adapter.
_flex_attention_forward: Callable = hf_flex_attention_forward


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
    """Run the pinned Transformers FlexAttention adapter without sequence parallelism."""
    if not isinstance(attention_mask, BlockMask):
        raise TypeError(f"FlexAttention requires a BlockMask, got {type(attention_mask).__name__}.")

    if any(dim == 0 for tensor in (query, key, value) for dim in tensor.shape):
        raise ValueError("FlexAttention does not support query/key/value tensors with zero dimensions.")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError(
            f"FlexAttention GQA requires query heads ({query.shape[1]}) to be divisible by "
            f"key/value heads ({key.shape[1]})."
        )

    ulysses_enabled = get_parallel_state().ulysses_enabled
    if ulysses_enabled and not skip_ulysses:
        raise RuntimeError("FlexAttention sequence parallelism is not enabled by this adapter.")
    if sliding_window is not None:
        raise ValueError("FlexAttention sliding-window semantics must be encoded in the supplied BlockMask.")

    return _flex_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        softcap=softcap,
        **kwargs,
    )
