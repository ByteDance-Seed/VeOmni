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

"""Dense async Ulysses output projection: A2A heads-to-seq, then linear."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn.functional as F
from torch import Tensor

from ......distributed.sequence_parallel.comm import get_ulysses_sequence_parallel_group
from ......distributed.sequence_parallel.ulysses import all_to_all_tensor
from ......distributed.sequence_parallel.utils import (
    padding_tensor_for_seqeunce_parallel,
    unpadding_tensor_for_seqeunce_parallel,
)
from .....registry import SavedState
from ...shared.backward import linear_input_backward, linear_parameter_backward


@dataclass
class _Meta:
    """Saved all-to-all dims plus the head layout used to unflatten the input grad."""

    seq_dimension: int
    head_dimension: int
    unpadded_dim_size: int
    group: Any
    num_heads: int
    head_dim: int


def forward(
    hidden_states: Tensor,
    proj_weight: Tensor,
    proj_bias: Tensor | None,
    *,
    seq_dimension: int,
    head_dimension: int,
    unpadded_dim_size: int,
    group: Any = None,
) -> tuple[Tensor, SavedState]:
    """All-to-all seq-to-heads, flatten heads, then the output linear."""
    sp_group = get_ulysses_sequence_parallel_group() if group is None else group
    hidden_states = padding_tensor_for_seqeunce_parallel(hidden_states, seq_dimension)
    hidden_states = all_to_all_tensor(
        hidden_states, scatter_dim=seq_dimension, gather_dim=head_dimension, group=sp_group
    )
    num_heads = hidden_states.shape[head_dimension]
    head_dim = hidden_states.shape[-1]
    # Output linear sees [B, S, H*D]. Backward reshapes the input grad back to heads.
    hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1)
    output = F.linear(hidden_states, proj_weight, proj_bias)
    return output, SavedState(
        (hidden_states, proj_weight, proj_bias),
        _Meta(seq_dimension, head_dimension, unpadded_dim_size, sp_group, num_heads, head_dim),
    )


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor | None, ...]:
    """Reshape the input grad to heads, reverse all-to-all, overlap weight/bias grads."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    hidden_states, proj_weight, proj_bias = saved.tensors
    grad_hidden = linear_input_backward(grad_output, hidden_states, proj_weight)
    # Reverse all-to-all needs the collected head layout, not the flattened [B, S, H*D].
    grad_hidden = grad_hidden.reshape(grad_hidden.shape[0], -1, meta.num_heads, meta.head_dim)
    grad_out_res = all_to_all_tensor(
        grad_hidden,
        scatter_dim=meta.head_dimension,
        gather_dim=meta.seq_dimension,
        group=meta.group,
        async_op=True,
    )
    grad_proj_weight, grad_proj_bias = linear_parameter_backward(
        grad_output,
        hidden_states,
        proj_weight,
        has_bias=proj_bias is not None,
    )
    grad_hidden = unpadding_tensor_for_seqeunce_parallel(grad_out_res(), meta.seq_dimension, meta.unpadded_dim_size)
    return grad_hidden, grad_proj_weight, grad_proj_bias
