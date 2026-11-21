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

"""BAGEL two-route MoT async Ulysses output projection: inverse A2A then routed linear."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
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
class _BagelOMeta:
    """Saved all-to-all group, input sequence length, head geometry, and bias flags."""

    sp_group: Any
    input_sequence_length: int
    num_heads: int
    head_dim: int
    has_proj_bias_0: bool
    has_proj_bias_1: bool


def forward(
    hidden_states: Tensor,
    route_0_indices: Tensor,
    route_1_indices: Tensor,
    proj_weight_0: Tensor,
    proj_bias_0: Tensor | None,
    proj_weight_1: Tensor,
    proj_bias_1: Tensor | None,
    *,
    group: Any = None,
) -> tuple[Tensor, SavedState]:
    """Inverse Ulysses exchange followed by a two-route MoT output projection."""
    sp_group = get_ulysses_sequence_parallel_group() if group is None else group
    if sp_group is None:
        raise ValueError("Async MoT Ulysses requires an initialized Ulysses process group.")
    input_sequence_length = hidden_states.shape[0]
    hidden_states = padding_tensor_for_seqeunce_parallel(hidden_states, 0, group=sp_group)
    hidden_states = all_to_all_tensor(
        hidden_states,
        scatter_dim=0,
        gather_dim=1,
        group=sp_group,
        async_op=False,
    )
    num_heads = hidden_states.shape[1]
    head_dim = hidden_states.shape[2]
    flattened_states = hidden_states.reshape(hidden_states.shape[0], num_heads * head_dim)

    output_0 = F.linear(flattened_states.index_select(0, route_0_indices), proj_weight_0, proj_bias_0)
    output_1 = F.linear(flattened_states.index_select(0, route_1_indices), proj_weight_1, proj_bias_1)
    output = output_0.new_zeros((flattened_states.shape[0], proj_weight_0.shape[0]))
    output.index_copy_(0, route_0_indices, output_0)
    output.index_copy_(0, route_1_indices, output_1)

    return output, SavedState(
        (
            flattened_states,
            route_0_indices,
            route_1_indices,
            proj_weight_0,
            proj_weight_1,
        ),
        _BagelOMeta(
            sp_group=sp_group,
            input_sequence_length=input_sequence_length,
            num_heads=num_heads,
            head_dim=head_dim,
            has_proj_bias_0=proj_bias_0 is not None,
            has_proj_bias_1=proj_bias_1 is not None,
        ),
    )


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor | None, ...]:
    """Compute routed input gradients, launch reverse all-to-all, and overlap parameter gradients."""
    meta = saved.metadata
    assert isinstance(meta, _BagelOMeta)
    flattened_states, route_0_indices, route_1_indices, proj_weight_0, proj_weight_1 = saved.tensors
    grad_output_0 = grad_output.index_select(0, route_0_indices)
    grad_output_1 = grad_output.index_select(0, route_1_indices)

    flattened_states_0 = flattened_states.index_select(0, route_0_indices)
    flattened_states_1 = flattened_states.index_select(0, route_1_indices)
    grad_flattened_states = torch.zeros_like(flattened_states)
    grad_flattened_states.index_copy_(
        0,
        route_0_indices,
        linear_input_backward(grad_output_0, flattened_states_0, proj_weight_0),
    )
    grad_flattened_states.index_copy_(
        0,
        route_1_indices,
        linear_input_backward(grad_output_1, flattened_states_1, proj_weight_1),
    )
    grad_hidden_states = grad_flattened_states.reshape(flattened_states.shape[0], meta.num_heads, meta.head_dim)
    grad_hidden_wait = all_to_all_tensor(
        grad_hidden_states,
        scatter_dim=1,
        gather_dim=0,
        group=meta.sp_group,
        async_op=True,
    )

    grad_proj_weight_0, grad_proj_bias_0 = linear_parameter_backward(
        grad_output_0,
        flattened_states_0,
        proj_weight_0,
        has_bias=meta.has_proj_bias_0,
    )
    grad_proj_weight_1, grad_proj_bias_1 = linear_parameter_backward(
        grad_output_1,
        flattened_states_1,
        proj_weight_1,
        has_bias=meta.has_proj_bias_1,
    )

    grad_hidden_states = unpadding_tensor_for_seqeunce_parallel(
        grad_hidden_wait(),
        0,
        meta.input_sequence_length,
        group=meta.sp_group,
    )
    return (
        grad_hidden_states,
        None,
        None,
        grad_proj_weight_0,
        grad_proj_bias_0,
        grad_proj_weight_1,
        grad_proj_bias_1,
    )
