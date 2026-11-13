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

"""Grouped block-diagonal linear eager implementation."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def wrapper(x: Tensor, weight: Tensor, bias: Tensor, *, n_groups: int, block_size: int = 128) -> Tensor:
    """Run block-diagonal grouped linear projection over ``n_groups`` head groups.

    ``x`` has shape ``[..., n_groups, in_per_group]`` and ``weight`` has shape
    ``[n_groups * out_per_group, in_per_group]``. An empty ``bias`` tensor
    denotes an unused bias.
    """
    del block_size
    if n_groups <= 0 or weight.ndim != 2 or weight.shape[0] % n_groups != 0:
        raise ValueError(f"weight shape {tuple(weight.shape)} is incompatible with n_groups={n_groups}.")
    out_per_group = weight.shape[0] // n_groups
    in_per_group = weight.shape[1]
    if x.ndim < 2 or x.shape[-2] != n_groups or x.shape[-1] != in_per_group:
        raise ValueError(f"Expected input shape [..., {n_groups}, {in_per_group}], got {tuple(x.shape)}.")
    if bias.numel() != 0 and bias.shape != (weight.shape[0],):
        raise ValueError("Non-empty bias must have shape [n_groups * out_per_group].")

    n_rows = x.numel() // (n_groups * in_per_group)
    grouped = x.reshape(n_rows, n_groups, in_per_group).transpose(0, 1)
    w = weight.view(n_groups, out_per_group, in_per_group).transpose(1, 2)
    out = torch.bmm(grouped, w).transpose(0, 1).reshape(*x.shape[:-2], n_groups, out_per_group)
    if bias.numel() > 0:
        out = out + bias.view(n_groups, out_per_group)
    return out


def module_wrapper(linear: nn.Module, x: Tensor, *, block_size: int = 128) -> Tensor:
    """Row entry point: call the projection module.

    The module is called rather than driven through :func:`wrapper` on its weight
    for the same reason as the ``standard`` eager row: a projection listed in
    ``basic_modules`` is ``fully_shard``-wrapped on its own, and its unshard hook
    runs in ``forward``. Reading ``linear.weight`` instead would hand this row a
    sharded ``DTensor``. It also covers adapter-wrapped projections, and the
    grouped layout stays a property of the module (``n_groups``), not of the call.

    :func:`wrapper` remains the tensor-level form the quantizing row builds on.
    """
    del block_size
    return linear(x)
