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

"""Hopper/Blackwell DeepGEMM FP8 block-diagonal grouped linear forward and backward."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .....utils.device import get_gpu_compute_capability, is_nvidia_cuda_device
from .._module import require_deepgemm_block_size, resolve_projection


def grouped_linear(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    n_groups: int,
    block_size: int = 128,
) -> Tensor:
    """Run FP8 block-diagonal grouped linear GEMMs with BF16 inputs, outputs, and gradients.

    ``x`` has shape ``[..., n_groups, in_per_group]`` and ``weight`` has shape
    ``[n_groups * out_per_group, in_per_group]``. Both ``in_per_group`` and
    ``out_per_group`` must be positive multiples of 128. ``block_size`` is fixed
    by DeepGEMM's blockwise recipe at 128; see the ``standard`` row.
    """
    require_deepgemm_block_size(block_size)
    if not is_nvidia_cuda_device(x.device) or get_gpu_compute_capability(x.device) not in (90, 100):
        raise RuntimeError("deepgemm_fp8 requires NVIDIA Hopper SM90 or Blackwell SM100.")
    if x.ndim < 2 or x.dtype != torch.bfloat16:
        raise ValueError("deepgemm_fp8 requires BF16 input with at least 2 dimensions.")
    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or weight.device != x.device:
        raise ValueError("deepgemm_fp8 requires a 2D BF16 weight on the same device as input.")
    if n_groups <= 0 or weight.shape[0] % n_groups != 0:
        raise ValueError(f"weight shape {tuple(weight.shape)} is incompatible with n_groups={n_groups}.")
    out_per_group = weight.shape[0] // n_groups
    in_per_group = weight.shape[1]
    if x.shape[-2] != n_groups or x.shape[-1] != in_per_group:
        raise ValueError(f"Expected input shape [..., {n_groups}, {in_per_group}], got {tuple(x.shape)}.")
    if min(out_per_group, in_per_group) == 0 or out_per_group % 128 or in_per_group % 128:
        raise ValueError("FP8 grouped linear requires positive per-group dimensions divisible by 128.")
    if bias.numel() != 0 and (bias.shape != (weight.shape[0],) or bias.device != x.device):
        raise ValueError("Non-empty bias must have shape [n_groups * out_per_group] on the same device as input.")

    from ...moe_experts.shared.deepgemm_fp8 import GroupedLinear, get_uniform_layout

    # One contiguous group per head group: the block-diagonal projection is a
    # grouped GEMM whose groups share the same token count, so the layout is a
    # uniform one and every group's row block is its own head group's tokens.
    n_rows = x.numel() // (n_groups * in_per_group)
    flat_x = x.reshape(n_rows, n_groups, in_per_group).transpose(0, 1).reshape(n_groups * n_rows, in_per_group)
    layout = get_uniform_layout(n_rows, n_groups, x.device)
    recompute_weight = isinstance(weight, torch.nn.Parameter) and weight.is_contiguous()
    w_3d = weight.contiguous().view(n_groups, out_per_group, in_per_group)
    out_flat = GroupedLinear.apply(flat_x, w_3d, layout, recompute_weight)
    out = (
        out_flat.view(n_groups, n_rows, out_per_group).transpose(0, 1).reshape(*x.shape[:-2], n_groups, out_per_group)
    )
    if bias.numel() > 0:
        out = out + bias.to(out.dtype).view(n_groups, out_per_group)
    return out


def wrapper(linear_module: nn.Module, x: Tensor, *, block_size: int = 128) -> Tensor:
    """Row entry point: reach ``linear_module``'s weight and run the grouped FP8 GEMM.

    The group count stays a property of the module (``n_groups``), not of the
    call site, which is the same reason the eager row calls ``module(x)``.
    """
    weight, shortcut = resolve_projection("deepgemm_fp8", linear_module, x)
    assert shortcut is None
    bias = getattr(linear_module, "bias", None)
    return grouped_linear(
        x,
        weight,
        bias if bias is not None else weight.new_empty(0),
        n_groups=linear_module.n_groups,
        block_size=block_size,
    )
