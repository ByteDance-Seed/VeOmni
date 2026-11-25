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
from .._module import resolve_projection
from ..standard.deepgemm import _BLOCK, SUPPORTED_DTYPES


def wrapper(linear: nn.Module, x: Tensor, *, dtype: str = "fp8", block_size: int | None = None) -> Tensor:
    """Run block-diagonal grouped FP8 GEMMs with BF16 inputs, outputs, and gradients.

    ``x`` has shape ``[..., n_groups, in_per_group]`` and the projection's weight
    ``[n_groups * out_per_group, in_per_group]``. Both per-group dimensions must be
    positive multiples of 128.

    The group count stays a property of the module (``n_groups``), not of the call
    site. ``allow_linear_subclass`` is on because the grouped layout is a property
    of the subclass's own weight, not extra math in its ``forward`` -- see
    :func:`.._module.resolve_projection` and the ``qat`` grouped row, which takes
    the same position.
    """
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"linear/grouped/deepgemm supports dtype={SUPPORTED_DTYPES}, got {dtype!r}. "
            "DeepGEMM ships blockwise FP8 kernels only."
        )
    if block_size is not None and block_size != _BLOCK:
        raise ValueError(
            f"linear/grouped/deepgemm supports only block_size={_BLOCK}, got {block_size}. "
            "The block width is part of DeepGEMM's blockwise FP8 recipe, not a per-call choice."
        )
    weight, shortcut = resolve_projection("deepgemm", linear, x, allow_linear_subclass=True)
    assert shortcut is None
    bias = getattr(linear, "bias", None)
    if bias is None:
        bias = weight.new_empty(0)
    n_groups = linear.n_groups

    if not is_nvidia_cuda_device(x.device) or get_gpu_compute_capability(x.device) not in (90, 100):
        raise RuntimeError("deepgemm requires NVIDIA Hopper SM90 or Blackwell SM100.")
    if x.ndim < 2 or x.dtype != torch.bfloat16:
        raise ValueError("deepgemm requires BF16 input with at least 2 dimensions.")
    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or weight.device != x.device:
        raise ValueError("deepgemm requires a 2D BF16 weight on the same device as input.")
    if n_groups <= 0 or weight.shape[0] % n_groups != 0:
        raise ValueError(f"weight shape {tuple(weight.shape)} is incompatible with n_groups={n_groups}.")
    out_per_group = weight.shape[0] // n_groups
    in_per_group = weight.shape[1]
    if x.shape[-2] != n_groups or x.shape[-1] != in_per_group:
        raise ValueError(f"Expected input shape [..., {n_groups}, {in_per_group}], got {tuple(x.shape)}.")
    if min(out_per_group, in_per_group) == 0 or out_per_group % _BLOCK or in_per_group % _BLOCK:
        raise ValueError("FP8 grouped linear requires positive per-group dimensions divisible by 128.")
    if bias.numel() != 0 and (bias.shape != (weight.shape[0],) or bias.device != x.device):
        raise ValueError("Non-empty bias must have shape [n_groups * out_per_group] on the same device as input.")

    from ...moe_experts.shared.deepgemm import GroupedLinear, get_uniform_layout

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
