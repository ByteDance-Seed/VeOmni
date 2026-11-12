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

"""Hopper/Blackwell DeepGEMM FP8 dense linear forward and backward."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .....utils.device import get_gpu_compute_capability, is_nvidia_cuda_device
from .._module import resolve_projection


#: GEMM channels this row requires: DeepGEMM's blockwise FP8 recipe is 128 wide on
#: both the reduction and the output axis, so neither dimension may be partial.
_BLOCK = 128


def linear(x: Tensor, weight: Tensor, bias: Tensor, *, block_size: int = 128) -> Tensor:
    """Run FP8 dense linear GEMMs with BF16 inputs, outputs, and first-order gradients.

    Input and output channel dimensions must be positive multiples of 128.
    An empty ``bias`` tensor denotes an unused bias.
    """
    del block_size
    if not is_nvidia_cuda_device(x.device) or get_gpu_compute_capability(x.device) not in (90, 100):
        raise RuntimeError("deepgemm_fp8 requires NVIDIA Hopper SM90 or Blackwell SM100.")
    if x.ndim < 2 or x.dtype != torch.bfloat16:
        raise ValueError("deepgemm_fp8 requires BF16 input with at least 2 dimensions.")
    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or weight.device != x.device:
        raise ValueError("deepgemm_fp8 requires a 2D BF16 weight on the same device as input.")
    out_features, in_features = weight.shape
    if x.shape[-1] != in_features:
        raise ValueError(f"Input last dimension {x.shape[-1]} does not match weight in_features {in_features}.")
    if min(out_features, in_features) == 0 or out_features % _BLOCK or in_features % _BLOCK:
        raise ValueError("FP8 dense linear requires positive dimensions and in/out multiples of 128.")
    if bias.numel() != 0 and (bias.shape != (out_features,) or bias.device != x.device):
        raise ValueError("Non-empty bias must have shape [out_features] on the same device as input.")

    from ...moe_experts.shared.deepgemm_fp8 import GroupedLinear, get_uniform_layout

    x_2d = x.reshape(-1, in_features)
    layout = get_uniform_layout(x_2d.shape[0], 1, x.device)
    # Reuse only the parameter's storage: a cast, a dequantization or a
    # ``contiguous()`` copy would keep a second BF16 weight alive across backward,
    # which is the largest temporary this row can create.
    recompute_weight = isinstance(weight, torch.nn.Parameter) and weight.is_contiguous()
    w_3d = weight.contiguous().unsqueeze(0)
    out_2d = GroupedLinear.apply(x_2d, w_3d, layout, recompute_weight)
    if bias.numel() > 0:
        out_2d = out_2d + bias.to(out_2d.dtype)
    return out_2d.reshape(*x.shape[:-1], out_features)


def wrapper(linear_module: nn.Module, x: Tensor, *, block_size: int = 128) -> Tensor:
    """Row entry point: reach ``linear_module``'s weight and run the FP8 GEMM.

    This row quantizes the weight it multiplies, so it needs a projection that
    owns one; ``resolve_projection`` refuses a wrapped projection rather than
    falling back to its full-precision forward.
    """
    weight, shortcut = resolve_projection("deepgemm_fp8", linear_module, x)
    assert shortcut is None
    bias = getattr(linear_module, "bias", None)
    return linear(x, weight, bias if bias is not None else weight.new_empty(0), block_size=block_size)
