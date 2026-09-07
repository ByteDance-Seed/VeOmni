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

"""Reusable mathematical backward units for asynchronous Ulysses schedules.

The functions in this module do not launch communication or retain autograd
context. Async Ulysses implementations remain responsible for arranging these
units around collective launch and wait points.
"""

import importlib
from typing import Optional

import torch
from torch import Tensor


_fused_layer_norm_cuda = None


def _flatten_linear_operands(
    grad_output: Tensor,
    input_tensor: Tensor,
    weight: Tensor,
) -> tuple[Tensor, Tensor]:
    grad_output_2d = grad_output.reshape(-1, weight.shape[0])
    input_2d = input_tensor.reshape(-1, weight.shape[1])
    if grad_output_2d.shape[0] != input_2d.shape[0]:
        raise ValueError(
            "Linear backward requires matching input and output row counts, got "
            f"{input_2d.shape[0]} and {grad_output_2d.shape[0]}."
        )
    return grad_output_2d, input_2d


def linear_input_backward(grad_output: Tensor, input_tensor: Tensor, weight: Tensor) -> Tensor:
    """Compute the input gradient of a linear projection."""
    grad_output_2d, _ = _flatten_linear_operands(grad_output, input_tensor, weight)
    return (grad_output_2d @ weight).reshape_as(input_tensor)


def linear_parameter_backward(
    grad_output: Tensor,
    input_tensor: Tensor,
    weight: Tensor,
    *,
    has_bias: bool,
) -> tuple[Tensor, Optional[Tensor]]:
    """Compute the weight and optional bias gradients of a linear projection."""
    grad_output_2d, input_2d = _flatten_linear_operands(grad_output, input_tensor, weight)
    grad_weight = grad_output_2d.transpose(0, 1) @ input_2d
    grad_bias = grad_output_2d.sum(dim=0) if has_bias else None
    return grad_weight, grad_bias


def linear_backward(
    grad_output: Tensor,
    input_tensor: Tensor,
    weight: Tensor,
    *,
    has_bias: bool,
) -> tuple[Tensor, Tensor, Optional[Tensor]]:
    """Compute input, weight, and optional bias gradients for a linear projection."""
    grad_output_2d, input_2d = _flatten_linear_operands(grad_output, input_tensor, weight)
    grad_input = (grad_output_2d @ weight).reshape_as(input_tensor)
    grad_weight = grad_output_2d.transpose(0, 1) @ input_2d
    grad_bias = grad_output_2d.sum(dim=0) if has_bias else None
    return grad_input, grad_weight, grad_bias


def _get_fused_layer_norm_cuda():
    global _fused_layer_norm_cuda
    if _fused_layer_norm_cuda is None:
        _fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
    return _fused_layer_norm_cuda


def layer_norm_backward(
    grad_output: Tensor,
    input_tensor: Tensor,
    mean: Tensor,
    invvar: Tensor,
    weight: Tensor,
    bias: Tensor,
    normalized_shape: torch.Size,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute affine LayerNorm gradients with the fixed CUDA kernel."""
    return _get_fused_layer_norm_cuda().backward_affine(
        grad_output.contiguous(),
        mean,
        invvar,
        input_tensor,
        normalized_shape,
        weight,
        bias,
        eps,
        False,
    )


def reduce_repeated_kv_gradient(
    grad_output: Tensor,
    original_num_heads: int,
    repeats: int,
    *,
    head_dimension: int,
) -> Tensor:
    """Reduce gradients from repeated KV heads back to their original heads."""
    if repeats == 1:
        return grad_output
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}.")

    head_dimension %= grad_output.ndim
    expected_heads = original_num_heads * repeats
    if grad_output.shape[head_dimension] != expected_heads:
        raise ValueError(
            f"Repeated KV head dimension must have size {expected_heads}, got {grad_output.shape[head_dimension]}."
        )

    shape = list(grad_output.shape)
    shape[head_dimension : head_dimension + 1] = [original_num_heads, repeats]
    return grad_output.reshape(shape).sum(dim=head_dimension + 1)


__all__ = [
    "layer_norm_backward",
    "linear_backward",
    "linear_input_backward",
    "linear_parameter_backward",
    "reduce_repeated_kv_gradient",
]
