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

"""Linear backward units for async Ulysses schedules.

These helpers do not launch collectives or keep autograd context. The QKV / O
eager pairs arrange them around all-to-all launch and wait points.
"""

from __future__ import annotations

from torch import Tensor


def _align_linear_backward_dtype(
    grad_output: Tensor,
    input_tensor: Tensor,
    weight: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Promote mixed forward/backward dtypes the way ``F.linear`` autograd does.

    Autocast can save FP32 operands and a lower-precision ``grad_output``.
    Match ``F.linear`` autograd: compute in ``grad_output`` dtype, then cast
    grads back to the saved operand dtypes.
    """
    compute_dtype = grad_output.dtype
    return grad_output, input_tensor.to(compute_dtype), weight.to(compute_dtype)


def _flatten_linear_operands(
    grad_output: Tensor,
    input_tensor: Tensor,
    weight: Tensor,
) -> tuple[Tensor, Tensor]:
    """Reshape linear operands to 2D and check that the leading sizes match."""
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
    grad_output, _, weight = _align_linear_backward_dtype(grad_output, input_tensor, weight)
    grad_output_2d, _ = _flatten_linear_operands(grad_output, input_tensor, weight)
    return (grad_output_2d @ weight).reshape_as(input_tensor).to(input_tensor.dtype)


def linear_parameter_backward(
    grad_output: Tensor,
    input_tensor: Tensor,
    weight: Tensor,
    *,
    has_bias: bool,
) -> tuple[Tensor, Tensor | None]:
    """Compute the weight and optional bias gradients of a linear projection."""
    grad_output, input_tensor, _ = _align_linear_backward_dtype(grad_output, input_tensor, weight)
    grad_output_2d, input_2d = _flatten_linear_operands(grad_output, input_tensor, weight)
    grad_weight = (grad_output_2d.transpose(0, 1) @ input_2d).to(weight.dtype)
    grad_bias = grad_output_2d.sum(dim=0).to(weight.dtype) if has_bias else None
    return grad_weight, grad_bias


def linear_backward(
    grad_output: Tensor,
    input_tensor: Tensor,
    weight: Tensor,
    *,
    has_bias: bool,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Compute input, weight, and optional bias gradients for a linear projection."""
    aligned_grad, aligned_input, aligned_weight = _align_linear_backward_dtype(grad_output, input_tensor, weight)
    grad_output_2d, input_2d = _flatten_linear_operands(aligned_grad, aligned_input, aligned_weight)
    grad_input = (grad_output_2d @ aligned_weight).reshape_as(input_tensor).to(input_tensor.dtype)
    grad_weight = (grad_output_2d.transpose(0, 1) @ input_2d).to(weight.dtype)
    grad_bias = grad_output_2d.sum(dim=0).to(weight.dtype) if has_bias else None
    return grad_input, grad_weight, grad_bias


def reduce_repeated_kv_gradient(
    grad_output: Tensor,
    original_num_heads: int,
    repeats: int,
    *,
    head_dimension: int,
) -> Tensor:
    """Sum repeated-KV grads back to ``original_num_heads`` along ``head_dimension``."""
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
    "linear_backward",
    "linear_input_backward",
    "linear_parameter_backward",
    "reduce_repeated_kv_gradient",
]
