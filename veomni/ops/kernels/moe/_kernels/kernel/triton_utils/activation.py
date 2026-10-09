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

import math
from enum import Enum

import torch
import triton
import triton.language as tl

from .......utils.device import get_torch_device


class ActivationType(str, Enum):
    GELU = "gelu"
    GELU_NEW = "gelu_new"  # gelu with tanh approximation
    SILU = "silu"


_ACTIVATION_ALIASES = {
    "gelu": ActivationType.GELU.value,
    "gelu_new": ActivationType.GELU_NEW.value,
    "gelu_pytorch_tanh": ActivationType.GELU_NEW.value,
    "gelu_tanh": ActivationType.GELU_NEW.value,
    "silu": ActivationType.SILU.value,
    # HF's own alias for SiLU (``ACT2FN["swish"] is ACT2FN["silu"]``). Omitting
    # it would refuse a config that is bit-for-bit the kernel's default.
    "swish": ActivationType.SILU.value,
}


def normalize_activation(activation: str | ActivationType) -> str:
    """Map model-config activation names to the Triton kernel names."""
    activation_name = activation.value if isinstance(activation, ActivationType) else activation
    try:
        return _ACTIVATION_ALIASES[activation_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_ACTIVATION_ALIASES))
        raise ValueError(
            f"Unsupported fused MoE activation {activation_name!r}. Expected one of: {supported}."
        ) from exc


@triton.jit
def activation_fwd(x: tl.tensor, ACTIVATION: tl.constexpr):
    orig_dtype = x.dtype
    x = x.to(tl.float32)
    if ACTIVATION == "gelu":
        y = gelu(x)
    elif ACTIVATION == "gelu_new":
        y = gelu_new(x)
    elif ACTIVATION == "silu":
        y = silu(x)
    else:
        tl.static_assert(False, f"Unsupported activation of {ACTIVATION}")
    return y.to(orig_dtype)


@triton.jit
def activation_bwd(dy: tl.tensor, x: tl.tensor, ACTIVATION: tl.constexpr):
    orig_dtype = dy.dtype
    x = x.to(tl.float32)
    dy = dy.to(tl.float32)
    if ACTIVATION == "gelu":
        dx = dy * gelu_grad(x)
    elif ACTIVATION == "gelu_new":
        dx = dy * gelu_new_grad(x)
    elif ACTIVATION == "silu":
        dx = dy * silu_grad(x)
    else:
        tl.static_assert(False, f"Unsupported activation of {ACTIVATION}")
    return dx.to(orig_dtype)


@triton.jit
def _activation_forward_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    num_columns: tl.constexpr,
    input_row_stride: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    input_offsets = (offsets // num_columns) * input_row_stride + offsets % num_columns
    values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, activation_fwd(values, ACTIVATION), mask=mask)


@triton.jit
def _activation_backward_kernel(
    grad_output_ptr,
    input_ptr,
    grad_input_ptr,
    num_elements,
    num_columns: tl.constexpr,
    input_row_stride: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
    input_offsets = (offsets // num_columns) * input_row_stride + offsets % num_columns
    values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    tl.store(grad_input_ptr + offsets, activation_bwd(grad_output, values, ACTIVATION), mask=mask)


def apply_activation(input: torch.Tensor, activation: str | ActivationType) -> torch.Tensor:
    """Apply the configured fused-MoE activation."""
    activation = normalize_activation(activation)
    if activation == ActivationType.SILU.value:
        return torch.ops.aten.silu(input)

    assert input.is_cuda, "Triton activation requires a CUDA tensor."
    assert input.ndim == 2 and input.stride(1) == 1, (
        f"Triton activation requires a row-major 2D tensor, got shape={input.shape}, stride={input.stride()}."
    )
    assert input.dtype in (torch.bfloat16, torch.float16), input.dtype

    output = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    if input.numel() == 0:
        return output

    block_size = 1024
    grid = (triton.cdiv(input.numel(), block_size),)
    with get_torch_device().device(input.device):
        _activation_forward_kernel[grid](
            input_ptr=input,
            output_ptr=output,
            num_elements=input.numel(),
            num_columns=input.shape[1],
            input_row_stride=input.stride(0),
            ACTIVATION=activation,
            BLOCK_SIZE=block_size,
        )
    return output


def apply_activation_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    activation: str | ActivationType,
) -> torch.Tensor:
    """Apply the derivative of the configured fused-MoE activation."""
    activation = normalize_activation(activation)
    if activation == ActivationType.SILU.value:
        return torch.ops.aten.silu_backward(grad_output, input)

    assert grad_output.shape == input.shape, (grad_output.shape, input.shape)
    assert grad_output.device == input.device, (grad_output.device, input.device)
    assert grad_output.dtype == input.dtype, (grad_output.dtype, input.dtype)
    assert grad_output.is_cuda, "Triton activation backward requires CUDA tensors."
    assert grad_output.is_contiguous(), "Triton activation backward requires contiguous grad_output."
    assert input.ndim == 2 and input.stride(1) == 1, (
        f"Triton activation backward requires a row-major 2D input, got shape={input.shape}, stride={input.stride()}."
    )
    assert input.dtype in (torch.bfloat16, torch.float16), input.dtype

    grad_input = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    if input.numel() == 0:
        return grad_input

    block_size = 1024
    grid = (triton.cdiv(input.numel(), block_size),)
    with get_torch_device().device(input.device):
        _activation_backward_kernel[grid](
            grad_output_ptr=grad_output,
            input_ptr=input,
            grad_input_ptr=grad_input,
            num_elements=input.numel(),
            num_columns=input.shape[1],
            input_row_stride=input.stride(0),
            ACTIVATION=activation,
            BLOCK_SIZE=block_size,
        )
    return grad_input


_sqrt2pi = tl.constexpr(math.sqrt(2.0 / math.pi))
_sqrt1_2 = tl.constexpr(math.sqrt(1.0 / 2))
_gaussian_pdf_normalization = tl.constexpr(1.0 / math.sqrt(2 * math.pi))


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def gelu(x):
    """Gaussian Error Linear Unit (GELU)"""
    x = x.to(tl.float32)
    return x * 0.5 * (1.0 + tl.erf(x * _sqrt1_2))


@triton.jit
def gelu_grad(x):
    x = x.to(tl.float32)
    cdf = 0.5 * (1.0 + tl.erf(x * _sqrt1_2))
    pdf = tl.exp(-0.5 * x * x) * _gaussian_pdf_normalization
    return cdf + x * pdf


@triton.jit
def gelu_new(x):
    """
    GeLU_ activation - Gaussian error linear unit, with tanh approximation

    .. _GeLU: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1.0 + tanh(_sqrt2pi * x * (1.0 + 0.044715 * x * x)))


@triton.jit
def gelu_new_grad(x):
    # CREDITS: Fast implementation proposed in
    # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/fused_bias_gelu.py#L30
    x = x.to(tl.float32)
    tanh_out = tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    return 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)


@triton.jit
def silu(x):
    """https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html"""
    x = x.to(tl.float32)
    return x * tl.sigmoid(x)


@triton.jit
def silu_grad(x):
    x = x.to(tl.float32)
    f = tl.sigmoid(x)
    return f + x * (f - f * f)
