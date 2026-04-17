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

import triton
import triton.language as tl


class ActivationType(str, Enum):
    GELU = "gelu"
    GELU_NEW = "gelu_new"  # gelu with tanh approximation
    SILU = "silu"


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


_sqrt2pi: triton.language.constexpr = math.sqrt(2.0 / math.pi)
_sqrt1_2: triton.language.constexpr = math.sqrt(1.0 / 2)
_gaussian_pdf_normalization: triton.language.constexpr = 1.0 / math.sqrt(2 * math.pi)


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
