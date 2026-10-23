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

"""situ_and_mul kernel family (Kimi K3 SituGLU activation).

``eager`` is the fp32-compute PyTorch reference. ``triton`` is the fused
Triton (Triton-Ascend) kernel vendored from MindSpeed-MM: fused forward +
backward with fp32 compute inside the kernel and fp16/bf16 input/output.
"""

import torch
from torch import Tensor

from ...platform import NpuKernelRequirement
from ...registry import register_op


def _eager(x: Tensor, beta: float = 1.0, linear_beta: float | None = None) -> Tensor:
    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float32)
    up = x[..., d:].to(torch.float32)
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (situ_a * up).to(x.dtype)


def _triton(x: Tensor, beta: float = 1.0, linear_beta: float | None = None) -> Tensor:
    from .situ_triton import situ_and_mul

    return situ_and_mul(x, beta=beta, linear_beta=linear_beta)


register_op(
    "situ_and_mul",
    "kimi_k3",
    "eager",
    wrapper=_eager,
    description="Kimi K3 SituGLU activation PyTorch reference (fp32 compute)",
)

register_op(
    "situ_and_mul",
    "kimi_k3",
    "triton",
    wrapper=_triton,
    description="Kimi K3 SituGLU fused Triton kernel (Triton-Ascend on NPU)",
    requirement=NpuKernelRequirement(),
    requires=("triton",),
    default=True,
)
