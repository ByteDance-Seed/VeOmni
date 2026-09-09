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
# See the License for the specific language governing limitations
# under the License.

"""MoE experts kernel family.

``standard`` is routed SwiGLU (split or merged fc1). ``gpt_oss`` is the
interleaved gate/up layout with bias and ``alpha`` / ``limit``. Eager is
the HF expert loop (regular autograd). Fused rows wrap the local Functions.
"""

from ...platform import (
    NVIDIA_SM70_PLUS,
    NVIDIA_SM90_PLUS,
    ROCM_GPU,
    GpuKernelRequirement,
    MluKernelRequirement,
    NpuKernelRequirement,
)
from ...registry import register_op
from .gpt_oss import eager as gpt_oss_eager
from .gpt_oss import quack as gpt_oss_quack
from .standard import eager as standard_eager
from .standard import mlu as standard_mlu
from .standard import npu as standard_npu
from .standard import quack as standard_quack
from .standard import triton as standard_triton


_GPU_SM70_OR_ROCM = GpuKernelRequirement(platforms=(NVIDIA_SM70_PLUS, ROCM_GPU))
_NVIDIA_SM90_PLUS = GpuKernelRequirement(platforms=(NVIDIA_SM90_PLUS,))


register_op(
    "moe_experts",
    "standard",
    "eager",
    description="PyTorch routed SwiGLU MoE experts with split or merged gate/up weights",
    wrapper=standard_eager.wrapper,
)

register_op(
    "moe_experts",
    "standard",
    "fused_triton",
    description="Triton grouped-GEMM routed SwiGLU MoE experts",
    wrapper=standard_triton.wrapper,
    requirement=_GPU_SM70_OR_ROCM,
)

register_op(
    "moe_experts",
    "standard",
    "fused_triton",
    description="Triton grouped-GEMM routed SwiGLU MoE experts",
    wrapper=standard_triton.wrapper,
    requirement=MluKernelRequirement(),
)

register_op(
    "moe_experts",
    "standard",
    "fused_quack",
    description="Quack CUTLASS/CuTe routed SwiGLU MoE experts",
    wrapper=standard_quack.wrapper,
    requirement=_NVIDIA_SM90_PLUS,
)

register_op(
    "moe_experts",
    "standard",
    "fused_npu",
    description="torch_npu grouped-GEMM routed SwiGLU MoE experts",
    wrapper=standard_npu.wrapper,
    requirement=NpuKernelRequirement(),
)

register_op(
    "moe_experts",
    "standard",
    "fused_mlu",
    description="Apex grouped-GEMM routed SwiGLU MoE experts",
    wrapper=standard_mlu.wrapper,
    requirement=MluKernelRequirement(),
)

register_op(
    "moe_experts",
    "gpt_oss",
    "eager",
    description="PyTorch GPT-OSS MoE experts with interleaved gate/up weights and bias",
    wrapper=gpt_oss_eager.wrapper,
)

register_op(
    "moe_experts",
    "gpt_oss",
    "fused_quack",
    description="Quack CUTLASS/CuTe GPT-OSS MoE experts with interleaved gate/up weights and bias",
    wrapper=gpt_oss_quack.wrapper,
    requirement=_NVIDIA_SM90_PLUS,
)
