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

from ...registry import register_kernel
from ...requirement import CudaKernelRequirement, NpuKernelRequirement
from .gpt_oss import eager as gpt_oss_eager
from .gpt_oss import quack as gpt_oss_quack
from .standard import eager as standard_eager
from .standard import npu as standard_npu
from .standard import quack as standard_quack
from .standard import triton as standard_triton


register_kernel("moe_experts", "standard", "eager", wrapper=standard_eager.wrapper)

register_kernel(
    "moe_experts",
    "standard",
    "triton",
    wrapper=standard_triton.wrapper,
    requirement=CudaKernelRequirement(min_cc=70),
)

register_kernel(
    "moe_experts",
    "standard",
    "quack",
    wrapper=standard_quack.wrapper,
    requirement=CudaKernelRequirement(min_cc=90),
)

register_kernel(
    "moe_experts",
    "standard",
    "npu",
    wrapper=standard_npu.wrapper,
    requirement=NpuKernelRequirement(),
)

register_kernel("moe_experts", "gpt_oss", "eager", wrapper=gpt_oss_eager.wrapper)

register_kernel(
    "moe_experts",
    "gpt_oss",
    "quack",
    wrapper=gpt_oss_quack.wrapper,
    requirement=CudaKernelRequirement(min_cc=90),
)
