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

"""MoE experts LoRA kernel family.

``shared`` is one LoRA pair per logical spec across experts. ``independent``
is a per-expert pair. Eager is the per-expert loop. Triton and NPU wrap the
local fused Functions. ``quack`` / ``mlu`` are not registered; callers remap
those ``moe_implementation`` values to eager.
"""

from ...registry import register_kernel
from ...requirement import CudaKernelRequirement, NpuKernelRequirement
from .independent import eager as independent_eager
from .independent import npu as independent_npu
from .independent import triton as independent_triton
from .shared import eager as shared_eager
from .shared import npu as shared_npu
from .shared import triton as shared_triton


register_kernel("moe_experts_lora", "shared", "eager", wrapper=shared_eager.wrapper)

register_kernel(
    "moe_experts_lora",
    "shared",
    "triton",
    wrapper=shared_triton.wrapper,
    requirement=CudaKernelRequirement(min_cc=70),
)

register_kernel(
    "moe_experts_lora",
    "shared",
    "npu",
    wrapper=shared_npu.wrapper,
    requirement=NpuKernelRequirement(),
)

register_kernel("moe_experts_lora", "independent", "eager", wrapper=independent_eager.wrapper)

register_kernel(
    "moe_experts_lora",
    "independent",
    "triton",
    wrapper=independent_triton.wrapper,
    requirement=CudaKernelRequirement(min_cc=70),
)

register_kernel(
    "moe_experts_lora",
    "independent",
    "npu",
    wrapper=independent_npu.wrapper,
    requirement=NpuKernelRequirement(),
)
