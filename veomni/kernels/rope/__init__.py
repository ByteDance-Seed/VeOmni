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

"""RoPE kernel family.

cuda
- full
  - eager
  - liger_kernel
- partial
  - eager
- deepseek_v4
  - eager
  - triton
- wan
  - eager
  - triton

npu
- full
  - eager
  - torch_npu
- partial
  - eager
  - torch_npu
- wan
  - eager
  - torch_npu
"""

from ..platform import CudaKernelRequirement, NpuKernelRequirement
from ..registry import register_kernel
from .deepseek_v4 import eager as dsv4_eager
from .deepseek_v4 import triton as dsv4_triton
from .full import eager as full_eager
from .full import liger_kernel as full_liger
from .full import torch_npu as full_npu
from .partial import eager as partial_eager
from .partial import torch_npu as partial_npu
from .wan import eager as wan_eager
from .wan import torch_npu as wan_npu
from .wan import triton as wan_triton


register_kernel("rope", "full", "eager", full_eager.forward, full_eager.backward)

register_kernel(
    "rope",
    "full",
    "liger_kernel",
    full_liger.forward,
    full_liger.backward,
    requirement=CudaKernelRequirement(),
)

register_kernel(
    "rope",
    "full",
    "torch_npu",
    full_npu.forward,
    full_npu.backward,
    requirement=NpuKernelRequirement(),
)

register_kernel("rope", "partial", "eager", partial_eager.forward, partial_eager.backward)

register_kernel(
    "rope",
    "partial",
    "torch_npu",
    partial_npu.forward,
    partial_npu.backward,
    requirement=NpuKernelRequirement(),
)

register_kernel("rope", "deepseek_v4", "eager", dsv4_eager.forward, dsv4_eager.backward)

register_kernel(
    "rope",
    "deepseek_v4",
    "triton",
    dsv4_triton.forward,
    dsv4_triton.backward,
    requirement=CudaKernelRequirement(),
)

register_kernel("rope", "wan", "eager", wan_eager.forward, wan_eager.backward)

register_kernel(
    "rope",
    "wan",
    "triton",
    wan_triton.forward,
    wan_triton.backward,
    requirement=CudaKernelRequirement(),
)

register_kernel(
    "rope",
    "wan",
    "torch_npu",
    wan_npu.forward,
    wan_npu.backward,
    requirement=NpuKernelRequirement(),
)
