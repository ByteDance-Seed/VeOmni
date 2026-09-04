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

"""RMSNorm kernel family.

Variants: ``standard`` (offset 0, llama-style cast), ``qwen3_5`` (offset 1,
gemma-style fp32 scale), and ``unweighted`` (no affine weight). Each variant
registers an eager row plus optional CUDA / NPU adapters.
"""

from ...registry import register_kernel
from ...requirement import CudaKernelRequirement, NpuKernelRequirement
from .qwen3_5 import eager as qwen3_5_eager
from .qwen3_5 import liger_kernel as qwen3_5_liger
from .qwen3_5 import npu as qwen3_5_npu
from .standard import eager as standard_eager
from .standard import liger_kernel as standard_liger
from .standard import npu as standard_npu
from .standard import triton as standard_triton
from .unweighted import eager as unweighted_eager
from .unweighted import liger_kernel as unweighted_liger


register_kernel("rms_norm", "standard", "eager", standard_eager.forward, standard_eager.backward)

register_kernel(
    "rms_norm",
    "standard",
    "liger_kernel",
    standard_liger.forward,
    standard_liger.backward,
    requirement=CudaKernelRequirement(),
)

register_kernel(
    "rms_norm",
    "standard",
    "npu",
    standard_npu.forward,
    standard_npu.backward,
    requirement=NpuKernelRequirement(),
)

register_kernel(
    "rms_norm",
    "standard",
    "triton",
    standard_triton.forward,
    standard_triton.backward,
    requirement=CudaKernelRequirement(),
)

register_kernel("rms_norm", "qwen3_5", "eager", qwen3_5_eager.forward, qwen3_5_eager.backward)

register_kernel(
    "rms_norm",
    "qwen3_5",
    "liger_kernel",
    qwen3_5_liger.forward,
    qwen3_5_liger.backward,
    requirement=CudaKernelRequirement(),
)

register_kernel(
    "rms_norm",
    "qwen3_5",
    "npu",
    qwen3_5_npu.forward,
    qwen3_5_npu.backward,
    requirement=NpuKernelRequirement(),
)

register_kernel(
    "rms_norm",
    "unweighted",
    "eager",
    unweighted_eager.forward,
    unweighted_eager.backward,
)

register_kernel(
    "rms_norm",
    "unweighted",
    "liger_kernel",
    unweighted_liger.forward,
    unweighted_liger.backward,
    requirement=CudaKernelRequirement(),
)
