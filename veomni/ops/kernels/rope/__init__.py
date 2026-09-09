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

Variants: ``full`` (rotate every channel), ``partial`` (rotate a prefix),
``deepseek_v4`` (trailing interleaved slice), and ``wan`` (complex multiply
by freqs). Each variant registers an eager row plus optional CUDA / NPU
adapters.
"""

from ...platform import GpuKernelRequirement, NpuKernelRequirement
from ...registry import register_op
from .deepseek_v4 import eager as dsv4_eager
from .deepseek_v4 import triton as dsv4_triton
from .full import eager as full_eager
from .full import liger_kernel as full_liger
from .full import npu as full_npu
from .partial import eager as partial_eager
from .partial import npu as partial_npu
from .wan import eager as wan_eager
from .wan import npu as wan_npu
from .wan import triton as wan_triton


_GPU = GpuKernelRequirement()


register_op(
    "rope",
    "full",
    "eager",
    full_eager.forward,
    full_eager.backward,
    description="PyTorch rotary embedding over every channel",
)

register_op(
    "rope",
    "full",
    "liger_kernel",
    full_liger.forward,
    full_liger.backward,
    description="Liger Kernel rotary embedding over every channel",
    requirement=_GPU,
)

register_op(
    "rope",
    "full",
    "npu",
    full_npu.forward,
    full_npu.backward,
    description="torch_npu rotary embedding over every channel",
    requirement=NpuKernelRequirement(),
)

register_op(
    "rope",
    "partial",
    "eager",
    partial_eager.forward,
    partial_eager.backward,
    description="PyTorch rotary embedding over a channel prefix",
)

register_op(
    "rope",
    "partial",
    "npu",
    partial_npu.forward,
    partial_npu.backward,
    description="torch_npu rotary embedding over a channel prefix",
    requirement=NpuKernelRequirement(),
)

register_op(
    "rope",
    "deepseek_v4",
    "eager",
    dsv4_eager.forward,
    dsv4_eager.backward,
    description="PyTorch DeepSeek-V4 rotary embedding over a trailing interleaved slice",
)

register_op(
    "rope",
    "deepseek_v4",
    "triton",
    dsv4_triton.forward,
    dsv4_triton.backward,
    description="Triton DeepSeek-V4 rotary embedding over a trailing interleaved slice",
    requirement=_GPU,
)

register_op(
    "rope",
    "wan",
    "eager",
    wan_eager.forward,
    wan_eager.backward,
    description="PyTorch Wan rotary embedding using complex multiplication",
)

register_op(
    "rope",
    "wan",
    "triton",
    wan_triton.forward,
    wan_triton.backward,
    description="Triton Wan rotary embedding using complex multiplication",
    requirement=_GPU,
)

register_op(
    "rope",
    "wan",
    "npu",
    wan_npu.forward,
    wan_npu.backward,
    description="torch_npu Wan rotary embedding using complex multiplication",
    requirement=NpuKernelRequirement(),
)
