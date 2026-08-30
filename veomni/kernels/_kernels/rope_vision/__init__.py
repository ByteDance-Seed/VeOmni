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

"""Vision RoPE kernel family.

``full`` uses the same rotate-every-channel math as ``rope.full``, but the
inputs are ``[S, H, D]`` query/key with ``cos`` / ``sin`` unsqueezed on the
head axis. Registers eager plus an NPU adapter.
"""

from ...registry import register_kernel
from ...requirement import NpuKernelRequirement
from .full import eager as full_eager
from .full import npu as full_npu


register_kernel("rope_vision", "full", "eager", full_eager.forward, full_eager.backward)

register_kernel(
    "rope_vision",
    "full",
    "npu",
    full_npu.forward,
    full_npu.backward,
    requirement=NpuKernelRequirement(),
)
