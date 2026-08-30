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

"""SwiGLU kernel family.

Variant ``standard`` is ``silu(gate) * up``. The three MLP linears stay in
the module, matching ``veomni.ops.liger``'s Liger factory.
"""

from ...registry import register_kernel
from ...requirement import CudaKernelRequirement
from .standard import eager as standard_eager
from .standard import liger_kernel as standard_liger


register_kernel("swiglu_mlp", "standard", "eager", standard_eager.forward, standard_eager.backward)

register_kernel(
    "swiglu_mlp",
    "standard",
    "liger_kernel",
    standard_liger.forward,
    standard_liger.backward,
    requirement=CudaKernelRequirement(),
)
