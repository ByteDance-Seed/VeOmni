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

"""SwiGLU MLP kernel family.

Variant ``standard`` is ``down(silu(gate(x)) * up(x))`` with optional DSV4
clamp. Variant ``geglu`` is ``down(gelu_pytorch_tanh(gate(x)) * up(x))``.
Empty biases are unused.
"""

from ...platform import GpuKernelRequirement
from ...registry import register_op
from .geglu import eager as geglu_eager
from .geglu import liger_kernel as geglu_liger
from .standard import eager as standard_eager
from .standard import liger_kernel as standard_liger


_GPU = GpuKernelRequirement()


register_op(
    "swiglu_mlp",
    "standard",
    "eager",
    standard_eager.forward,
    standard_eager.backward,
    description="PyTorch SwiGLU MLP with optional activation clamping",
)

register_op(
    "swiglu_mlp",
    "standard",
    "liger_kernel",
    standard_liger.forward,
    standard_liger.backward,
    description="Liger Kernel SwiGLU MLP with optional activation clamping",
    requirement=_GPU,
    requires=("liger_kernel",),
)

register_op(
    "swiglu_mlp",
    "geglu",
    "eager",
    geglu_eager.forward,
    geglu_eager.backward,
    description="PyTorch GeGLU MLP with tanh-approximate GELU",
)

register_op(
    "swiglu_mlp",
    "geglu",
    "liger_kernel",
    geglu_liger.forward,
    geglu_liger.backward,
    description="Liger Kernel GeGLU MLP with tanh-approximate GELU",
    requirement=_GPU,
    requires=("liger_kernel",),
)
