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

"""Gated MLP kernel family.

Variant ``standard`` is ``down(act(gate(x)) * up(x))`` for any activation
``veomni.ops.activation`` evaluates, with the optional DSV4 clamp. Variant
``geglu`` is the same shape fixed to ``gelu_pytorch_tanh``.

Every row takes the **MLP module** and returns the block's output, so a model's
forward is one handle call with no branch: the row decides for itself whether it
can read the projection weights or has to run the projection modules. See
``._module`` for that decision and why it belongs to the row.
"""

from ...platform import NVIDIA_SM90_PLUS, GpuKernelRequirement, MluKernelRequirement
from ...registry import register_op
from .geglu import eager as geglu_eager
from .geglu import liger_kernel as geglu_liger
from .standard import eager as standard_eager
from .standard import liger_kernel as standard_liger
from .standard import qat as standard_qat
from .standard import torchao_nvfp4 as standard_torchao_nvfp4


_GPU = GpuKernelRequirement()
_MLU = MluKernelRequirement()
_NVIDIA_SM90_PLUS = GpuKernelRequirement(platforms=(NVIDIA_SM90_PLUS,))

register_op(
    "gated_mlp",
    "standard",
    "eager",
    wrapper=standard_eager.wrapper,
    description="PyTorch gated MLP with optional activation clamping, autograd backward",
)

register_op(
    "gated_mlp",
    "standard",
    "liger_kernel",
    wrapper=standard_liger.wrapper,
    description="Liger Kernel SwiGLU MLP with optional activation clamping",
    requirement=_GPU,
    requires=("liger_kernel",),
    default=True,
)

register_op(
    "gated_mlp",
    "standard",
    "liger_kernel",
    wrapper=standard_liger.wrapper,
    description="Liger Kernel SwiGLU MLP with optional activation clamping",
    requirement=_MLU,
    requires=("liger_kernel",),
    default=True,
)

register_op(
    "gated_mlp",
    "standard",
    "qat",
    description="TileLang blockwise FP8 fake-quantized SwiGLU MLP with optional activation clamping",
    wrapper=standard_qat.wrapper,
    requirement=_NVIDIA_SM90_PLUS,
    requires=("tilelang",),
)

register_op(
    "gated_mlp",
    "standard",
    "torchao_nvfp4",
    description="TorchAO NVFP4 fake-quantized SwiGLU MLP with optional activation clamping",
    wrapper=standard_torchao_nvfp4.wrapper,
    requires=("torchao",),
)


register_op(
    "gated_mlp",
    "geglu",
    "eager",
    wrapper=geglu_eager.wrapper,
    description="PyTorch GeGLU MLP with tanh-approximate GELU",
)

register_op(
    "gated_mlp",
    "geglu",
    "liger_kernel",
    wrapper=geglu_liger.wrapper,
    description="Liger Kernel GeGLU MLP with tanh-approximate GELU",
    requirement=_GPU,
    requires=("liger_kernel",),
    default=True,
)

register_op(
    "gated_mlp",
    "geglu",
    "liger_kernel",
    wrapper=geglu_liger.wrapper,
    description="Liger Kernel GeGLU MLP with tanh-approximate GELU",
    requirement=_MLU,
    requires=("liger_kernel",),
    default=True,
)
