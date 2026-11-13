# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Linear projection kernel family.

Variant ``standard`` is a 2D linear projection ``x @ weight.T + bias``. Variant
``grouped`` is a block-diagonal grouped projection over ``n_groups`` head
groups.

Rows take the projection **module**, not raw tensors: ``op(self.q_a_proj, x)``.
``_module.resolve_projection`` decides how to reach the weight, so a model never
has to know whether it is looking at an ``nn.Linear`` or a wrapped one, and never
branches on the row itself.

The ``qat`` row sits next to the eager reference: it fake-quantizes both GEMM
operands with the DeepSeek-V4 TileLang quantizers and keeps the multiply in the
model dtype, which is what an emulated low-precision deployment GEMM computes.
``torchao_nvfp4`` does the same with torchao's NVFP4 round trip; it is
``standard`` only, so a model that also binds ``grouped`` (DeepSeek-V4) cannot
select it.
"""

from ...platform import NVIDIA_SM90_PLUS, GpuKernelRequirement
from ...registry import register_op
from .grouped import eager as grouped_eager
from .grouped import qat as grouped_qat
from .standard import eager as standard_eager
from .standard import qat as standard_qat
from .standard import torchao_nvfp4 as standard_torchao_nvfp4


_NVIDIA_SM90_PLUS = GpuKernelRequirement(platforms=(NVIDIA_SM90_PLUS,))


register_op(
    "linear",
    "standard",
    "eager",
    description="PyTorch dense linear projection",
    wrapper=standard_eager.wrapper,
)

register_op(
    "linear",
    "standard",
    "qat",
    description="TileLang blockwise FP8 fake-quantized dense linear projection",
    wrapper=standard_qat.wrapper,
    requirement=_NVIDIA_SM90_PLUS,
    requires=("tilelang",),
)

register_op(
    "linear",
    "standard",
    "torchao_nvfp4",
    description="TorchAO NVFP4 fake-quantized dense linear projection",
    wrapper=standard_torchao_nvfp4.wrapper,
    requires=("torchao",),
)

register_op(
    "linear",
    "grouped",
    "eager",
    description="PyTorch block-diagonal grouped linear projection",
    wrapper=grouped_eager.module_wrapper,
)

register_op(
    "linear",
    "grouped",
    "qat",
    description="TileLang blockwise FP8 fake-quantized grouped linear projection",
    wrapper=grouped_qat.module_wrapper,
    requirement=_NVIDIA_SM90_PLUS,
    requires=("tilelang",),
)
