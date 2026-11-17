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
operands and keeps the multiply in the model dtype, which is what an emulated
low-precision deployment GEMM computes. NVIDIA uses TileLang; AMD gfx942/gfx950
uses the equivalent Triton quantizers.
``torchao`` serves TorchAO's dynamic FP8 GEMMs and FP4 fake quantization; it is
``standard`` only, so a model that also binds ``grouped`` (DeepSeek-V4) uses it
only on ``standard`` sites.

The precision is a row *argument* (``dtype``), not part of the name: ``qat`` is
FP8-only (``dtype="fp8"``), and ``torchao`` serves FP8, NVFP4, and MXFP4.
"""

from ...platform import NVIDIA_SM90_PLUS, ROCM_GFX942_GFX950, GpuKernelRequirement
from ...registry import register_op
from .grouped import eager as grouped_eager
from .grouped import qat as grouped_qat
from .standard import eager as standard_eager
from .standard import qat as standard_qat
from .standard import torchao as standard_torchao


_QAT_GPU = GpuKernelRequirement(platforms=(NVIDIA_SM90_PLUS, ROCM_GFX942_GFX950))


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
    description="Blockwise FP8 fake-quantized dense linear projection",
    wrapper=standard_qat.wrapper,
    requirement=_QAT_GPU,
    requires=("triton",),
)

register_op(
    "linear",
    "standard",
    "torchao",
    description="TorchAO low-precision (FP8 / NVFP4 / MXFP4) dense linear projection",
    wrapper=standard_torchao.wrapper,
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
    description="Blockwise FP8 fake-quantized grouped linear projection",
    wrapper=grouped_qat.module_wrapper,
    requirement=_QAT_GPU,
    requires=("triton",),
)
