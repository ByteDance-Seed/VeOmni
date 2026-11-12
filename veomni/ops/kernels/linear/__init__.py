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

Two rows quantize the operands they multiply, and they are not the same thing:

* ``qat`` fake-quantizes both GEMM operands with the DeepSeek-V4 TileLang
  quantizers and keeps the multiply in the model dtype, which is what an emulated
  low-precision deployment GEMM computes.
* ``deepgemm_fp8`` quantizes them and then multiplies on the FP8 tensor cores
  through DeepGEMM, which is real low-precision compute rather than an emulation.
  It covers forward, input gradient and weight gradient.

torchao-backed formats are registered separately, on top of this family.
"""

from ...platform import NVIDIA_SM90_PLUS, GpuKernelRequirement, NvidiaGpuPlatform
from ...registry import register_op
from .grouped import deepgemm as grouped_deepgemm
from .grouped import eager as grouped_eager
from .grouped import qat as grouped_qat
from .standard import deepgemm as standard_deepgemm
from .standard import eager as standard_eager
from .standard import qat as standard_qat


_NVIDIA_SM90_PLUS = GpuKernelRequirement(platforms=(NVIDIA_SM90_PLUS,))
#: DeepGEMM ships Hopper and Blackwell kernels only. SM90 and SM100 are named
#: explicitly so an SM89 (FP8-capable but unsupported by the library) or an SM120
#: host is refused at resolve time instead of failing inside the first launch.
_NVIDIA_SM90_OR_SM100 = GpuKernelRequirement(
    platforms=(NvidiaGpuPlatform(min_cc=90, max_cc=90), NvidiaGpuPlatform(min_cc=100, max_cc=100)),
)


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
    "deepgemm_fp8",
    description="DeepGEMM blockwise FP8 dense linear forward, input gradients and weight gradients",
    wrapper=standard_deepgemm.wrapper,
    requirement=_NVIDIA_SM90_OR_SM100,
    requires=("deep_gemm", "triton"),
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

register_op(
    "linear",
    "grouped",
    "deepgemm_fp8",
    description="DeepGEMM blockwise FP8 grouped linear forward, input gradients and weight gradients",
    wrapper=grouped_deepgemm.wrapper,
    requirement=_NVIDIA_SM90_OR_SM100,
    requires=("deep_gemm", "triton"),
)
