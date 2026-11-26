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
``torchao`` serves TorchAO's dynamic FP8 GEMMs and FP4 fake quantization; it is
``standard`` only, so a model that also binds ``grouped`` (DeepSeek-V4) uses it
only on ``standard`` sites.

``deepgemm`` is not a recipe at all: it *is* the deployment GEMM, running forward,
input gradient and weight gradient on the FP8 tensor cores via DeepGEMM. ``qat``
emulates the same rounding and multiplies in the model dtype, so the two are
different kernels behind one field rather than two settings of one kernel.

The precision is a row *argument* (``dtype``), not part of the name: ``qat`` is
FP8-only (``dtype="fp8"``), and ``torchao`` serves FP8, NVFP4, and MXFP4.
"""

from ...platform import NVIDIA_SM90_PLUS, GpuKernelRequirement, NvidiaGpuPlatform
from ...registry import register_op
from .grouped import deepgemm as grouped_deepgemm
from .grouped import eager as grouped_eager
from .grouped import qat as grouped_qat
from .standard import deepgemm as standard_deepgemm
from .standard import eager as standard_eager
from .standard import qat as standard_qat
from .standard import torchao as standard_torchao


_NVIDIA_SM90_PLUS = GpuKernelRequirement(platforms=(NVIDIA_SM90_PLUS,))
#: DeepGEMM ships Hopper and Blackwell kernels only. SM90 and SM100 are named
#: explicitly so an SM89 host -- which has an E4M3 type but no DeepGEMM kernel --
#: is refused at resolve time instead of failing inside the first launch.
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
    description="TileLang blockwise FP8 fake-quantized grouped linear projection",
    wrapper=grouped_qat.module_wrapper,
    requirement=_NVIDIA_SM90_PLUS,
    requires=("tilelang",),
)

register_op(
    "linear",
    "standard",
    "deepgemm",
    description="DeepGEMM blockwise FP8 dense linear forward, input gradients and weight gradients",
    wrapper=standard_deepgemm.wrapper,
    requirement=_NVIDIA_SM90_OR_SM100,
    requires=("deep_gemm", "triton"),
)

register_op(
    "linear",
    "grouped",
    "deepgemm",
    description="DeepGEMM blockwise FP8 grouped linear forward, input gradients and weight gradients",
    wrapper=grouped_deepgemm.wrapper,
    requirement=_NVIDIA_SM90_OR_SM100,
    requires=("deep_gemm", "triton"),
)


def paired_linear(x, linear_a, linear_b):
    """Run two projections sharing input ``x``, fusing work when both handles match.

    Dispatches purely on the bound ``veomni_linear`` handles on ``linear_a`` and
    ``linear_b``: fuses paired ``deepgemm`` (BF16 FP8-training) projections when
    eligible, and otherwise invokes each projection module directly.
    """
    import torch

    op_a = getattr(linear_a, "veomni_linear", None)
    op_b = getattr(linear_b, "veomni_linear", None)
    if (
        op_a is not None
        and op_b is not None
        and op_a.impl == op_b.impl
        and op_a.kwargs == op_b.kwargs
        and op_a.impl == "deepgemm"
        and op_a.kwargs.get("dtype", "fp8") == "fp8"
        and op_a.kwargs.get("block_size", 128) == 128
        and linear_a.bias is None
        and linear_b.bias is None
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and linear_a.weight.dtype == torch.bfloat16
        and linear_b.weight.dtype == torch.bfloat16
        and linear_a.weight.shape[0] % 128 == 0
        and linear_b.weight.shape[0] % 128 == 0
        and linear_a.weight.shape[1] % 128 == 0
    ):
        from ..moe_experts.shared.deepgemm import GroupedLinear, get_uniform_layout

        in_shape = x.shape
        hidden = in_shape[-1]
        out_a = linear_a.weight.shape[0]
        out_b = linear_b.weight.shape[0]
        x_2d = x.reshape(-1, hidden)
        layout = get_uniform_layout(x_2d.shape[0], 1, x.device)
        w_cat = torch.cat((linear_a.weight, linear_b.weight), dim=0).unsqueeze(0).contiguous()
        out_cat = GroupedLinear.apply(x_2d, w_cat, layout, False)
        res_a, res_b = out_cat.split([out_a, out_b], dim=-1)
        return res_a.reshape(*in_shape[:-1], out_a), res_b.reshape(*in_shape[:-1], out_b)

    return linear_a(x), linear_b(x)
