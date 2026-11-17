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

"""MoE experts kernel family.

``standard`` is routed SwiGLU (split or merged fc1). ``gpt_oss`` is the
interleaved gate/up layout with bias and ``alpha`` / ``limit``. Eager is
the HF expert loop (regular autograd). Fused rows wrap the local Functions.
"""

from torch import nn

from ...platform import (
    NVIDIA_SM70_PLUS,
    NVIDIA_SM90_PLUS,
    ROCM_GPU,
    GpuKernelRequirement,
    MluKernelRequirement,
    NpuKernelRequirement,
    NvidiaGpuPlatform,
)
from ...registry import RowPrecision, register_op
from .gpt_oss import eager as gpt_oss_eager
from .gpt_oss import quack as gpt_oss_quack
from .shared import unpermute
from .standard import deepgemm as standard_deepgemm
from .standard import eager as standard_eager
from .standard import mlu as standard_mlu
from .standard import npu as standard_npu
from .standard import qat as standard_qat
from .standard import quack as standard_quack
from .standard import triton as standard_triton


_GPU_SM70_OR_ROCM = GpuKernelRequirement(platforms=(NVIDIA_SM70_PLUS, ROCM_GPU))
_NVIDIA_SM90_PLUS = GpuKernelRequirement(platforms=(NVIDIA_SM90_PLUS,))


def rounding_impls() -> frozenset[str]:
    """``moe_experts`` rows that round the operands they are handed, native ones included.

    The activation guard needs this set rather than :meth:`OpRegistry.emulated_impls`:
    every fused row bakes SiLU into both directions, ``deepgemm`` included, so an
    activation the family cannot run has to be refused for a native row just as much as
    for an emulated one. Only ``EXACT`` rows are satisfied by the Python act_fn loop,
    because they are the ones that never round.

    Adapter rejection reads the row's precision directly (``veomni/lora/moe_layers.py``)
    rather than a set from here: it needs the same "does this round?" answer, and a
    second name for it would be one more place to drift.
    """
    from ...registry import OP_REGISTRY

    return OP_REGISTRY.rounding_impls("moe_experts")


def reject_quantizing_act_fn(owner: nn.Module) -> None:
    """Raise when a non-SwiGLU activation would bypass a quantizing ``moe_experts`` row.

    Every fused row bakes SiLU into both directions -- ``standard/triton.py`` calls
    ``aten.silu`` in the forward and ``aten.silu_backward`` in the backward, and the
    TileLang ``qat`` rows have the same structure -- so an activation outside
    ``{"silu", "swish"}`` has no fused row and falls back to
    ``merged_experts_act_fn_forward``, which loops over experts in Python and cannot
    reach the family row.

    Call this on that fallback. Refusing keeps a quantizing selection from training
    the routed experts at full precision while the config claims otherwise. Support
    for other activations is tracked in #1430.

    The activation is read from ``owner.use_swiglu_mlp`` rather than a config:
    ``DeepseekV3Experts`` and its siblings are plain ``nn.Module`` objects that
    record the answer at construction and hold no config reference.

    Args:
        owner: The experts module that owns the ``veomni_moe`` handle and carries
            ``use_swiglu_mlp``.

    Raises:
        ValueError: when the selection rounds its operands and the activation is not SwiGLU.
    """
    if getattr(owner, "use_swiglu_mlp", True):
        return
    handle = getattr(owner, "veomni_moe", None)
    impl = getattr(handle, "impl", None)
    if impl not in rounding_impls():
        return
    raise ValueError(
        f"moe_implementation={impl!r} cannot run with a non-SwiGLU activation: every `moe_experts` row "
        "with a fused kernel implements SwiGLU only, in both forward and backward, so these routed "
        "experts would train through the Python act_fn loop and round nothing. Use hidden_act='silu' "
        "to quantize them, or moe_implementation='eager' to train them in the model dtype."
    )


register_op(
    "moe_unpermute",
    "standard",
    "eager",
    wrapper=unpermute.eager,
    description="Weighted token scatter with FP32 accumulation",
)
register_op(
    "moe_unpermute",
    "standard",
    "triton",
    wrapper=unpermute.triton,
    description="Fused weighted token scatter with FP32 accumulation",
    requirement=GpuKernelRequirement(platforms=(NVIDIA_SM70_PLUS,)),
    requires=("triton",),
)


register_op(
    "moe_experts",
    "standard",
    "eager",
    description="PyTorch routed SwiGLU MoE experts with split or merged gate/up weights",
    wrapper=standard_eager.wrapper,
)

register_op(
    "moe_experts",
    "standard",
    "fused_triton",
    description="Triton grouped-GEMM routed SwiGLU MoE experts",
    wrapper=standard_triton.wrapper,
    requirement=_GPU_SM70_OR_ROCM,
    requires=("triton",),
    default=True,
)

register_op(
    "moe_experts",
    "standard",
    "fused_mlu",
    description="Apex grouped-GEMM routed SwiGLU MoE experts",
    wrapper=standard_mlu.wrapper,
    requirement=MluKernelRequirement(),
    requires=("apex",),
    default=True,
)

register_op(
    "moe_experts",
    "standard",
    "fused_triton",
    description="Triton grouped-GEMM routed SwiGLU MoE experts",
    wrapper=standard_triton.wrapper,
    requirement=MluKernelRequirement(),
    requires=("triton",),
    default=True,
)

register_op(
    "moe_experts",
    "standard",
    "fused_quack",
    description="Quack CUTLASS/CuTe routed SwiGLU MoE experts",
    wrapper=standard_quack.wrapper,
    requirement=_NVIDIA_SM90_PLUS,
    requires=("quack",),
)

register_op(
    "moe_experts",
    "standard",
    "qat",
    description="TileLang blockwise FP8/FP4 fake-quantized Triton grouped-GEMM routed SwiGLU MoE experts",
    precision=RowPrecision.EMULATED,
    wrapper=standard_qat.wrapper,
    requirement=_NVIDIA_SM90_PLUS,
    requires=("tilelang", "triton"),
    dtypes=("fp8", "fp4"),
)

register_op(
    "moe_experts",
    "standard",
    "deepgemm",
    description="DeepGEMM blockwise FP8 expert forward, input gradients and weight gradients",
    # ``NATIVE``, not ``EMULATED``: the row casts the operands to FP8 and runs
    # DeepGEMM's grouped GEMM in that format, so it owns the cast instead of
    # receiving already-rounded operands. That is what separates it from ``qat``
    # even though both round.
    precision=RowPrecision.NATIVE,
    wrapper=standard_deepgemm.wrapper,
    requirement=GpuKernelRequirement(
        platforms=(NvidiaGpuPlatform(min_cc=90, max_cc=90), NvidiaGpuPlatform(min_cc=100, max_cc=100)),
    ),
    requires=("deep_gemm", "triton"),
    # One format today. It is still an argument rather than a name because that is
    # the shape every rounding row uses, and DeepGEMM gaining another layout is a
    # new value, not a new row.
    dtypes=("fp8",),
)

register_op(
    "moe_experts",
    "standard",
    "fused_npu",
    description="torch_npu grouped-GEMM routed SwiGLU MoE experts",
    wrapper=standard_npu.wrapper,
    requirement=NpuKernelRequirement(),
    default=True,
)

register_op(
    "moe_experts",
    "gpt_oss",
    "eager",
    description="PyTorch GPT-OSS MoE experts with interleaved gate/up weights and bias",
    wrapper=gpt_oss_eager.wrapper,
)

register_op(
    "moe_experts",
    "gpt_oss",
    "fused_quack",
    description="Quack CUTLASS/CuTe GPT-OSS MoE experts with interleaved gate/up weights and bias",
    wrapper=gpt_oss_quack.wrapper,
    requirement=_NVIDIA_SM90_PLUS,
    requires=("quack",),
)
