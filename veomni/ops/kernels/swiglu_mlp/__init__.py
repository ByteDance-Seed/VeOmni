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

from torch import nn

from ...platform import NVIDIA_SM90_PLUS, GpuKernelRequirement, MluKernelRequirement
from ...qat.torchao_fp4 import SUPPORTED_DTYPES
from ...registry import NumericMode, register_op
from .geglu import eager as geglu_eager
from .geglu import liger_kernel as geglu_liger
from .standard import eager as standard_eager
from .standard import liger_kernel as standard_liger
from .standard import qat as standard_qat
from .standard import torchao as standard_torchao


_GPU = GpuKernelRequirement()
_MLU = MluKernelRequirement()
_NVIDIA_SM90_PLUS = GpuKernelRequirement(platforms=(NVIDIA_SM90_PLUS,))

#: Activations the quantizing rows implement. ``standard`` is SiLU in both
#: directions (forward and backward); ``geglu`` is GELU-tanh but registers no
#: quantizing row, so its activation is deliberately absent. A model outside this
#: set has no quantizing row at all, and selecting one would round nothing -- which
#: is why ``configuration_base`` refuses the combination at load time (#1430).
FUSED_ACTIVATIONS = frozenset({"silu", "swish"})


def has_plain_linear_projections(mlp: nn.Module) -> bool:
    """Whether ``swiglu_mlp`` may read ``mlp``'s projection weights directly.

    Only an exact ``nn.Linear`` qualifies. A wrapped projection such as
    ``LoraLinear`` has no ``weight``, and a ``Linear`` subclass such as torchao's
    ``Float8Linear`` exposes one but does extra computation in its ``forward``
    (float8 GEMMs). Either must run its own forward, so the MLP has to call the
    projection modules instead of the weight-only fused path.

    Raises:
        ValueError: when a fallback is needed while ``mlp``'s own ``swiglu_mlp``
            selection is a fake-quantizing row. A wrapped projection has no
            ``weight`` for the row to read: ``LoraLinear`` keeps the base layer at
            ``base_layer``, so falling back would run at full precision while the
            config claimed quantization-aware training -- a silent correctness bug,
            not a graceful degradation. Every non-quantizing row keeps the
            fallback.

    """
    plain = all(type(getattr(mlp, name)) is nn.Linear for name in ("gate_proj", "up_proj", "down_proj"))
    if plain:
        return True

    reject_quantizing_row(
        mlp,
        "a wrapped projection or an nn.Linear subclass: LoRA keeps the base weights at `base_layer`, and a "
        "subclass such as torchao's Float8Linear computes its own GEMM in forward, so the handle cannot "
        "fake-quantize the weight the projection actually uses",
        remedy="Set swiglu_mlp_implementation='eager' for adapter or float8 training on gate/up/down.",
    )
    return False


def reject_quantizing_row(owner: nn.Module, reason: str, *, remedy: str) -> None:
    """Raise when ``owner``'s ``swiglu_mlp`` row rounds and its call site cannot run it.

    A wrapped projection (LoRA keeps the base layer at ``base_layer``) has no
    ``weight`` for the row to read, so falling through would train at full
    precision on a config that says otherwise. ``has_plain_linear_projections``
    calls this on that path.

    An unsupported *activation* is refused earlier, at config load, by
    ``configuration_base._reject_quantized_swiglu_on_an_unfused_activation`` --
    before a handle is even built. Do not route that case through here.
    """
    handle = getattr(owner, "veomni_swiglu_mlp", None)
    # The handle knows its own arguments, so it answers for the mode the config
    # selected rather than for whichever mode the row could run in.
    mode = getattr(handle, "mode", None)
    if mode is None or not mode.rounds_operands:
        return
    raise ValueError(f"swiglu_mlp_implementation={getattr(handle, 'impl', None)!r} cannot run on {reason}. {remedy}")


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
    default=True,
)

register_op(
    "swiglu_mlp",
    "standard",
    "liger_kernel",
    standard_liger.forward,
    standard_liger.backward,
    description="Liger Kernel SwiGLU MLP with optional activation clamping",
    requirement=_MLU,
    requires=("liger_kernel",),
    default=True,
)

register_op(
    "swiglu_mlp",
    "standard",
    "qat",
    description="TileLang blockwise FP8 fake-quantized SwiGLU MLP with optional activation clamping",
    modes=(NumericMode.EMULATED,),
    wrapper=standard_qat.wrapper,
    requirement=_NVIDIA_SM90_PLUS,
    requires=("tilelang",),
)

register_op(
    "swiglu_mlp",
    "standard",
    "torchao",
    description="TorchAO FP4 fake-quantized SwiGLU MLP with optional activation clamping",
    modes=(NumericMode.EMULATED,),
    wrapper=standard_torchao.wrapper,
    requires=("torchao",),
    dtypes=SUPPORTED_DTYPES,
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
    default=True,
)

register_op(
    "swiglu_mlp",
    "geglu",
    "liger_kernel",
    geglu_liger.forward,
    geglu_liger.backward,
    description="Liger Kernel GeGLU MLP with tanh-approximate GELU",
    requirement=_MLU,
    requires=("liger_kernel",),
    default=True,
)
