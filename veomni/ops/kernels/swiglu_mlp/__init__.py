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
from ...registry import register_op
from .geglu import eager as geglu_eager
from .geglu import liger_kernel as geglu_liger
from .standard import eager as standard_eager
from .standard import liger_kernel as standard_liger
from .standard import qat as standard_qat


_GPU = GpuKernelRequirement()
_MLU = MluKernelRequirement()
_NVIDIA_SM90_PLUS = GpuKernelRequirement(platforms=(NVIDIA_SM90_PLUS,))

#: ``swiglu_mlp`` rows that fake-quantize the weight tensors they are handed.
#: Only these need a projection's own ``weight``; every other row is a plain
#: arithmetic rewrite that keeps working on a wrapped projection's fallback path.
QUANTIZING_IMPLS = frozenset({"qat"})

#: Which ``swiglu_mlp`` rows evaluate which activations. A row missing from this
#: map is held to the same set as ``eager`` -- an unknown backend's activation
#: support cannot be assumed.
#:
#: ``liger_kernel`` is the GPU default and calls ``liger_kernel.ops.swiglu``, which
#: is SiLU only; a ``gelu`` model on the default row would otherwise raise on its
#: first forward. ``veomni.ops.activation`` is the authority for what ``eager`` and
#: ``qat`` can evaluate.
_ROW_ACTIVATIONS: dict[str, frozenset[str] | None] = {
    "eager": None,  # every activation `veomni.ops.activation` supports
    "qat": None,
    "liger_kernel": frozenset({"silu"}),
}


def row_supports_activation(impl: str | None, activation: str) -> bool:
    """Whether the row named *impl* can compute *activation*.

    Args:
        impl: The resolved implementation name of the row, or ``None`` when the
            selection has not been resolved yet (treated as "may not support it").
        activation: The model's configured activation.

    Returns:
        ``True`` when the row evaluates this activation itself. A caller that gets
        ``False`` must take its projection fallback rather than call the row.
    """
    from ...activation import normalize_activation, supports

    try:
        name = normalize_activation(activation)
    except ValueError:
        # An activation nothing here implements -- a row cannot serve it either.
        return False
    allowed = _ROW_ACTIVATIONS.get(impl, frozenset({"silu"})) if impl else frozenset({"silu"})
    if allowed is not None:
        return name in allowed
    return supports(name)


def use_fused_row(mlp: nn.Module, activation: str) -> bool:
    """Whether to call ``mlp``'s ``swiglu_mlp`` row, or take the projection fallback.

    Returns ``False`` when the row cannot serve this call, so the caller uses its
    projection modules. That is a legitimate choice for a reference row: the
    fallback evaluates the config's ``act_fn`` for any activation.

    Raises:
        ValueError: when the row *cannot* serve the call and it fake-quantizes. A
            quantizing row that fell back would train at full precision while the
            config claimed otherwise, which is the failure this whole check exists
            to prevent. Nothing here silently degrades.

    Two reasons a row cannot serve a call:

    * the projections are wrapped (LoRA keeps the base layer at
      ``base_layer``), which ``has_plain_linear_projections`` reports; and
    * the row does not evaluate the model's activation -- ``liger_kernel`` is the
      GPU default and implements SiLU only.

    Args:
        mlp: The MLP block that owns the projections and the handle.
        activation: The activation the config asks for, normally
            ``config.hidden_act``.
    """
    from ...activation import supports

    handle = getattr(mlp, "veomni_swiglu_mlp", None)
    impl = getattr(handle, "impl", None)

    if not supports(activation):
        # No row evaluates it, so only a fallback can run this model at all.
        if impl in QUANTIZING_IMPLS:
            raise ValueError(
                f"swiglu_mlp_implementation={impl!r} cannot run on hidden_act={activation!r}: no "
                f"gated-MLP row evaluates that activation, and a quantizing row cannot fall back "
                f"without training at full precision. Set swiglu_mlp_implementation='eager' and "
                f"check that the fallback path is acceptable for this model."
            )
        return False

    if not has_plain_linear_projections(mlp):
        return False
    if row_supports_activation(impl, activation):
        return True
    if impl in QUANTIZING_IMPLS:  # pragma: no cover - quantizing rows take every activation
        raise ValueError(
            f"swiglu_mlp_implementation={impl!r} cannot run on hidden_act={activation!r}. "
            f"Select 'eager' or 'qat', which evaluate the config's activation."
        )
    return False


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
    """Raise when ``owner``'s ``swiglu_mlp`` row quantizes and its call site cannot run it.

    A wrapped projection (LoRA keeps the base layer at ``base_layer``) has no
    ``weight`` for the row to read, so falling through would train at full
    precision on a config that says otherwise. ``has_plain_linear_projections``
    calls this on that path.

    An unsupported *activation* is refused earlier, at config load, by
    ``configuration_base._reject_quantized_swiglu_on_an_unfused_activation`` --
    before a handle is even built. Do not route that case through here.
    """
    handle = getattr(owner, "veomni_swiglu_mlp", None)
    impl = getattr(handle, "impl", None)
    if impl not in QUANTIZING_IMPLS:
        return
    raise ValueError(f"swiglu_mlp_implementation={impl!r} cannot run on {reason}. {remedy}")


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
    wrapper=standard_qat.wrapper,
    requirement=_NVIDIA_SM90_PLUS,
    requires=("tilelang",),
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
