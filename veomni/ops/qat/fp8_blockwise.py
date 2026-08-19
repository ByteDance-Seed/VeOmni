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

"""Block-wise FP8 fake quantization for quantization-aware training.

QAT needs the forward pass to see exactly the values an FP8 inference kernel
would compute while the backward pass keeps updating BF16 master weights. Both
halves come from reusing the DeepSeek-V4 TileLang quantizers for the
quantize-dequantize round trip and wrapping them in a straight-through
estimator, which keeps training bit-aligned with inference by construction
instead of by a second, drifting reimplementation of the same numerics.

The backward pass is the usual straight-through approximation: a
quantize-dequantize round trip is a step function whose true derivative is zero
almost everywhere, so the gradient is passed through unchanged. What this
recipe does avoid is the *second* approximation that clipped quantizers need.
Every scale is derived from its own block's amax, and ``ue8m0`` rounds that
scale up, so ``x / scale`` stays inside the representable FP8 range and no
element sits in a saturated region whose gradient would have to be masked.

The TileLang quantizers are SM90-only and BF16-only, so every entry point here
inherits both restrictions.
"""

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from ..kernels.deepseek_v4 import act_quant, fp8_weight_quant


# DeepSeek-V4 rounds every scale up to a power of two so that it survives
# storage as a bare E8M0 exponent. Every published V4 checkpoint ships pow2
# scales, and `checkpoint_tensor_converter` exports them the same way, so QAT
# defaults to the same format rather than to the kernels' unrounded scales.
DEFAULT_SCALE_FMT = "ue8m0"

__all__ = [
    "DEFAULT_SCALE_FMT",
    "fp8_fake_quant_act",
    "fp8_fake_quant_act_prefix",
    "fp8_fake_quant_weight",
    "qat_linear",
]


def _check_operand(tensor: torch.Tensor, scale_fmt: str | None, what: str) -> None:
    # The TileLang quantizers hard-code a BF16 operand dtype, so a FP32 tensor
    # would be reinterpreted rather than converted. Refuse instead of casting:
    # a silent downcast here would change training numerics invisibly.
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"FP8 fake quantization expects a bfloat16 {what}, got {tensor.dtype}")
    # The kernels only test `scale_fmt is not None`, so any typo would silently
    # select power-of-two scales instead of failing.
    if scale_fmt not in (None, "ue8m0"):
        raise ValueError(f"scale_fmt must be None or 'ue8m0', got {scale_fmt!r}")
    # A DTensor carries the global shape, so the divisibility checks and the
    # tile reshape below would be computed against dimensions the local shard
    # does not have. Expert-parallel linears reach forward as DTensors.
    if isinstance(tensor, DTensor):
        raise TypeError(f"FP8 fake quantization does not support a DTensor {what}; pass the local shard")


class _Fp8FakeQuantAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, block_size: int, scale_fmt: str | None) -> torch.Tensor:
        return act_quant(x.detach(), block_size, scale_fmt, dequant=True)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        return grad_output, None, None


class _Fp8FakeQuantWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, block_size: int, scale_fmt: str | None) -> torch.Tensor:
        return fp8_weight_quant(weight.detach(), block_size, scale_fmt, dequant=True)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        return grad_output, None, None


def fp8_fake_quant_act(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = DEFAULT_SCALE_FMT,
) -> torch.Tensor:
    """Quantize-dequantize an activation over ``1 x block_size`` blocks.

    Blocks run along the last dimension, one scale per block per row, matching
    how the inference GEMMs consume their activation operand.

    Args:
        x: BF16 activation whose last dimension is divisible by ``block_size``.
        block_size: Number of channels sharing a scale. DeepSeek-V4 uses 128 for
            GEMM operands and 64 for the NoPE half of the attention KV entry.
        scale_fmt: ``"ue8m0"`` rounds each scale up to a power of two; ``None``
            keeps the unrounded ``amax / 448`` scale.

    Returns:
        A BF16 tensor holding the dequantized values, differentiable into ``x``
        through a straight-through estimator.
    """
    _check_operand(x, scale_fmt, "activation")
    if x.shape[-1] % block_size:
        raise ValueError(f"activation of {x.shape[-1]} channels is not divisible by block_size {block_size}")
    return _Fp8FakeQuantAct.apply(x, block_size, scale_fmt)


def fp8_fake_quant_act_prefix(
    x: torch.Tensor,
    quant_features: int,
    block_size: int = 128,
    scale_fmt: str | None = DEFAULT_SCALE_FMT,
) -> torch.Tensor:
    """Fake-quantize only the leading ``quant_features`` channels of ``x``.

    DeepSeek-V4 stores the NoPE half of an attention KV entry in FP8 and leaves
    the trailing RoPE channels in BF16, so the quantizer has to stop at the
    split rather than run over the whole head.

    Args:
        x: BF16 activation.
        quant_features: Size of the leading slice to quantize; must be a
            multiple of ``block_size``. Passing the full last dimension is
            allowed and degenerates to :func:`fp8_fake_quant_act`.
        block_size: Number of channels sharing a scale.
        scale_fmt: See :func:`fp8_fake_quant_act`.

    Returns:
        A BF16 tensor of the same shape as ``x``, quantized below
        ``quant_features`` and untouched above it.
    """
    _check_operand(x, scale_fmt, "activation")
    features = x.shape[-1]
    if not 0 < quant_features <= features:
        raise ValueError(f"quant_features must fall in (0, {features}], got {quant_features}")
    if quant_features % block_size:
        raise ValueError(f"quant_features {quant_features} is not divisible by block_size {block_size}")
    if quant_features == features:
        return fp8_fake_quant_act(x, block_size, scale_fmt)
    quantized = fp8_fake_quant_act(x[..., :quant_features], block_size, scale_fmt)
    return torch.cat([quantized, x[..., quant_features:]], dim=-1)


def fp8_fake_quant_weight(
    weight: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = DEFAULT_SCALE_FMT,
) -> torch.Tensor:
    """Quantize-dequantize a 2D weight over square ``block_size x block_size`` tiles.

    Args:
        weight: BF16 weight whose both dimensions are divisible by ``block_size``.
        block_size: Side length of the quantization tile.
        scale_fmt: See :func:`fp8_fake_quant_act`.

    Returns:
        A BF16 tensor holding the dequantized weight, differentiable into
        ``weight`` through a straight-through estimator.
    """
    _check_operand(weight, scale_fmt, "weight")
    if weight.dim() != 2:
        raise ValueError(f"FP8 weight fake quantization expects a 2D weight, got shape {tuple(weight.shape)}")
    rows, cols = weight.shape
    if rows % block_size or cols % block_size:
        raise ValueError(f"weight shape {(rows, cols)} is not divisible by block_size {block_size}")
    return _Fp8FakeQuantWeight.apply(weight, block_size, scale_fmt)


def qat_linear(
    linear: nn.Module,
    x: torch.Tensor,
    *,
    enabled: bool = True,
    quantize_activation: bool = True,
    act_block_size: int = 128,
    weight_block_size: int = 128,
    scale_fmt: str | None = DEFAULT_SCALE_FMT,
) -> torch.Tensor:
    """Run ``linear`` with FP8 fake-quantized operands.

    This is the single substitution point for every linear that an FP8
    inference kernel would run as a true FP8 GEMM. With ``enabled=False`` it is
    the identity wrapper over ``linear(x)``, so a call site can be converted
    once and stay valid whether or not QAT is switched on.

    ``linear`` is invoked through its own ``forward``, so layers that are not a
    plain ``F.linear`` -- DeepSeek-V4's grouped output projection runs a
    ``bmm`` over reshaped weights -- keep their arithmetic.

    Args:
        linear: Any module exposing a ``weight`` parameter, typically an
            ``nn.Linear`` or a subclass of one.
        x: BF16 input activation.
        enabled: When false, bypass quantization entirely.
        quantize_activation: Whether the activation operand is quantized too.
            False gives a weight-only (W8A16) round trip.
        act_block_size: Activation block size, along the last dimension.
        weight_block_size: Side length of the square weight tile.
        scale_fmt: See :func:`fp8_fake_quant_act`.
    """
    if not enabled:
        return linear(x)
    if "weight" not in linear._parameters:
        raise TypeError(f"{type(linear).__name__} does not own a 'weight' parameter, so it cannot be fake-quantized")
    if quantize_activation:
        x = fp8_fake_quant_act(x, act_block_size, scale_fmt)
    weight = fp8_fake_quant_weight(linear.weight, weight_block_size, scale_fmt)
    # `functional_call` swaps the parameter for the duration of the call and
    # restores it afterwards, so the fake-quantized weight reaches the module's
    # own forward while gradients still flow back to the real parameter.
    return torch.func.functional_call(linear, {"weight": weight}, (x,))
