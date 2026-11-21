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

"""Shared TorchAO low-precision primitives for the ``linear`` and ``gated_mlp`` rows.

Covers both formats exposed by the ``torchao`` op row:

* **Dynamic FP8** (``torchao_fp8_linear``): drives torchao's functional
  ``matmul_with_hp_or_float8_args`` directly from plain ``nn.Linear`` weights
  without module swapping, supporting ``tensorwise``, ``rowwise``, and
  ``rowwise_with_gw_hp`` scaling in native ``aten::_scaled_mm`` (SM89+) or
  emulated mode.
* **FP4 fake quantization** (``torchao_fake_quant``, ``nvfp4_fake_quant``):
  emulated NVFP4 (16-wide groups, E4M3 block scale + FP32 per-tensor scale) and
  MXFP4 (32-wide groups, E8M0 power-of-two block scale) straight-through
  quantizers built on torchao tensor subclasses.

``torchao`` is imported lazily on the first call, so importing ``veomni.ops`` on
a host without the optional ``float8`` extra stays free.
"""

from __future__ import annotations

from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version

import torch
import torch.nn.functional as F
from packaging.version import Version
from torch import Tensor
from torch.distributed.tensor import DTensor

from ..utils import device


__all__ = [
    "DEFAULT_FLOAT8_RECIPE",
    "FLOAT8_RECIPES",
    "FP4_SUPPORTED_DTYPES",
    "MXFP4_BLOCK_SIZE",
    "NVFP4_BLOCK_SIZE",
    "TORCHAO_REQUIRED_VERSION",
    "nvfp4_fake_quant",
    "torchao_fake_quant",
    "torchao_fp8_linear",
    "validate_float8_recipe",
]

#: Supported torchao dynamic float8 scaling recipes.
FLOAT8_RECIPES: tuple[str, ...] = ("tensorwise", "rowwise", "rowwise_with_gw_hp")
DEFAULT_FLOAT8_RECIPE: str = "tensorwise"
TORCHAO_REQUIRED_VERSION: str = "0.18.0"

#: Group widths for the two torchao FP4 fake-quantization layouts.
NVFP4_BLOCK_SIZE = 16
MXFP4_BLOCK_SIZE = 32
FP4_SUPPORTED_DTYPES: tuple[str, ...] = ("nvfp4", "mxfp4")
_FP4_BLOCK_SIZES = {"nvfp4": NVFP4_BLOCK_SIZE, "mxfp4": MXFP4_BLOCK_SIZE}


def validate_float8_recipe(recipe: str) -> str:
    """Validate and return a torchao float8 scaling recipe name."""
    if not isinstance(recipe, str) or recipe not in FLOAT8_RECIPES:
        raise ValueError(f"float8 recipe must be one of {FLOAT8_RECIPES}, got {recipe!r}.")
    return recipe


def _require_torchao_version() -> None:
    """Verify the pinned torchao extra is installed before importing ``torchao.float8``."""
    try:
        installed_version = version("torchao")
    except PackageNotFoundError as exc:
        raise ImportError(
            f"float8 training requires torchao=={TORCHAO_REQUIRED_VERSION}, which is an optional extra: "
            "uv sync --extra gpu --extra float8 --dev"
        ) from exc
    if Version(installed_version).public != TORCHAO_REQUIRED_VERSION:
        raise ImportError(f"float8 training requires torchao=={TORCHAO_REQUIRED_VERSION}; found {installed_version}.")


_FLOAT8_CONFIG_CACHE: dict[tuple[str, bool], tuple] = {}


@torch._dynamo.assume_constant_result
def _float8_linear_configs(recipe: str, emulated: bool):
    """Build and cache torchao's ``(Float8LinearConfig, LinearMMConfig)`` pair.

    Marked ``assume_constant_result`` so ``torch.compile(fullgraph=True)`` treats
    the immutable config pair as a compile-time constant instead of tracing
    ``importlib.metadata.version`` I/O.
    """
    key = (recipe, emulated)
    cached = _FLOAT8_CONFIG_CACHE.get(key)
    if cached is not None:
        return cached

    _require_torchao_version()

    from torchao.float8.config import Float8LinearConfig
    from torchao.float8.float8_training_tensor import LinearMMConfig, ScaledMMConfig

    # Weight-gradient GEMMs contract over flattened token rows, which may not
    # be divisible by 16 for packed/ragged batches. Native zero padding keeps
    # those batches valid without changing the returned shapes.
    config = replace(
        Float8LinearConfig.from_recipe_name(recipe),
        pad_inner_dim=True,
        emulate=emulated,
    )
    # Keep FSDP parameter AllGather and master parameters in their existing
    # dtypes; the op row changes only the Linear forward/backward GEMMs.
    assert not config.enable_fsdp_float8_all_gather and config.emulate is emulated
    linear_mm_config = LinearMMConfig(
        # output
        ScaledMMConfig(
            config.emulate,
            config.gemm_config_output.use_fast_accum,
            False,
            config.pad_inner_dim,
        ),
        # grad_input
        ScaledMMConfig(
            config.emulate,
            config.gemm_config_grad_input.use_fast_accum,
            False,
            config.pad_inner_dim,
        ),
        # grad_weight
        ScaledMMConfig(
            config.emulate,
            config.gemm_config_grad_weight.use_fast_accum,
            False,
            config.pad_inner_dim,
        ),
    )
    result = (config, linear_mm_config)
    _FLOAT8_CONFIG_CACHE[key] = result
    return result


_float8_linear_configs.cache_clear = _FLOAT8_CONFIG_CACHE.clear


def torchao_fp8_linear(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    recipe: str = DEFAULT_FLOAT8_RECIPE,
    emulated: bool = False,
) -> Tensor:
    """Compute ``x @ weight.T + bias`` with torchao dynamic float8 GEMMs.

    Args:
        x: Input activation tensor ``[..., in_features]``.
        weight: Plain 2D linear weight ``[out_features, in_features]`` with both
            dimensions divisible by 16.
        bias: Optional 1D bias ``[out_features]``.
        recipe: Dynamic scaling recipe: ``"tensorwise"``, ``"rowwise"``, or
            ``"rowwise_with_gw_hp"``.
        emulated: When ``False`` (default), dispatch native ``aten::_scaled_mm``
            on NVIDIA CUDA SM89+. When ``True``, run torchao's emulated float8
            math (no SM89+ hardware required).

    Returns:
        Output tensor of shape ``[*x.shape[:-1], out_features]``.
    """
    validate_float8_recipe(recipe)
    if type(emulated) is not bool:
        raise ValueError(f"float8 emulated flag must be a bool, got {emulated!r}.")
    if isinstance(x, DTensor) or isinstance(weight, DTensor):
        raise TypeError("torchao float8 linear does not support a DTensor operand; pass the local tensor.")
    if weight.ndim != 2:
        raise ValueError(f"torchao float8 linear expects a 2D weight, got shape {tuple(weight.shape)}.")
    if weight.shape[0] % 16 or weight.shape[1] % 16:
        raise ValueError(
            f"torchao float8 linear requires in_features ({weight.shape[1]}) and "
            f"out_features ({weight.shape[0]}) divisible by 16."
        )
    if not emulated and (
        not device.IS_CUDA_AVAILABLE or torch.version.hip or device.get_gpu_compute_capability() < 89
    ):
        raise RuntimeError(
            "Native float8 linear (dtype='fp8', emulated=False) requires an NVIDIA CUDA SM89 or newer device."
        )

    config, linear_mm_config = _float8_linear_configs(recipe, emulated)
    if x.numel() == 0:
        return F.linear(x, weight, bias)

    from torchao.float8.float8_linear import matmul_with_hp_or_float8_args

    # Mirror Float8Linear.forward's autocast handling so both operands keep the
    # active autocast precision just like F.linear.
    if torch.is_autocast_enabled(x.device.type):
        autocast_dtype = torch.get_autocast_dtype(x.device.type)
        x = x.to(autocast_dtype)
        weight = weight.to(autocast_dtype)

    output = matmul_with_hp_or_float8_args.apply(
        x,
        weight.t(),
        linear_mm_config,
        config,
    )
    if bias is not None:
        output = output + bias.to(output.dtype)
    return output


def _nvfp4_roundtrip(x: Tensor, block_size: int) -> Tensor:
    """Quantize-dequantize ``x`` in ``1 x block_size`` groups to NVFP4 with two-level scaling."""
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor, per_tensor_amax_to_scale

    # `to_nvfp4` expects a 2D or 3D contiguous tensor; flatten leading dimensions
    # into rows and restore the original shape after dequantizing.
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    # Clamp per_tensor_scale away from 0 so an all-zero operand dequantizes to
    # zeros instead of dividing 0.0 / 0.0 inside torchao's nvfp4_quantize.
    per_tensor_scale = per_tensor_amax_to_scale(torch.max(torch.abs(x_2d))).clamp(min=1e-12)
    quantized = NVFP4Tensor.to_nvfp4(
        x_2d,
        block_size=block_size,
        per_tensor_scale=per_tensor_scale,
        is_swizzled_scales=False,
        use_triton_kernel=False,
    )
    return quantized.dequantize(x.dtype).reshape(x.shape)


def _mxfp4_roundtrip(x: Tensor, block_size: int) -> Tensor:
    """Quantize-dequantize ``x`` in ``1 x block_size`` groups to MXFP4 with an E8M0 scale."""
    from torchao.prototype.mx_formats.config import ScaleCalculationMode
    from torchao.prototype.mx_formats.mx_tensor import MXTensor
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    # ``MXFP4`` is torchao's `e2m1` element type with an `e8m0` scale, one per group;
    # there is no per-tensor scale, which is the whole difference from NVFP4.
    quantized = MXTensor.to_mx(
        x_2d,
        torch.float4_e2m1fn_x2,
        block_size=block_size,
        scaling_mode=ScaleCalculationMode.FLOOR,
        kernel_preference=KernelPreference.EMULATED,
        is_swizzled_scales=False,
    )
    return quantized.dequantize(x.dtype).reshape(x.shape)


def _check_fp4_operand(x: Tensor, dtype: str, block_size: int) -> None:
    """Validate the shared FP4 QAT contract: local operand, supported dtype, whole groups."""
    # A DTensor reports the global shape, so the divisibility check below would
    # validate a shape this rank does not hold, and a per-tensor ``amax`` would
    # silently depend on how the tensor is partitioned. Same contract as the FP8
    # primitives: the caller passes the local tensor.
    if isinstance(x, DTensor):
        raise TypeError(f"{dtype} fake quantization does not support a DTensor operand; pass the local tensor")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"{dtype} fake quantization expects a bfloat16 or float32 operand, got {x.dtype}")
    width = _FP4_BLOCK_SIZES[dtype]
    if block_size != width:
        raise ValueError(f"{dtype} fake quantization requires block_size={width}, got {block_size}.")
    if x.shape[-1] % block_size:
        raise ValueError(
            f"{dtype} fake quantization needs the last dimension ({x.shape[-1]}) divisible by block_size "
            f"{block_size}; the deployment kernels index one scale per group and cannot pad a partial one."
        )


def torchao_fake_quant(dtype: str):
    """Return ``(x, block_size=None) -> x`` fake-quantizing to ``dtype`` (``nvfp4`` or ``mxfp4``)."""
    if dtype not in _FP4_BLOCK_SIZES:
        raise ValueError(
            f"torchao fake quantization does not support dtype={dtype!r}; it supports {FP4_SUPPORTED_DTYPES}"
        )

    def quantize(x: Tensor, block_size: int | None = None) -> Tensor:
        width = _FP4_BLOCK_SIZES[dtype] if block_size is None else block_size
        if isinstance(x, DTensor):
            _check_fp4_operand(x, dtype, width)
        if x.numel() == 0:
            return x
        _check_fp4_operand(x, dtype, width)
        with torch.no_grad():
            rounded = _nvfp4_roundtrip(x, width) if dtype == "nvfp4" else _mxfp4_roundtrip(x, width)
        return rounded.detach() + (x - x.detach())

    return quantize


def nvfp4_fake_quant(x: Tensor, block_size: int = NVFP4_BLOCK_SIZE) -> Tensor:
    """Fake-quantize ``x`` in NVFP4, differentiably into ``x`` via straight-through estimation."""
    return torchao_fake_quant("nvfp4")(x, block_size)
