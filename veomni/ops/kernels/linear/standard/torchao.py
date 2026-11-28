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

"""Standard dense linear torchao implementation (FP8, MXFP8, NVFP4, MXFP4).

``dtype`` selects the low-precision format:

* ``"fp8"``: torchao dynamic float8 GEMMs (``tensorwise``, ``rowwise``, or
  ``rowwise_with_gw_hp`` via ``recipe``). Defaults to ``emulated=False`` (native
  ``aten::_scaled_mm`` on NVIDIA SM89+); ``emulated=True`` runs torchao's
  emulated float8 path without SM89+ hardware.
* ``"mxfp8"``: block-scaled E4M3 GEMMs with 32 values per E8M0 scale. Defaults
  to native execution on NVIDIA SM100+; set ``emulated=True`` for emulation.
* ``"nvfp4"`` / ``"mxfp4"``: torchao FP4 fake-quantization (QAT). Defaults to
  ``emulated=True``; native FP4 (``emulated=False``) is not implemented yet.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from ....torchao import (
    DEFAULT_FLOAT8_RECIPE,
    MXFP8_BLOCK_SIZE,
    default_emulated,
    mxfp8_model_projection,
    torchao_fake_quant,
    torchao_fp8_linear,
    validate_float8_recipe,
    validate_mxfp8_projection_hooks,
)
from ....torchao import FP4_SUPPORTED_DTYPES as _FP4_DTYPES
from .._module import resolve_projection


#: Formats this row serves via its ``dtype`` argument.
SUPPORTED_DTYPES: tuple[str, ...] = ("fp8", "mxfp8", *_FP4_DTYPES)


def _resolve_emulated(dtype: str, emulated: bool | None) -> bool:
    """Apply the format default or validate an explicit emulation setting."""
    if emulated is None:
        return default_emulated(dtype)
    if type(emulated) is not bool:
        raise ValueError(f"linear_implementation='torchao': emulated must be a bool or None, got {emulated!r}.")
    return emulated


def wrapper(
    linear: nn.Module,
    x: Tensor,
    *,
    dtype: str = "nvfp4",
    block_size: int | None = None,
    recipe: str | None = None,
    emulated: bool | None = None,
) -> Tensor:
    """Run ``linear`` through torchao at the precision ``dtype`` names.

    ``dtype`` picks the format: ``"fp8"`` (dynamic float8 with ``recipe``
    scaling), ``"mxfp8"`` (32-wide E4M3 blocks), ``"nvfp4"`` (16-wide groups, E4M3 block scale + FP32 per-tensor
    scale), or ``"mxfp4"`` (32-wide groups, E8M0 power-of-two scale).
    Both FP8 formats default to native GEMMs. FP4 layouts default to emulated QAT.
    """
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"linear_implementation='torchao' cannot run with dtype={dtype!r}; supported dtypes: {SUPPORTED_DTYPES}."
        )
    use_emulated = _resolve_emulated(dtype, emulated)

    if dtype == "fp8":
        if block_size is not None:
            raise ValueError(
                "linear_implementation='torchao' with dtype='fp8' uses recipe scaling and does not accept block_size."
            )
        resolved_recipe = DEFAULT_FLOAT8_RECIPE if recipe is None else validate_float8_recipe(recipe)
        weight, shortcut = resolve_projection("torchao", linear, x)
        assert shortcut is None
        bias = getattr(linear, "bias", None)
        return torchao_fp8_linear(x, weight, bias, recipe=resolved_recipe, emulated=use_emulated)

    if recipe is not None:
        raise ValueError(
            f"linear_implementation='torchao' with dtype={dtype!r} does not accept recipe={recipe!r}; "
            "recipe is only valid for dtype='fp8'."
        )
    if dtype == "mxfp8" and block_size not in (None, MXFP8_BLOCK_SIZE):
        raise ValueError(f"linear_implementation='torchao' with dtype='mxfp8' requires block_size={MXFP8_BLOCK_SIZE}.")
    if dtype in _FP4_DTYPES and not use_emulated:
        return _native_fp4(dtype)
    weight, shortcut = resolve_projection("torchao", linear, x)
    assert shortcut is None
    bias = getattr(linear, "bias", None)
    if dtype == "mxfp8":
        validate_mxfp8_projection_hooks(linear, "linear")
        return mxfp8_model_projection(x, weight, bias, emulated=use_emulated)
    quantize = torchao_fake_quant(dtype)
    return F.linear(quantize(x, block_size), quantize(weight, block_size), bias)


def _native_fp4(dtype: str) -> Tensor:
    """Refuse unimplemented native FP4 GEMM rather than silently falling back to emulation."""
    raise NotImplementedError(
        f"linear_implementation='torchao' with dtype={dtype!r} and emulated=False is not implemented yet: "
        "the native FP4 path needs Blackwell SM100+ dispatch. Drop emulated=False (or set emulated=True) "
        "to train against the emulated QAT numerics, which needs no special hardware."
    )


__all__ = [
    "SUPPORTED_DTYPES",
    "wrapper",
]
