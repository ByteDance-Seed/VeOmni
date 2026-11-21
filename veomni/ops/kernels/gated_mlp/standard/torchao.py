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

"""Standard gated-MLP torchao implementation (FP8, NVFP4, MXFP4).

``dtype`` selects the low-precision format for all three projections
(``gate_proj``, ``up_proj``, ``down_proj``), while clamping and
``act_fn(gate) * up`` stay in the model dtype:

* ``"fp8"``: torchao dynamic float8 GEMMs (``tensorwise``, ``rowwise``, or
  ``rowwise_with_gw_hp`` via ``recipe``). Defaults to ``emulated=False`` (native
  ``aten::_scaled_mm`` on NVIDIA SM89+); ``emulated=True`` runs torchao's
  emulated float8 path.
* ``"nvfp4"`` / ``"mxfp4"``: torchao FP4 fake-quantization (QAT). Defaults to
  ``emulated=True``; native FP4 (``emulated=False``) is not implemented yet.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from ....activation import apply_activation
from ....torchao import (
    DEFAULT_FLOAT8_RECIPE,
    torchao_fake_quant,
    torchao_fp8_linear,
    validate_float8_recipe,
)
from ....torchao import FP4_SUPPORTED_DTYPES as _FP4_DTYPES
from . import eager as _eager


#: Formats this row serves via its ``dtype`` argument.
SUPPORTED_DTYPES: tuple[str, ...] = ("fp8", *_FP4_DTYPES)


def _resolve_emulated(dtype: str, emulated: bool | None) -> bool:
    """Resolve ``emulated`` default per format: native for FP8, emulated for FP4."""
    if emulated is None:
        return dtype != "fp8"
    if type(emulated) is not bool:
        raise ValueError(f"gated_mlp_implementation='torchao': emulated must be a bool or None, got {emulated!r}.")
    return emulated


def wrapper(
    mlp: nn.Module,
    x: Tensor,
    *,
    swiglu_limit: float | None = None,
    dtype: str = "nvfp4",
    block_size: int | None = None,
    recipe: str | None = None,
    emulated: bool | None = None,
    activation: str = "silu",
) -> Tensor:
    """Run a gated MLP with torchao low-precision projections and the optional clamp.

    ``mlp`` is the model's MLP block; ``render`` reads the plain projection
    weights and refuses wrapped projections rather than silently running at full
    precision. ``dtype`` picks ``"fp8"``, ``"nvfp4"``, or ``"mxfp4"``.
    """
    from .._module import render

    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"gated_mlp_implementation='torchao' cannot run with dtype={dtype!r}; supported dtypes: {SUPPORTED_DTYPES}."
        )
    use_emulated = _resolve_emulated(dtype, emulated)

    if dtype == "fp8":
        if block_size is not None:
            raise ValueError(
                "gated_mlp_implementation='torchao' with dtype='fp8' uses recipe scaling "
                "and does not accept block_size."
            )
        resolved_recipe = DEFAULT_FLOAT8_RECIPE if recipe is None else validate_float8_recipe(recipe)
        gate_w, gate_b, up_w, up_b, down_w, down_b = render(mlp, impl="torchao", activation=activation).weights
        gate = torchao_fp8_linear(
            x, gate_w, _eager.optional_bias(gate_b), recipe=resolved_recipe, emulated=use_emulated
        )
        up = torchao_fp8_linear(x, up_w, _eager.optional_bias(up_b), recipe=resolved_recipe, emulated=use_emulated)
        gate_c, up_c = _eager.clamp_gate_up(gate, up, swiglu_limit)
        hidden = apply_activation(gate_c, activation) * up_c
        if hidden.dtype != x.dtype:
            hidden = hidden.to(dtype=x.dtype)
        return torchao_fp8_linear(
            hidden, down_w, _eager.optional_bias(down_b), recipe=resolved_recipe, emulated=use_emulated
        )

    if recipe is not None:
        raise ValueError(
            f"gated_mlp_implementation='torchao' with dtype={dtype!r} does not accept recipe={recipe!r}; "
            "recipe is only valid for dtype='fp8'."
        )
    if not use_emulated:
        raise NotImplementedError(
            f"gated_mlp_implementation='torchao' with dtype={dtype!r} and emulated=False is not implemented yet: "
            "the native FP4 path needs Blackwell SM100+ dispatch. Drop emulated=False (or set emulated=True) "
            "to train against the emulated QAT numerics, which needs no special hardware."
        )

    gate_w, gate_b, up_w, up_b, down_w, down_b = render(mlp, impl="torchao", activation=activation).weights
    quantize = torchao_fake_quant(dtype)
    x_q = quantize(x, block_size)
    gate = F.linear(x_q, quantize(gate_w, block_size), _eager.optional_bias(gate_b))
    up = F.linear(x_q, quantize(up_w, block_size), _eager.optional_bias(up_b))

    gate_c, up_c = _eager.clamp_gate_up(gate, up, swiglu_limit)
    hidden = apply_activation(gate_c, activation) * up_c
    if hidden.dtype != x.dtype:
        hidden = hidden.to(dtype=x.dtype)

    hidden_q = quantize(hidden, block_size)
    return F.linear(hidden_q, quantize(down_w, block_size), _eager.optional_bias(down_b))


__all__ = [
    "SUPPORTED_DTYPES",
    "wrapper",
]
