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

"""Standard dense linear torchao implementation, emulated or native.

``emulated=True`` (the default) rounds both GEMM operands to the target format and
multiplies in the model dtype -- the same shape as the TileLang ``qat`` row next to
it, and the only mode that needs no low-precision hardware. ``emulated=False``
hands the operands to torchao's real low-precision GEMM for that layout.

The two modes differ in what a guard may assume, which is why the row declares both
rather than hiding one behind a second name: an emulated row reads the weight to
round it and refuses a wrapped projection, while a native row that owns its own
arithmetic does not need to.

FP8 is not a ``dtype`` here. torchao's FP8 training is ``Float8Linear``, a module
swap that happens before FSDP wrapping; see ``veomni/distributed/float8.py`` and
``mixed_precision.float8``.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from ....qat.torchao_fp4 import SUPPORTED_DTYPES as _FP4_DTYPES
from ....qat.torchao_fp4 import torchao_fake_quant
from .._module import resolve_projection


#: Formats this row can round to, as its ``dtype`` argument.
#:
#: The FP4 layouts are the ones an op row can serve. ``fp8`` is deliberately absent:
#: torchao's FP8 training path is ``Float8Linear``, a *module swap* done before FSDP
#: wraps, and this row receives one already-built module per call. Driving FP8
#: through a call-local wrapper would quantize per invocation rather than own the
#: module, which is a different feature with different numerics -- it belongs behind
#: ``mixed_precision.float8`` (``veomni/distributed/float8.py``), not here.
SUPPORTED_DTYPES = _FP4_DTYPES


def wrapper(
    linear: nn.Module,
    x: Tensor,
    *,
    dtype: str = "nvfp4",
    block_size: int | None = None,
    emulated: bool = True,
) -> Tensor:
    """Run ``linear`` at the precision ``dtype`` names.

    ``dtype`` picks the format: ``nvfp4`` (16-wide groups, an FP32 per-tensor
    scale) or ``mxfp4`` (32-wide groups, an E8M0 power-of-two scale).
    ``block_size`` defaults to that format's group width, which is the only width it
    defines. ``emulated`` picks the mode: round-then-multiply-in-model-dtype, or the
    real low-precision GEMM for that layout.
    """
    if not emulated:
        return _native(linear, x, dtype, block_size)
    # An emulated row always gets a weight: ``resolve_projection`` raises for a
    # wrapped projection instead of handing back a shortcut.
    weight, shortcut = resolve_projection("torchao", linear, x, emulated=True)
    assert shortcut is None
    bias = getattr(linear, "bias", None)
    quantize = torchao_fake_quant(dtype)
    return F.linear(quantize(x, block_size), quantize(weight, block_size), bias)


def _native(linear: nn.Module, x: Tensor, dtype: str, block_size: int | None) -> Tensor:
    """Run the projection through torchao's real low-precision GEMM.

    Not implemented yet. The native paths land with the kernel work: ``fp8`` builds
    on torchao's ``Float8Linear`` (already used by
    ``veomni/distributed/float8.py`` for module-swap training), and the FP4 layouts
    need Blackwell SM100+ dispatch. Refusing rather than falling back keeps a config
    that asks for native from silently training on emulation, which is the whole
    distinction the two modes exist to make.
    """
    raise NotImplementedError(
        f"linear_implementation='torchao' with dtype={dtype!r} and emulated=False is not implemented yet: "
        "the native path needs the low-precision GEMM for this format. Drop emulated=False to train "
        "against the emulated numerics, which needs no special hardware."
    )
