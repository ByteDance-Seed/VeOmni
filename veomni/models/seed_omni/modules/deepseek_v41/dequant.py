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

"""Undo the released DeepSeek-V4.1 checkpoint's two quantizations, and redo them for export.

The checkpoint ships every dense projection as fp8 with a block scale, and the
routed experts as fp4 packed two values to a byte with a block scale of their
own. Training wants bf16 master weights in both cases -- the fp4 QAT in
``DeepseekV4Experts`` fake-quantizes *from* bf16, so handing it fp4 storage
would quantize an already-quantized tensor and lose a second time.

Both routines follow ``inference/convert.py`` and the released
``config.json``'s ``quantization_config``: block size ``[32, 32]``, scale
format ``ue8m0``. The block size is read from the config rather than assumed,
because the same file also admits ``[128, 128]`` and the reference asserts on
exactly those two.

The fp4 value table is not a computation: e2m1 has 16 representable values and
the table lists them in bit order, sign in the high bit. It is reproduced here
rather than derived so that it can be compared against the reference's copy
element by element.
"""

from __future__ import annotations

from typing import Sequence

import torch


#: e2m1 in bit order: 8 magnitudes, then the same 8 negated. From
#: ``inference/convert.py``; ``tests`` asserts the two agree.
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def fp4_table(device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor(FP4_VALUES, dtype=torch.float32, device=device)


def dequant_fp8_blockwise(
    weight: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """``[out, in]`` fp8 plus its ``[out/bo, in/bi]`` block scale -> ``out_dtype``.

    The block shape is inferred from the two tensors rather than passed in: the
    checkpoint's own scale tensor is the authority on how its weight was
    blocked, and a config value that disagreed with it would silently rescale
    every block. The reference derives it the same way.

    Two block shapes occur, and the difference is structural rather than a
    tuning choice. The 2-D projections are blocked squarely, ``(32, 32)`` --
    which is the case the reference asserts on, because ``wo_a`` is the only
    tensor it dequantizes. The engram tables are blocked ``(1, 32)``: one scale
    per row per 32 columns. A row of an embedding table is an independent
    memory, so sharing an exponent across 32 unrelated rows would cost range on
    every one of them.
    """
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(f"expected 2-D weight and scale, got {tuple(weight.shape)} and {tuple(scale.shape)}")
    if weight.size(0) % scale.size(0) or weight.size(1) % scale.size(1):
        raise ValueError(
            f"weight {tuple(weight.shape)} is not an integer number of blocks of scale {tuple(scale.shape)}"
        )
    out_block = weight.size(0) // scale.size(0)
    in_block = weight.size(1) // scale.size(1)
    if in_block not in (32, 128) or out_block not in (1, 32, 128):
        raise ValueError(
            f"unsupported fp8 block {(out_block, in_block)}: the released checkpoint uses (32, 32) for the "
            "projections and (1, 32) for the engram tables, and the reference also accepts (128, 128). A "
            "block outside that set means the scale was read for the wrong tensor, which would rescale "
            "every block rather than fail."
        )
    blocked = weight.unflatten(0, (-1, out_block)).unflatten(-1, (-1, in_block)).float()
    blocked = blocked * scale[:, None, :, None].float()
    return blocked.flatten(2, 3).flatten(0, 1).to(out_dtype)


def dequant_fp4_blockwise(
    packed: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """``[out, in/2]`` packed e2m1 plus its block scale -> ``[out, in]`` ``out_dtype``.

    Two values to a byte, **low nibble first**: the reference stacks
    ``[table[low], table[high]]`` and flattens, so within a byte the lower index
    along the input dimension is the low nibble. Swapping them transposes every
    adjacent pair of input features, which no shape check would catch.
    """
    if packed.dtype not in (torch.int8, torch.uint8):
        raise ValueError(f"packed fp4 must arrive as int8/uint8, got {packed.dtype}")
    if packed.ndim != 2:
        raise ValueError(f"expected a 2-D packed weight, got {tuple(packed.shape)}")

    as_bytes = packed.view(torch.uint8)
    low = as_bytes & 0x0F
    high = (as_bytes >> 4) & 0x0F
    table = fp4_table(packed.device)
    values = torch.stack([table[low.long()], table[high.long()]], dim=-1).flatten(1)

    out_dim, in_dim = values.shape
    if scale.size(0) != out_dim or in_dim % scale.size(1):
        raise ValueError(f"fp4 scale {tuple(scale.shape)} does not block the unpacked weight {(out_dim, in_dim)}")
    in_block = in_dim // scale.size(1)
    values = values.unflatten(-1, (-1, in_block)) * scale.float()[:, :, None]
    return values.flatten(1).to(out_dtype)


#: Largest finite magnitudes of the two element formats.
FP8_E4M3_MAX = 448.0
FP4_E2M1_MAX = 6.0

#: Midpoints between consecutive e2m1 magnitudes. A magnitude rounds to the
#: smaller neighbour on a tie.
_FP4_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)

#: Smallest block amax a scale is derived from, as in the reference, so an
#: all-zero block gets a finite scale.
_MIN_AMAX = 1e-4


def _ue8m0_scale(amax: torch.Tensor, element_max: float) -> torch.Tensor:
    """The power-of-two scale ``ue8m0`` stores: the smallest that fits ``amax`` in the element range."""
    return torch.exp2(torch.ceil(torch.log2(amax.clamp_min(_MIN_AMAX) / element_max)))


def quant_fp8_blockwise(
    weight: torch.Tensor,
    block: tuple[int, int] = (32, 32),
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[out, in]`` -> fp8 e4m3 weight and its ``[out/bo, in/bi]`` ue8m0 block scale.

    The inverse of :func:`dequant_fp8_blockwise` on its own output: a weight it
    dequantized requantizes to values that dequantize to the same numbers,
    because the requantized scale is a power of two no larger than the
    original and dividing by it only shifts exponents. ``block`` is ``(32, 32)``
    for the projections and ``(1, 32)`` for the engram tables.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got {tuple(weight.shape)}")
    out_block, in_block = block
    if weight.size(0) % out_block or weight.size(1) % in_block:
        raise ValueError(f"weight {tuple(weight.shape)} is not an integer number of {block} blocks")
    blocked = weight.float().unflatten(0, (-1, out_block)).unflatten(-1, (-1, in_block))
    scale = _ue8m0_scale(blocked.abs().amax(dim=(1, 3)), FP8_E4M3_MAX)
    values = (blocked / scale[:, None, :, None]).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    quantized = values.flatten(2, 3).flatten(0, 1).to(torch.float8_e4m3fn)
    return quantized, scale.to(torch.float8_e8m0fnu)


def quant_fp4_blockwise(weight: torch.Tensor, block: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """``[out, in]`` -> packed e2m1 ``[out, in/2]`` int8 and its ``[out, in/block]`` ue8m0 scale.

    The inverse of :func:`dequant_fp4_blockwise`, with the same nibble order:
    the lower input index is the low nibble. Each value rounds to the nearest
    e2m1 magnitude and keeps its sign bit, so a small negative value is stored
    as ``-0``, as the release stores it.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got {tuple(weight.shape)}")
    if weight.size(1) % block or block % 2:
        raise ValueError(f"weight {tuple(weight.shape)} does not split into blocks of {block} columns")
    blocked = weight.float().unflatten(-1, (-1, block))
    scale = _ue8m0_scale(blocked.abs().amax(dim=-1), FP4_E2M1_MAX)
    values = (blocked / scale[..., None]).flatten(1)
    midpoints = torch.tensor(_FP4_MIDPOINTS, dtype=torch.float32, device=weight.device)
    codes = torch.bucketize(values.abs(), midpoints).to(torch.uint8)
    codes = codes | torch.signbit(values).to(torch.uint8) << 3
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.view(torch.int8), scale.to(torch.float8_e8m0fnu)


def fuse_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``w1`` and ``w3`` -> the ``[2I, H]`` slab ``DeepseekV4Experts`` expects.

    Order is not a convention to pick: ``_apply_gate`` does
    ``gate, up = gate_up.chunk(2, dim=-1)`` on the *output* of the projection,
    so the gate rows come first. Reversing them swaps which half gets the
    one-sided clamp and which gets the two-sided one -- the model still runs,
    and the outputs are wrong everywhere the clamp binds.
    """
    if gate.shape != up.shape:
        raise ValueError(f"w1 {tuple(gate.shape)} and w3 {tuple(up.shape)} must match")
    return torch.cat([gate, up], dim=0)


def stack_experts(per_expert: Sequence[torch.Tensor]) -> torch.Tensor:
    """``[E, ...]`` from one tensor per expert, in expert-index order."""
    return torch.stack(list(per_expert), dim=0)


def split_block_shape(weight: torch.Tensor, scale: torch.Tensor) -> tuple[int, int]:
    """The ``(out, in)`` block the scale implies. Exposed for diagnostics."""
    return weight.size(0) // scale.size(0), weight.size(1) // scale.size(1)


__all__ = [
    "FP4_E2M1_MAX",
    "FP4_VALUES",
    "FP8_E4M3_MAX",
    "dequant_fp4_blockwise",
    "dequant_fp8_blockwise",
    "fp4_table",
    "fuse_gate_up",
    "quant_fp4_blockwise",
    "quant_fp8_blockwise",
    "split_block_shape",
    "stack_experts",
]
