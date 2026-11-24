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

"""Triton blockwise FP4 (E2M1 + E4M3/UE8M0 scale) storage round-trip."""

from __future__ import annotations

import torch
from torch import Tensor

from .....utils.device import get_torch_device, is_nvidia_cuda_device
from ....registry import SavedState
from ..standard.triton import _row_stride_if_1d_rows
from . import eager as _eager


def _pow2_block(block_size: int) -> bool:
    """Whether ``block_size`` is a power of two, i.e. usable as a kernel extent."""
    return block_size > 0 and not (block_size & (block_size - 1))


def triton_fp4_roundtrip(x: Tensor, block_size: int = 16, scale_fmt: str = "e4m3") -> Tensor:
    """Run the Triton FP4 storage round-trip, falling back to eager elsewhere.

    What falls back is the requests the *kernel* cannot serve at all: a
    non-power-of-two ``block_size`` (Triton needs a power-of-two ``tl.arange``
    extent), or a dtype it does not compile for. Callers get the same values either
    way; only the path differs.

    A layout the kernel's static strides cannot address does **not** fall back --
    the operand is copied with ``.contiguous()`` and rounded by the kernel. Treating
    layout as a reason to use the eager row would silently push a whole class of real
    inputs (a transposed 4D tensor, a strided 3D one with ``batch > 1``) onto the slow
    path, which is the opposite of what selecting this row asks for.

    There is no ``inplace`` / ``out`` form. Both existed in an earlier revision of
    this file with no caller, and a store that writes into the caller's KV slot
    would need the caller to hand over that slot -- nothing does yet. Adding it
    back means deciding what happens when the layout cannot be written in place;
    the FP8 row's answer (forward ``inplace`` and reject when unsupported) is the
    precedent.
    """
    if scale_fmt not in _eager.SCALE_FORMATS:
        raise ValueError(f"scale_fmt must be one of {_eager.SCALE_FORMATS}, got {scale_fmt!r}")
    last = x.shape[-1]
    if block_size <= 0 or last % block_size != 0:
        raise ValueError(f"last dim {last} is not divisible by FP4 group size {block_size}")
    if x.numel() == 0:
        # Bail out before the row count: ``0 // 0`` raises, and an empty operand is a
        # legitimate input the eager row accepts.
        return x.clone()

    n_rows = x.numel() // last
    if not (
        is_nvidia_cuda_device(x.device)
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and _pow2_block(block_size)
    ):
        return _eager.fp4_roundtrip(x, block_size=block_size, scale_fmt=scale_fmt)

    from .._triton_kernels import _triton_fp4_roundtrip_kernel

    n_blocks = last // block_size
    # A layout ``_row_stride_if_1d_rows`` cannot express -- a transposed 4D tensor
    # such as ``q.transpose(1, 2)``, or a strided 3D one with ``batch > 1`` -- is
    # made contiguous and rounded here rather than handed to the eager row. The
    # eager row is the reference for *numerics*, not a substitute for the kernel:
    # falling back on layout alone would silently move a whole class of real inputs
    # onto the slow path. ``kv_round/standard/triton.py`` answers a layout it cannot
    # address the same way, so the two rows agree.
    x_stride = _row_stride_if_1d_rows(x, last, n_rows)
    z = x if x_stride is not None else x.contiguous()
    if x_stride is None:
        x_stride = last
    y = torch.empty(z.shape, dtype=z.dtype, device=z.device)
    if n_rows > 0:
        with get_torch_device().device(z.device):
            _triton_fp4_roundtrip_kernel[(n_rows, n_blocks)](
                z,
                y,
                x_stride,
                last,
                BLOCK_SIZE=block_size,
                USE_E4M3_SCALE=scale_fmt == "e4m3",
            )
    return y


def forward(x: Tensor, *, block_size: int = 16, scale_fmt: str = "e4m3") -> tuple[Tensor, SavedState]:
    """Run Triton FP4 storage round-trip."""
    return triton_fp4_roundtrip(x, block_size=block_size, scale_fmt=scale_fmt), SavedState(())


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor]:
    """Pass gradient straight through to the input."""
    del saved
    return (grad_output,)
