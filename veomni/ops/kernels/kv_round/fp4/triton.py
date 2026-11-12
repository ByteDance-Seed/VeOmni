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


def triton_fp4_roundtrip(
    x: Tensor,
    block_size: int = 16,
    scale_fmt: str = "e4m3",
    *,
    inplace: bool = False,
    out: Tensor | None = None,
) -> Tensor:
    """Run the Triton FP4 storage round-trip, falling back to eager elsewhere.

    ``inplace`` writes over ``x``; ``out`` writes into a caller-provided buffer.
    Both exist because a decode step wants the round trip to land in the KV slot
    it belongs to rather than in a fresh allocation per token.

    A request the kernel cannot serve falls back to the eager row rather than
    failing, because the eager row is the reference and this one is an
    optimization of it: a non-power-of-two ``block_size`` (Triton needs a
    power-of-two ``tl.arange`` extent), or an ``out`` whose row layout the kernel
    cannot address. Callers get the correct values either way.

    Raises:
        ValueError: ``inplace=True`` with an input the kernel and the eager row
            both cannot write in place. Silently returning a fresh tensor would
            leave the caller's buffer unchanged while the docstring promised
            otherwise.
    """
    if scale_fmt not in _eager.SCALE_FORMATS:
        raise ValueError(f"scale_fmt must be one of {_eager.SCALE_FORMATS}, got {scale_fmt!r}")
    last = x.shape[-1]
    if block_size <= 0 or last % block_size != 0:
        raise ValueError(f"last dim {last} is not divisible by FP4 group size {block_size}")
    if inplace and out is not None:
        raise ValueError("pass at most one of `inplace` and `out`; they describe two write targets.")

    n_rows = x.numel() // last
    # What the kernel needs: a pow2 block axis, and an addressable row layout for
    # both source and destination. Anything else is eager's job.
    triton_can_run = (
        is_nvidia_cuda_device(x.device)
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and _pow2_block(block_size)
        and _row_stride_if_1d_rows(x, last, n_rows) is not None
        and (out is None or _row_stride_if_1d_rows(out, last, n_rows) is not None)
    )
    if not triton_can_run:
        if inplace:
            raise ValueError(
                "inplace=True requires a CUDA tensor with a row-major contiguous last dimension; "
                "this call cannot be written in place. Use the default (out-of-place) form."
            )
        result = _eager.fp4_roundtrip(x, block_size=block_size, scale_fmt=scale_fmt)
        if out is not None:
            out.copy_(result)
            return out
        return result

    from .._triton_kernels import _triton_fp4_roundtrip_kernel

    use_e4m3 = scale_fmt == "e4m3"
    n_blocks = last // block_size
    x_stride = _row_stride_if_1d_rows(x, last, n_rows)
    if out is not None:
        y = out
        y_stride = _row_stride_if_1d_rows(y, last, n_rows)
    elif inplace:
        y = x
        y_stride = x_stride
    else:
        y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        y_stride = last
    if n_rows > 0:
        with get_torch_device().device(x.device):
            _triton_fp4_roundtrip_kernel[(n_rows, n_blocks)](
                x,
                y,
                x_stride,
                y_stride,
                n_blocks,
                BLOCK_SIZE=block_size,
                USE_E4M3_SCALE=use_e4m3,
            )
    return y


def forward(x: Tensor, *, block_size: int = 16, scale_fmt: str = "e4m3") -> tuple[Tensor, SavedState]:
    """Run Triton FP4 storage round-trip."""
    return triton_fp4_roundtrip(x, block_size=block_size, scale_fmt=scale_fmt), SavedState(())


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor]:
    """Pass gradient straight through to the input."""
    del saved
    return (grad_output,)
