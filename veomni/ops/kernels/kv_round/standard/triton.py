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

"""Triton blockwise FP8 (E4M3 + UE8M0 scale) storage round-trip."""

from __future__ import annotations

import torch
from torch import Tensor

from .....utils.device import get_torch_device, is_nvidia_cuda_device
from ....registry import SavedState
from . import eager as _eager


def _row_stride_if_1d_rows(t: Tensor, last: int, n_rows: int) -> int | None:
    if t.stride(-1) != 1:
        return None
    if t.is_contiguous() or n_rows <= 1:
        return last
    if t.ndim == 2:
        return t.stride(0)
    if t.ndim == 3 and t.shape[0] == 1:
        return t.stride(1)
    return None


def triton_fp8_roundtrip(
    x: Tensor,
    block_size: int = 32,
    *,
    quant_features: int = -1,
    inplace: bool = False,
    out: Tensor | None = None,
) -> Tensor:
    """Run Triton FP8 storage round-trip on CUDA or fall back to eager on CPU/other dtypes.

    ``quant_features`` limits the round trip to the leading channels; ``-1``
    quantizes the whole last dimension. The split is handled before the kernel
    because a group may not straddle it.
    """
    last = x.shape[-1]
    if block_size <= 0 or last % block_size != 0:
        raise ValueError(f"last dim {last} is not divisible by FP8 block size {block_size}")
    if quant_features != -1 and not (0 < quant_features <= last and quant_features % block_size == 0):
        raise ValueError(
            f"quant_features must fall in (0, {last}] and be divisible by block_size {block_size}, "
            f"got {quant_features}"
        )
    if quant_features != -1 and quant_features < last:
        quantized = triton_fp8_roundtrip(x[..., :quant_features].contiguous(), block_size, inplace=inplace)
        return torch.cat([quantized, x[..., quant_features:]], dim=-1)
    if is_nvidia_cuda_device(x.device) and x.dtype in (torch.bfloat16, torch.float16, torch.float32):
        from .._triton_kernels import _triton_fp8_roundtrip_kernel

        n_rows = x.numel() // last
        n_blocks = last // block_size
        x_stride = _row_stride_if_1d_rows(x, last, n_rows)
        z = x if x_stride is not None else x.contiguous()
        if x_stride is None:
            x_stride = last
        if out is not None:
            y = out
            y_stride = _row_stride_if_1d_rows(y, last, n_rows) or last
        elif inplace and z is x:
            y = x
            y_stride = x_stride
        else:
            y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
            y_stride = last
        if n_rows > 0:
            with get_torch_device().device(z.device):
                _triton_fp8_roundtrip_kernel[(n_rows, n_blocks)](
                    z, y, x_stride, y_stride, n_blocks, BLOCK_SIZE=block_size
                )
        return y
    res = _eager.fp8_roundtrip(x, block_size=block_size, quant_features=quant_features)
    if out is not None:
        out.copy_(res)
        return out
    return res


def forward(x: Tensor, *, block_size: int = 32, quant_features: int = -1) -> tuple[Tensor, SavedState]:
    """Run Triton FP8 storage round-trip."""
    return triton_fp8_roundtrip(x, block_size=block_size, quant_features=quant_features), SavedState(())


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor]:
    """Pass gradient straight through to the input."""
    del saved
    return (grad_output,)
