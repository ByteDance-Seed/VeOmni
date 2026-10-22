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

"""SwiGLU clamping and merged-FC1 forward activation shared across MoE experts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from .....utils.device import get_torch_device, is_nvidia_cuda_device


if TYPE_CHECKING:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
else:
    triton: Any = None
    tl: Any = None
    libdevice: Any = None

_MERGED_SWIGLU_FORWARD_KERNEL: Any = None


def apply_swiglu_clamp(fc1_1_output, fc1_2_output, swiglu_limit):
    """gpt-oss / DeepSeek-V4 style clamped SwiGLU pre-activation.

    Returns ``(fc1_1_clamped, fc1_2_clamped, mask_fc1_1, mask_fc1_2)``.
    When ``swiglu_limit`` is ``None`` this is a no-op and masks are ``None``;
    callers may skip the corresponding mask multiplications in backward.

    Semantics mirror ``torch.clamp`` autograd: gradients vanish where the
    pre-clamp value falls outside the bound. ``gate`` is upper-bounded only
    (``max=limit``) so the negative tail still feeds SiLU; ``up`` is
    symmetrically clamped to ``[-limit, +limit]``.
    """
    if swiglu_limit is None:
        return fc1_1_output, fc1_2_output, None, None
    mask_fc1_1 = fc1_1_output <= swiglu_limit
    mask_fc1_2 = (fc1_2_output >= -swiglu_limit) & (fc1_2_output <= swiglu_limit)
    fc1_1_output = fc1_1_output.clamp(max=swiglu_limit)
    fc1_2_output = fc1_2_output.clamp(min=-swiglu_limit, max=swiglu_limit)
    return fc1_1_output, fc1_2_output, mask_fc1_1, mask_fc1_2


def _merged_swiglu_forward_kernel(
    gate_ptr,
    up_ptr,
    output_ptr,
    numel,
    columns: tl.constexpr,
    gate_stride_m: tl.constexpr,
    gate_stride_n: tl.constexpr,
    up_stride_m: tl.constexpr,
    up_stride_n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    rows = offsets // columns
    cols = offsets % columns
    gate = tl.load(gate_ptr + rows * gate_stride_m + cols * gate_stride_n, mask=mask, other=0).to(tl.float32)
    up = tl.load(up_ptr + rows * up_stride_m + cols * up_stride_n, mask=mask, other=0).to(tl.float32)

    # Preserve ATen's materialized SiLU rounding before the separate product.
    silu = libdevice.div_rn(gate, 1.0 + libdevice.exp(-gate)).to(gate_ptr.dtype.element_ty)
    output = silu.to(tl.float32) * up
    tl.store(output_ptr + offsets, output, mask=mask)


def _get_merged_swiglu_forward_kernel():
    global _MERGED_SWIGLU_FORWARD_KERNEL, libdevice, tl, triton
    if _MERGED_SWIGLU_FORWARD_KERNEL is None:
        import triton as _triton
        import triton.language as _tl
        from triton.language.extra import libdevice as _libdevice

        triton = _triton
        tl = _tl
        libdevice = _libdevice
        _MERGED_SWIGLU_FORWARD_KERNEL = _triton.jit(_merged_swiglu_forward_kernel)
    return _MERGED_SWIGLU_FORWARD_KERNEL


def merged_swiglu_forward(gate: Tensor, up: Tensor) -> Tensor:
    """Fuse staged ``silu(gate) * up`` for merged-FC1 halves on NVIDIA CUDA.

    Called inside both non-EP and EP merged-FC1 custom autograd functions, which
    continue to own saved tensors and backward math. Falls back to the staged
    ATen expression on non-NVIDIA backends (ROCm, MLU).
    """
    assert gate.ndim == 2 and up.shape == gate.shape
    assert gate.device == up.device
    if not is_nvidia_cuda_device(gate.device):
        return torch.ops.aten.silu(gate) * up
    assert gate.dtype == up.dtype and gate.dtype in (torch.bfloat16, torch.float16)

    output = gate.new_empty(gate.shape)
    if output.numel() == 0:
        return output
    kernel = _get_merged_swiglu_forward_kernel()
    with get_torch_device().device(gate.device):
        kernel[(triton.cdiv(output.numel(), 1024),)](
            gate,
            up,
            output,
            output.numel(),
            columns=gate.shape[1],
            gate_stride_m=gate.stride(0),
            gate_stride_n=gate.stride(1),
            up_stride_m=up.stride(0),
            up_stride_n=up.stride(1),
            BLOCK_SIZE=1024,
            enable_fp_fusion=False,
        )
    return output
