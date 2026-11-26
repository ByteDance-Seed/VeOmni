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

"""Optional Triton kernels for the shared merged-FC1 SwiGLU forward/backward."""

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
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


@triton.jit
def _merged_swiglu_backward_kernel(
    gate_ptr,
    up_ptr,
    grad_ptr,
    gate_mask_ptr,
    up_mask_ptr,
    activation_ptr,
    grad_output_ptr,
    numel,
    columns: tl.constexpr,
    gate_stride_m: tl.constexpr,
    gate_stride_n: tl.constexpr,
    up_stride_m: tl.constexpr,
    up_stride_n: tl.constexpr,
    grad_stride_m: tl.constexpr,
    grad_stride_n: tl.constexpr,
    CLAMPED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < numel
    rows, cols = offsets // columns, offsets % columns
    gate = tl.load(gate_ptr + rows * gate_stride_m + cols * gate_stride_n, valid, other=0).to(tl.float32)
    up = tl.load(up_ptr + rows * up_stride_m + cols * up_stride_n, valid, other=0).to(tl.float32)
    grad = tl.load(grad_ptr + rows * grad_stride_m + cols * grad_stride_n, valid, other=0).to(tl.float32)
    denominator = 1.0 + libdevice.exp(-gate)
    silu = libdevice.div_rn(gate, denominator).to(gate_ptr.dtype.element_ty).to(tl.float32)
    activation = silu * up
    # Match both materialized products before ATen's SiLU backward.
    grad_gate_activation = (grad * up).to(gate_ptr.dtype.element_ty).to(tl.float32)
    sigmoid = libdevice.div_rn(1.0, denominator)
    # ATen's CUDA SiLU derivative contracts this multiply-add.
    derivative = libdevice.fma(gate, 1.0 - sigmoid, 1.0)
    grad_gate = grad_gate_activation * sigmoid * derivative
    grad_up = silu * grad
    if CLAMPED:
        gate_mask = tl.load(gate_mask_ptr + offsets, valid, other=False)
        up_mask = tl.load(up_mask_ptr + offsets, valid, other=False)
        grad_gate = tl.where(gate_mask, grad_gate, 0.0)
        grad_up = tl.where(up_mask, grad_up, 0.0)
    tl.store(activation_ptr + offsets, activation, valid)
    tl.store(grad_output_ptr + rows * (2 * columns) + cols, grad_gate, valid)
    tl.store(grad_output_ptr + rows * (2 * columns) + columns + cols, grad_up, valid)
