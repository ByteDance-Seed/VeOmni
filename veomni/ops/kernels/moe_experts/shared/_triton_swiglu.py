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

"""Optional Triton kernel for the shared merged-FC1 SwiGLU activation."""

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
