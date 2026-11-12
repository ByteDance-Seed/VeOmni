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

"""Triton JIT kernel for the FP8 KV/indexer storage round-trip."""

import triton
import triton.language as tl


@triton.jit
def _triton_fp8_roundtrip_kernel(
    X,
    Y,
    X_STRIDE_ROW,
    Y_STRIDE_ROW,
    N_BLOCKS,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col_block = tl.program_id(1).to(tl.int64)
    col_offs = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + row * X_STRIDE_ROW + col_offs).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1.0e-4)
    scaled = amax * (1.0 / 448.0)
    bits = scaled.to(tl.int32, bitcast=True)
    exp_byte = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = (exp_byte << 23).to(tl.float32, bitcast=True)
    q = tl.minimum(tl.maximum(x / scale, -448.0), 448.0).to(tl.float8e4nv).to(tl.float32)
    tl.store(Y + row * Y_STRIDE_ROW + col_offs, (q * scale).to(Y.dtype.element_ty))
