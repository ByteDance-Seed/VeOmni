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

"""KV cache and indexer activation storage-rounding kernel family.

Variant ``fp8`` rounds contiguous ``block_size``-wide channel groups to
``float8_e4m3fn`` with ``ue8m0`` power-of-two scales. That is the only variant
because it is the only storage format in use: the routed experts of a
V4-Flash checkpoint are FP4, but they are weights consumed by a GEMM, which the
``moe_experts`` family quantizes, not a cache entry.

These are *storage* round trips, not GEMM operand quantization: the point is to
show training the values an inference cache would hold. ``quant_features``
selects a leading prefix so a call can stop at the boundary between the NoPE
channels DeepSeek-V4 stores quantized and the RoPE channels it keeps in BF16.
Backward is a straight-through estimator, which is sound here because every
scale derives from its own group's amax and rounds up, so no element saturates.
"""

from ...platform import GpuKernelRequirement
from ...registry import register_op
from .fp8 import eager as fp8_eager
from .fp8 import triton as fp8_triton


_GPU = GpuKernelRequirement()

register_op(
    "kv_round",
    "fp8",
    "eager",
    fp8_eager.forward,
    fp8_eager.backward,
    description="PyTorch blockwise FP8 (E4M3 + UE8M0 scale) storage round-trip with straight-through backward",
)

register_op(
    "kv_round",
    "fp8",
    "triton",
    fp8_triton.forward,
    fp8_triton.backward,
    description="Triton blockwise FP8 (E4M3 + UE8M0 scale) storage round-trip with straight-through backward",
    requirement=_GPU,
    requires=("triton",),
    default=True,
)
