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

A row here answers "round this stored tensor the way inference would". The
variant is the *recipe* -- which geometry a model stores -- and the
implementation is the backend that performs it.

* ``standard`` is model-agnostic: the caller passes ``block_size`` and, if it
  wants to stop partway through the last dimension, ``quant_features``.
* ``deepseek_v4`` fixes DeepSeek-V4's two geometries in one place: 64-wide FP8
  groups over the NoPE channels of a KV entry, and 128-wide groups over a whole
  indexer head. A caller selects which by passing ``rope_features``.

Both variants also register ``none``, which returns the tensor unchanged. Every
other family spells "off" as ``eager``; here ``eager`` is a real rounding row, so
"off" needs its own name. That keeps the field shaped like the others -- the model
always builds a handle and never branches on a missing one.

These are *storage* round trips, not GEMM operand quantization: the point is to
show training the values an inference cache would hold. Backward is a
straight-through estimator, which is sound here because every scale derives from
its own group's amax and rounds up, so no element saturates.
"""

from ...platform import GpuKernelRequirement
from ...registry import register_op
from .deepseek_v4 import eager as dsv4_eager
from .deepseek_v4 import none as dsv4_none
from .deepseek_v4 import triton as dsv4_triton
from .standard import eager as standard_eager
from .standard import none as standard_none
from .standard import triton as standard_triton


_GPU = GpuKernelRequirement()

register_op(
    "kv_round",
    "standard",
    "none",
    standard_none.forward,
    standard_none.backward,
    description="Leave stored tensors unchanged; the off state for this family",
)

register_op(
    "kv_round",
    "standard",
    "eager",
    standard_eager.forward,
    standard_eager.backward,
    description="PyTorch blockwise FP8 (E4M3 + UE8M0 scale) storage round-trip with straight-through backward",
)

register_op(
    "kv_round",
    "standard",
    "triton",
    standard_triton.forward,
    standard_triton.backward,
    description="Triton blockwise FP8 (E4M3 + UE8M0 scale) storage round-trip with straight-through backward",
    requirement=_GPU,
    requires=("triton",),
    default=True,
)

register_op(
    "kv_round",
    "deepseek_v4",
    "none",
    dsv4_none.forward,
    dsv4_none.backward,
    description="Leave stored tensors unchanged; the off state for this family",
)

register_op(
    "kv_round",
    "deepseek_v4",
    "eager",
    dsv4_eager.forward,
    dsv4_eager.backward,
    description=(
        "PyTorch DeepSeek-V4 storage rounding: 64-wide FP8 groups over the NoPE channels of a KV "
        "entry, 128-wide over a whole indexer head, straight-through backward"
    ),
)

register_op(
    "kv_round",
    "deepseek_v4",
    "triton",
    dsv4_triton.forward,
    dsv4_triton.backward,
    description=(
        "Triton DeepSeek-V4 storage rounding: the same two geometries as the eager row on the "
        "Triton round trip, straight-through backward"
    ),
    requirement=_GPU,
    requires=("triton",),
    default=True,
)
