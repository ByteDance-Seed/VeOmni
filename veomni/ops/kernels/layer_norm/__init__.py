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

"""Affine LayerNorm kernel family.

The ``standard`` variant is affine LayerNorm with a last-dim ``normalized_shape``.
It registers an eager ``native_layer_norm`` row plus an optional Apex fused row.
"""

from ...platform import GpuKernelRequirement
from ...registry import register_op
from .standard import apex as standard_apex
from .standard import eager as standard_eager


register_op(
    "layer_norm",
    "standard",
    "eager",
    standard_eager.forward,
    standard_eager.backward,
    description="PyTorch affine LayerNorm via native_layer_norm",
)

register_op(
    "layer_norm",
    "standard",
    "apex",
    standard_apex.forward,
    standard_apex.backward,
    description="Apex fused affine LayerNorm via fused_layer_norm_cuda",
    requirement=GpuKernelRequirement(),
    requires=("fused_layer_norm_cuda",),
)
