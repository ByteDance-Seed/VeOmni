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

"""Qwen4-Exp QSA indexer (block selection) kernels.

The top-k block selection has two implementations: ``eager``, the reference
``einsum + ReLU + topk`` loop that lives in the generated modeling file and
is the default on every device, and ``npu_ascendc``, the CANN
LightningIndexerV2 operator registered here. The model binds ``npu_ascendc``
as an instance-local ``VeomniOp`` and calls it only when that implementation
is selected; the eager loop never goes through the registry.
"""

from ...platform import NpuKernelRequirement
from ...registry import register_op
from .standard import npu_ascendc as qsa_ascendc


register_op(
    "qsa_indexer",
    "standard",
    "npu_ascendc",
    description="Ascend CANN Lightning Indexer QSA block selection",
    wrapper=qsa_ascendc.wrapper,
    requirement=NpuKernelRequirement(),
    requires=("cann_ops_transformer",),
)
