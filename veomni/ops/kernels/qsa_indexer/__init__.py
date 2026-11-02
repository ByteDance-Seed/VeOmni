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

"""Qwen4-Exp QSA indexer block selection (``eager``, ``cann``).

Both rows consume the pooled/normed/RoPE-rotated compressed keys and the
normed/RoPE-rotated index queries the model prepares, and return per-row
compressed-block ids with ``-1`` padding. The incomplete tail block, the
dense-prefix rows, and the Ulysses/CP exchange belong to the model, because
only the model knows the sequence layout; these rows see one selection call
for the single-segment causal case (``world_size == 1``, equal-length,
non-packed batch). ``cann`` is the ``cann_ops_transformer`` Lightning Indexer
(Ascend NPU).
"""

from ...platform import NpuKernelRequirement
from ...registry import register_op
from . import cann as qsa_indexer_cann
from . import eager as qsa_indexer_eager


register_op(
    "qsa_indexer",
    "standard",
    "eager",
    description="PyTorch dense-score QSA block-selection reference",
    wrapper=qsa_indexer_eager.wrapper,
)

register_op(
    "qsa_indexer",
    "standard",
    "cann",
    description="Ascend CANN Lightning Indexer QSA block selection",
    wrapper=qsa_indexer_cann.wrapper,
    requirement=NpuKernelRequirement(),
    requires=("cann_ops_transformer",),
)
