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


"""Qwen4-Exp query-sparse attention (QSA).

Both rows consume the compact ``[B, S, K]`` global token selection the QSA
indexer produces. ``eager`` expands it to a dense mask and runs quadratic GQA
as a portable numerical reference. ``tilelang`` gathers the selected K/V rows
and never materializes the score tensor, which is what makes 16K-token
training feasible.

The selection itself, its packed-block boundaries, and the Ulysses exchange
belong to the model, because only the model knows the sequence layout. These
rows see one local attention call.
"""

from ...platform import NVIDIA_GPU, GpuKernelRequirement
from ...registry import register_op
from .standard import eager as qsa_eager
from .standard import tilelang as qsa_tilelang


register_op(
    "qsa_attention",
    "standard",
    "eager",
    description="PyTorch dense-mask QSA reference",
    wrapper=qsa_eager.wrapper,
)

register_op(
    "qsa_attention",
    "standard",
    "tilelang",
    description="TileLang gather-based QSA sparse attention",
    wrapper=qsa_tilelang.wrapper,
    requirement=GpuKernelRequirement(platforms=(NVIDIA_GPU,)),
    requires=("tilelang",),
)
