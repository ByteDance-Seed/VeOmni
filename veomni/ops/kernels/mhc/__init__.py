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

"""Manifold-constrained Hyper-Connection kernels used by DeepSeek-V4.

Three independent variants share this package. They are not a compound
kernel. Eager is the modeling math (regular autograd). ``tilelang`` is
TileKernels on SM90+. ``post`` / ``tilelang`` is a raw pair; ``pre`` and
``head`` keep an opaque wrapper because TileKernels owns that autograd.
``triton`` is the AMD ROCm counterpart: Triton kernels adapted from AMD
Primus around the FP32 mix GEMM, each owning its autograd. ``cann`` is the
Ascend CANN fused pre and post. CANN has no head kernel, so ``head`` /
``cann`` runs the eager math and lets one ``mhc_implementation`` name
cover all three variants.
"""

from ...platform import NVIDIA_SM90_PLUS, ROCM_GPU, GpuKernelRequirement, NpuKernelRequirement
from ...registry import register_op
from .head import eager as head_eager
from .head import tilelang as head_tilelang
from .head import triton as head_triton
from .post import cann as post_cann
from .post import eager as post_eager
from .post import tilelang as post_tilelang
from .post import triton as post_triton
from .pre import cann as pre_cann
from .pre import eager as pre_eager
from .pre import tilelang as pre_tilelang
from .pre import triton as pre_triton


_TILELANG = GpuKernelRequirement(platforms=(NVIDIA_SM90_PLUS,))
_TRITON = GpuKernelRequirement(platforms=(ROCM_GPU,))
_CANN = NpuKernelRequirement()

register_op(
    "mhc",
    "pre",
    "eager",
    description="PyTorch DeepSeek-V4 mHC pre transform with Sinkhorn normalization and collapse",
    wrapper=pre_eager.wrapper,
)

register_op(
    "mhc",
    "pre",
    "tilelang",
    description="TileKernels DeepSeek-V4 mHC pre transform with Sinkhorn normalization and collapse",
    wrapper=pre_tilelang.wrapper,
    requirement=_TILELANG,
    requires=("tile_kernels",),
)

register_op(
    "mhc",
    "pre",
    "triton",
    description="Triton DeepSeek-V4 mHC pre transform with Sinkhorn normalization and collapse for AMD ROCm",
    wrapper=pre_triton.wrapper,
    requirement=_TRITON,
    requires=("triton",),
)

register_op(
    "mhc",
    "pre",
    "cann",
    description="Ascend CANN DeepSeek-V4 mHC pre transform with Sinkhorn normalization and collapse",
    wrapper=pre_cann.wrapper,
    requirement=_CANN,
    requires=("cann_ops_transformer",),
)

register_op(
    "mhc",
    "post",
    "eager",
    description="PyTorch DeepSeek-V4 mHC residual post-mix",
    wrapper=post_eager.wrapper,
)

register_op(
    "mhc",
    "post",
    "tilelang",
    post_tilelang.forward,
    post_tilelang.backward,
    description="TileKernels DeepSeek-V4 mHC residual post-mix",
    requirement=_TILELANG,
    requires=("tile_kernels",),
)

register_op(
    "mhc",
    "post",
    "triton",
    description="Triton DeepSeek-V4 mHC residual post-mix for AMD ROCm",
    wrapper=post_triton.wrapper,
    requirement=_TRITON,
    requires=("triton",),
)

register_op(
    "mhc",
    "post",
    "cann",
    description="Ascend CANN DeepSeek-V4 mHC residual post-mix",
    wrapper=post_cann.wrapper,
    requirement=_CANN,
    requires=("cann_ops_transformer",),
)

register_op(
    "mhc",
    "head",
    "eager",
    description="PyTorch DeepSeek-V4 final mHC collapse",
    wrapper=head_eager.wrapper,
)

register_op(
    "mhc",
    "head",
    "tilelang",
    description="TileKernels DeepSeek-V4 final mHC collapse",
    wrapper=head_tilelang.wrapper,
    requirement=_TILELANG,
    requires=("tile_kernels",),
)

register_op(
    "mhc",
    "head",
    "triton",
    description="Triton DeepSeek-V4 final mHC collapse for AMD ROCm",
    wrapper=head_triton.wrapper,
    requirement=_TRITON,
    requires=("triton",),
)

register_op(
    "mhc",
    "head",
    "cann",
    description="PyTorch DeepSeek-V4 final mHC collapse for CANN builds (no CANN head kernel)",
    wrapper=head_eager.wrapper,
    requirement=_CANN,
)
