# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""

from ...registry import register_kernel
from ...requirement import CudaKernelRequirement
from .head import eager as head_eager
from .head import tilelang as head_tilelang
from .post import eager as post_eager
from .post import tilelang as post_tilelang
from .pre import eager as pre_eager
from .pre import tilelang as pre_tilelang


_TILELANG = CudaKernelRequirement(min_cc=90)

register_kernel("mhc", "pre", "eager", wrapper=pre_eager.wrapper)

register_kernel("mhc", "pre", "tilelang", wrapper=pre_tilelang.wrapper, requirement=_TILELANG)

register_kernel("mhc", "post", "eager", wrapper=post_eager.wrapper)

register_kernel("mhc", "post", "tilelang", post_tilelang.forward, post_tilelang.backward, requirement=_TILELANG)

register_kernel("mhc", "head", "eager", wrapper=head_eager.wrapper)

register_kernel("mhc", "head", "tilelang", wrapper=head_tilelang.wrapper, requirement=_TILELANG)
