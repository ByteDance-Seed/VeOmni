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

"""Registry entries for manifold-constrained Hyper-Connection kernels."""

from ...kernel_registry import KERNEL_REGISTRY, HardwareRequirement, KernelSpec


def _mhc_pre_tile_kernels_factory():
    from .tile_kernels import mhc_pre_tile_kernels

    return mhc_pre_tile_kernels


def _mhc_post_tile_kernels_factory():
    from .tile_kernels import mhc_post_tile_kernels

    return mhc_post_tile_kernels


def _mhc_head_tile_kernels_factory():
    from .tile_kernels import mhc_head_tile_kernels

    return mhc_head_tile_kernels


for variant, factory, description in (
    ("pre", _mhc_pre_tile_kernels_factory, "TileKernels DeepSeek V4 mHC pre/Sinkhorn/collapse"),
    ("post", _mhc_post_tile_kernels_factory, "TileKernels DeepSeek V4 mHC residual post-mix"),
    ("head", _mhc_head_tile_kernels_factory, "TileKernels DeepSeek V4 final mHC collapse"),
):
    KERNEL_REGISTRY.register(
        KernelSpec(
            name="tilelang",
            op_name="mhc",
            variant=variant,
            factory=factory,
            hardware=HardwareRequirement(device_type="gpu", min_compute_capability=90),
            description=description,
        )
    )


__all__ = []
