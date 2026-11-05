# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SM100+ CUDA FA4 kernel selection for MagiAttention."""

from functools import cache

import torch

from ......utils.device import get_gpu_compute_capability


KERNEL_UNSUPPORTED = "unsupported"
KERNEL_CUTE_JIT = "cute_jit"
CUDA_DEVICE_TYPE = "cuda"


def get_kernel_mode(device: torch.device) -> str:
    """Resolve the CUDA FA4 implementation selected for the query device."""
    if device.type != CUDA_DEVICE_TYPE or torch.version.hip is not None:
        return KERNEL_UNSUPPORTED

    compute_capability = get_gpu_compute_capability(device)
    if compute_capability >= 100:
        return KERNEL_CUTE_JIT
    return KERNEL_UNSUPPORTED


@cache
def prepare_kernel(device: torch.device) -> None:
    """Validate that the query device supports MagiAttention's CUTE JIT path."""
    kernel_mode = get_kernel_mode(device)
    if kernel_mode == KERNEL_UNSUPPORTED:
        if device.type == CUDA_DEVICE_TYPE and torch.version.hip is not None:
            hardware = "ROCm"
        else:
            compute_capability = get_gpu_compute_capability(device) if device.type == CUDA_DEVICE_TYPE else 0
            hardware = f"SM{compute_capability}" if compute_capability else device.type
        raise RuntimeError(
            f"VeOmni `magi_attention` does not support {hardware}; use an NVIDIA SM100+ GPU with CUTE DSL/JIT."
        )
    # MagiAttention compiles and caches the SM100+ kernel on the first FA4 call.
