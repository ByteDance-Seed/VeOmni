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
# See the License for the specific language governing limitations
# under the License.

"""DSV4 checkpoint export quantization.

TileLang is optional and GPU-only. Importing this module must not load
``act_quant``. Callers go through the wrappers, which reject non-SM90 first.
"""

import torch

from veomni.utils.device import IS_CUDA_AVAILABLE, get_gpu_compute_capability


def _require_tilelang_sm90() -> None:
    if torch.version.hip is not None or not IS_CUDA_AVAILABLE or get_gpu_compute_capability() < 90:
        raise RuntimeError("DeepSeek V4 TileLang kernels require an SM90 or later NVIDIA CUDA GPU")


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    _require_tilelang_sm90()
    from .act_quant import fp4_act_quant as impl

    return impl(x, block_size, inplace)


def fp8_weight_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_tilelang_sm90()
    from .act_quant import fp8_weight_quant as impl

    return impl(x, block_size, scale_fmt, scale_dtype)


__all__ = [
    "fp4_act_quant",
    "fp8_weight_quant",
]
