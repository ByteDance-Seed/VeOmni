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

"""DSV4 checkpoint export quantization.

TileLang is optional and GPU-only. Importing this module must not load
``act_quant``. Callers go through the wrappers, which reject unsupported GPUs first.
"""

import torch

from veomni.ops.qat._hardware import TILELANG_GPU_DESCRIPTION, is_tilelang_gpu_supported, require_tilelang_fp4_gpu


def _require_tilelang_gpu() -> None:
    if not is_tilelang_gpu_supported():
        raise RuntimeError(f"DeepSeek V4 TileLang kernels require {TILELANG_GPU_DESCRIPTION}")


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    dequant: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to packed E2M1 FP4 with E8M0 block scales on SM90+ CUDA GPUs."""
    _require_tilelang_gpu()
    require_tilelang_fp4_gpu()
    from veomni.ops.qat.quant import fp4_act_quant as impl

    return impl(x, block_size, dequant)


def fp8_weight_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D bfloat16 weight tensor to block-scaled FP8 on a TileLang GPU."""
    _require_tilelang_gpu()
    from veomni.ops.qat.quant import fp8_weight_quant as impl

    return impl(x, block_size, scale_fmt, scale_dtype)


__all__ = [
    "fp4_act_quant",
    "fp8_weight_quant",
]
