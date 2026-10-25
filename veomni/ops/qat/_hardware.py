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

"""Hardware checks shared by the QAT TileLang facades."""

from functools import cache

from ..platform import NVIDIA_SM90_PLUS, TILELANG_GPUS


TILELANG_GPU_DESCRIPTION = "an SM90 or later NVIDIA CUDA GPU or an AMD gfx942/gfx950 GPU"


def is_tilelang_gpu_supported() -> bool:
    """Return whether this GPU runs the DeepSeek-V4 TileLang kernels."""
    return any(platform.matches() for platform in TILELANG_GPUS)


@cache
def require_tilelang_gpu() -> None:
    """Reject unsupported hardware before importing the TileLang quantizers."""
    if not is_tilelang_gpu_supported():
        raise RuntimeError(f"QAT TileLang quantizers require {TILELANG_GPU_DESCRIPTION}")


def is_tilelang_fp4_gpu_supported() -> bool:
    """Return whether this GPU runs the TileLang FP4 quantizer.

    TileLang's HIP backend lowers a float-to-FP4 cast by reinterpreting the value's integer
    part as E2M1 bits and cannot store 8-lane FP4 vectors, so on ROCm the FP4 quantizer
    either fails to compile or returns wrong values.
    """
    return NVIDIA_SM90_PLUS.matches()


@cache
def require_tilelang_fp4_gpu() -> None:
    """Reject unsupported hardware before importing the TileLang FP4 quantizer."""
    require_tilelang_gpu()
    if not is_tilelang_fp4_gpu_supported():
        raise RuntimeError(
            "The QAT TileLang FP4 quantizer requires an SM90 or later NVIDIA CUDA GPU: TileLang's HIP backend "
            "does not lower FP4 casts correctly yet"
        )
