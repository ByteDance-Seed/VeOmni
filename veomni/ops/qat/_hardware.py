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

"""Hardware checks shared by the QAT quantizer facades.

FP8 quantizers run the TileLang kernels on SM90+ NVIDIA GPUs and the Triton
kernels on AMD gfx942/gfx950; both are bit-exact with the same reference. The
FP4 quantizer exists only in TileLang and stays NVIDIA-only.
"""

from functools import cache

from ...utils.device import get_rocm_gcn_arch
from ..platform import NVIDIA_SM90_PLUS, ROCM_GPU


TRITON_FP8_ARCHS = ("gfx942", "gfx950")
FP8_QUANTIZER_GPUS = "an SM90 or later NVIDIA CUDA GPU or an AMD ROCm gfx942/gfx950 GPU"


@cache
def quantizer_backend() -> str | None:
    """Return ``"tilelang"`` or ``"triton"`` for the quantizers this GPU runs, or ``None``."""
    if NVIDIA_SM90_PLUS.matches():
        return "tilelang"
    if ROCM_GPU.matches() and get_rocm_gcn_arch() in TRITON_FP8_ARCHS:
        return "triton"
    return None


def require_tilelang_sm90() -> None:
    """Reject unsupported hardware before importing the TileLang quantizers."""
    if quantizer_backend() != "tilelang":
        raise RuntimeError("QAT TileLang quantizers require an SM90 or later NVIDIA CUDA GPU")


def fp8_quantizer_backend() -> str:
    """Return the FP8 quantizer backend for this GPU, rejecting unsupported hardware first."""
    backend = quantizer_backend()
    if backend is None:
        raise RuntimeError(f"QAT FP8 quantizers require {FP8_QUANTIZER_GPUS}")
    return backend
