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

from __future__ import annotations

import torch
from torch import nn

from ...utils.device import IS_CUDA_AVAILABLE, get_gpu_compute_capability, get_rocm_gcn_arch


TRITON_FP8_ARCHS = ("gfx942", "gfx950")
FP8_QUANTIZER_GPUS = "an SM90 or later NVIDIA CUDA GPU or an AMD ROCm gfx942/gfx950 GPU"


#: Rows whose fake quantization needs a device with native BF16 support: the
#: operand they round is a BF16 tensor and the trained GEMM runs in that dtype.
_NATIVE_BF16_QAT_IMPLS = frozenset({"torchao_mxfp4", "torchao_nvfp4"})


def _native_bf16_qat_selections(model: nn.Module) -> set[str]:
    """Selected torchao QAT rows on ``model``'s config tree, child configs included."""

    def walk(config) -> set[str]:
        found: set[str] = set()
        ops = getattr(config, "ops_config", None)
        if ops is not None:
            for field in ("linear_implementation", "swiglu_mlp_implementation"):
                impl = getattr(ops, field, None)
                if impl in _NATIVE_BF16_QAT_IMPLS:
                    found.add(impl)
        for name in getattr(config, "sub_configs", {}) or {}:
            child = getattr(config, name, None)
            if child is not None:
                found |= walk(child)
        return found

    return walk(getattr(model, "config", None))


def reject_qat_without_native_bf16(model: nn.Module) -> None:
    """Refuse a torchao QAT row on a CUDA device older than SM80.

    A training-path precondition rather than a row ``requirement``: the fake
    quantizer itself is plain tensor math that also runs on CPU, which is what the
    op-level tests use. Only the distributed training GEMM needs native BF16, and
    SM80 is the first NVIDIA generation that has it.
    """
    selected = _native_bf16_qat_selections(model)
    if not selected:
        return
    from ...utils.device import IS_CUDA_AVAILABLE, get_gpu_compute_capability

    if not IS_CUDA_AVAILABLE:
        return
    if get_gpu_compute_capability() < 80:
        raise RuntimeError(
            f"torchao QAT rows ({', '.join(sorted(selected))}) require a CUDA device with native "
            "BF16 support (SM80 or newer); this host reports an older compute capability."
        )


def quantizer_backend(device: torch.types.Device | int | None = None) -> str | None:
    """Return the FP8 backend for ``device``, or ``None`` when that device is unsupported."""
    if not IS_CUDA_AVAILABLE:
        return None
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    elif isinstance(device, int):
        device = torch.device("cuda", device)
    else:
        device = torch.device(device)
    if device.type != "cuda":
        return None
    if torch.version.hip is not None:
        return "triton" if get_rocm_gcn_arch(device) in TRITON_FP8_ARCHS else None
    if get_gpu_compute_capability(device) >= 90:
        return "tilelang"
    return None


def require_tilelang_sm90(device: torch.types.Device | int | None = None) -> None:
    """Reject unsupported hardware before importing the TileLang quantizers."""
    if quantizer_backend(device) != "tilelang":
        raise RuntimeError("QAT TileLang quantizers require an SM90 or later NVIDIA CUDA GPU")


def fp8_quantizer_backend(device: torch.types.Device | int | None = None) -> str:
    """Return the FP8 quantizer backend for this GPU, rejecting unsupported hardware first."""
    backend = quantizer_backend(device)
    if backend is None:
        raise RuntimeError(f"QAT FP8 quantizers require {FP8_QUANTIZER_GPUS}")
    return backend
