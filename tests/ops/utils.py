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

"""Shared tensor and hardware helpers for operation tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from veomni.utils.device import IS_CUDA_AVAILABLE, get_gpu_compute_capability


def make_grad_leaf(tensor: Tensor) -> Tensor:
    """Detach a tensor and make the detached tensor a gradient leaf."""
    return tensor.detach().requires_grad_(True)


def make_grad_leaves(*tensors: Tensor) -> tuple[Tensor, ...]:
    """Make detached gradient leaves for a sequence of tensors."""
    return tuple(make_grad_leaf(tensor) for tensor in tensors)


def cosine_similarity(actual: Tensor, expected: Tensor) -> float:
    """Return cosine similarity between flattened float32 tensors."""
    return F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0).item()


def is_nvidia_cuda_available(*, min_cc: int | None = None, max_cc: int | None = None) -> bool:
    """Return whether an NVIDIA CUDA device satisfies the compute-capability range."""
    if not IS_CUDA_AVAILABLE or torch.version.hip is not None:
        return False
    capability = get_gpu_compute_capability()
    return (min_cc is None or capability >= min_cc) and (max_cc is None or capability <= max_cc)


def require_nvidia_cuda(*packages: str, min_cc: int | None = None, max_cc: int | None = None) -> None:
    """Skip unless optional packages and a compatible NVIDIA CUDA device are available."""
    for package in packages:
        pytest.importorskip(package)
    if not IS_CUDA_AVAILABLE or torch.version.hip is not None:
        pytest.skip("test requires an NVIDIA CUDA GPU")

    capability = get_gpu_compute_capability()
    if min_cc is not None and capability < min_cc:
        pytest.skip(f"test requires SM{min_cc} or later")
    if max_cc is not None and capability > max_cc:
        pytest.skip(f"test requires SM{max_cc} or earlier")
