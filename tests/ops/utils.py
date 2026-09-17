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


def assert_gradient_direction_and_scale(
    actual: Tensor,
    expected: Tensor,
    *,
    min_cosine: float,
    norm_rtol: float,
) -> None:
    """Check both direction and norm for an approximate fused gradient."""
    assert torch.isfinite(actual).all()
    assert cosine_similarity(actual, expected) > min_cosine
    torch.testing.assert_close(
        actual.float().norm(),
        expected.float().norm(),
        rtol=norm_rtol,
        atol=1e-6,
    )


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


def tensor_error_stats(actual: Tensor, expected: Tensor) -> dict[str, float]:
    """Return max absolute and relative error of ``actual`` versus ``expected``."""
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    return {
        "max_abs": diff.max().item(),
        "max_rel": (diff / expected_f.abs().clamp(min=1e-8)).max().item(),
    }


def assert_reference_signal(name: str, reference: Tensor, atol: float, rtol: float) -> None:
    """Reject a reference that an all-zero implementation would also match."""
    assert not torch.allclose(torch.zeros_like(reference), reference, atol=atol, rtol=rtol), (
        f"{name} signal is too small to reject an all-zero implementation"
    )


def assert_close_with_error(
    name: str,
    actual: Tensor,
    expected: Tensor,
    *,
    atol: float,
    rtol: float,
) -> None:
    """Compare tensors and include measured error when the budget fails."""
    stats = tensor_error_stats(actual, expected)
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as exc:
        raise AssertionError(
            f"{name}: max_abs={stats['max_abs']:.6g} max_rel={stats['max_rel']:.6g} (budget atol={atol} rtol={rtol})"
        ) from exc
