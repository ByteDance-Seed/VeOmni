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

"""RMSNorm eager vs HF, and fused impls vs eager."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4UnweightedRMSNorm
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

from tests.kernels.tol import (
    EAGER_ATOL,
    EAGER_GRAD_ATOL,
    EAGER_GRAD_RTOL,
    EAGER_RTOL,
    RMS_FUSED_ATOL,
    RMS_FUSED_GRAD_ATOL,
    RMS_FUSED_GRAD_RTOL,
    RMS_FUSED_QWEN35_ATOL,
    RMS_FUSED_QWEN35_RTOL,
    RMS_FUSED_RTOL,
    RMS_NPU_ATOL,
    RMS_NPU_RTOL,
    RMS_TRITON_ATOL,
    RMS_TRITON_GRAD_ATOL,
    RMS_TRITON_GRAD_RTOL,
    RMS_TRITON_RTOL,
    RMS_UNWEIGHTED_ATOL,
    RMS_UNWEIGHTED_RTOL,
)
from veomni.kernels import resolve_kernel
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE


def _hf_rms_norm(variant: str, hidden: int, eps: float) -> nn.Module:
    if variant == "standard":
        return Qwen3RMSNorm(hidden, eps=eps)
    if variant == "qwen3_5":
        return Qwen3_5RMSNorm(hidden, eps=eps)
    raise KeyError(variant)


def _clone_inputs(x: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
    return x.detach().requires_grad_(True), weight.detach().requires_grad_(True)


def _fused_weight(variant: str, hidden: int, device: str, dtype: torch.dtype) -> Tensor:
    if variant == "qwen3_5":
        weight = torch.zeros(hidden, device=device, dtype=dtype)
        return weight + 0.01 * torch.randn_like(weight)
    return torch.randn(hidden, device=device, dtype=dtype)


@pytest.mark.parametrize("variant", ["standard", "qwen3_5"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_eager_matches_hf(variant: str, dtype: torch.dtype):
    torch.manual_seed(0)
    hidden = 64
    eps = 1e-6
    x = torch.randn(2, 16, hidden, dtype=dtype, requires_grad=True)
    weight = torch.randn(hidden, dtype=dtype, requires_grad=True)

    module = _hf_rms_norm(variant, hidden, eps).to(dtype=dtype)
    with torch.no_grad():
        module.weight.copy_(weight)

    x_h = x.detach().requires_grad_(True)
    out_h = module(x_h)

    x_e, w_e = _clone_inputs(x, weight)
    out_e = resolve_kernel("rms_norm", variant, "eager").wrapper(x_e, w_e, eps=eps)
    assert torch.allclose(out_e.float(), out_h.float(), atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = torch.randn_like(out_e)
    out_h.backward(go)
    out_e.backward(go)
    assert torch.allclose(x_e.grad.float(), x_h.grad.float(), atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(w_e.grad.float(), module.weight.grad.float(), atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_unweighted_eager_matches_hf():
    torch.manual_seed(0)
    hidden = 64
    eps = 1e-6
    x = torch.randn(2, 16, hidden, dtype=torch.float32, requires_grad=True)

    module = DeepseekV4UnweightedRMSNorm(eps=eps)
    x_h = x.detach().requires_grad_(True)
    out_h = module(x_h)

    x_e = x.detach().requires_grad_(True)
    out_e = resolve_kernel("rms_norm", "unweighted", "eager").wrapper(x_e, eps=eps)
    assert torch.allclose(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = torch.randn_like(out_e)
    out_h.backward(go)
    out_e.backward(go)
    assert torch.allclose(x_e.grad, x_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def _fused_matches_eager(
    variant: str,
    impl: str,
    device: str,
    dtype: torch.dtype,
    *,
    atol: float,
    rtol: float,
    grad_atol: float | None = None,
    grad_rtol: float | None = None,
    cast_fp32: bool = False,
) -> None:
    eager = resolve_kernel("rms_norm", variant, "eager").wrapper
    other = resolve_kernel("rms_norm", variant, impl).wrapper
    torch.manual_seed(0)
    hidden = 128
    base_x = torch.randn(2, 16, hidden, device=device, dtype=dtype)
    base_w = _fused_weight(variant, hidden, device, dtype)
    eps = 1e-6
    if grad_atol is None:
        grad_atol = atol
    if grad_rtol is None:
        grad_rtol = rtol

    x_e, w_e = _clone_inputs(base_x, base_w)
    x_o, w_o = _clone_inputs(base_x, base_w)
    out_e = eager(x_e, w_e, eps=eps)
    out_o = other(x_o, w_o, eps=eps)
    left, right = (out_e.float(), out_o.float()) if cast_fp32 else (out_e, out_o)
    assert torch.allclose(left, right, atol=atol, rtol=rtol)

    go = torch.randn_like(out_e)
    out_e.backward(go)
    out_o.backward(go)
    assert torch.allclose(x_e.grad, x_o.grad, atol=grad_atol, rtol=grad_rtol)
    assert torch.allclose(w_e.grad, w_o.grad, atol=grad_atol, rtol=grad_rtol)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="liger RMSNorm needs CUDA")
@pytest.mark.parametrize("variant", ["standard", "qwen3_5"])
def test_liger_matches_eager(variant: str):
    pytest.importorskip("liger_kernel")
    _fused_matches_eager(
        variant,
        "liger_kernel",
        "cuda",
        torch.bfloat16,
        atol=RMS_FUSED_QWEN35_ATOL if variant == "qwen3_5" else RMS_FUSED_ATOL,
        rtol=RMS_FUSED_QWEN35_RTOL if variant == "qwen3_5" else RMS_FUSED_RTOL,
        grad_atol=RMS_FUSED_GRAD_ATOL,
        grad_rtol=RMS_FUSED_GRAD_RTOL,
        cast_fp32=variant == "qwen3_5",
    )


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="liger RMSNorm needs CUDA")
def test_unweighted_liger_matches_eager():
    pytest.importorskip("liger_kernel")
    eager = resolve_kernel("rms_norm", "unweighted", "eager").wrapper
    other = resolve_kernel("rms_norm", "unweighted", "liger_kernel").wrapper
    torch.manual_seed(0)
    base_x = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
    eps = 1e-6

    x_e = base_x.detach().requires_grad_(True)
    x_o = base_x.detach().requires_grad_(True)
    out_e = eager(x_e, eps=eps)
    out_o = other(x_o, eps=eps)
    assert torch.allclose(out_e, out_o, atol=RMS_UNWEIGHTED_ATOL, rtol=RMS_UNWEIGHTED_RTOL)

    go = torch.randn_like(out_e)
    out_e.backward(go)
    out_o.backward(go)
    assert torch.allclose(x_e.grad, x_o.grad, atol=RMS_FUSED_GRAD_ATOL, rtol=RMS_FUSED_GRAD_RTOL)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="triton RMSNorm needs CUDA")
def test_triton_matches_eager():
    pytest.importorskip("triton")
    _fused_matches_eager(
        "standard",
        "triton",
        "cuda",
        torch.bfloat16,
        atol=RMS_TRITON_ATOL,
        rtol=RMS_TRITON_RTOL,
        grad_atol=RMS_TRITON_GRAD_ATOL,
        grad_rtol=RMS_TRITON_GRAD_RTOL,
    )


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="NPU RMSNorm needs NPU")
@pytest.mark.parametrize("variant", ["standard", "qwen3_5"])
def test_npu_matches_eager(variant: str):
    _fused_matches_eager(
        variant,
        "npu",
        "npu",
        torch.bfloat16,
        atol=RMS_NPU_ATOL,
        rtol=RMS_NPU_RTOL,
        cast_fp32=variant == "qwen3_5",
    )
