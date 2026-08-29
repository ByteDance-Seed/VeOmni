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


@pytest.mark.parametrize("variant", ["standard", "qwen3_5"])
def test_eager_matches_hf(variant: str):
    torch.manual_seed(0)
    hidden = 64
    eps = 1e-6
    x = torch.randn(2, 16, hidden, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(hidden, dtype=torch.float32, requires_grad=True)

    module = _hf_rms_norm(variant, hidden, eps)
    with torch.no_grad():
        module.weight.copy_(weight)

    x_h = x.detach().requires_grad_(True)
    out_h = module(x_h)

    x_e, w_e = _clone_inputs(x, weight)
    out_e = resolve_kernel("rms_norm", variant, "eager").wrapper(x_e, w_e, eps=eps)
    assert torch.allclose(out_e, out_h, atol=1e-6, rtol=1e-6)

    go = torch.randn_like(out_e)
    out_h.backward(go)
    out_e.backward(go)
    assert torch.allclose(x_e.grad, x_h.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(w_e.grad, module.weight.grad, atol=1e-5, rtol=1e-5)


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
    assert torch.allclose(out_e, out_h, atol=1e-6, rtol=1e-6)

    go = torch.randn_like(out_e)
    out_h.backward(go)
    out_e.backward(go)
    assert torch.allclose(x_e.grad, x_h.grad, atol=1e-5, rtol=1e-5)


def _fused_matches_eager(
    variant: str, impl: str, device: str, dtype: torch.dtype, *, atol: float, rtol: float
) -> None:
    eager = resolve_kernel("rms_norm", variant, "eager").wrapper
    other = resolve_kernel("rms_norm", variant, impl).wrapper
    torch.manual_seed(0)
    base_x = torch.randn(2, 16, 64, device=device, dtype=dtype)
    base_w = torch.randn(64, device=device, dtype=dtype)
    eps = 1e-6

    x_e, w_e = _clone_inputs(base_x, base_w)
    x_o, w_o = _clone_inputs(base_x, base_w)
    out_e = eager(x_e, w_e, eps=eps)
    out_o = other(x_o, w_o, eps=eps)
    assert torch.allclose(out_e, out_o, atol=atol, rtol=rtol)

    go = torch.randn_like(out_e)
    out_e.backward(go)
    out_o.backward(go)
    assert torch.allclose(x_e.grad, x_o.grad, atol=atol, rtol=rtol)
    assert torch.allclose(w_e.grad, w_o.grad, atol=atol, rtol=rtol)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="liger RMSNorm needs CUDA")
@pytest.mark.parametrize("variant", ["standard", "qwen3_5"])
def test_liger_matches_eager(variant: str):
    pytest.importorskip("liger_kernel")
    # One bf16 ulp is ~3e-2 here. A swapped variant differs by ~0.5.
    _fused_matches_eager(variant, "liger_kernel", "cuda", torch.bfloat16, atol=4e-2, rtol=2e-2)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="liger RMSNorm needs CUDA")
def test_unweighted_liger_matches_eager():
    pytest.importorskip("liger_kernel")
    eager = resolve_kernel("rms_norm", "unweighted", "eager").wrapper
    other = resolve_kernel("rms_norm", "unweighted", "liger_kernel").wrapper
    torch.manual_seed(0)
    base_x = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)
    eps = 1e-6

    x_e = base_x.detach().requires_grad_(True)
    x_o = base_x.detach().requires_grad_(True)
    out_e = eager(x_e, eps=eps)
    out_o = other(x_o, eps=eps)
    assert torch.allclose(out_e, out_o, atol=4e-2, rtol=2e-2)

    go = torch.randn_like(out_e)
    out_e.backward(go)
    out_o.backward(go)
    assert torch.allclose(x_e.grad, x_o.grad, atol=4e-2, rtol=2e-2)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="triton RMSNorm needs CUDA")
def test_triton_matches_eager():
    pytest.importorskip("triton")
    _fused_matches_eager("standard", "triton", "cuda", torch.bfloat16, atol=4e-2, rtol=2e-2)


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="torch_npu RMSNorm needs NPU")
@pytest.mark.parametrize("variant", ["standard", "qwen3_5"])
def test_torch_npu_matches_eager(variant: str):
    _fused_matches_eager(variant, "torch_npu", "npu", torch.bfloat16, atol=4e-2, rtol=2e-2)
