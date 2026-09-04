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

"""Tests for the opt-in ``nabla`` kernels (RMSNorm, SwiGLU SiLU-mul).

Structure mirrors ``test_kernel_registry_numerical.py``: CPU-runnable parts
(registry wiring, torch fallback) run everywhere; the Triton paths are gated
on CUDA.  Because these are *training* kernels, the CUDA tests check the
backward pass too (grad_x / grad_weight / grad_gate / grad_up) against an
fp32 eager oracle, not just the forward output.
"""

from types import SimpleNamespace

import pytest
import torch

import veomni.ops  # noqa: F401 -- trigger KERNEL_REGISTRY registrations
from veomni.ops.config.registry import get_op
from veomni.ops.dispatch import OpSlot
from veomni.ops.kernel_registry import KERNEL_REGISTRY
from veomni.utils.device import IS_CUDA_AVAILABLE


DEVICE = "cuda" if IS_CUDA_AVAILABLE else "cpu"

# ---------------------------------------------------------------------------
# Eager fp32 oracles
# ---------------------------------------------------------------------------


def _eager_rms_norm(x, weight, eps):
    # fp32-accumulation oracle with a single cast at the end — the same
    # rounding order as the nabla kernels. The cast-early HF-style oracle
    # (normalize -> cast to bf16 -> multiply) injects ~1 bf16 ULP per element
    # into the output and O(sqrt(M) * ULP) noise into grad_weight, which
    # measures the oracle's rounding style, not the kernel's math.
    x_f = x.to(torch.float32)
    variance = x_f.pow(2).mean(-1, keepdim=True)
    x_n = x_f * torch.rsqrt(variance + eps)
    return (weight.to(torch.float32) * x_n).to(x.dtype)


def _eager_silu_mul(gate, up):
    dtype = gate.dtype
    return (torch.nn.functional.silu(gate.to(torch.float32)) * up.to(torch.float32)).to(dtype)


# ---------------------------------------------------------------------------
# CPU-runnable: registry wiring
# ---------------------------------------------------------------------------


def test_registry_lists_nabla_kernels():
    assert "nabla" in KERNEL_REGISTRY.list_available("rms_norm", "standard")
    assert "nabla" in KERNEL_REGISTRY.list_available("swiglu_mlp", "standard")


def test_opspec_has_nabla_backend_and_default_unchanged():
    rms = get_op("rms_norm")
    assert rms.default == "liger_kernel"
    assert "nabla" in rms.backends
    assert rms.backends["nabla"].requires == ("triton",)

    swiglu = get_op("swiglu_mlp")
    assert swiglu.default == "liger_kernel"
    assert "nabla" in swiglu.backends
    assert swiglu.backends["nabla"].requires == ("triton",)


# ---------------------------------------------------------------------------
# CPU-runnable: torch fallback correctness (fwd + bwd)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rms_norm_fallback_matches_eager_fwd_bwd(dtype):
    from veomni.ops.kernels.rms_norm.nabla import rms_norm

    x = torch.randn(4, 8, 128, dtype=dtype, requires_grad=True)
    w = torch.randn(128, dtype=dtype, requires_grad=True)
    dout = torch.randn(4, 8, 128, dtype=dtype)

    out = rms_norm(x, w, 1e-6)
    gx, gw = torch.autograd.grad(out, (x, w), dout)

    x_e = x.detach().clone().requires_grad_(True)
    w_e = w.detach().clone().requires_grad_(True)
    out_e = _eager_rms_norm(x_e, w_e, 1e-6)
    gx_e, gw_e = torch.autograd.grad(out_e, (x_e, w_e), dout)

    # Same rounding order on both sides (fp32 end-to-end, single final cast):
    # only fp32 accumulation order differs.
    assert torch.allclose(out, out_e, atol=2e-3, rtol=2e-3)
    # grads accumulate over rows; bf16 rounding on the leaf cast dominates.
    assert torch.allclose(gx, gx_e, atol=2e-2, rtol=2e-2)
    assert torch.allclose(gw, gw_e, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_silu_mul_fallback_matches_eager_fwd_bwd(dtype):
    from veomni.ops.kernels.swiglu.nabla import silu_mul

    gate = torch.randn(4, 8, 96, dtype=dtype, requires_grad=True)
    up = torch.randn(4, 8, 96, dtype=dtype, requires_grad=True)
    dout = torch.randn(4, 8, 96, dtype=dtype)

    out = silu_mul(gate, up)
    gg, gu = torch.autograd.grad(out, (gate, up), dout)

    gate_e = gate.detach().clone().requires_grad_(True)
    up_e = up.detach().clone().requires_grad_(True)
    out_e = _eager_silu_mul(gate_e, up_e)
    gg_e, gu_e = torch.autograd.grad(out_e, (gate_e, up_e), dout)

    assert torch.allclose(out, out_e, atol=2e-3, rtol=2e-3)
    assert torch.allclose(gg, gg_e, atol=2e-2, rtol=2e-2)
    assert torch.allclose(gu, gu_e, atol=2e-2, rtol=2e-2)


def test_nabla_rmsnorm_module_matches_eager():
    from veomni.ops.kernels.rms_norm.nabla import NablaRMSNorm

    mod = NablaRMSNorm(64, eps=1e-6)
    x = torch.randn(2, 8, 64, dtype=torch.float32)
    assert torch.allclose(mod(x), _eager_rms_norm(x, mod.weight, 1e-6), atol=1e-5, rtol=1e-5)


def test_nabla_swiglu_mlp_module_matches_eager():
    from veomni.ops.kernels.swiglu.nabla import NablaSwiGLUMLP

    config = SimpleNamespace(hidden_size=32, intermediate_size=64, hidden_act="silu")
    mlp = NablaSwiGLUMLP(config)
    eager = SimpleNamespace(gate_proj=mlp.gate_proj, up_proj=mlp.up_proj, down_proj=mlp.down_proj)
    x = torch.randn(2, 4, 32, dtype=torch.float32)
    out_kernel = mlp(x)
    out_eager = eager.down_proj(torch.nn.functional.silu(eager.gate_proj(x)) * eager.up_proj(x))
    assert torch.allclose(out_kernel, out_eager, atol=1e-5, rtol=1e-5)

    with pytest.raises(ValueError, match="not supported"):
        NablaSwiGLUMLP(SimpleNamespace(hidden_size=32, intermediate_size=64, hidden_act="gelu"))


# ---------------------------------------------------------------------------
# CUDA-gated: Triton paths through the OpSlot dispatch
# ---------------------------------------------------------------------------

pytestmark_cuda = pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="nabla Triton kernels require CUDA")


@pytestmark_cuda
def test_rms_norm_nabla_triton_matches_eager_fwd_bwd():
    slot = OpSlot("rms_norm", "standard")
    slot.bind("nabla")

    # 3-D call shape used by the generated modeling code ((B, S, H)).
    x = torch.randn(2, 16, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(128, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    dout = torch.randn(2, 16, 128, device=DEVICE, dtype=torch.bfloat16)

    out = slot(x, w, 1e-6)
    gx, gw = torch.autograd.grad(out, (x, w), dout)

    x_e = x.detach().clone().requires_grad_(True)
    w_e = w.detach().clone().requires_grad_(True)
    out_e = _eager_rms_norm(x_e, w_e, 1e-6)
    gx_e, gw_e = torch.autograd.grad(out_e, (x_e, w_e), dout)

    # Single-pass reduction + fp32 kept end-to-end with a single final cast,
    # same rounding order as the oracle above.
    assert torch.allclose(out, out_e, atol=2e-3, rtol=2e-3)
    # grads: fp32 accumulation inside the kernel; bf16 rounding on the leaf
    # cast and on the 2x16-row reduction for grad_weight dominate.
    assert torch.allclose(gx, gx_e, atol=2e-2, rtol=2e-2)
    assert torch.allclose(gw, gw_e, atol=2e-2, rtol=2e-2)


@pytestmark_cuda
def test_rms_norm_nabla_triton_2d_3d_consistent():
    slot = OpSlot("rms_norm", "standard")
    slot.bind("nabla")

    x3 = torch.randn(4, 8, 128, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn(128, device=DEVICE, dtype=torch.bfloat16)
    assert torch.allclose(slot(x3, w, 1e-6), slot(x3.reshape(-1, 128), w, 1e-6).view_as(x3), atol=0, rtol=0)


@pytestmark_cuda
def test_swiglu_nabla_triton_matches_eager_fwd_bwd():
    slot = OpSlot("swiglu_mlp", "standard")
    slot.bind("nabla")

    class _MLP(torch.nn.Module):
        def __init__(self, dim, hidden):
            super().__init__()
            self.gate_proj = torch.nn.Linear(dim, hidden, bias=False)
            self.up_proj = torch.nn.Linear(dim, hidden, bias=False)
            self.down_proj = torch.nn.Linear(hidden, dim, bias=False)
            self.act_fn = torch.nn.SiLU()

    torch.manual_seed(0)
    mlp = _MLP(128, 256).to(DEVICE).to(torch.bfloat16)
    x = torch.randn(2, 16, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    dout = torch.randn(2, 16, 128, device=DEVICE, dtype=torch.bfloat16)

    out = slot(mlp, x)
    (gx,) = torch.autograd.grad(out, (x,), dout, retain_graph=True)
    gg = torch.autograd.grad(out, (mlp.gate_proj.weight, mlp.up_proj.weight), dout, allow_unused=True)

    x_e = x.detach().clone().requires_grad_(True)
    # Eager reference with the SAME activation rounding order as the kernel
    # (fp32 silu, single cast). A bf16 nn.SiLU reference injects ~1 bf16 ULP
    # into every intermediate element, which the token-summing weight grads
    # amplify beyond the tolerance — measuring rounding style, not math.
    out_e = mlp.down_proj(_eager_silu_mul(mlp.gate_proj(x_e), mlp.up_proj(x_e)))
    (gx_e,) = torch.autograd.grad(out_e, (x_e,), dout, retain_graph=True)
    gg_e = torch.autograd.grad(out_e, (mlp.gate_proj.weight, mlp.up_proj.weight), dout, allow_unused=True)

    # Three bf16 matmuls stack rounding; 5e-3 as in the liger swiglu test.
    assert torch.allclose(out, out_e, atol=5e-3, rtol=5e-3)
    assert torch.allclose(gx, gx_e, atol=5e-3, rtol=5e-3)
    for g_k, g_e in zip(gg, gg_e):
        assert torch.allclose(g_k, g_e, atol=5e-3, rtol=5e-3)
