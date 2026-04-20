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

"""Numerical alignment tests for KERNEL_REGISTRY-registered kernels.

For each (op_name, variant, impl_name) tuple that ships a non-eager kernel,
bind an OpSlot to that kernel and compare its output against the canonical
eager implementation on random inputs. This is what guards against the
KernelRegistry silently routing the wrong implementation into an OpSlot
(e.g. a standard-variant kernel into a qwen3_5 slot).
"""

import pytest
import torch

import veomni.ops  # noqa: F401 - trigger kernel registrations
from veomni.ops.dispatch import OpSlot
from veomni.utils.device import IS_CUDA_AVAILABLE


pytestmark = pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="kernels require CUDA")


def _fresh_slot(op_name, variant, impl_name):
    slot = OpSlot(op_name, variant)
    slot.bind(impl_name)
    return slot


def _eager_rms_norm_standard(x, weight, eps):
    dtype = x.dtype
    x_f = x.to(torch.float32)
    variance = x_f.pow(2).mean(-1, keepdim=True)
    x_f = x_f * torch.rsqrt(variance + eps)
    return (weight * x_f.to(dtype)).to(dtype)


def _eager_rms_norm_qwen3_5(x, weight, eps):
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x_norm = x.to(torch.float32) * torch.rsqrt(variance + eps)
    return ((1.0 + weight.to(torch.float32)) * x_norm).to(x.dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _eager_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def test_rms_norm_standard_liger_matches_eager():
    pytest.importorskip("liger_kernel")
    slot = _fresh_slot("rms_norm", "standard", "liger_kernel")
    x = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    assert torch.allclose(slot(x, w, 1e-6), _eager_rms_norm_standard(x, w, 1e-6), atol=1e-2, rtol=1e-2)


def test_rms_norm_qwen3_5_liger_matches_eager():
    pytest.importorskip("liger_kernel")
    slot = _fresh_slot("rms_norm", "qwen3_5", "liger_kernel")
    x = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(128, device="cuda", dtype=torch.bfloat16)  # Qwen3.5 initializes to zeros
    w += 0.01 * torch.randn_like(w)
    out_kernel = slot(x, w, 1e-6).to(torch.float32)
    out_eager = _eager_rms_norm_qwen3_5(x, w, 1e-6).to(torch.float32)
    assert torch.allclose(out_kernel, out_eager, atol=1e-2, rtol=1e-2)


def test_rotary_pos_emb_liger_matches_eager():
    pytest.importorskip("liger_kernel")
    slot = _fresh_slot("rotary_pos_emb", "full", "liger_kernel")
    B, H, S, D = 2, 4, 16, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    # HF RoPE convention: cos/sin are duplicated across the two halves of head_dim.
    half = torch.randn(B, S, D // 2, device="cuda", dtype=torch.bfloat16)
    cos = torch.cat([half, half], dim=-1)
    half_s = torch.randn(B, S, D // 2, device="cuda", dtype=torch.bfloat16)
    sin = torch.cat([half_s, half_s], dim=-1)
    q_k, k_k = slot(q, k, cos, sin)
    q_e, k_e = _eager_rope(q, k, cos, sin)
    assert torch.allclose(q_k, q_e, atol=1e-2, rtol=1e-2)
    assert torch.allclose(k_k, k_e, atol=1e-2, rtol=1e-2)


def test_swiglu_mlp_liger_matches_eager():
    pytest.importorskip("liger_kernel")
    slot = _fresh_slot("swiglu_mlp", "standard", "liger_kernel")

    class _MLP(torch.nn.Module):
        def __init__(self, dim, hidden):
            super().__init__()
            self.gate_proj = torch.nn.Linear(dim, hidden, bias=False)
            self.up_proj = torch.nn.Linear(dim, hidden, bias=False)
            self.down_proj = torch.nn.Linear(hidden, dim, bias=False)
            self.act_fn = torch.nn.SiLU()

    mlp = _MLP(128, 256).cuda().to(torch.bfloat16)
    x = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
    out_kernel = slot(mlp, x)
    out_eager = mlp.down_proj(mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x))
    assert torch.allclose(out_kernel, out_eager, atol=1e-2, rtol=1e-2)


def test_load_balancing_loss_triton_matches_eager():
    from veomni.ops.kernels.load_balancing_loss.eager import load_balancing_loss_pytorch

    slot = _fresh_slot("load_balancing_loss", "standard", "triton")
    num_experts, top_k, num_layers, N = 8, 2, 3, 256
    gate_logits = tuple(torch.randn(N, num_experts, device="cuda", dtype=torch.float32) for _ in range(num_layers))
    out_kernel = slot(gate_logits, num_experts, top_k, None)
    out_eager = load_balancing_loss_pytorch(gate_logits, num_experts, top_k, None)
    assert torch.allclose(out_kernel, out_eager, atol=1e-4, rtol=1e-4)
