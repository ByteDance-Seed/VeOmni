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

import importlib.util

import pytest
import torch

import veomni.ops  # noqa: F401 - trigger kernel registrations
from veomni.ops.dispatch import OpSlot
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_gpu_compute_capability


pytestmark = pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="kernels require CUDA")

DEVICE = get_device_type()


def _fresh_slot(op_name, variant, impl_name):
    slot = OpSlot(op_name, variant)
    slot.bind(impl_name)
    return slot


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

    mlp = _MLP(128, 256).to(DEVICE).to(torch.bfloat16)
    x = torch.randn(2, 16, 128, device=DEVICE, dtype=torch.bfloat16)
    out_kernel = slot(mlp, x)
    out_eager = mlp.down_proj(mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x))
    # bf16 gate/up linears + silu + down linear — three matmuls accumulate
    # more bf16 rounding than the single-op kernels above; 5e-3 is the
    # smallest margin that held across local runs.
    assert torch.allclose(out_kernel, out_eager, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(
    get_gpu_compute_capability() != 90 or importlib.util.find_spec("flash_qla") is None,
    reason="flash_qla only ships Hopper SM90 kernels — SM10x WIP upstream "
    "(https://github.com/QwenLM/FlashQLA/issues/2). Skipped on Ampere / L20 / Blackwell.",
)
def test_chunk_gated_delta_rule_flash_qla_matches_fla():
    # Both `fla` and `flash_qla` are non-eager Triton implementations of the
    # same chunk gated delta-rule kernel — there is no torch reference that
    # supports cu_seqlens, so this test compares the two Triton implementations
    # against each other on identical inputs. Catches a registry mis-routing
    # (e.g. flash_qla bound where fla is expected) and any future drift in
    # FlashQLA's call convention from FLA's.
    fla_slot = _fresh_slot("chunk_gated_delta_rule", "standard", "fla")
    flash_qla_slot = _fresh_slot("chunk_gated_delta_rule", "standard", "flash_qla")

    # Qwen3.5 GatedDeltaNet shapes: B=1, T=64 short packed sequence,
    # H=4 heads, head dim K=V=128, bf16 (model default).
    B, T, H, K, V = 1, 64, 4, 128, 128
    torch.manual_seed(0)
    q = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.bfloat16)
    # `g` is the forget gate in log space — keep it well below zero so the
    # recurrence stays numerically stable.
    g = -torch.rand(B, T, H, device=DEVICE, dtype=torch.float32).abs() * 0.5
    beta = torch.rand(B, T, H, device=DEVICE, dtype=torch.bfloat16)

    out_fla, _ = fla_slot(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    out_flash_qla, _ = flash_qla_slot(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    # Two independent Triton kernels accumulating in fp32 with bf16 IO; chunk
    # boundaries differ between the implementations, so error compounds beyond
    # a single bf16 rounding. 5e-2 was the smallest margin that held for two
    # independent kernels of this shape across seeds.
    assert torch.allclose(out_fla, out_flash_qla, atol=5e-2, rtol=5e-2)


def test_load_balancing_loss_triton_matches_eager():
    from veomni.ops.kernels.load_balancing_loss.eager import load_balancing_loss_pytorch

    slot = _fresh_slot("load_balancing_loss", "standard", "triton")
    num_experts, top_k, num_layers, N = 8, 2, 3, 256
    gate_logits = tuple(torch.randn(N, num_experts, device=DEVICE, dtype=torch.float32) for _ in range(num_layers))
    out_kernel = slot(gate_logits, num_experts, top_k, None)
    out_eager = load_balancing_loss_pytorch(gate_logits, num_experts, top_k, None)
    assert torch.allclose(out_kernel, out_eager, atol=1e-4, rtol=1e-4)
