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

"""Tests for fused load balancing loss against the HuggingFace reference."""

import pytest
import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    load_balancing_loss_func as _reference_load_balancing_loss,
)

from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_torch_device


# (num_experts, top_k, num_layers, batch_size, seq_len)
_CONFIGS = [
    (8, 2, 1, 4, 128),
    (32, 4, 2, 2, 256),
    (60, 8, 4, 4, 512),
    (60, 8, 28, 2, 4096),
    (128, 4, 32, 1, 8192),
]

_DEVICE = get_device_type()


def _skip_no_cuda():
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA not available")


def _get_triton_impl():
    from veomni.ops.fused_load_balancing_loss.triton_kernel import load_balancing_loss_triton

    return load_balancing_loss_triton


def _make_gate_logits(batch_size, seq_len, num_experts, num_layers):
    N = batch_size * seq_len
    return tuple(torch.randn(N, num_experts, device=_DEVICE, dtype=torch.float32) for _ in range(num_layers))


def _measure_peak_memory(fn):
    """Run fn after resetting peak memory stats and return peak memory in bytes."""
    dev = get_torch_device()
    dev.reset_peak_memory_stats()
    dev.synchronize()
    fn()
    dev.synchronize()
    return dev.max_memory_allocated()


class TestFusedLoadBalancingLoss:
    """Test suite comparing fused Triton kernel against HF reference."""

    def test_none_input(self):
        _skip_no_cuda()
        triton_fn = _get_triton_impl()
        assert triton_fn(None, 8, 2) == 0

    def test_non_tuple_input(self):
        _skip_no_cuda()
        triton_fn = _get_triton_impl()
        logits = torch.randn(32, 8, device=_DEVICE)
        assert triton_fn(logits, 8, 2) == 0

    def test_forward_full_mask(self):
        """All tokens masked out should return 0 in the Triton implementation."""
        _skip_no_cuda()
        triton_fn = _get_triton_impl()
        gate_logits = tuple(torch.randn(8, 4, device=_DEVICE) for _ in range(2))
        attention_mask = torch.zeros(2, 4, device=_DEVICE)

        out = triton_fn(gate_logits, 4, 2, attention_mask)
        assert out.item() == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("num_experts,top_k,num_layers,batch_size,seq_len", _CONFIGS)
    def test_forward_no_mask(self, num_experts, top_k, num_layers, batch_size, seq_len):
        _skip_no_cuda()
        triton_fn = _get_triton_impl()

        torch.manual_seed(42)
        gate_logits = _make_gate_logits(batch_size, seq_len, num_experts, num_layers)

        ref = _reference_load_balancing_loss(gate_logits, num_experts, top_k)
        out = triton_fn(gate_logits, num_experts, top_k)

        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("num_experts,top_k,num_layers,batch_size,seq_len", _CONFIGS)
    def test_forward_with_mask(self, num_experts, top_k, num_layers, batch_size, seq_len):
        _skip_no_cuda()
        triton_fn = _get_triton_impl()

        torch.manual_seed(123)
        gate_logits = _make_gate_logits(batch_size, seq_len, num_experts, num_layers)
        attention_mask = torch.ones(batch_size, seq_len, device=_DEVICE)
        attention_mask[:, seq_len // 2 :] = 0

        ref = _reference_load_balancing_loss(gate_logits, num_experts, top_k, attention_mask)
        out = triton_fn(gate_logits, num_experts, top_k, attention_mask)

        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("num_experts,top_k,num_layers,batch_size,seq_len", _CONFIGS)
    def test_backward(self, num_experts, top_k, num_layers, batch_size, seq_len):
        """Verify gradients match the reference implementation."""
        _skip_no_cuda()
        triton_fn = _get_triton_impl()

        torch.manual_seed(99)
        N = batch_size * seq_len
        gate_logits_ref = tuple(
            torch.randn(N, num_experts, device=_DEVICE, dtype=torch.float32, requires_grad=True)
            for _ in range(num_layers)
        )
        ref_loss = _reference_load_balancing_loss(gate_logits_ref, num_experts, top_k)
        ref_loss.backward()
        ref_grads = [g.grad.clone() for g in gate_logits_ref]

        gate_logits_fused = tuple(g.detach().clone().requires_grad_(True) for g in gate_logits_ref)
        fused_loss = triton_fn(gate_logits_fused, num_experts, top_k)
        fused_loss.backward()
        fused_grads = [g.grad.clone() for g in gate_logits_fused]

        torch.testing.assert_close(fused_loss, ref_loss, atol=1e-4, rtol=1e-4)
        for i, (rg, fg) in enumerate(zip(ref_grads, fused_grads)):
            torch.testing.assert_close(fg, rg, atol=1e-4, rtol=1e-4, msg=f"Gradient mismatch at layer {i}")

    @pytest.mark.parametrize("num_experts,top_k,num_layers,batch_size,seq_len", _CONFIGS)
    def test_backward_with_mask(self, num_experts, top_k, num_layers, batch_size, seq_len):
        """Verify gradients with attention mask."""
        _skip_no_cuda()
        triton_fn = _get_triton_impl()

        torch.manual_seed(77)
        N = batch_size * seq_len
        attention_mask = torch.ones(batch_size, seq_len, device=_DEVICE)
        attention_mask[:, seq_len // 2 :] = 0

        gate_logits_ref = tuple(
            torch.randn(N, num_experts, device=_DEVICE, dtype=torch.float32, requires_grad=True)
            for _ in range(num_layers)
        )
        ref_loss = _reference_load_balancing_loss(gate_logits_ref, num_experts, top_k, attention_mask)
        ref_loss.backward()
        ref_grads = [g.grad.clone() for g in gate_logits_ref]

        gate_logits_fused = tuple(g.detach().clone().requires_grad_(True) for g in gate_logits_ref)
        fused_loss = triton_fn(gate_logits_fused, num_experts, top_k, attention_mask)
        fused_loss.backward()
        fused_grads = [g.grad.clone() for g in gate_logits_fused]

        torch.testing.assert_close(fused_loss, ref_loss, atol=1e-4, rtol=1e-4)
        for i, (rg, fg) in enumerate(zip(ref_grads, fused_grads)):
            torch.testing.assert_close(fg, rg, atol=1e-4, rtol=1e-4, msg=f"Gradient mismatch at layer {i}")

    @pytest.mark.parametrize("num_experts,top_k,num_layers,batch_size,seq_len", _CONFIGS)
    def test_memory_saving(self, num_experts, top_k, num_layers, batch_size, seq_len):
        """Triton kernel should use less peak memory than HF reference."""
        _skip_no_cuda()
        triton_fn = _get_triton_impl()

        torch.manual_seed(0)
        gate_logits = _make_gate_logits(batch_size, seq_len, num_experts, num_layers)

        # Warm-up triton compilation
        _warmup = tuple(torch.randn(16, num_experts, device=_DEVICE) for _ in range(2))
        triton_fn(_warmup, num_experts, top_k)
        get_torch_device().synchronize()

        ref_mem = _measure_peak_memory(lambda: _reference_load_balancing_loss(gate_logits, num_experts, top_k))
        triton_mem = _measure_peak_memory(lambda: triton_fn(gate_logits, num_experts, top_k))

        ref_mb = ref_mem / (1024 * 1024)
        triton_mb = triton_mem / (1024 * 1024)
        saved_mb = ref_mb - triton_mb
        print(
            f"\n[E={num_experts}, K={top_k}, L={num_layers}, BS={batch_size}, seq={seq_len}] "
            f"HF: {ref_mb:.1f} MB | Triton: {triton_mb:.1f} MB | Saved: {saved_mb:.1f} MB"
        )
        assert triton_mem < ref_mem, (
            f"Triton kernel should use less memory than HF reference: triton={triton_mb:.1f} MB >= ref={ref_mb:.1f} MB"
        )
