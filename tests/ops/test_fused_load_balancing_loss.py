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


DEFAULT_ATOL = 1e-4

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
    from veomni.ops.kernels.load_balancing_loss.triton import load_balancing_loss_triton

    return load_balancing_loss_triton


def _get_pytorch_impl():
    from veomni.ops.kernels.load_balancing_loss.eager import load_balancing_loss_pytorch

    return load_balancing_loss_pytorch


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


# ---------------------------------------------------------------------------
# Triton kernel tests
# ---------------------------------------------------------------------------


class TestTritonLoadBalancingLoss:
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
        """All tokens masked out should return 0.

        MoE load balancing loss supports masking padded tokens via attention_mask
        so that padding does not skew expert routing statistics.
        """
        _skip_no_cuda()
        triton_fn = _get_triton_impl()
        gate_logits = tuple(torch.randn(8, 4, device=_DEVICE) for _ in range(2))
        attention_mask = torch.zeros(2, 4, device=_DEVICE)

        out = triton_fn(gate_logits, 4, 2, attention_mask)
        assert out.item() == 0.0

    @pytest.mark.parametrize("num_experts,top_k,num_layers,batch_size,seq_len", _CONFIGS)
    def test_forward_no_mask(self, num_experts, top_k, num_layers, batch_size, seq_len):
        _skip_no_cuda()
        triton_fn = _get_triton_impl()

        torch.manual_seed(42)
        gate_logits = _make_gate_logits(batch_size, seq_len, num_experts, num_layers)

        ref = _reference_load_balancing_loss(gate_logits, num_experts, top_k)
        out = triton_fn(gate_logits, num_experts, top_k)

        torch.testing.assert_close(out, ref, atol=DEFAULT_ATOL, rtol=DEFAULT_ATOL)

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

        torch.testing.assert_close(out, ref, atol=DEFAULT_ATOL, rtol=DEFAULT_ATOL)

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

        torch.testing.assert_close(fused_loss, ref_loss, atol=DEFAULT_ATOL, rtol=DEFAULT_ATOL)
        for i, (rg, fg) in enumerate(zip(ref_grads, fused_grads)):
            torch.testing.assert_close(
                fg, rg, atol=DEFAULT_ATOL, rtol=DEFAULT_ATOL, msg=f"Gradient mismatch at layer {i}"
            )

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

        torch.testing.assert_close(fused_loss, ref_loss, atol=DEFAULT_ATOL, rtol=DEFAULT_ATOL)
        for i, (rg, fg) in enumerate(zip(ref_grads, fused_grads)):
            torch.testing.assert_close(
                fg, rg, atol=DEFAULT_ATOL, rtol=DEFAULT_ATOL, msg=f"Gradient mismatch at layer {i}"
            )

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

    @pytest.mark.parametrize(
        "num_experts,top_k,num_layers,batch_size,seq_len",
        [(8, 2, 2, 4, 128), (60, 8, 4, 2, 512)],
    )
    def test_determinism(self, num_experts, top_k, num_layers, batch_size, seq_len):
        """Triton kernel must produce bitwise-identical results across runs."""
        _skip_no_cuda()
        triton_fn = _get_triton_impl()

        torch.manual_seed(42)
        gate_logits = _make_gate_logits(batch_size, seq_len, num_experts, num_layers)

        results = [triton_fn(gate_logits, num_experts, top_k) for _ in range(5)]
        for r in results[1:]:
            assert torch.equal(results[0], r), "Triton forward is non-deterministic"

    @pytest.mark.parametrize(
        "num_experts,top_k,num_layers,batch_size,seq_len",
        [(8, 2, 2, 4, 128), (60, 8, 4, 2, 512)],
    )
    def test_determinism_with_mask(self, num_experts, top_k, num_layers, batch_size, seq_len):
        """Triton kernel must produce bitwise-identical results with mask."""
        _skip_no_cuda()
        triton_fn = _get_triton_impl()

        torch.manual_seed(42)
        gate_logits = _make_gate_logits(batch_size, seq_len, num_experts, num_layers)
        attention_mask = torch.ones(batch_size, seq_len, device=_DEVICE)
        attention_mask[:, seq_len // 2 :] = 0

        results = [triton_fn(gate_logits, num_experts, top_k, attention_mask) for _ in range(5)]
        for r in results[1:]:
            assert torch.equal(results[0], r), "Triton forward with mask is non-deterministic"


# ---------------------------------------------------------------------------
# PyTorch eager tests
# ---------------------------------------------------------------------------


class TestPytorchLoadBalancingLoss:
    """Test suite comparing PyTorch for-loop implementation against HF reference."""

    def test_none_input(self):
        pytorch_fn = _get_pytorch_impl()
        assert pytorch_fn(None, 8, 2) == 0

    def test_non_tuple_input(self):
        pytorch_fn = _get_pytorch_impl()
        logits = torch.randn(32, 8, device=_DEVICE)
        assert pytorch_fn(logits, 8, 2) == 0

    def test_forward_full_mask(self):
        """All tokens masked out should return 0.

        MoE load balancing loss supports masking padded tokens via attention_mask
        so that padding does not skew expert routing statistics.
        """
        pytorch_fn = _get_pytorch_impl()
        gate_logits = tuple(torch.randn(8, 4, device=_DEVICE) for _ in range(2))
        attention_mask = torch.zeros(2, 4, device=_DEVICE)

        out = pytorch_fn(gate_logits, 4, 2, attention_mask)
        assert out.item() == 0.0

    @pytest.mark.parametrize("num_experts,top_k,num_layers,batch_size,seq_len", _CONFIGS)
    def test_forward_no_mask(self, num_experts, top_k, num_layers, batch_size, seq_len):
        pytorch_fn = _get_pytorch_impl()

        torch.manual_seed(42)
        gate_logits = _make_gate_logits(batch_size, seq_len, num_experts, num_layers)

        ref = _reference_load_balancing_loss(gate_logits, num_experts, top_k)
        out = pytorch_fn(gate_logits, num_experts, top_k)

        torch.testing.assert_close(out, ref, atol=DEFAULT_ATOL, rtol=DEFAULT_ATOL)

    @pytest.mark.parametrize("num_experts,top_k,num_layers,batch_size,seq_len", _CONFIGS)
    def test_forward_with_mask(self, num_experts, top_k, num_layers, batch_size, seq_len):
        pytorch_fn = _get_pytorch_impl()

        torch.manual_seed(123)
        gate_logits = _make_gate_logits(batch_size, seq_len, num_experts, num_layers)
        attention_mask = torch.ones(batch_size, seq_len, device=_DEVICE)
        attention_mask[:, seq_len // 2 :] = 0

        ref = _reference_load_balancing_loss(gate_logits, num_experts, top_k, attention_mask)
        out = pytorch_fn(gate_logits, num_experts, top_k, attention_mask)

        torch.testing.assert_close(out, ref, atol=DEFAULT_ATOL, rtol=DEFAULT_ATOL)

    @pytest.mark.parametrize(
        "num_experts,top_k,num_layers,batch_size,seq_len",
        [(8, 2, 2, 4, 128), (60, 8, 4, 2, 512)],
    )
    def test_determinism(self, num_experts, top_k, num_layers, batch_size, seq_len):
        """PyTorch implementation must produce bitwise-identical results across runs."""
        pytorch_fn = _get_pytorch_impl()

        torch.manual_seed(42)
        gate_logits = _make_gate_logits(batch_size, seq_len, num_experts, num_layers)

        results = [pytorch_fn(gate_logits, num_experts, top_k) for _ in range(5)]
        for r in results[1:]:
            assert torch.equal(results[0], r), "PyTorch forward is non-deterministic"
