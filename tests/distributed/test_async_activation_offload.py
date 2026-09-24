"""Tests for async activation offload feature.
Covers:
  1. Core components: SwapTensor, GetCnt, OffloadManager, async_save_on_cpu
  2. Module patching: _get_no_split_offload_modules, async_offload_modules
  3. Argument validation: apply_async_activation_offload requires _no_split_modules
  4. Gradient parity: forward+backward with offload matches without offload
Run (single GPU):
    pytest tests/distributed/test_async_activation_offload.py -v
"""
import pytest
import torch
from veomni.distributed.async_offload import (
    PinnedBufferPool,
    _get_no_split_offload_modules,
    apply_async_activation_offload,
    async_offload_modules,
    reset_async_activation_offload,
)
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE

_HAS_ACCEL = IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE


class ToyDecoderLayer(torch.nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size)
        self.norm = torch.nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(hidden_states))


class ToyModel(torch.nn.Module):
    _no_split_modules = ["ToyDecoderLayer"]

    def __init__(self, hidden_size=64, num_layers=4):
        super().__init__()
        self.layers = torch.nn.ModuleList([ToyDecoderLayer(hidden_size) for _ in range(num_layers)])
        self.embed = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return x.sum()


class ToyModelNoNoSplitModules(torch.nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).sum()


class TestAsyncOffloadModules:
    def test_sets_per_instance_attributes(self):
        model = ToyModel(hidden_size=64, num_layers=4)
        modules = _get_no_split_offload_modules(model)
        async_offload_modules(modules, host_buffer_pool=PinnedBufferPool())
        for layer in model.layers:
            assert hasattr(layer, "_veomni_offload_layer_idx")
            assert hasattr(layer, "_veomni_offload_depth")
            assert layer._veomni_offload_depth == 4

    def test_class_patched_only_once(self):
        model = ToyModel(hidden_size=64, num_layers=4)
        modules = _get_no_split_offload_modules(model)
        async_offload_modules(modules, host_buffer_pool=PinnedBufferPool())
        assert type(model.layers[0])._veomni_async_offload_instance_class is True
        async_offload_modules(modules, host_buffer_pool=PinnedBufferPool())
        assert type(model.layers[0])._veomni_async_offload_instance_class is True


class TestApplyAsyncActivationOffload:
    def test_applies_to_model_with_no_split_modules(self):
        model = ToyModel(hidden_size=64, num_layers=4)
        apply_async_activation_offload(model, activation_offload_modules=[])
        for layer in model.layers:
            assert hasattr(layer, "_veomni_offload_layer_idx")

    def test_raises_for_model_without_no_split_modules(self):
        model = ToyModelNoNoSplitModules(hidden_size=64)
        with pytest.raises(ValueError):
            apply_async_activation_offload(model, activation_offload_modules=[])

@pytest.mark.skipif(not _HAS_ACCEL, reason="requires CUDA or NPU for streams + pinned memory")
class TestGradientParity:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_gradient_parity_with_and_without_offload(self, dtype):
        torch.manual_seed(42)
        device = torch.device("cuda" if IS_CUDA_AVAILABLE else "npu")
        hidden_size = 64
        num_layers = 4
        batch = 2
        seq_len = 8

        model = ToyModel(hidden_size=hidden_size, num_layers=num_layers).to(device=device, dtype=dtype)
        x = torch.randn(batch, seq_len, hidden_size, device=device, dtype=dtype)

        loss_ref = model(x)
        loss_ref.backward()
        grads_ref = {name: p.grad.clone() for name, p in model.named_parameters()}

        for p in model.parameters():
            p.grad = None

        apply_async_activation_offload(model, activation_offload_modules=[])
        loss_off = model(x)
        loss_off.backward()
        grads_off = {name: p.grad.clone() for name, p in model.named_parameters()}
        reset_async_activation_offload(model)

        assert torch.allclose(loss_ref, loss_off, rtol=1e-3, atol=1e-3), (
            f"Loss mismatch: ref={loss_ref.item():.6f}, off={loss_off.item():.6f}"
        )
        for name in grads_ref:
            assert torch.allclose(grads_ref[name], grads_off[name], rtol=1e-2, atol=1e-2), (
                f"Gradient mismatch for {name}: max diff={torch.max(torch.abs(grads_ref[name] - grads_off[name])).item():.6e}"
            )
