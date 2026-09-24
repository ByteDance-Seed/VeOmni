"""Tests for async activation offload feature.
Covers:
  1. Core components: SwapTensor, GetCnt, OffloadManager, async_save_on_cpu
  2. Module patching: _get_no_split_offload_modules, async_offload_modules
  3. Argument validation: apply_async_activation_offload requires _no_split_modules
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
)


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
