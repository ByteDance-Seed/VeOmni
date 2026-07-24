"""Tests for async activation offload feature.

Covers:
  1. Core components: SwapTensor, GetCnt, OffloadManager, async_save_on_cpu
  2. Module patching: get_offload_modules_from_class_names, async_offload_modules
  3. Argument validation: enable_async_activation_offload requires gradient_checkpointing

Run (single GPU):
    pytest tests/distributed/test_async_activation_offload.py -v
"""

import os

import torch
import pytest

from veomni.distributed.async_offloading import (
    GetCnt,
    OffloadManager,
    SwapTensor,
    async_save_on_cpu,
    async_offload_modules,
    get_offload_modules_from_class_names,
    apply_async_activation_offload,
    base_check_fn,
    _Singleton,
)
from veomni.arguments import OffloadConfig, GradientCheckpointingConfig, TrainingArguments, AcceleratorConfig


# ---------------------------------------------------------------------------
# Toy models for testing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# async_offload_modules tests
# ---------------------------------------------------------------------------


class TestAsyncOffloadModules:
    def setup_method(self):
        _Singleton._instances.clear()

    def test_sets_per_instance_attributes(self):
        model = ToyModel(hidden_size=64, num_layers=4)
        modules = get_offload_modules_from_class_names(model, ["ToyDecoderLayer"])
        async_offload_modules(modules)

        for layer in model.layers:
            assert hasattr(layer, "_veomni_offload_layer_idx")
            assert hasattr(layer, "_veomni_offload_depth")
            assert layer._veomni_offload_depth == 4

    def test_class_patched_only_once(self):
        model = ToyModel(hidden_size=64, num_layers=4)
        modules = get_offload_modules_from_class_names(model, ["ToyDecoderLayer"])

        async_offload_modules(modules)
        assert ToyDecoderLayer._veomni_async_offload_patched is True

        async_offload_modules(modules)
        assert ToyDecoderLayer._veomni_async_offload_patched is True


# ---------------------------------------------------------------------------
# apply_async_activation_offload tests
# ---------------------------------------------------------------------------


class TestApplyAsyncActivationOffload:
    def setup_method(self):
        _Singleton._instances.clear()
        if hasattr(ToyDecoderLayer, "_veomni_async_offload_patched"):
            delattr(ToyDecoderLayer, "_veomni_async_offload_patched")

    def test_applies_to_model_with_no_split_modules(self):
        model = ToyModel(hidden_size=64, num_layers=4)
        apply_async_activation_offload(model)

        for layer in model.layers:
            assert hasattr(layer, "_veomni_offload_layer_idx")

    def test_warns_for_model_without_no_split_modules(self, capfd):
        model = ToyModelNoNoSplitModules(hidden_size=64)
        apply_async_activation_offload(model)
