# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates.
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
"""Unit tests for DT-FSDP2 (disaggregated-tensor FSDP2 monkey-patch)."""

import copy
import os
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from veomni.distributed.dt_fsdp2 import (
    DTFSDP2_SUB_MODULE_NAMES,
    apply_dt_fsdp2_patch,
    discover_dtfsdp2_submodules,
    fully_shard as dt_fully_shard,
)
from veomni.distributed.parallel_state import init_parallel_state
from veomni.utils.device import get_device_type, get_torch_device


# ---------------------------------------------------------------------------
# Toy models
# ---------------------------------------------------------------------------

class _ToyAttention(nn.Module):
    """Minimal attention-like sub-module with a meaningful parameter count."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.v_proj(self.k_proj(self.q_proj(x))))


class _ToyMLP(nn.Module):
    """Minimal MLP sub-module."""

    def __init__(self, hidden: int = 32, intermediate: int = 64):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            nn.functional.gelu(self.gate_proj(x)) * self.up_proj(x)
        )


class _ToyDecoderLayer(nn.Module):
    """A tiny transformer decoder layer matching the standard sub-module pattern.

    Named children: ``self_attn``, ``mlp``, ``input_layernorm``,
    ``post_attention_layernorm``.
    """

    _no_split_modules = ["_ToyDecoderLayer"]

    def __init__(self, hidden: int = 32, intermediate: int = 64):
        super().__init__()
        self.self_attn = _ToyAttention(hidden)
        self.mlp = _ToyMLP(hidden, intermediate)
        self.input_layernorm = nn.LayerNorm(hidden)
        self.post_attention_layernorm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class _ToyDecoderModel(nn.Module):
    """A small multi-layer transformer model for integration testing."""

    _no_split_modules = ["_ToyDecoderLayer"]

    def __init__(self, num_layers: int = 3, hidden: int = 32, intermediate: int = 64):
        super().__init__()
        self.embed = nn.Embedding(256, hidden)
        self.layers = nn.ModuleList(
            [_ToyDecoderLayer(hidden, intermediate) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, 256, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.lm_head(x).sum()


# ---------------------------------------------------------------------------
# Layer variants for discovery tests
# ---------------------------------------------------------------------------

class _LayerWithLinearAttn(nn.Module):
    """Qwen3_5-style layer: ``linear_attn`` instead of ``self_attn``."""

    def __init__(self, hidden: int = 32, intermediate: int = 64):
        super().__init__()
        self.linear_attn = _ToyAttention(hidden)
        self.mlp = _ToyMLP(hidden, intermediate)
        self.input_layernorm = nn.LayerNorm(hidden)
        self.post_attention_layernorm = nn.LayerNorm(hidden)


class _LayerWithHyperConnection(nn.Module):
    """DeepSeekV4-style layer: ``attn_hc`` + ``ffn_hc`` in addition to standard modules."""

    def __init__(self, hidden: int = 32, intermediate: int = 64):
        super().__init__()
        self.self_attn = _ToyAttention(hidden)
        self.mlp = _ToyMLP(hidden, intermediate)
        self.attn_hc = nn.Linear(hidden, hidden, bias=False)
        self.ffn_hc = nn.Linear(hidden, hidden, bias=False)
        self.input_layernorm = nn.LayerNorm(hidden)
        self.post_attention_layernorm = nn.LayerNorm(hidden)


class _LayerMinimal(nn.Module):
    """A layer with only ``self_attn`` and norms — no ``mlp``."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.self_attn = _ToyAttention(hidden)
        self.input_layernorm = nn.LayerNorm(hidden)
        self.post_attention_layernorm = nn.LayerNorm(hidden)


# ---------------------------------------------------------------------------
# Unit tests — sub-module discovery (single-process, no GPU needed)
# ---------------------------------------------------------------------------

class TestSubmoduleDiscovery:
    """Tests for :func:`discover_dtfsdp2_submodules`."""

    def test_standard_layer(self):
        """Standard attn + mlp + norms."""
        layer = _ToyDecoderLayer()
        result = discover_dtfsdp2_submodules(layer)
        names = [n for n, _ in result]
        assert "self_attn" in names
        assert "mlp" in names
        assert "input_layernorm" in names
        assert "post_attention_layernorm" in names
        # linear_attn, attn_hc, ffn_hc should NOT be present
        assert "linear_attn" not in names
        assert "attn_hc" not in names

    def test_linear_attn_layer(self):
        """Qwen3_5-style: linear_attn instead of self_attn."""
        layer = _LayerWithLinearAttn()
        result = discover_dtfsdp2_submodules(layer)
        names = [n for n, _ in result]
        assert "linear_attn" in names
        assert "self_attn" not in names  # not present
        assert "mlp" in names

    def test_hyperconnection_layer(self):
        """DeepSeekV4-style: attn_hc + ffn_hc."""
        layer = _LayerWithHyperConnection()
        result = discover_dtfsdp2_submodules(layer)
        names = [n for n, _ in result]
        assert "attn_hc" in names
        assert "ffn_hc" in names
        # Verify order: attn-like before mlp-like before hc-modules before norms
        attn_idx = names.index("self_attn")
        mlp_idx = names.index("mlp")
        attn_hc_idx = names.index("attn_hc")
        post_norm_idx = names.index("post_attention_layernorm")
        assert attn_idx < mlp_idx < attn_hc_idx < post_norm_idx

    def test_minimal_layer(self):
        """Layer without mlp."""
        layer = _LayerMinimal()
        result = discover_dtfsdp2_submodules(layer)
        names = [n for n, _ in result]
        assert "self_attn" in names
        assert "mlp" not in names  # missing → skipped
        assert "input_layernorm" in names

    def test_empty_layer(self):
        """A bare nn.Module with no known sub-modules returns an empty list."""

        class BareModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 8)

        layer = BareModule()
        result = discover_dtfsdp2_submodules(layer)
        # ``linear`` is not in DTFSDP2_SUB_MODULE_NAMES
        assert len(result) == 0

    def test_submodule_names_are_methods_not_modules(self):
        """A module that has a regular method with a sub-module name is handled correctly.

        getattr will return the method, but isinstance(method, nn.Module) is False.
        """

        class WeirdLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.real_mod = nn.Linear(8, 8)

            def self_attn(self, x):
                return x  # this is a method, not a module

        layer = WeirdLayer()
        result = discover_dtfsdp2_submodules(layer)
        # The method ``self_attn`` should NOT be returned (it's not an nn.Module)
        names = [n for n, _ in result]
        assert "self_attn" not in names


class TestDTFSDP2SubModuleNames:
    """Validate the :data:`DTFSDP2_SUB_MODULE_NAMES` constant."""

    def test_is_tuple(self):
        assert isinstance(DTFSDP2_SUB_MODULE_NAMES, tuple)

    def test_contains_essential_modules(self):
        """The constant must include the most common sub-module names."""
        essential = {"self_attn", "linear_attn", "mlp"}
        assert essential <= set(DTFSDP2_SUB_MODULE_NAMES)

    def test_attn_before_mlp(self):
        """Attn modules should appear before mlp/ffn (forward execution order)."""
        names = DTFSDP2_SUB_MODULE_NAMES
        for attn_name in ("self_attn", "linear_attn"):
            if attn_name in names:
                assert names.index(attn_name) < names.index("mlp")


# ---------------------------------------------------------------------------
# Integration tests (require FSDP2 / torch.distributed)
# ---------------------------------------------------------------------------

def _init_test_dist(rank: int, world_size: int, port: int):
    """Initialise process group for testing on CPU (backend=gloo)."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # Use gloo for CPU tests — FSDP2 supports gloo on CPU
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _build_and_run_dtfsdp2(rank, world_size, port, enable_dt: bool):
    """Worker: build model with or without DT-FSDP2, run fwd/bwd, return loss."""
    _init_test_dist(rank, world_size, port)

    init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        dp_mode="fsdp2",
        device_type="cpu",
    )

    # Create model on CPU
    model = _ToyDecoderModel(num_layers=3, hidden=32, intermediate=64)
    # Create a copy for the native path so we start from the same initial state
    model_copy = copy.deepcopy(model) if not enable_dt else None

    if enable_dt:
        apply_dt_fsdp2_patch()

    # Import after potentially patching
    from torch.distributed._composable.fsdp import fully_shard

    # Apply FSDP2
    if enable_dt:
        # DT-FSDP2 path: per-sub-module
        for layer in model.layers:
            submods = discover_dtfsdp2_submodules(layer)
            for _, sub_mod in submods:
                fully_shard(sub_mod, hook_module=layer)
        fully_shard(model)
    else:
        # Native FSDP2 path
        for layer in model_copy.layers:
            fully_shard(layer)
        fully_shard(model_copy)

    # Forward pass
    target = model if enable_dt else model_copy
    inp = torch.randint(0, 256, (2, 16))
    out = target(inp)
    if hasattr(out, "sum"):
        out = out.sum()
    out.backward()

    # Verify gradients exist and are finite
    grads_ok = True
    for name, param in target.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                grads_ok = False
                break

    dist.barrier()
    dist.destroy_process_group()

    if not grads_ok:
        raise AssertionError("Non-finite gradients detected!")
    return out.item()


def _build_and_run_native_fsdp2(rank, world_size, port):
    """Worker for native FSDP2 path (separate function for clarity)."""
    return _build_and_run_dtfsdp2(rank, world_size, port, enable_dt=False)


def _build_and_run_dt_fsdp2(rank, world_size, port):
    """Worker for DT-FSDP2 path."""
    return _build_and_run_dtfsdp2(rank, world_size, port, enable_dt=True)


class TestDTFSDP2Integration:
    """End-to-end integration tests using torch.distributed."""

    @pytest.mark.skipif(
        not dist.is_available(),
        reason="torch.distributed is not available",
    )
    def test_dt_fsdp2_forward_backward(self):
        """DT-FSDP2 forward + backward completes without error on a toy model."""
        import socket

        import torch.multiprocessing as mp

        world_size = 2
        port = _find_free_port()

        mp.spawn(
            _build_and_run_dt_fsdp2,
            args=(world_size, port),
            nprocs=world_size,
            join=True,
        )

    @pytest.mark.skipif(
        not dist.is_available(),
        reason="torch.distributed is not available",
    )
    def test_native_fsdp2_forward_backward(self):
        """Sanity-check: native FSDP2 forward + backward works on the same model."""
        import socket

        import torch.multiprocessing as mp

        world_size = 2
        port = _find_free_port()

        mp.spawn(
            _build_and_run_native_fsdp2,
            args=(world_size, port),
            nprocs=world_size,
            join=True,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
