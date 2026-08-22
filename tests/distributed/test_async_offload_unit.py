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

"""Unit tests for MindSpeed-style async activation offload helpers."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_layers import GradientCheckpointingLayer

from veomni.arguments.arguments_types import OffloadConfig
from veomni.distributed.async_offload import (
    GetCnt,
    PinnedBufferPool,
    apply_async_activation_offload,
    base_check_fn,
    get_offload_modules,
    reset_async_activation_offload,
)
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE, get_device_type, get_torch_device


class _AsyncOffloadFSDPBlock(GradientCheckpointingLayer):
    def __init__(self, device=None):
        super().__init__()
        self.proj = nn.Linear(16, 16, device=device)

    def forward(self, hidden_states):
        return torch.nn.functional.gelu(self.proj(hidden_states))


class _AsyncOffloadFSDPModel(nn.Module):
    _no_split_modules = ["_AsyncOffloadFSDPBlock"]

    def __init__(self, device=None):
        super().__init__()
        self.layers = nn.ModuleList([_AsyncOffloadFSDPBlock(device=device) for _ in range(4)])

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        checkpoint_func = partial(checkpoint, **(gradient_checkpointing_kwargs or {}))
        for layer in self.layers:
            layer.gradient_checkpointing = True
            layer._gradient_checkpointing_func = checkpoint_func

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def test_offload_config_allows_no_modules_for_auto_discovery():
    cfg = OffloadConfig(enable_async_activation=True)
    assert cfg.enable_async_activation is True
    assert cfg.activation_offload_modules == []


def test_offload_config_rejects_sync_and_async_activation_together():
    with pytest.raises(ValueError, match="mutually exclusive"):
        OffloadConfig(
            enable_activation=True,
            enable_async_activation=True,
            activation_offload_modules=["model.layers.{*}"],
        )


@pytest.mark.parametrize("cache_limit", [-1.0, float("inf"), float("nan"), float.fromhex("0x1.fffffffffffffp+1023")])
def test_offload_config_rejects_invalid_host_cache_limit(cache_limit):
    with pytest.raises(ValueError, match="activation_offload_host_cache_limit_gb"):
        OffloadConfig(activation_offload_host_cache_limit_gb=cache_limit)


def test_get_offload_modules_glob_is_segment_aware():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])

    matched = get_offload_modules(Toy(), ["model.layers.*"])

    assert [item[0] for item in matched] == ["model.layers.0", "model.layers.1"]


def test_get_cnt_unique_keys_across_second_pass():
    """Second forward over the same layer indices must not collide keys."""
    cnt = GetCnt()
    first = [cnt.get_cnt(i)[0] for i in range(3)]
    second = [cnt.get_cnt(i)[0] for i in range(3)]
    assert first == ["0_0", "1_0", "2_0"]
    assert second == ["0_1", "1_1", "2_1"]
    assert set(first).isdisjoint(second)

    cnt = GetCnt()
    assert [cnt.get_cnt(i)[1] for i in (0, 5, 2)] == [None, 0, 5]


def test_get_prefetch_keys_use_previous_offloaded_layer_and_its_tensor_count():
    cnt = GetCnt()
    cnt.get_cnt(2)
    cnt.get_cnt(2)
    cnt.get_cnt(5)

    assert cnt.get_prefetch_keys(5) == ["2_0", "2_1"]


def test_get_offload_modules_brace_star_expands_sequential():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    model = Toy()
    matched = get_offload_modules(model, ["model.layers.{*}"])
    names = [item[0] for item in matched]
    assert names == ["model.layers.0", "model.layers.1", "model.layers.2"]
    # depth field is rewritten to total offload layer count
    assert all(item[-1] == 3 for item in matched)
    assert [item[2] for item in matched] == [0, 1, 2]


def test_apply_async_activation_offload_rejects_unmatched_patterns():
    model = nn.Sequential(nn.Linear(4, 4))

    with pytest.raises(ValueError, match="did not match any model modules"):
        apply_async_activation_offload(model, ["model.layers.{*}"])


def test_apply_async_activation_offload_auto_discovers_no_split_modules():
    class DecoderLayer(nn.Module):
        pass

    class Toy(nn.Module):
        _no_split_modules = ["DecoderLayer"]

        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([DecoderLayer(), DecoderLayer()])

    model = Toy()
    apply_async_activation_offload(model, [])

    assert [layer._veomni_offload_layer_idx for layer in model.layers] == [0, 1]
    # FSDP2 and torch.compile discover blocks by exact class name after this
    # hook is installed, so the private wrapper must preserve that identity.
    assert [layer.__class__.__name__ for layer in model.layers] == ["DecoderLayer", "DecoderLayer"]
    assert len([module for module in model.modules() if module.__class__.__name__ in model._no_split_modules]) == 2
    assert all(layer._veomni_offload_depth == 2 for layer in model.layers)
    assert model.layers[0]._veomni_offload_manager is model.layers[1]._veomni_offload_manager


@pytest.mark.parametrize(
    "no_split_modules, error_match",
    [
        (None, "requires model._no_split_modules"),
        (["MissingDecoderLayer"], "did not match any modules"),
    ],
)
def test_apply_async_activation_offload_auto_discovery_fails_closed(no_split_modules, error_match):
    model = nn.Sequential(nn.Linear(4, 4))
    if no_split_modules is not None:
        model._no_split_modules = no_split_modules

    with pytest.raises(ValueError, match=error_match):
        apply_async_activation_offload(model, [])


def test_async_offload_patch_is_confined_to_selected_instances_and_models():
    class DecoderLayer(nn.Module):
        def forward(self, hidden_states):
            return hidden_states * 2

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.selected = DecoderLayer()
            self.unselected = DecoderLayer()

    first_model = Toy()
    second_model = Toy()
    apply_async_activation_offload(first_model, ["selected"])
    apply_async_activation_offload(second_model, ["selected"])

    assert type(first_model.selected) is not DecoderLayer
    assert type(first_model.unselected) is DecoderLayer
    assert type(second_model.unselected) is DecoderLayer
    assert first_model.selected._veomni_offload_manager is not second_model.selected._veomni_offload_manager
    torch.testing.assert_close(first_model.selected(torch.ones(2)), torch.full((2,), 2.0))


def test_async_offload_manager_resets_after_forward_error():
    class FailingLayer(nn.Module):
        def forward(self, hidden_states):
            raise RuntimeError("expected failure")

    model = nn.Sequential(FailingLayer())
    apply_async_activation_offload(model, ["0"])
    manager = model[0]._veomni_offload_manager
    manager.get_cnt(7)

    with pytest.raises(RuntimeError, match="expected failure"):
        model(torch.ones(2))

    assert not manager.items
    assert manager.getcnt._block_idx == -1
    assert manager.getcnt._block_tensor_nums == {}


def test_async_offload_rejects_tensor_views_with_shared_storage():
    base = torch.arange(8.0)
    view = base[2:6]

    assert not base_check_fn(view)
    torch.testing.assert_close(base, torch.arange(8.0))


def test_async_offload_rejects_cpu_tensors():
    assert not base_check_fn(torch.ones(4))


def test_pinned_buffer_pool_reuses_matching_layout():
    pool = PinnedBufferPool()
    source = torch.empty((2, 3), dtype=torch.float32)
    first, key = pool.acquire(source)
    pool.release(first, key)
    second, second_key = pool.acquire(source)

    assert first.data_ptr() == second.data_ptr()
    assert key == second_key
    assert pool.allocations == 1
    assert pool.reuses == 1


def test_pinned_buffer_pool_cache_is_bounded_across_dynamic_shapes():
    pool = PinnedBufferPool(max_cached_bytes=64)

    for width in (4, 8, 12):
        source = torch.empty((width,), dtype=torch.float32)
        buffer, key = pool.acquire(source)
        pool.release(buffer, key)

    assert pool.cached_bytes <= 64
    assert pool.evictions > 0


def test_pinned_buffer_pool_refreshes_layout_recency_on_reuse():
    pool = PinnedBufferPool(max_cached_bytes=48)
    layouts = [
        torch.empty((4,), dtype=torch.float32),
        torch.empty((2, 2), dtype=torch.float32),
        torch.empty((1, 4), dtype=torch.float32),
        torch.empty((4, 1), dtype=torch.float32),
    ]

    first_key_buffers = [pool.acquire(layouts[0]) for _ in range(2)]
    for buffer, key in first_key_buffers:
        pool.release(buffer, key)
    second_buffer, second_key = pool.acquire(layouts[1])
    pool.release(second_buffer, second_key)

    reused_buffer, reused_key = pool.acquire(layouts[0])
    pool.release(reused_buffer, reused_key)
    for layout in layouts[2:]:
        buffer, key = pool.acquire(layout)
        pool.release(buffer, key)

    allocations_before = pool.allocations
    pool.acquire(layouts[1])

    assert pool.allocations == allocations_before + 1


def test_async_offload_manager_resets_at_step_boundary():
    model = nn.Sequential(nn.Linear(2, 2))
    apply_async_activation_offload(model, ["0"])
    manager = model[0]._veomni_offload_manager
    manager.get_cnt(9)

    reset_async_activation_offload(model)

    assert not manager.items
    assert manager.getcnt._block_idx == -1
    assert manager.getcnt._block_tensor_nums == {}


@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE),
    reason="CUDA or NPU is required for async D2H/H2D validation",
)
def test_async_offload_accelerator_gradient_parity_and_peak_memory():
    device_api = get_torch_device()
    device = torch.device(get_device_type())

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(512, 512)

        def forward(self, hidden_states):
            return torch.nn.functional.gelu(self.proj(hidden_states))

    def make_model():
        return nn.Sequential(*(Block() for _ in range(4))).to(device)

    torch.manual_seed(0)
    baseline = make_model()
    initial_state = {name: parameter.detach().cpu().clone() for name, parameter in baseline.state_dict().items()}
    input_cpu = torch.randn(4, 128, 512)

    device_api.reset_peak_memory_stats()
    baseline_before = device_api.memory_allocated()
    baseline_input = input_cpu.to(device).requires_grad_(True)
    baseline_output = baseline(baseline_input)
    baseline_output.square().mean().backward()
    device_api.synchronize()
    baseline_peak = device_api.max_memory_allocated() - baseline_before

    baseline_output_cpu = baseline_output.detach().cpu()
    baseline_input_grad_cpu = baseline_input.grad.detach().cpu()
    baseline_parameter_grads = [parameter.grad.detach().cpu() for parameter in baseline.parameters()]

    del baseline_output, baseline_input, baseline
    device_api.empty_cache()

    offloaded = make_model()
    offloaded.load_state_dict(initial_state)
    apply_async_activation_offload(offloaded, ["0", "1", "2", "3"])

    device_api.reset_peak_memory_stats()
    offloaded_before = device_api.memory_allocated()
    offloaded_input = input_cpu.to(device).requires_grad_(True)
    offloaded_output = offloaded(offloaded_input)
    offloaded_output.square().mean().backward()
    device_api.synchronize()
    offloaded_peak = device_api.max_memory_allocated() - offloaded_before

    manager = offloaded[0]._veomni_offload_manager
    assert not manager.items
    assert manager.host_buffer_pool.allocations > 0

    torch.testing.assert_close(offloaded_output.detach().cpu(), baseline_output_cpu)
    torch.testing.assert_close(offloaded_input.grad.detach().cpu(), baseline_input_grad_cpu)
    for offloaded_parameter, baseline_parameter_grad in zip(offloaded.parameters(), baseline_parameter_grads):
        torch.testing.assert_close(offloaded_parameter.grad.detach().cpu(), baseline_parameter_grad)

    # This is a numerical and memory gate, not only an execution smoke test:
    # offloading must preserve gradients and must not increase peak allocated
    # device memory for the same model and input.
    assert offloaded_peak <= baseline_peak, (
        f"async offload increased peak allocated memory: baseline={baseline_peak}, offloaded={offloaded_peak}"
    )

    del offloaded_output, offloaded_input
    offloaded.zero_grad(set_to_none=True)
    second_input = input_cpu.to(device).requires_grad_(True)
    second_output = offloaded(second_input)
    second_output.square().mean().backward()
    device_api.synchronize()
    assert manager.host_buffer_pool.reuses > 0


def _run_async_offload_base_trainer_fsdp2_gc():
    import torch.distributed as dist

    from veomni.arguments import (
        ChunkMBSConfig,
        FSDPConfig,
        GradientCheckpointingConfig,
        MixedPrecisionConfig,
        OptimizerConfig,
        TorchCompileConfig,
    )
    from veomni.distributed.parallel_state import init_parallel_state, use_parallel_state
    from veomni.trainer.base import BaseTrainer

    world_size = dist.get_world_size()
    init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        dp_mode="fsdp2",
        device_type=get_device_type(),
        name="base",
    )

    trainer = object.__new__(BaseTrainer)
    trainer.model = _AsyncOffloadFSDPModel(device="meta")
    trainer.args = SimpleNamespace(
        model=SimpleNamespace(
            lora_config=None,
            fqn_to_index_mapping=None,
            model_path=None,
            basic_modules=[],
        ),
        train=SimpleNamespace(
            accelerator=SimpleNamespace(
                offload_config=OffloadConfig(
                    enable_async_activation=True,
                    activation_offload_host_cache_limit_gb=0.01,
                ),
                fsdp_config=FSDPConfig(mixed_precision=MixedPrecisionConfig(enable=False)),
            ),
            optimizer=OptimizerConfig(),
            chunk_mbs_config=ChunkMBSConfig(enable=False),
            checkpoint=SimpleNamespace(load_path=None),
            init_device="meta",
            gradient_checkpointing=GradientCheckpointingConfig(enable=True),
            broadcast_model_weights_from_rank0=False,
            ep_sharded_stream_load=False,
            torch_compile=TorchCompileConfig(enable=False),
        ),
    )

    torch.manual_seed(0)
    with use_parallel_state("base"):
        trainer._build_parallelized_model()

    assert all(layer.gradient_checkpointing for layer in trainer.model.layers)
    manager = trainer.model.layers[0]._veomni_offload_manager
    hidden_states = torch.randn(2, 8, 16, device=get_device_type(), requires_grad=True)
    with use_parallel_state("base"):
        output = trainer.model(hidden_states)
    with use_parallel_state("base"):
        output.float().square().mean().backward()

    assert torch.isfinite(output).all()
    assert manager.host_buffer_pool.allocations > 0
    assert not manager.items
    for parameter in trainer.model.parameters():
        if parameter.grad is not None:
            local_grad = parameter.grad.to_local() if hasattr(parameter.grad, "to_local") else parameter.grad
            assert torch.isfinite(local_grad).all()
    dist.barrier()


@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE),
    reason="CUDA or NPU is required for FSDP2 async-offload integration",
)
def test_async_offload_base_trainer_fsdp2_gc_integration():
    from ..tools.launch_utils import torchrun

    torchrun(_run_async_offload_base_trainer_fsdp2_gc, world_size=2)
