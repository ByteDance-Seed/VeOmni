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

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments.arguments_types import OffloadConfig
from veomni.distributed.async_offload import (
    GetCnt,
    OffloadManager,
    PinnedBufferPool,
    SwapTensor,
    _has_private_dense_storage,
    _unpack_swap_tensor,
    apply_async_activation_offload,
    base_check_fn,
    get_offload_modules,
    reset_async_activation_offload,
)
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE, get_device_type, get_torch_device


_CI_ACCELERATOR_2 = LaunchSpec(nproc=2, device="accelerator")


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

    patched_class = type(first_model.selected)
    apply_async_activation_offload(first_model, ["selected"])
    assert type(first_model.selected) is patched_class
    value = torch.randn(2, 4, requires_grad=True)
    output = first_model.selected(value)
    torch.testing.assert_close(output, value * 2)
    output.sum().backward()
    torch.testing.assert_close(value.grad, torch.full_like(value, 2))

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

    assert not _has_private_dense_storage(view)
    assert not base_check_fn(view)
    torch.testing.assert_close(base, torch.arange(8.0))


def test_private_dense_storage_allows_padded_contiguous_owner():
    tensor = torch.arange(4.0)
    tensor.untyped_storage().resize_(tensor.untyped_storage().nbytes() * 2)

    assert _has_private_dense_storage(tensor)
    alias = tensor.as_strided((2,), (1,), storage_offset=2)
    assert not _has_private_dense_storage(alias)


@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE),
    reason="CUDA or NPU is required for async D2H/H2D validation",
)
def test_wait_d2h_finished_preserves_offset_zero_set_alias():
    device = torch.device(get_device_type())
    original = torch.randn(8, 8, device=device)
    original.untyped_storage().resize_(original.untyped_storage().nbytes() * 2)
    expected = original.detach().cpu().clone()
    alias = torch.empty_like(original)
    alias.set_(original.untyped_storage(), 0, original.size(), original.stride())
    assert alias._base is None
    assert alias.storage_offset() == 0

    manager = OffloadManager(PinnedBufferPool(max_cached_bytes=1 << 20))
    swap = SwapTensor(original, "0_0", manager.host_buffer_pool)
    swap.launch_d2h(manager.swap_stream)
    swap.wait_d2h_finished()

    torch.testing.assert_close(alias.cpu(), expected)
    restored = _unpack_swap_tensor(manager, swap, prefetch=False)
    torch.testing.assert_close(restored.cpu(), expected)


def test_async_offload_rejects_cpu_tensors():
    assert not base_check_fn(torch.ones(4))


class _FakeEvent:
    """Minimal stand-in for the device events the swap path records and waits on."""

    def __init__(self):
        self.recorded = 0
        self.synced = 0

    def record(self):
        self.recorded += 1

    def synchronize(self):
        self.synced += 1


class _FakeStream:
    """Stand-in for a swap stream: only ``wait_event`` is touched."""

    def wait_event(self, _event):
        pass


def _patch_swap_stream(monkeypatch):
    """Route event creation and stream switching to the stand-ins above."""
    from contextlib import nullcontext

    from veomni.distributed import async_offload

    monkeypatch.setattr(async_offload, "create_event", _FakeEvent)
    monkeypatch.setattr(async_offload, "switch_to_specified_stream", lambda _stream: nullcontext())


def test_deferred_free_requires_an_allocator_owned_storage(monkeypatch):
    """``_deferred_free_supported`` accepts only storage the native allocator owns.

    Foreign storage (``from_numpy`` / ``frombuffer`` / ``from_dlpack``) and COW
    storage (``torch._lazy_clone``) must be rejected: ``record_stream`` is a no-op
    for them, so detaching would free memory an in-flight copy is reading.
    """
    from veomni.distributed import async_offload

    monkeypatch.setattr(async_offload, "is_caching_allocator_active", lambda: True)

    owned = torch.empty(4)
    assert async_offload._deferred_free_supported(owned)

    import numpy as np

    foreign = torch.from_numpy(np.zeros(4, dtype=np.float32))
    assert not foreign.untyped_storage().resizable()
    assert not async_offload._deferred_free_supported(foreign)

    if hasattr(torch, "_lazy_clone") and hasattr(torch._C, "_is_cow_tensor"):
        cow_source = torch.empty(4)
        cow_clone = torch._lazy_clone(cow_source)
        assert cow_clone.untyped_storage().resizable()
        assert not async_offload._deferred_free_supported(cow_clone)
        assert not async_offload._deferred_free_supported(cow_source)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="NVIDIA CUDA required for the allocator-backend probe")
def test_is_caching_allocator_active_accepts_only_the_native_backend(monkeypatch):
    """Only the native caching allocator may take the deferred-free path.

    ``backend:cudaMallocAsync`` has its own ``recordStream`` that asserts the
    pointer is in its allocation map, so an external DLPack tensor aborts there.
    """
    from veomni.utils import device

    monkeypatch.delenv("PYTORCH_NO_CUDA_MEMORY_CACHING", raising=False)
    for backend, expected in (("native", True), ("cudaMallocAsync", False)):
        fake = SimpleNamespace(memory=SimpleNamespace(get_allocator_backend=lambda b=backend: b))
        monkeypatch.setattr(device, "get_torch_device", lambda f=fake: f)
        assert device.is_caching_allocator_active() is expected


def test_launch_d2h_gates_record_stream_on_deferred_free(monkeypatch):
    """``record_stream`` and the host wait in ``wait_d2h_finished`` must be paired.

    Skipping ``record_stream`` without the synchronize frees memory mid-copy;
    calling it where the allocator ignores it is a no-op hiding the same bug.
    """
    import numpy as np

    from veomni.distributed import async_offload

    _patch_swap_stream(monkeypatch)

    recorded_streams = []
    monkeypatch.setattr(
        torch.Tensor,
        "record_stream",
        lambda self, s: recorded_streams.append((self, s)),
    )

    pool = PinnedBufferPool(max_cached_bytes=1 << 20)
    stream = _FakeStream()

    # Unsupported allocator -> no record_stream call, host synchronize in wait_d2h_finished.
    monkeypatch.setattr(async_offload, "is_caching_allocator_active", lambda: False)
    t_no_alloc = torch.arange(4.0)
    swap_no_alloc = SwapTensor(t_no_alloc, "0_0", pool)
    swap_no_alloc.launch_d2h(stream)
    assert recorded_streams == []
    assert swap_no_alloc._deferred_free is False
    swap_no_alloc.wait_d2h_finished()
    assert swap_no_alloc.d2h_event.synced == 1

    # Native allocator with foreign storage -> no record_stream call, host synchronize.
    monkeypatch.setattr(async_offload, "is_caching_allocator_active", lambda: True)
    t_foreign = torch.from_numpy(np.arange(4, dtype=np.float32))
    swap_foreign = SwapTensor(t_foreign, "0_1", pool)
    swap_foreign.launch_d2h(stream)
    assert recorded_streams == []
    assert swap_foreign._deferred_free is False
    swap_foreign.wait_d2h_finished()
    assert swap_foreign.d2h_event.synced == 1

    # Native allocator with owned storage -> record_stream called, host synchronize skipped.
    t_owned = torch.arange(4.0)
    swap_owned = SwapTensor(t_owned, "0_2", pool)
    swap_owned.launch_d2h(stream)
    assert recorded_streams == [(t_owned, stream)]
    assert swap_owned._deferred_free is True
    swap_owned.wait_d2h_finished()
    assert swap_owned.d2h_event.synced == 0


def test_release_host_buffer_syncs_abandoned_d2h_without_deferred_free(monkeypatch):
    """A wrapper torn down mid-D2H must not leave a copy reading freed device memory.

    Without ``record_stream``, dropping the last device reference before the copy
    finishes is unsafe, so teardown synchronizes. The deferred path must not: the
    pool's buffer tag covers reuse.
    """
    from veomni.distributed import async_offload

    _patch_swap_stream(monkeypatch)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, _s: None)

    stream = _FakeStream()

    # Non-deferred path: abandoned forward graph (stat == "d2h") must sync d2h_event.
    monkeypatch.setattr(async_offload, "is_caching_allocator_active", lambda: False)
    pool_sync = PinnedBufferPool(max_cached_bytes=1 << 20)
    swap_sync = SwapTensor(torch.arange(4.0), "0_0", pool_sync)
    host_buf_sync = swap_sync.tensor_cpu
    swap_sync.launch_d2h(stream)
    assert swap_sync.stat == "d2h"
    assert swap_sync._deferred_free is False
    swap_sync.release_host_buffer()
    assert swap_sync.d2h_event.synced == 1
    assert "_last_swap_event" not in host_buf_sync.__dict__

    # Native deferred path: abandoned forward graph skips host sync and keeps the tag for pool reuse.
    monkeypatch.setattr(async_offload, "is_caching_allocator_active", lambda: True)
    pool_async = PinnedBufferPool(max_cached_bytes=1 << 20)
    swap_async = SwapTensor(torch.arange(4.0), "0_1", pool_async)
    host_buf_async = swap_async.tensor_cpu
    swap_async.launch_d2h(stream)
    assert swap_async.stat == "d2h"
    assert swap_async._deferred_free is True
    swap_async.release_host_buffer()
    assert swap_async.d2h_event.synced == 0
    assert host_buf_async.__dict__.get("_last_swap_event") is swap_async.d2h_event


def test_shared_host_buffer_pool_survives_repeated_application():
    """The per-module application path must not multiply the configured limit.

    A composed model applies offload one module at a time, and each call needs
    its own manager because ``layer_idx`` restarts at 0. The host limit bounds
    pinned memory for one pool. Sharing that pool across calls keeps one budget;
    omitting it (passing only a limit) would give each call its own pool.
    """
    thinker = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    talker = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    pool = PinnedBufferPool(max_cached_bytes=1 << 20)

    apply_async_activation_offload(thinker, ["0", "1"], host_buffer_pool=pool)
    apply_async_activation_offload(talker, ["0", "1"], host_buffer_pool=pool)

    offloaded = [thinker[0], thinker[1], talker[0], talker[1]]
    assert len({id(module._veomni_offload_manager) for module in offloaded}) == 2
    assert all(module._veomni_offload_manager.host_buffer_pool is pool for module in offloaded)


def test_apply_async_activation_offload_builds_a_pool_when_given_a_limit():
    model = nn.Sequential(nn.Linear(4, 4))

    apply_async_activation_offload(model, ["0"], host_cache_limit_bytes=1 << 20)

    assert model[0]._veomni_offload_manager.host_buffer_pool.max_cached_bytes == 1 << 20


def test_offload_manager_rejects_a_non_pool_argument():
    """Fail at wiring time, not from inside the pack hook on the first forward."""
    with pytest.raises(TypeError, match="PinnedBufferPool"):
        OffloadManager(1 << 20)


def test_apply_async_activation_offload_rejects_a_limit_and_a_pool_together():
    model = nn.Sequential(nn.Linear(4, 4))

    with pytest.raises(ValueError, match="not both"):
        apply_async_activation_offload(
            model,
            ["0"],
            host_cache_limit_bytes=1 << 20,
            host_buffer_pool=PinnedBufferPool(),
        )


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


def test_pinned_buffer_pool_waits_for_pending_event_on_eviction():
    """An evicted buffer must be synchronized even though it never returns via ``acquire``.

    Eviction frees the buffer for good, so ``release`` itself must wait, and do it
    outside the pool lock since the pool is shared across managers and threads.
    """
    pool = PinnedBufferPool(max_cached_bytes=32)

    class _FakeEvent:
        def __init__(self):
            self.sync_count = 0
            self.lock_free_during_sync = None

        def synchronize(self):
            self.sync_count += 1
            acquired = pool._lock.acquire(blocking=False)
            self.lock_free_during_sync = acquired
            if acquired:
                pool._lock.release()

    first, first_key = pool.acquire(torch.empty((4,), dtype=torch.float32))  # 16 bytes
    second, second_key = pool.acquire(torch.empty((8,), dtype=torch.float32))  # 32 bytes
    pending = _FakeEvent()
    first._last_swap_event = pending

    # Release fits in the cache: wait is deferred and the tag stays attached.
    pool.release(first, first_key)
    assert pending.sync_count == 0
    assert "_last_swap_event" in first.__dict__

    # Releasing a 32-byte buffer forces eviction of ``first`` without ever returning via ``acquire()``.
    pool.release(second, second_key)
    assert pool.evictions == 1
    assert pending.sync_count == 1
    assert pending.lock_free_during_sync is True
    assert "_last_swap_event" not in first.__dict__


def test_pinned_buffer_pool_waits_for_pending_event_on_drop():
    """An oversized buffer that bypasses the cache must be synchronized on ``release``.

    A drop is chosen by size, not cache pressure, so it exercises a different
    branch than eviction.
    """
    pool = PinnedBufferPool(max_cached_bytes=16)

    class _FakeEvent:
        def __init__(self, target_pool):
            self._pool = target_pool
            self.sync_count = 0
            self.lock_free_during_sync = None

        def synchronize(self):
            self.sync_count += 1
            acquired = self._pool._lock.acquire(blocking=False)
            self.lock_free_during_sync = acquired
            if acquired:
                self._pool._lock.release()

    # Drop path: oversized buffer bypasses cache and synchronizes on release outside the lock.
    big_src = torch.empty((8,), dtype=torch.float32)  # 32 bytes > max_cached_bytes=16
    big_buf, big_key = pool.acquire(big_src)
    drop_event = _FakeEvent(pool)
    big_buf._last_swap_event = drop_event
    pool.release(big_buf, big_key)
    assert pool.drops == 1
    assert drop_event.sync_count == 1
    assert drop_event.lock_free_during_sync is True
    assert "_last_swap_event" not in big_buf.__dict__


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
def test_unpack_swap_tensor_survives_repeated_access():
    """PyTorch may unpack the same saved tensor twice (``retain_graph=True``)."""
    device = torch.device(get_device_type())
    manager = OffloadManager(PinnedBufferPool(max_cached_bytes=1 << 20))
    original = torch.randn(8, 8, device=device)
    expected = original.detach().cpu().clone()
    swap = SwapTensor(original, "0_0", manager.host_buffer_pool)
    swap.launch_d2h(manager.swap_stream)
    swap.wait_d2h_finished()
    manager.put("0_0", swap)

    first = _unpack_swap_tensor(manager, swap, prefetch=False)
    assert not manager.exist("0_0")
    torch.testing.assert_close(first.cpu(), expected)

    second = _unpack_swap_tensor(manager, swap, prefetch=False)
    torch.testing.assert_close(second.cpu(), expected)
    assert second.data_ptr() == first.data_ptr()


@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE),
    reason="CUDA or NPU is required for async D2H/H2D validation",
)
def test_async_offload_accelerator_gradient_parity_and_peak_memory():
    device_api = get_torch_device()
    device = torch.device(get_device_type())
    device_api.synchronize()
    device_api.empty_cache()

    class Block(GradientCheckpointingLayer):
        def __init__(self):
            super().__init__()
            # Same nesting as production: patched ``__call__`` (async pack) wraps
            # ``GradientCheckpointingLayer.__call__`` (checkpoint). NPU Linear
            # often saves a tensor whose ``data_ptr`` is not the module input, so
            # a checkpoint-less Sequential never packs (allocations == 0).
            self.gradient_checkpointing = True
            self._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
            self.proj = nn.Linear(512, 512)

        def forward(self, hidden_states):
            return torch.nn.functional.gelu(self.proj(hidden_states))

    def make_model():
        model = nn.Sequential(*(Block() for _ in range(4))).to(device)
        model.train()
        return model

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
        FSDPConfig,
        GradientCheckpointingConfig,
        MixedPrecisionConfig,
        OptimizerConfig,
        TorchCompileConfig,
    )
    from veomni.distributed.parallel_state import _init_parallel_state, use_parallel_state
    from veomni.models.model_runtime import VeOmniModelRuntime

    world_size = dist.get_world_size()
    _init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        dp_mode="fsdp2",
        device_type=get_device_type(),
        name="base",
    )

    runtime = object.__new__(VeOmniModelRuntime)
    runtime.model = _AsyncOffloadFSDPModel(device="meta")
    runtime.model_name = "base"
    runtime.args = SimpleNamespace(
        lora_config=None,
        fqn_to_index_mapping=None,
        model_path=None,
        basic_modules=[],
        optimizer=OptimizerConfig(),
        broadcast_model_weights_from_rank0=False,
        extra_parallel_stream_load=False,
        accelerator=SimpleNamespace(
            offload_config=OffloadConfig(
                enable_async_activation=True,
                activation_offload_host_cache_limit_gb=0.01,
            ),
            fsdp_config=FSDPConfig(mixed_precision=MixedPrecisionConfig(enable=False)),
            init_device="meta",
            gradient_checkpointing=GradientCheckpointingConfig(enable=True),
            torch_compile=TorchCompileConfig(enable=False),
        ),
    )
    runtime.train_args = SimpleNamespace(
        checkpoint=SimpleNamespace(load_path=None),
    )

    torch.manual_seed(0)
    with use_parallel_state("base"):
        runtime._build_parallelized_model()

    assert all(layer.gradient_checkpointing for layer in runtime.model.layers)
    manager = runtime.model.layers[0]._veomni_offload_manager
    hidden_states = torch.randn(2, 8, 16, device=get_device_type(), requires_grad=True)
    with use_parallel_state("base"):
        output = runtime.model(hidden_states)
    with use_parallel_state("base"):
        output.float().square().mean().backward()

    assert torch.isfinite(output).all()
    assert manager.host_buffer_pool.allocations > 0
    assert not manager.items
    for parameter in runtime.model.parameters():
        if parameter.grad is not None:
            local_grad = parameter.grad.to_local() if hasattr(parameter.grad, "to_local") else parameter.grad
            assert torch.isfinite(local_grad).all()
    dist.barrier()


@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_2)
@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE),
    reason="CUDA or NPU is required for FSDP2 async-offload integration",
)
def test_async_offload_base_trainer_fsdp2_gc_integration():
    from ..tools.launch_utils import torchrun

    torchrun(_run_async_offload_base_trainer_fsdp2_gc, world_size=_CI_ACCELERATOR_2.nproc)
