# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""MultiOptimizer checkpoint compatibility is checked before any training step."""

import os
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_capabilities
from veomni.models import model_runtime
from veomni.utils.device import get_device_type


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _runtime(multi, save_async, manager="dcp"):
    class StubRuntime(model_runtime.VeOmniModelRuntime):
        def setup(self):
            pass

        def _build_model(self):
            self.model = torch.nn.Linear(2, 2)

        def _freeze_model_module(self):
            pass

        def _build_parallelized_model(self):
            pass

        def _build_optimizer(self):
            if multi == "fail":
                raise RuntimeError("injected optimizer build failure")
            self.optimizer = None if multi is None else SimpleNamespace(_is_multi_optimizer=multi)

        def _build_model_assets(self):
            pass

        def build_checkpoint(self):
            pass

    args = SimpleNamespace(optimizer=SimpleNamespace(type="adamw"))
    train = SimpleNamespace(checkpoint=SimpleNamespace(load_path=None, save_async=save_async, manager=manager))
    return StubRuntime(args, train=train)


@pytest.mark.parametrize("multi,save_async", [(None, True), (False, False), (False, True), (True, False)])
def test_compatible_optimizer_checkpoint_modes(monkeypatch, multi, save_async):
    monkeypatch.setattr(model_runtime, "use_parallel_state", lambda *_: nullcontext())
    optimizer = _runtime(multi, save_async).optimizer
    assert (None if optimizer is None else optimizer._is_multi_optimizer) is multi


def test_async_multi_optimizer_fails_at_runtime_build(monkeypatch):
    monkeypatch.setattr(model_runtime, "use_parallel_state", lambda *_: nullcontext())
    with pytest.raises(RuntimeError, match="synchronous"):
        _runtime(True, True)


def test_other_checkpoint_managers_keep_their_own_async_policy(monkeypatch):
    monkeypatch.setattr(model_runtime, "use_parallel_state", lambda *_: nullcontext())
    assert _runtime(True, True, manager="custom").optimizer._is_multi_optimizer is True


def test_omni_composed_wrap_rejects_async_multi_optimizer_before_checkpoint(monkeypatch):
    from veomni.models.seed_omni.accelerated.omni_module.omni_module_runtime import ModuleRuntime

    runtime = ModuleRuntime.__new__(ModuleRuntime)
    runtime.model_name = "llm"
    runtime.model = torch.nn.Linear(2, 2)
    runtime.train_args = SimpleNamespace(checkpoint=SimpleNamespace(save_async=True))
    monkeypatch.setattr(runtime, "_scoped", nullcontext)
    monkeypatch.setattr(runtime, "_apply_recompute_policy", lambda: None)
    monkeypatch.setattr(
        runtime, "_build_optimizer", lambda: setattr(runtime, "optimizer", SimpleNamespace(_is_multi_optimizer=True))
    )
    reached_checkpoint = []
    monkeypatch.setattr(runtime, "build_checkpoint", lambda: reached_checkpoint.append(True))
    with pytest.raises(RuntimeError, match="synchronous"):
        runtime.build_after_omni_model_wrap()
    assert not reached_checkpoint


@pytest.mark.parametrize("multi", [None, False, True])
def test_omni_module_scope_checks_async_before_checkpoint(monkeypatch, multi):
    from veomni.models.seed_omni.accelerated.omni_module.omni_module_runtime import ModuleRuntime

    monkeypatch.setattr(ModuleRuntime, "setup", lambda self: None)
    monkeypatch.setattr(ModuleRuntime, "_scoped", lambda self: nullcontext())
    monkeypatch.setattr(ModuleRuntime, "_build_model", lambda self: setattr(self, "model", torch.nn.Linear(2, 2)))
    monkeypatch.setattr(ModuleRuntime, "_build_model_assets", lambda self: None)
    monkeypatch.setattr(ModuleRuntime, "_freeze_model_module", lambda self: None)
    monkeypatch.setattr(ModuleRuntime, "_build_parallelized_model", lambda self: None)
    monkeypatch.setattr(
        ModuleRuntime,
        "_build_optimizer",
        lambda self: setattr(self, "optimizer", None if multi is None else SimpleNamespace(_is_multi_optimizer=multi)),
    )
    reached_checkpoint = []
    monkeypatch.setattr(ModuleRuntime, "build_checkpoint", lambda self: reached_checkpoint.append(True))
    accelerator = SimpleNamespace(fsdp_config=SimpleNamespace(fsdp_scope="module", fsdp_mode="fsdp2"))
    train = SimpleNamespace(training_task="sft", checkpoint=SimpleNamespace(save_async=True))
    kwargs = dict(module_config=SimpleNamespace(), global_accelerator=accelerator, train_args=train)
    if multi:
        with pytest.raises(RuntimeError, match="synchronous"):
            ModuleRuntime(SimpleNamespace(accelerator=accelerator), "llm", **kwargs)
        assert not reached_checkpoint
    else:
        ModuleRuntime(SimpleNamespace(accelerator=accelerator), "llm", **kwargs)
        assert reached_checkpoint == [True]


def _preflight_failure_worker(rank, endpoint):
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        model_runtime.use_parallel_state = lambda *_: nullcontext()
        _runtime(True, False)
        with pytest.raises(RuntimeError, match="synchronous|checking MultiOptimizer checkpoint mode"):
            _runtime(True, True)
        with pytest.raises(RuntimeError, match="injected optimizer build failure|building model optimizer"):
            _runtime("fail" if rank == 0 else False, False)
        done = torch.tensor(1)
        dist.all_reduce(done)
        assert done.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_optimizer_checkpoint_failures_reach_all_ranks():
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_preflight_failure_worker, args=(), nprocs=2)


def _accelerator_worker(rank, endpoint):
    from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device

    device = get_device_type()
    get_torch_device().set_device(f"{device}:{rank}")
    dist.init_process_group(
        get_dist_comm_backend(), store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        model_runtime.use_parallel_state = lambda *_: nullcontext()
        _runtime(True, False)
        with pytest.raises(RuntimeError, match="synchronous|checking MultiOptimizer checkpoint mode"):
            _runtime(True, True)
        with pytest.raises(RuntimeError, match="injected optimizer build failure|building model optimizer"):
            _runtime("fail" if rank == 0 else False, False)
        done = torch.ones(1, device=device)
        dist.all_reduce(done)
        assert done.item() == 2
    finally:
        dist.destroy_process_group()


def _require_two_native_devices():
    """Only optional local runs may skip the two-device preflight."""
    capabilities = runner_capabilities() if os.environ.get("VEOMNI_CI_CAPABILITIES") else None
    assigned = capabilities is not None and capabilities.backend != "cpu"
    if not torch.accelerator.is_available() or torch.accelerator.device_count() < 2:
        if assigned:
            pytest.fail("Assigned accelerator runtime must provide two usable devices")
        pytest.skip("requires two accelerators")
    if assigned and get_device_type() != capabilities.backend:
        pytest.fail("Assigned accelerator backend does not match the runtime")


@pytest.mark.parametrize("count", [0, 1])
def test_native_preflight_fails_when_assigned_devices_are_unusable(monkeypatch, count):
    monkeypatch.setenv("VEOMNI_CI_CAPABILITIES", "declared")
    monkeypatch.setitem(globals(), "runner_capabilities", lambda: SimpleNamespace(backend="npu"))
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: count > 0)
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: count)
    with pytest.raises(pytest.fail.Exception, match="two usable devices"):
        _require_two_native_devices()


def test_native_preflight_skips_without_optional_local_devices(monkeypatch):
    monkeypatch.delenv("VEOMNI_CI_CAPABILITIES", raising=False)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    with pytest.raises(pytest.skip.Exception, match="two accelerators"):
        _require_two_native_devices()


def test_native_preflight_accepts_two_assigned_devices(monkeypatch):
    monkeypatch.setenv("VEOMNI_CI_CAPABILITIES", "declared")
    monkeypatch.setitem(globals(), "runner_capabilities", lambda: SimpleNamespace(backend="npu"))
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: 2)
    monkeypatch.setitem(globals(), "get_device_type", lambda: "npu")
    _require_two_native_devices()


def test_native_preflight_rejects_mismatched_backend(monkeypatch):
    monkeypatch.setenv("VEOMNI_CI_CAPABILITIES", "declared")
    monkeypatch.setitem(globals(), "runner_capabilities", lambda: SimpleNamespace(backend="npu"))
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: 2)
    monkeypatch.setitem(globals(), "get_device_type", lambda: "cuda")
    with pytest.raises(pytest.fail.Exception, match="backend does not match"):
        _require_two_native_devices()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="accelerator"))
def test_native_accelerator_optimizer_checkpoint_preflight():
    from tests.tools.launch_utils import spawn_with_rendezvous

    _require_two_native_devices()
    spawn_with_rendezvous(_accelerator_worker, args=(), nprocs=2)
