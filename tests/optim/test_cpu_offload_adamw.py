# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""CPU reference checks for selective FP32 optimizer offload."""

import copy
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.optim.cpu_offload_adamw import CPUOffloadAdamW, select_cpu_offload_groups
from veomni.optim.master_fp32_adamw import FullFP32AdamW, MasterFP32AdamW
from veomni.optim.optimizer import MultiOptimizer, build_optimizer


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_offload_pattern_requires_trainable_match():
    model = torch.nn.Linear(4, 3)
    model.weight.requires_grad_(False)
    groups = [{"params": list(model.parameters())}]
    with pytest.raises(ValueError, match="no trainable parameters"):
        select_cpu_offload_groups(model, groups, ["weight"])
    selected = select_cpu_offload_groups(model, groups, ["bias"])
    assert selected[0]["cpu_offload"] and selected[0]["params"] == [model.bias]
    assert not selected[1]["cpu_offload"] and selected[1]["params"] == [model.weight]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("offload", [False, True])
def test_moment_dtype_strings_match_torch_dtype_updates(dtype, offload):
    parameters = [torch.nn.Parameter(torch.tensor([1.0, -0.5], dtype=torch.bfloat16)) for _ in range(2)]
    configured = [str(dtype).removeprefix("torch."), dtype]
    optimizers = [
        CPUOffloadAdamW(
            [{"params": [param], "cpu_offload": offload}],
            resident_moment_dtype=moment_dtype,
            cpu_moment_dtype=moment_dtype,
        )
        for param, moment_dtype in zip(parameters, configured, strict=True)
    ]
    for _ in range(3):
        for param, optimizer in zip(parameters, optimizers, strict=True):
            param.grad = torch.tensor([0.25, -0.125], dtype=param.dtype)
            optimizer.step()
    torch.testing.assert_close(parameters[0], parameters[1], rtol=0, atol=0)
    for key, expected_dtype in (("master_weight", torch.float32), ("exp_avg", dtype), ("exp_avg_sq", dtype)):
        left, right = (optimizer.state[param][key] for param, optimizer in zip(parameters, optimizers, strict=True))
        assert left.dtype == right.dtype == expected_dtype
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_builder_applies_configured_moment_dtypes_to_each_group():
    model = torch.nn.Linear(2, 2)
    optimizer = build_optimizer(
        model,
        optimizer_type="cpu_offload_adamw",
        cpu_offload_param_patterns=["weight"],
        cpu_offload_resident_moment_dtype="bfloat16",
        cpu_offload_cpu_moment_dtype="float32",
    )
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    optimizer.step()
    leaf = optimizer.optimizers_dict["adamw"]
    for param, dtype in ((model.weight, torch.float32), (model.bias, torch.bfloat16)):
        assert leaf.state[param]["master_weight"].dtype == torch.float32
        assert leaf.state[param]["exp_avg"].dtype == leaf.state[param]["exp_avg_sq"].dtype == dtype


def test_yaml_beta_sequence_has_immutable_optimizer_groups(tmp_path):
    from veomni.arguments.arguments_types import OptimizerConfig
    from veomni.arguments.parser import _instantiate_recursive
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

    config = _instantiate_recursive(OptimizerConfig, {"type": "full_fp32_adamw", "betas": [0.9, 0.95]})
    assert config.betas == (0.9, 0.95) and isinstance(config.betas, tuple)
    model = torch.nn.Linear(2, 2)
    optimizer = build_optimizer(model, optimizer_type=config.type, betas=config.betas)
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=SimpleNamespace(dp_mode="ddp"),
    )


@pytest.mark.parametrize("offload", [False, True])
def test_cpu_update_and_resume_match_fp32_reference(offload):
    torch.manual_seed(3)
    param = torch.nn.Parameter(torch.randn(16, dtype=torch.bfloat16))
    reference = torch.nn.Parameter(param.float().detach())
    optimizer = CPUOffloadAdamW([{"params": [param], "cpu_offload": offload}], lr=0.001)
    reference_optimizer = torch.optim.AdamW([reference], lr=0.001, betas=(0.9, 0.95), weight_decay=0)
    for _ in range(3):
        param.grad = torch.randn_like(param)
        reference.grad = param.grad.float()
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(optimizer.state[param]["master_weight"], reference, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(param, reference.bfloat16(), rtol=0, atol=0)
    clone = torch.nn.Parameter(param.detach().clone())
    resumed = CPUOffloadAdamW([{"params": [clone], "cpu_offload": offload}], lr=0.001)
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    clone.grad = param.grad.clone()
    optimizer.step()
    resumed.step()
    torch.testing.assert_close(clone, param, rtol=0, atol=0)
    for name in ("master_weight", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(resumed.state[clone][name], optimizer.state[param][name], rtol=0, atol=0)


@pytest.mark.parametrize("optimizer_type", ["master_fp32_adamw", "full_fp32_adamw", "cpu_offload_adamw"])
def test_builder_uses_sparse_resume_wrapper_for_exact_precision(optimizer_type):
    model = torch.nn.Linear(2, 2)
    optimizer = build_optimizer(model, optimizer_type=optimizer_type, cpu_offload_param_patterns=[".*"])
    assert isinstance(optimizer, MultiOptimizer)


def test_unwrapped_exact_optimizer_dcp_is_rejected_before_write(tmp_path):
    from types import SimpleNamespace

    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

    model = torch.nn.Linear(2, 2)
    optimizer = CPUOffloadAdamW([{"params": list(model.parameters()), "cpu_offload": True}])
    with pytest.raises(RuntimeError, match="require MultiOptimizer"):
        DistributedCheckpointer.save(
            str(tmp_path),
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
    assert not list(tmp_path.iterdir())


def test_frozen_optimizer_ownership_is_rebuilt_for_non_ep_checkpoint(tmp_path):
    from types import SimpleNamespace

    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.checkpoint.rank_local_optimizer import recover_frozen_optimizer_ownership

    parallel = SimpleNamespace(dp_mode="ddp")
    source = torch.nn.Linear(2, 2)
    optimizer = build_optimizer(source, optimizer_type="full_fp32_adamw")
    source.weight.grad = torch.ones_like(source.weight)
    optimizer.step()
    optimizer.zero_grad()
    source.weight.requires_grad_(False)
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": source, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target = torch.nn.Linear(2, 2)
    target.weight.requires_grad_(False)
    with recover_frozen_optimizer_ownership(target, str(tmp_path)):
        resumed = build_optimizer(target, optimizer_type="full_fp32_adamw")
    assert not target.weight.requires_grad
    DistributedCheckpointer.load(
        str(tmp_path),
        {"model": target, "optimizer": resumed, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    source.weight.requires_grad_(True)
    target.weight.requires_grad_(True)
    source.weight.grad = torch.ones_like(source.weight)
    target.weight.grad = torch.ones_like(target.weight)
    optimizer.step()
    resumed.step()
    torch.testing.assert_close(target.weight, source.weight, rtol=0, atol=0)


def test_extra_parallel_builder_preserves_explicit_group_weight_decay(monkeypatch):
    from types import SimpleNamespace

    import veomni.optim.optimizer as optimizer_module

    parallel = SimpleNamespace(dp_mode="fsdp2", any_extra_parallel_enabled=True, extra_parallel_names=("ep",))
    monkeypatch.setattr(optimizer_module, "get_parallel_state", lambda: parallel)
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(1.0)
    groups = [
        {"params": [model.weight], "weight_decay": 0.2},
        {"params": [model.bias], "weight_decay": 0.0},
    ]
    optimizer = build_optimizer(
        model, lr=0.1, optimizer_type="cpu_offload_adamw", param_groups=groups, cpu_offload_param_patterns=[".*"]
    )
    model.weight.grad = torch.zeros_like(model.weight)
    model.bias.grad = torch.zeros_like(model.bias)
    optimizer.step()
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 0.98))
    torch.testing.assert_close(model.bias, torch.ones_like(model.bias))


def test_extra_parallel_custom_group_keeps_legacy_decay_boundaries(monkeypatch):
    from types import SimpleNamespace

    import veomni.optim.optimizer as optimizer_module

    parallel = SimpleNamespace(dp_mode="fsdp2", any_extra_parallel_enabled=True, extra_parallel_names=("ep",))
    monkeypatch.setattr(optimizer_module, "get_parallel_state", lambda: parallel)
    model = torch.nn.Linear(2, 2)
    optimizer = build_optimizer(
        model,
        optimizer_type="cpu_offload_adamw",
        weight_decay=0.01,
        no_decay_params=["bias"],
        param_groups=[{"params": list(model.parameters())}],
        cpu_offload_param_patterns=[".*"],
    )
    leaf = optimizer.optimizers_dict["non_extra_parallel"]
    assert [group["weight_decay"] for group in leaf.param_groups] == [0.01, 0.0]


def test_staging_wait_uses_recorded_device_event(monkeypatch):
    calls = []

    class Event:
        def record(self, stream):
            calls.append(("record", stream))

        def synchronize(self):
            calls.append(("wait",))

    fake = SimpleNamespace(Event=Event, current_stream=lambda device: ("stream", device))
    monkeypatch.setattr(torch, "npu", fake, raising=False)
    device = SimpleNamespace(type="npu")
    CPUOffloadAdamW._wait_staging_copy(device)
    assert calls == [("record", ("stream", device)), ("wait",)]


def test_staging_buffer_growth_waits_before_replacement(monkeypatch):
    model = torch.nn.Linear(4, 4)
    optimizer = CPUOffloadAdamW([{"params": [model.weight], "cpu_offload": True}], lr=0.01)
    previous = optimizer._buffer(model.weight, 2)
    waits = []
    monkeypatch.setattr(optimizer, "_wait_staging_copy", lambda device: waits.append(device))
    replacement = optimizer._buffer(model.weight, 4)
    assert replacement.numel() == 4
    assert replacement.data_ptr() != previous.data_ptr()
    assert waits == [model.weight.device]


def test_multi_optimizer_forwards_zero_grad_and_runs_closure_once():
    model = torch.nn.Linear(2, 2)
    weight = torch.optim.AdamW([model.weight], lr=0.01)
    bias = torch.optim.AdamW([model.bias], lr=0.01)
    optimizer = MultiOptimizer(model, {"weight": weight, "bias": bias}, ["weight", "bias"])
    calls = []

    def closure():
        calls.append(1)
        loss = model(torch.ones(1, 2)).sum()
        loss.backward()
        return loss

    result = optimizer.step(closure)
    assert torch.is_tensor(result)
    assert calls == [1]
    optimizer.zero_grad(set_to_none=False)
    assert model.weight.grad is not None and torch.count_nonzero(model.weight.grad) == 0
    assert model.bias.grad is not None and torch.count_nonzero(model.bias.grad) == 0


@pytest.mark.parametrize("optimizer_type", [MasterFP32AdamW, FullFP32AdamW, CPUOffloadAdamW])
@pytest.mark.parametrize(
    "invalid_step", [torch.tensor(256, dtype=torch.bfloat16), torch.tensor(float("nan")), torch.tensor(1.5)]
)
def test_leaf_load_rejects_invalid_step_without_casting(optimizer_type, invalid_step):
    source_param = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    source = optimizer_type([source_param])
    source_param.grad = torch.ones_like(source_param)
    source.step()
    saved = copy.deepcopy(source.state_dict())
    saved["state"][0]["step"] = invalid_step
    target_param = torch.nn.Parameter(torch.ones_like(source_param))
    target = optimizer_type([target_param])
    with pytest.raises(RuntimeError, match="step must be"):
        target.load_state_dict(saved)
    assert not target.state


@pytest.mark.parametrize("optimizer_type", [MasterFP32AdamW, FullFP32AdamW, CPUOffloadAdamW])
def test_leaf_step_stays_fp32_with_bf16_default_dtype(optimizer_type):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        param = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
        optimizer = optimizer_type([param], lr=0.001)
        param.grad = torch.ones_like(param)
        for _ in range(257):
            optimizer.step()
        assert optimizer.state[param]["step"].dtype == torch.float32
        assert optimizer.state[param]["step"].item() == 257
    finally:
        torch.set_default_dtype(previous)
