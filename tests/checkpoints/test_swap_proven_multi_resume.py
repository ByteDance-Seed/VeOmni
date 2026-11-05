# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""CPU round-trip checks for host-offloaded state in proven MultiOptimizer DCP."""

from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.optim.optimizer import MultiOptimizer
from veomni.optim.swap_adamw import SwapAdamW


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _build(*, amsgrad=False, dtype=torch.float32):
    model = torch.nn.Linear(3, 2).to(dtype)
    leaf = SwapAdamW(model.parameters(), lr=0.01, betas=(0.9, 0.95), amsgrad=amsgrad, pin_memory=False)
    return model, leaf, MultiOptimizer(model, {"swap": leaf}, ["swap"])


def _step(model, optimizer):
    for index, parameter in enumerate(model.parameters()):
        if parameter.requires_grad:
            parameter.grad = torch.full_like(parameter, (index + 1) * 0.125)
    optimizer.step()
    optimizer.zero_grad()


@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_swap_proven_resume_matches_next_update(tmp_path, amsgrad, dtype):
    torch.manual_seed(18)
    model, source, optimizer = _build(amsgrad=amsgrad, dtype=dtype)
    _step(model, optimizer)
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target, restored, resumed = _build(amsgrad=amsgrad, dtype=dtype)
    DistributedCheckpointer.load(
        str(tmp_path),
        {"model": target, "optimizer": resumed, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    for left, right in zip(model.parameters(), target.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert source.state[left]["step"] == restored.state[right]["step"]
        for field in source._host_states[left]:
            torch.testing.assert_close(
                source._host_states[left][field], restored._host_states[right][field], rtol=0, atol=0
            )
    _step(model, optimizer)
    _step(target, resumed)
    for left, right in zip(model.parameters(), target.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert source.state[left]["step"] == restored.state[right]["step"]
        for field in source._host_states[left]:
            torch.testing.assert_close(
                source._host_states[left][field], restored._host_states[right][field], rtol=0, atol=0
            )


@pytest.mark.parametrize("history", [False, True])
def test_swap_preserves_frozen_or_absent_parameter_history(tmp_path, history):
    model = torch.nn.Linear(3, 2)
    if not history:
        model.bias.requires_grad_(False)
    source = SwapAdamW(model.parameters(), lr=0.01, pin_memory=False)
    optimizer = MultiOptimizer(model, {"swap": source}, ["swap"])
    if not history:
        model.bias.requires_grad_(True)
        model.weight.grad = torch.full_like(model.weight, 0.125)
        optimizer.step()
        optimizer.zero_grad()
        assert model.bias not in source.state
    else:
        _step(model, optimizer)
        model.bias.requires_grad_(False)

    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target = torch.nn.Linear(3, 2)
    target.bias.requires_grad_(not history)
    restored = SwapAdamW(target.parameters(), lr=0.01, pin_memory=False)
    resumed = MultiOptimizer(target, {"swap": restored}, ["swap"])
    DistributedCheckpointer.load(
        str(tmp_path),
        {"model": target, "optimizer": resumed, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    assert (model.bias in source.state) is (target.bias in restored.state)
    assert target.bias.requires_grad is (not history)
    if history:
        assert source.state[model.bias]["step"] == restored.state[target.bias]["step"]
        for field in source._host_states[model.bias]:
            torch.testing.assert_close(
                source._host_states[model.bias][field], restored._host_states[target.bias][field], rtol=0, atol=0
            )
        model.bias.requires_grad_(True)
        target.bias.requires_grad_(True)
    _step(model, optimizer)
    _step(target, resumed)
    for left, right in zip(model.parameters(), target.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_swap_empty_leaf_roundtrip_has_no_synthetic_state(tmp_path):
    model = torch.nn.Linear(3, 2)
    for param in model.parameters():
        param.requires_grad_(False)
    source = SwapAdamW(model.parameters(), lr=0.01, pin_memory=False)
    optimizer = MultiOptimizer(model, {"swap": source}, ["swap"])
    assert not source.state
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target = torch.nn.Linear(3, 2)
    for param in target.parameters():
        param.requires_grad_(False)
    restored = SwapAdamW(target.parameters(), lr=0.01, pin_memory=False)
    resumed = MultiOptimizer(target, {"swap": restored}, ["swap"])
    DistributedCheckpointer.load(
        str(tmp_path),
        {"model": target, "optimizer": resumed, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    assert not source.state and not restored.state
    for left, right in zip(model.parameters(), target.parameters(), strict=True):
        left.requires_grad_(True)
        right.requires_grad_(True)
    _step(model, optimizer)
    _step(target, resumed)
    for left, right in zip(model.parameters(), target.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_swap_all_frozen_history_recovers_without_synthetic_step(tmp_path):
    model, source, optimizer = _build()
    _step(model, optimizer)
    for param in model.parameters():
        param.requires_grad_(False)
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target = torch.nn.Linear(3, 2)
    for param in target.parameters():
        param.requires_grad_(False)
    restored = SwapAdamW(target.parameters(), lr=0.01, pin_memory=False)
    resumed = MultiOptimizer(target, {"swap": restored}, ["swap"])
    assert not restored.state
    DistributedCheckpointer.load(
        str(tmp_path),
        {"model": target, "optimizer": resumed, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    for left, right in zip(model.parameters(), target.parameters(), strict=True):
        assert not right.requires_grad
        assert source.state[left]["step"] == restored.state[right]["step"] == 1
        for field in source._state_keys:
            torch.testing.assert_close(
                source._host_states[left][field], restored._host_states[right][field], rtol=0, atol=0
            )
        left.requires_grad_(True)
        right.requires_grad_(True)
    _step(model, optimizer)
    _step(target, resumed)
    for left, right in zip(model.parameters(), target.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_swap_restore_hooks_observe_live_state(tmp_path):
    model, source, optimizer = _build()
    _step(model, optimizer)
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target, leaf, resumed = _build()
    live_state = leaf.state
    seen = []

    def before(current, state_dict):
        assert current.state is live_state
        seen.append("before")

    def after(current):
        assert current.state is live_state
        source_param = next(model.parameters())
        target_param = next(target.parameters())
        torch.testing.assert_close(
            current._host_states[target_param]["exp_avg"], source._host_states[source_param]["exp_avg"], rtol=0, atol=0
        )
        seen.append("after")

    leaf.register_load_state_dict_pre_hook(before)
    leaf.register_load_state_dict_post_hook(after)
    DistributedCheckpointer.load(
        str(tmp_path),
        {"model": target, "optimizer": resumed, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    assert seen == ["before", "after"]


def test_swap_checkpoint_state_hooks_observe_live_state(tmp_path):
    model, _, optimizer = _build()
    _step(model, optimizer)
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target, leaf, resumed = _build()
    seen = []

    def before(opt):
        assert all(isinstance(state["step"], int) for state in opt.state.values())
        seen.append("before")

    def after(opt, state_dict):
        assert all(isinstance(state["step"], int) for state in opt.state.values())
        assert all(isinstance(state["step"], int) for state in state_dict["state"].values() if "step" in state)
        seen.append("after")
        return state_dict

    leaf.register_state_dict_pre_hook(before)
    leaf.register_state_dict_post_hook(after)
    DistributedCheckpointer.load(
        str(tmp_path),
        {"model": target, "optimizer": resumed, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    assert seen and seen.count("before") == seen.count("after")


def test_swap_and_fp32_master_leaf_preserve_both_updates(tmp_path):
    from tests.checkpoints.test_optimizer_checkpoint_adapter import ToyAdamW

    def build():
        model = torch.nn.Linear(3, 2)
        swap = SwapAdamW([model.weight], lr=0.01, pin_memory=False)
        master = ToyAdamW([model.bias])
        return model, swap, master, MultiOptimizer(model, {"swap": swap, "master": master}, ["swap", "master"])

    source, swap, master, optimizer = build()
    _step(source, optimizer)
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": source, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target, restored_swap, restored_master, resumed = build()
    DistributedCheckpointer.load(
        str(tmp_path),
        {"model": target, "optimizer": resumed, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    for field in ("master_weight", "exp_avg", "exp_avg_sq"):
        assert restored_master.state[target.bias][field].dtype == torch.float32
        torch.testing.assert_close(
            master.state[source.bias][field], restored_master.state[target.bias][field], rtol=0, atol=0
        )
    for field in swap._state_keys:
        torch.testing.assert_close(
            swap._host_states[source.weight][field], restored_swap._host_states[target.weight][field], rtol=0, atol=0
        )
    _step(source, optimizer)
    _step(target, resumed)
    for left, right in zip(source.parameters(), target.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(
        master.state[source.bias]["master_weight"], restored_master.state[target.bias]["master_weight"], rtol=0, atol=0
    )


def test_swap_rejects_nonfresh_host_state_before_loading(tmp_path):
    model, _, optimizer = _build()
    _step(model, optimizer)
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target, leaf, resumed = _build()
    first = next(target.parameters())
    leaf._host_states[first]["exp_avg"].fill_(0.25)
    before = {name: param.detach().clone() for name, param in target.named_parameters()}
    with pytest.raises(RuntimeError, match="pristine host state"):
        DistributedCheckpointer.load(
            str(tmp_path),
            {"model": target, "optimizer": resumed, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
    for name, param in target.named_parameters():
        torch.testing.assert_close(param, before[name], rtol=0, atol=0)
    assert torch.all(leaf._host_states[first]["exp_avg"] == 0.25)
    assert not getattr(leaf, "_veomni_dcp_restore_failed", False)


def test_swap_failed_apply_keeps_host_buffers_and_blocks_update(tmp_path, monkeypatch):
    model, _, optimizer = _build()
    _step(model, optimizer)
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target, leaf, resumed = _build()
    original = {
        param: {key: value.clone() for key, value in states.items()} for param, states in leaf._host_states.items()
    }

    def fail_before_apply(current, state_dict):
        for param, states in current._veomni_dcp_staged_state.items():
            for field in current._state_keys:
                assert states[field].data_ptr() != current._host_states[param][field].data_ptr()
                states[field].fill_(17)
        raise RuntimeError("injected staged restore failure")

    monkeypatch.setattr(SwapAdamW, "load_state_dict", fail_before_apply)
    with pytest.raises(RuntimeError, match="injected staged restore failure"):
        DistributedCheckpointer.load(
            str(tmp_path),
            {"model": target, "optimizer": resumed, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
    assert "_veomni_dcp_staged_state" not in vars(leaf)
    for param, states in original.items():
        for field, value in states.items():
            torch.testing.assert_close(leaf._host_states[param][field], value, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="failed checkpoint restore"):
        leaf.step()
    with pytest.raises(RuntimeError, match="failed checkpoint restore"):
        resumed.step()


def test_failed_swap_restore_blocks_all_multi_optimizer_leaves(tmp_path, monkeypatch):
    def build():
        model = torch.nn.Linear(3, 2)
        adam = torch.optim.AdamW([model.weight], lr=0.01)
        swap = SwapAdamW([model.bias], lr=0.01, pin_memory=False)
        return model, MultiOptimizer(model, {"adam": adam, "swap": swap}, ["adam", "swap"])

    model, optimizer = build()
    _step(model, optimizer)
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target, resumed = build()

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected later optimizer leaf failure")

    monkeypatch.setattr(SwapAdamW, "load_state_dict", fail)
    with pytest.raises(RuntimeError, match="later optimizer leaf failure"):
        DistributedCheckpointer.load(
            str(tmp_path),
            {"model": target, "optimizer": resumed, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
    for param in target.parameters():
        param.grad = torch.ones_like(param)
    before = target.weight.detach().clone()
    adam_step = resumed.optimizers_dict["adam"].state[target.weight]["step"].clone()
    with pytest.raises(RuntimeError, match="failed checkpoint restore"):
        resumed.step()
    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)
    torch.testing.assert_close(resumed.optimizers_dict["adam"].state[target.weight]["step"], adam_step, rtol=0, atol=0)


def test_swap_rejects_inexact_checkpoint_step_before_saving(tmp_path):
    model, leaf, optimizer = _build()
    first = next(model.parameters())
    leaf.state[first]["step"] = 2**24 + 1
    with pytest.raises(RuntimeError, match="cannot be represented exactly"):
        DistributedCheckpointer.save(
            str(tmp_path),
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
    assert not list(tmp_path.iterdir())


def test_single_swap_state_dict_rejects_inexact_checkpoint_step(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = SwapAdamW(model.parameters(), pin_memory=False)
    param = next(model.parameters())
    optimizer.state[param]["step"] = 2**24
    assert optimizer.state_dict()["state"][0]["step"].item() == 2**24
    optimizer.state[param]["step"] = 2**24 + 1
    with pytest.raises(RuntimeError, match="cannot be represented exactly"):
        optimizer.state_dict()
    with pytest.raises(Exception, match="cannot be represented exactly"):
        DistributedCheckpointer.save(
            str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
        )


def test_single_swap_failed_partial_restore_blocks_update(tmp_path, monkeypatch):
    import veomni.checkpoint.dcp_checkpointer as implementation

    model = torch.nn.Linear(3, 2)
    optimizer = SwapAdamW(model.parameters(), pin_memory=False)
    _step(model, optimizer)
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    target = torch.nn.Linear(3, 2)
    resumed = SwapAdamW(target.parameters(), pin_memory=False)
    original_load = implementation.dcp.load

    def fail_after_partial_write(*args, **kwargs):
        if "optimizer" in kwargs["state_dict"] and not isinstance(kwargs["state_dict"]["optimizer"], dict):
            resumed._host_states[target.weight]["exp_avg"].fill_(0.25)
            raise RuntimeError("injected partial optimizer read failure")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(implementation.dcp, "load", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="partial optimizer read failure"):
        DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": resumed}, parallel_state=parallel)
    with pytest.raises(RuntimeError, match="failed checkpoint restore"):
        resumed.step()
    with pytest.raises(RuntimeError, match="failed checkpoint restore"):
        resumed.state_dict()
    with pytest.raises(BaseException, match="failed checkpoint restore"):
        DistributedCheckpointer.save(
            str(tmp_path / "invalid"), {"model": target, "optimizer": resumed}, parallel_state=parallel
        )


def test_single_swap_preserves_frozen_history_on_resume(tmp_path):
    model = torch.nn.Linear(3, 2)
    source = SwapAdamW(model.parameters(), lr=0.01, pin_memory=False)
    _step(model, source)
    model.bias.requires_grad_(False)
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": source}, parallel_state=parallel)

    target = torch.nn.Linear(3, 2)
    target.bias.requires_grad_(False)
    restored = SwapAdamW(target.parameters(), lr=0.01, pin_memory=False)
    DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)
    assert not target.bias.requires_grad
    assert restored.state[target.bias]["step"] == source.state[model.bias]["step"]
    for field in source._state_keys:
        torch.testing.assert_close(
            source._host_states[model.bias][field], restored._host_states[target.bias][field], rtol=0, atol=0
        )
    model.bias.requires_grad_(True)
    target.bias.requires_grad_(True)
    _step(model, source)
    _step(target, restored)
    torch.testing.assert_close(model.bias, target.bias, rtol=0, atol=0)


def test_single_swap_absent_frozen_history_stays_absent(tmp_path):
    model = torch.nn.Linear(3, 2)
    model.bias.requires_grad_(False)
    optimizer = SwapAdamW(model.parameters(), pin_memory=False)
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    target = torch.nn.Linear(3, 2)
    target.bias.requires_grad_(False)
    resumed = SwapAdamW(target.parameters(), pin_memory=False)
    DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": resumed}, parallel_state=parallel)
    assert not target.bias.requires_grad
    assert target.bias not in resumed.state
    assert target.bias not in resumed._host_states


def test_single_swap_rejects_saved_history_missing_from_runtime_groups(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = SwapAdamW(model.parameters(), pin_memory=False)
    _step(model, optimizer)
    model.bias.requires_grad_(False)
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    target = torch.nn.Linear(3, 2)
    target.bias.requires_grad_(False)
    resumed = SwapAdamW((p for p in target.parameters() if p.requires_grad), pin_memory=False)
    before = {name: value.clone() for name, value in target.state_dict().items()}
    with pytest.raises(RuntimeError, match="checkpoint parameter groups differ"):
        DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": resumed}, parallel_state=parallel)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert target.bias not in resumed.state


def test_single_swap_rejects_omitted_owner_without_history(tmp_path):
    model = torch.nn.Linear(3, 2)
    model.bias.requires_grad_(False)
    optimizer = SwapAdamW(model.parameters(), pin_memory=False)
    _step(model, optimizer)
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    target = torch.nn.Linear(3, 2)
    target.bias.requires_grad_(False)
    resumed = SwapAdamW((p for p in target.parameters() if p.requires_grad), pin_memory=False)
    before = {name: value.clone() for name, value in target.state_dict().items()}
    with pytest.raises(RuntimeError, match="checkpoint parameter groups differ"):
        DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": resumed}, parallel_state=parallel)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_single_swap_rejects_reordered_trainable_groups(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = SwapAdamW([{"params": [model.weight]}, {"params": [model.bias]}], pin_memory=False)
    _step(model, optimizer)
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    target = torch.nn.Linear(3, 2)
    resumed = SwapAdamW([{"params": [target.bias]}, {"params": [target.weight]}], pin_memory=False)
    before = {name: value.clone() for name, value in target.state_dict().items()}
    with pytest.raises(RuntimeError, match="checkpoint parameter groups differ"):
        DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": resumed}, parallel_state=parallel)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_single_swap_accepts_unowned_frozen_parameter_without_history(tmp_path):
    model = torch.nn.Linear(3, 2)
    model.bias.requires_grad_(False)
    optimizer = SwapAdamW((p for p in model.parameters() if p.requires_grad), pin_memory=False)
    _step(model, optimizer)
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    target = torch.nn.Linear(3, 2)
    target.bias.requires_grad_(False)
    resumed = SwapAdamW((p for p in target.parameters() if p.requires_grad), pin_memory=False)
    DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": resumed}, parallel_state=parallel)
    assert target.bias not in resumed.state
    torch.testing.assert_close(target.weight, model.weight, rtol=0, atol=0)


def test_single_swap_frozen_history_preserves_model_load_hook_view(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = SwapAdamW(model.parameters(), pin_memory=False)
    _step(model, optimizer)
    model.bias.requires_grad_(False)
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    target = torch.nn.Linear(3, 2)
    target.bias.requires_grad_(False)
    resumed = SwapAdamW(target.parameters(), pin_memory=False)
    seen = []

    def model_hook(module, _incompatible_keys):
        assert not module.bias.requires_grad
        seen.append(True)

    target.register_load_state_dict_post_hook(model_hook)
    DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": resumed}, parallel_state=parallel)
    assert seen
    assert not target.bias.requires_grad
    assert resumed.state[target.bias]["step"] == optimizer.state[model.bias]["step"]


def test_single_swap_frozen_history_rejects_optimizer_load_hook_before_loading(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = SwapAdamW(model.parameters(), pin_memory=False)
    _step(model, optimizer)
    model.bias.requires_grad_(False)
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    target = torch.nn.Linear(3, 2)
    target.bias.requires_grad_(False)
    resumed = SwapAdamW(target.parameters(), pin_memory=False)
    resumed.register_load_state_dict_pre_hook(lambda *_: None)
    before = target.weight.clone()
    with pytest.raises(RuntimeError, match="does not support optimizer load-state hooks"):
        DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": resumed}, parallel_state=parallel)
    assert not target.bias.requires_grad
    assert target.bias not in resumed.state
    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)


def _mismatched_leaf_worker(rank, checkpoint, endpoint):
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        model = torch.nn.Linear(3, 2)
        if rank == 0:
            leaves = {
                "weight": SwapAdamW([model.weight], pin_memory=False),
                "bias": SwapAdamW([model.bias], pin_memory=False),
            }
        else:
            leaves = {"both": SwapAdamW(model.parameters(), pin_memory=False)}
        optimizer = MultiOptimizer(model, leaves, list(leaves))
        with pytest.raises(RuntimeError, match="leaf order differs"):
            DistributedCheckpointer.save(
                checkpoint,
                {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
                parallel_state=SimpleNamespace(dp_mode="ddp"),
            )
        value = torch.ones(1)
        dist.all_reduce(value)
        assert value.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_mismatched_swap_leaves_fail_before_checkpoint(tmp_path):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_mismatched_leaf_worker, nprocs=2, args=(str(tmp_path / "ckpt"),))


def _single_swap_save_failure_worker(rank, checkpoint, failure, save_async, endpoint):
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        model = torch.nn.Linear(3, 2)
        optimizer = SwapAdamW(model.parameters(), pin_memory=False)
        if rank == 0:
            if failure == "step":
                optimizer.state[model.weight]["step"] = 2**24 + 1
            elif failure == "poison":
                optimizer._veomni_dcp_restore_failed = True
            else:

                def fail(_optimizer):
                    raise RuntimeError("injected optimizer state hook failure")

                optimizer.register_state_dict_pre_hook(fail)
        with pytest.raises(RuntimeError):
            DistributedCheckpointer.save(
                checkpoint,
                {"model": model, "optimizer": optimizer},
                save_async=save_async,
                parallel_state=SimpleNamespace(dp_mode="ddp"),
            )
        value = torch.ones(1)
        dist.all_reduce(value)
        assert value.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
@pytest.mark.parametrize("failure", ["step", "poison", "hook"])
@pytest.mark.parametrize("save_async", [False, True])
def test_single_swap_save_failure_is_coordinated(tmp_path, failure, save_async):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(
        _single_swap_save_failure_worker, nprocs=2, args=(str(tmp_path / "ckpt"), failure, save_async)
    )
