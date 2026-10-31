"""CPU/Gloo coverage for pre-read validation of offload optimizer groups."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.dcp_checkpointer import (
    DistributedCheckpointer,
    ModelState,
    OptimizerState,
    _coordinated_optimizer_flag,
    _coordinated_optimizer_policy,
    _ModelStrictLoadPlanner,
    _preflight_cpu_offload_groups,
)
from veomni.checkpoint.layout import optimizer_dir, weights_dir
from veomni.optim.cpu_offload_adamw import CPUOffloadAdamW


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _model_and_optimizer(reverse=False, split=False):
    model = torch.nn.Linear(4, 3)
    params = list(model.parameters())
    if reverse:
        params.reverse()
    groups = [{"params": [params[0]], "cpu_offload": True}, {"params": [params[1]]}] if split else params
    optimizer = CPUOffloadAdamW(groups, lr=0.01)
    return model, optimizer


def _step(model, optimizer, gradient):
    for param in model.parameters():
        param.grad = torch.full_like(param, gradient)
    optimizer.step()
    optimizer.zero_grad()


def _save(path, model, optimizer, fused):
    parallel = SimpleNamespace(dp_mode="ddp")
    model_state = ModelState(model, parallel_state=parallel)
    optim_state = OptimizerState(model, optimizer, parallel_state=parallel)
    if fused:
        dcp.save({"model": model_state, "optimizer": optim_state}, checkpoint_id=path)
    else:
        dcp.save({"model": model_state}, checkpoint_id=weights_dir(str(path)))
        dcp.save({"optimizer": optim_state}, checkpoint_id=optimizer_dir(str(path)))


def _clone_state(model, optimizer):
    return (
        [param.detach().clone() for param in model.parameters()],
        [{key: value.clone() for key, value in optimizer.state[param].items()} for param in model.parameters()],
    )


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("mismatch", ["order", "count"])
def test_preflight_rejects_before_changing_model_or_optimizer(tmp_path, fused, mismatch):
    torch.manual_seed(1)
    source_model, source = _model_and_optimizer()
    _step(source_model, source, 1.0)
    _save(tmp_path, source_model, source, fused)
    torch.manual_seed(2)
    target_model, target = _model_and_optimizer(reverse=mismatch == "order", split=mismatch == "count")
    _step(target_model, target, 0.25)
    before_weights, before_state = _clone_state(target_model, target)
    with pytest.raises(RuntimeError, match="parameter group"):
        DistributedCheckpointer.load(
            str(tmp_path), {"model": target_model, "optimizer": target}, parallel_state=SimpleNamespace(dp_mode="ddp")
        )
    after_weights, after_state = _clone_state(target_model, target)
    for before, after in zip(before_weights, after_weights, strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    for before, after in zip(before_state, after_state, strict=True):
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


@pytest.mark.parametrize("fused", [False, True])
def test_preflight_matching_groups_restore_and_step(tmp_path, fused):
    source_model, source = _model_and_optimizer()
    _step(source_model, source, 1.0)
    _save(tmp_path, source_model, source, fused)
    target_model, target = _model_and_optimizer()
    DistributedCheckpointer.load(
        str(tmp_path), {"model": target_model, "optimizer": target}, parallel_state=SimpleNamespace(dp_mode="ddp")
    )
    _step(source_model, source, 0.5)
    _step(target_model, target, 0.5)
    expected_weights, expected_state = _clone_state(source_model, source)
    actual_weights, actual_state = _clone_state(target_model, target)
    for expected, actual in zip(expected_weights, actual_weights, strict=True):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    for expected, actual in zip(expected_state, actual_state, strict=True):
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)


def test_preflight_checks_checkpoint_save_identity(tmp_path):
    model, optimizer = _model_and_optimizer()
    _step(model, optimizer, 1.0)
    _save(tmp_path, model, optimizer, fused=False)
    target_model, target = _model_and_optimizer()
    with pytest.raises((CheckpointException, RuntimeError), match="checkpoint changed after group preflight"):
        dcp.load(
            {
                "optimizer": OptimizerState(
                    target_model, target, parallel_state=SimpleNamespace(dp_mode="ddp"), load=True
                )
            },
            checkpoint_id=optimizer_dir(str(tmp_path)),
            planner=_ModelStrictLoadPlanner(
                strict_model=False, optimizer=target, model=target_model, expected_optimizer_save_id="different-save"
            ),
        )


def test_preflight_does_not_initialize_fresh_optimizer_state(tmp_path):
    model, optimizer = _model_and_optimizer()
    _step(model, optimizer, 1.0)
    _save(tmp_path, model, optimizer, fused=False)
    target_model, target = _model_and_optimizer()
    assert not target.state
    _preflight_cpu_offload_groups(
        target_model, target, DistributedCheckpointer._create_storage_reader, optimizer_dir(str(tmp_path))
    )
    assert not target.state


def test_policy_coordination_selects_accelerator_for_hccl(monkeypatch):
    import veomni.checkpoint.dcp_checkpointer as checkpoint

    model, optimizer = _model_and_optimizer()
    monkeypatch.setattr(checkpoint.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(checkpoint.dist, "get_world_size", lambda _group=None: 2)
    monkeypatch.setattr(checkpoint.dist, "get_backend", lambda _group=None: "hccl")
    monkeypatch.setattr(checkpoint.dist, "all_reduce", lambda _tensor, **_kwargs: None)
    monkeypatch.setattr(checkpoint, "get_device_type", lambda: "cpu")
    assert _coordinated_optimizer_flag(True, "test") is True
    assert _coordinated_optimizer_policy(optimizer) == (True, True, True)
    assert model.weight.requires_grad


def test_preflight_rejects_custom_optimizer_state_dict_before_read(tmp_path):
    class CustomOptimizer(CPUOffloadAdamW):
        def state_dict(self):
            raise AssertionError("custom state_dict must not run during preflight")

    model, optimizer = _model_and_optimizer()
    _step(model, optimizer, 1.0)
    _save(tmp_path, model, optimizer, fused=False)
    target_model = torch.nn.Linear(4, 3)
    target = CustomOptimizer(target_model.parameters())
    with pytest.raises(RuntimeError, match="standard optimizer state_dict"):
        DistributedCheckpointer.load(
            str(tmp_path), {"model": target_model, "optimizer": target}, parallel_state=SimpleNamespace(dp_mode="ddp")
        )


def test_preflight_allows_standard_scheduler_step_wrapper(tmp_path):
    model, optimizer = _model_and_optimizer()
    _step(model, optimizer, 1.0)
    _save(tmp_path, model, optimizer, fused=False)
    target_model, target = _model_and_optimizer()
    torch.optim.lr_scheduler.LambdaLR(target, lambda _: 1.0)
    DistributedCheckpointer.load(
        str(tmp_path), {"model": target_model, "optimizer": target}, parallel_state=SimpleNamespace(dp_mode="ddp")
    )
    _step(model, optimizer, 0.5)
    _step(target_model, target, 0.5)
    for original, restored in zip(model.parameters(), target_model.parameters(), strict=True):
        torch.testing.assert_close(original, restored, rtol=0, atol=0)


def test_preflight_rejects_a_custom_wrapper_copying_scheduler_marker(tmp_path):
    from functools import wraps

    model, optimizer = _model_and_optimizer(split=True)
    optimizer.param_groups[1]["lr"] = 0.1
    _step(model, optimizer, 1.0)
    _save(tmp_path, model, optimizer, fused=False)
    target_model, target = _model_and_optimizer(split=True)
    torch.optim.lr_scheduler.LambdaLR(target, lambda _: 1.0)
    scheduler_step = target.step
    called = []

    @wraps(scheduler_step)
    def custom_step(*args, **kwargs):
        called.append(True)
        target.param_groups[0]["params"], target.param_groups[1]["params"] = (
            target.param_groups[1]["params"],
            target.param_groups[0]["params"],
        )
        return scheduler_step(*args, **kwargs)

    target.step = custom_step
    with pytest.raises(RuntimeError, match="custom optimizer step implementation"):
        DistributedCheckpointer.load(
            str(tmp_path), {"model": target_model, "optimizer": target}, parallel_state=SimpleNamespace(dp_mode="ddp")
        )
    assert not called
    target.step = scheduler_step
    DistributedCheckpointer.load(
        str(tmp_path), {"model": target_model, "optimizer": target}, parallel_state=SimpleNamespace(dp_mode="ddp")
    )
    _step(model, optimizer, 0.5)
    _step(target_model, target, 0.5)
    for original, restored in zip(model.parameters(), target_model.parameters(), strict=True):
        torch.testing.assert_close(original, restored, rtol=0, atol=0)


@pytest.mark.parametrize("hook_type", ["step_pre", "step_post", "global_pre", "global_post"])
def test_preflight_rejects_fresh_optimizer_step_hooks_before_synthetic_step(tmp_path, hook_type):
    from torch.optim import optimizer as torch_optimizer

    source_model, source = _model_and_optimizer(split=True)
    source.param_groups[1]["lr"] = 0.1
    _step(source_model, source, 1.0)
    _save(tmp_path, source_model, source, fused=False)
    target_model, target = _model_and_optimizer(split=True)
    called = []

    def hook(*_args, **_kwargs):
        called.append(True)
        target.param_groups[0]["params"], target.param_groups[1]["params"] = (
            target.param_groups[1]["params"],
            target.param_groups[0]["params"],
        )

    register = {
        "step_pre": target.register_step_pre_hook,
        "step_post": target.register_step_post_hook,
        "global_pre": torch_optimizer.register_optimizer_step_pre_hook,
        "global_post": torch_optimizer.register_optimizer_step_post_hook,
    }[hook_type]
    handle = register(hook)
    try:
        with pytest.raises(RuntimeError, match="step hooks"):
            DistributedCheckpointer.load(
                str(tmp_path),
                {"model": target_model, "optimizer": target},
                parallel_state=SimpleNamespace(dp_mode="ddp"),
            )
        assert not called
        assert not target.state
    finally:
        handle.remove()
    DistributedCheckpointer.load(
        str(tmp_path), {"model": target_model, "optimizer": target}, parallel_state=SimpleNamespace(dp_mode="ddp")
    )
    _step(source_model, source, 0.5)
    _step(target_model, target, 0.5)
    for original, restored in zip(source_model.parameters(), target_model.parameters(), strict=True):
        torch.testing.assert_close(original, restored, rtol=0, atol=0)


def test_checkpoint_rejects_frozen_parameter_with_existing_optimizer_history(tmp_path):
    model, optimizer = _model_and_optimizer()
    _step(model, optimizer, 1.0)
    saved = _clone_state(model, optimizer)
    model.weight.requires_grad_(False)
    with pytest.raises(RuntimeError, match="frozen precision-sensitive optimizer state"):
        DistributedCheckpointer.save(
            str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
        )
    assert all(
        torch.equal(left, right) for left, right in zip(saved[0], _clone_state(model, optimizer)[0], strict=True)
    )
    model.weight.requires_grad_(True)
    _save(tmp_path / "unfrozen", model, optimizer, fused=False)
    resumed_model, resumed = _model_and_optimizer()
    DistributedCheckpointer.load(
        str(tmp_path / "unfrozen"),
        {"model": resumed_model, "optimizer": resumed},
        parallel_state=SimpleNamespace(dp_mode="ddp"),
    )
    _step(model, optimizer, 0.5)
    _step(resumed_model, resumed, 0.5)
    for source_param, target_param in zip(model.parameters(), resumed_model.parameters(), strict=True):
        torch.testing.assert_close(source_param, target_param, rtol=0, atol=0)


def test_frozen_history_rejection_keeps_previous_checkpoint_markers(tmp_path):
    model, optimizer = _model_and_optimizer()
    _step(model, optimizer, 1.0)
    parallel = SimpleNamespace(dp_mode="ddp")
    state = {"model": model, "optimizer": optimizer}
    DistributedCheckpointer.save(str(tmp_path), state, parallel_state=parallel)
    model_marker = (tmp_path / "model" / "ckpt" / ".metadata").read_bytes()
    optimizer_marker = (tmp_path / "model" / "optimizer" / ".metadata").read_bytes()
    model.weight.requires_grad_(False)
    with pytest.raises(RuntimeError, match="frozen precision-sensitive optimizer state"):
        DistributedCheckpointer.save(str(tmp_path), state, parallel_state=parallel)
    assert (tmp_path / "model" / "ckpt" / ".metadata").read_bytes() == model_marker
    assert (tmp_path / "model" / "optimizer" / ".metadata").read_bytes() == optimizer_marker


def test_checkpoint_rejects_frozen_restore_before_loading_optimizer_history(tmp_path):
    model, optimizer = _model_and_optimizer()
    _step(model, optimizer, 1.0)
    _save(tmp_path, model, optimizer, fused=False)
    target_model, target = _model_and_optimizer()
    _step(target_model, target, 0.25)
    target_model.weight.requires_grad_(False)
    before_weights, before_state = _clone_state(target_model, target)
    with pytest.raises(RuntimeError, match="restore frozen precision-sensitive optimizer parameter"):
        DistributedCheckpointer.load(
            str(tmp_path), {"model": target_model, "optimizer": target}, parallel_state=SimpleNamespace(dp_mode="ddp")
        )
    after_weights, after_state = _clone_state(target_model, target)
    for before, after in zip(before_weights, after_weights, strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    for before, after in zip(before_state, after_state, strict=True):
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    target_model.weight.requires_grad_(True)
    DistributedCheckpointer.load(
        str(tmp_path), {"model": target_model, "optimizer": target}, parallel_state=SimpleNamespace(dp_mode="ddp")
    )
    _step(model, optimizer, 0.5)
    _step(target_model, target, 0.5)
    for original, restored in zip(model.parameters(), target_model.parameters(), strict=True):
        torch.testing.assert_close(original, restored, rtol=0, atol=0)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("change", ["freeze", "reorder"])
def test_model_load_hook_cannot_silently_change_optimizer_resume(tmp_path, fused, change):
    source_model, source = _model_and_optimizer()
    _step(source_model, source, 1.0)
    _save(tmp_path, source_model, source, fused)
    target_model, target = _model_and_optimizer()
    _step(target_model, target, 0.25)
    _, before_state = _clone_state(target_model, target)

    def change_target(model, _incompatible):
        if change == "freeze":
            model.weight.requires_grad_(False)
        else:
            target.param_groups[0]["params"].reverse()

    handle = target_model.register_load_state_dict_post_hook(change_target)
    try:
        with pytest.raises(RuntimeError, match="frozen precision-sensitive|parameter groups"):
            DistributedCheckpointer.load(
                str(tmp_path),
                {"model": target_model, "optimizer": target},
                parallel_state=SimpleNamespace(dp_mode="ddp"),
            )
    finally:
        handle.remove()
    _, after_state = _clone_state(target_model, target)
    for before, after in zip(before_state, after_state, strict=True):
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    target_model.weight.requires_grad_(True)
    target.param_groups[0]["params"] = list(target_model.parameters())
    DistributedCheckpointer.load(
        str(tmp_path), {"model": target_model, "optimizer": target}, parallel_state=SimpleNamespace(dp_mode="ddp")
    )
    _step(source_model, source, 0.5)
    _step(target_model, target, 0.5)
    for original, restored in zip(source_model.parameters(), target_model.parameters(), strict=True):
        torch.testing.assert_close(original, restored, rtol=0, atol=0)


@pytest.mark.parametrize("fused", [False, True])
def test_model_load_hook_cannot_install_optimizer_step_hook_during_resume(tmp_path, fused):
    source_model, source = _model_and_optimizer(split=True)
    source.param_groups[1]["lr"] = 0.1
    _step(source_model, source, 1.0)
    _save(tmp_path, source_model, source, fused)
    target_model, target = _model_and_optimizer(split=True)
    called, handles = [], []

    def swap_groups(*_args):
        called.append(True)
        target.param_groups[0]["params"], target.param_groups[1]["params"] = (
            target.param_groups[1]["params"],
            target.param_groups[0]["params"],
        )

    def install_hook(_model, _incompatible):
        handles.append(target.register_step_pre_hook(swap_groups))

    model_hook = target_model.register_load_state_dict_post_hook(install_hook)
    try:
        with pytest.raises(RuntimeError, match="step hooks"):
            DistributedCheckpointer.load(
                str(tmp_path),
                {"model": target_model, "optimizer": target},
                parallel_state=SimpleNamespace(dp_mode="ddp"),
            )
        assert not called
        assert not target.state
    finally:
        model_hook.remove()
        for handle in handles:
            handle.remove()


def test_model_hook_replacing_optimizer_clears_original_resume_marker(tmp_path):
    from veomni.optim.optimizer import MultiOptimizer

    source_model, source_leaf = _model_and_optimizer()
    source = MultiOptimizer(source_model, {"leaf": source_leaf}, ["leaf"])
    _save(tmp_path, source_model, source, fused=True)
    model, leaf = _model_and_optimizer()
    optimizer = MultiOptimizer(model, {"leaf": leaf}, ["leaf"])
    replacement_model, replacement_leaf = _model_and_optimizer()
    replacement = MultiOptimizer(replacement_model, {"leaf": replacement_leaf}, ["leaf"])
    state = {"model": model, "optimizer": optimizer}

    def replace_optimizer(_model, _incompatible):
        state["optimizer"] = replacement

    model.register_load_state_dict_post_hook(replace_optimizer)
    with pytest.raises(RuntimeError, match="Optimizer object changed"):
        DistributedCheckpointer.load(str(tmp_path), state, parallel_state=SimpleNamespace(dp_mode="ddp"))
    assert not getattr(optimizer, "_veomni_resume_zero_optimizer_state", False)


@pytest.mark.parametrize("fused", [False, True])
def test_precision_mismatch_in_other_exact_optimizer_is_rejected(tmp_path, fused):
    from veomni.optim.master_fp32_adamw import FullFP32AdamW, MasterFP32AdamW

    source_model = torch.nn.Linear(4, 3, dtype=torch.bfloat16)
    source = FullFP32AdamW(source_model.parameters())
    _step(source_model, source, 1.0)
    _save(tmp_path, source_model, source, fused)
    model = torch.nn.Linear(4, 3, dtype=torch.bfloat16)
    optimizer = MasterFP32AdamW(model.parameters())
    with pytest.raises((RuntimeError, CheckpointException), match="dtype|precision"):
        DistributedCheckpointer.load(
            str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
        )


def test_checkpoint_rejects_frozen_multioptimizer_leaf_before_read(tmp_path):
    from veomni.optim.optimizer import MultiOptimizer

    model, leaf = _model_and_optimizer()
    optimizer = MultiOptimizer(model, {"leaf": leaf}, ["leaf"])
    model.weight.requires_grad_(False)
    with pytest.raises(RuntimeError, match="restore frozen precision-sensitive optimizer parameter"):
        DistributedCheckpointer.load(
            str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
        )


def _rank_mismatch_worker(rank, root):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2)
    try:
        torch.manual_seed(5)
        source_model, source = _model_and_optimizer()
        source_model = DistributedDataParallel(source_model)
        _step(source_model, source, 1.0)
        checkpoint = f"{root}/checkpoint"
        _save(checkpoint, source_model, source, fused=False)
        torch.manual_seed(11)
        target_model, target = _model_and_optimizer(reverse=rank == 1)
        target_model = DistributedDataParallel(target_model)
        _step(target_model, target, 0.25)
        before_weights, before_state = _clone_state(target_model, target)
        with pytest.raises(RuntimeError, match="parameter group|another rank"):
            DistributedCheckpointer.load(
                checkpoint,
                {"model": target_model, "optimizer": target},
                process_group=dist.group.WORLD,
                parallel_state=SimpleNamespace(dp_mode="ddp"),
            )
        after_weights, after_state = _clone_state(target_model, target)
        for before, after in zip(before_weights, after_weights, strict=True):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        for before, after in zip(before_state, after_state, strict=True):
            for key in before:
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_preflight_coordinates_one_rank_mismatch(tmp_path):
    torch.multiprocessing.spawn(_rank_mismatch_worker, args=(str(tmp_path),), nprocs=2, join=True)


def _rank_frozen_save_worker(rank, root):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2)
    try:
        model, optimizer = _model_and_optimizer()
        _step(model, optimizer, 1.0)
        if rank == 1:
            model.weight.requires_grad_(False)
        with pytest.raises((CheckpointException, RuntimeError), match="frozen precision-sensitive|another rank"):
            DistributedCheckpointer.save(
                f"{root}/checkpoint",
                {"model": model, "optimizer": optimizer},
                parallel_state=SimpleNamespace(dp_mode="ddp"),
            )
    finally:
        dist.destroy_process_group()


def test_frozen_state_save_rejection_is_coordinated_across_ranks(tmp_path):
    torch.multiprocessing.spawn(_rank_frozen_save_worker, args=(str(tmp_path),), nprocs=2, join=True)


def _rank_frozen_restore_worker(rank, root):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2)
    try:
        torch.manual_seed(1)
        model, optimizer = _model_and_optimizer()
        _step(model, optimizer, 1.0)
        checkpoint = f"{root}/checkpoint"
        _save(checkpoint, model, optimizer, fused=False)
        torch.manual_seed(2)
        target_model, target = _model_and_optimizer()
        _step(target_model, target, 0.25)
        before_weights, before_state = _clone_state(target_model, target)
        if rank == 1:
            target_model.weight.requires_grad_(False)
        with pytest.raises(RuntimeError, match="frozen precision-sensitive|another rank"):
            DistributedCheckpointer.load(
                checkpoint,
                {"model": target_model, "optimizer": target},
                process_group=dist.group.WORLD,
                parallel_state=SimpleNamespace(dp_mode="ddp"),
            )
        after_weights, after_state = _clone_state(target_model, target)
        for before, after in zip(before_weights, after_weights, strict=True):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        for before, after in zip(before_state, after_state, strict=True):
            for key in before:
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_frozen_restore_rejection_is_coordinated_across_ranks(tmp_path):
    torch.multiprocessing.spawn(_rank_frozen_restore_worker, args=(str(tmp_path),), nprocs=2, join=True)


def _asymmetric_optimizer_worker(rank, root, operation, mismatch):
    import torch.distributed as dist

    from veomni.optim.master_fp32_adamw import FullFP32AdamW

    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2)
    try:
        model, optimizer = _model_and_optimizer()
        if rank == 1:
            if mismatch == "presence":
                optimizer = None
            elif mismatch == "precision":
                optimizer = torch.optim.AdamW(model.parameters())
            else:
                optimizer = FullFP32AdamW(model.parameters())
        checkpoint = f"{root}/checkpoint"
        state = {"model": model, "optimizer": optimizer}
        with pytest.raises(RuntimeError, match="DCP ranks disagree"):
            if operation == "save":
                DistributedCheckpointer.save(checkpoint, state, parallel_state=SimpleNamespace(dp_mode="ddp"))
            else:
                DistributedCheckpointer.load(checkpoint, state, parallel_state=SimpleNamespace(dp_mode="ddp"))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("operation", ["save", "load"])
@pytest.mark.parametrize("mismatch", ["presence", "precision", "offload"])
def test_rank_asymmetric_optimizer_configuration_fails_together(tmp_path, operation, mismatch):
    torch.multiprocessing.spawn(
        _asymmetric_optimizer_worker, args=(str(tmp_path), operation, mismatch), nprocs=2, join=True
    )


def _reader_failure_worker(rank, root):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2)
    try:
        model, optimizer = _model_and_optimizer()
        _step(model, optimizer, 1.0)
        checkpoint = f"{root}/checkpoint"
        _save(checkpoint, model, optimizer, fused=False)
        count = 0

        def reader_factory(path):
            nonlocal count
            count += 1
            if rank == 1 and count == 2:
                raise OSError("rank-local reader construction failure")
            return dcp.FileSystemReader(path)

        with pytest.raises((OSError, RuntimeError), match="reader construction failure|another rank"):
            _preflight_cpu_offload_groups(model, optimizer, reader_factory, optimizer_dir(checkpoint))
    finally:
        dist.destroy_process_group()


def test_rank_local_reader_construction_failure_is_coordinated(tmp_path):
    torch.multiprocessing.spawn(_reader_failure_worker, args=(str(tmp_path),), nprocs=2, join=True)


def _rank_changed_checkpoint_identity_worker(rank, root):
    from datetime import timedelta
    from unittest.mock import patch

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        model, optimizer = _model_and_optimizer()
        _step(model, optimizer, 1.0)
        checkpoint = f"{root}/checkpoint"
        _save(checkpoint, model, optimizer, fused=False)
        target_model, target = _model_and_optimizer()
        calls = 0

        def preflight(*args, **kwargs):
            nonlocal calls
            calls += 1
            save_id = _preflight_cpu_offload_groups(*args, **kwargs)
            return f"{save_id}-changed" if rank == 1 and calls == 2 else save_id

        with patch("veomni.checkpoint.dcp_checkpointer._preflight_cpu_offload_groups", side_effect=preflight):
            with pytest.raises(RuntimeError, match="checkpoint changed|another rank"):
                DistributedCheckpointer.load(
                    checkpoint,
                    {"model": target_model, "optimizer": target},
                    parallel_state=SimpleNamespace(dp_mode="ddp"),
                    process_group=dist.group.WORLD,
                )
        assert not target.state
    finally:
        dist.destroy_process_group()


def test_rank_local_checkpoint_identity_change_is_coordinated(tmp_path):
    torch.multiprocessing.spawn(_rank_changed_checkpoint_identity_worker, args=(str(tmp_path),), nprocs=2, join=True)


def _rank_optimizer_reader_setup_failure_worker(rank, root):
    from datetime import timedelta
    from unittest.mock import patch

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        model, optimizer = _model_and_optimizer()
        _step(model, optimizer, 1.0)
        checkpoint = f"{root}/checkpoint"
        _save(checkpoint, model, optimizer, fused=False)
        target_model, target = _model_and_optimizer()
        optimizer_path = str(optimizer_dir(checkpoint))
        optimizer_readers = 0

        def reader(path):
            nonlocal optimizer_readers
            if str(path) == optimizer_path:
                optimizer_readers += 1
                if rank == 1 and optimizer_readers == 5:
                    raise RuntimeError("rank-local optimizer reader setup failed")
            return dcp.FileSystemReader(path)

        with patch.object(DistributedCheckpointer, "_create_storage_reader", side_effect=reader):
            with pytest.raises(RuntimeError, match="optimizer reader setup failed|another rank"):
                DistributedCheckpointer.load(
                    checkpoint,
                    {"model": target_model, "optimizer": target},
                    parallel_state=SimpleNamespace(dp_mode="ddp"),
                    process_group=dist.group.WORLD,
                )
        assert not target.state
    finally:
        dist.destroy_process_group()


def test_rank_local_optimizer_reader_setup_failure_is_coordinated(tmp_path):
    torch.multiprocessing.spawn(
        _rank_optimizer_reader_setup_failure_worker, args=(str(tmp_path),), nprocs=2, join=True
    )


def _rank_model_hook_failure_worker(rank, root, hook_phase):
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        model, optimizer = _model_and_optimizer()
        _step(model, optimizer, 1.0)
        checkpoint = f"{root}/checkpoint"
        _save(checkpoint, model, optimizer, fused=False)
        target_model, target = _model_and_optimizer()
        if rank == 1:

            def fail(*_args):
                raise RuntimeError(f"local {hook_phase} model hook failed")

            if hook_phase == "save":
                target_model.register_state_dict_pre_hook(fail)
            else:
                target_model.register_load_state_dict_post_hook(fail)
        with pytest.raises(RuntimeError, match="model hook failed|another rank"):
            DistributedCheckpointer.load(
                checkpoint,
                {"model": target_model, "optimizer": target},
                parallel_state=SimpleNamespace(dp_mode="ddp"),
                process_group=dist.group.WORLD,
            )
        assert not target.state
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("hook_phase", ["save", "load"])
def test_rank_local_model_state_hook_failure_is_coordinated(tmp_path, hook_phase):
    torch.multiprocessing.spawn(_rank_model_hook_failure_worker, args=(str(tmp_path), hook_phase), nprocs=2, join=True)


def _rank_optimizer_hook_failure_worker(rank, root, hook_phase):
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        model, optimizer = _model_and_optimizer()
        _step(model, optimizer, 1.0)
        checkpoint = f"{root}/checkpoint"
        _save(checkpoint, model, optimizer, fused=False)
        target_model, target = _model_and_optimizer()
        if rank == 1:

            def fail(*_args):
                raise RuntimeError(f"local optimizer {hook_phase} hook failed")

            if hook_phase == "pre":
                target.register_load_state_dict_pre_hook(fail)
            else:
                target.register_load_state_dict_post_hook(fail)
        with pytest.raises(RuntimeError, match="optimizer .* hook failed|another rank"):
            DistributedCheckpointer.load(
                checkpoint,
                {"model": target_model, "optimizer": target},
                parallel_state=SimpleNamespace(dp_mode="ddp"),
                process_group=dist.group.WORLD,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("hook_phase", ["pre", "post"])
def test_rank_local_optimizer_load_hook_failure_is_coordinated(tmp_path, hook_phase):
    torch.multiprocessing.spawn(
        _rank_optimizer_hook_failure_worker, args=(str(tmp_path), hook_phase), nprocs=2, join=True
    )
