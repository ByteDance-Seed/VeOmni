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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_saving_before_first_optimizer_step_does_not_change_first_update(tmp_path, dtype):
    model, optimizer = _model_and_optimizer()
    reference_model, reference = _model_and_optimizer()
    model.to(dtype=dtype)
    reference_model.to(dtype=dtype)
    reference_model.load_state_dict(model.state_dict())
    DistributedCheckpointer.save(
        str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
    )
    assert not optimizer.state
    resumed_model, resumed = _model_and_optimizer()
    resumed_model.to(dtype=dtype)
    DistributedCheckpointer.load(
        str(tmp_path), {"model": resumed_model, "optimizer": resumed}, parallel_state=SimpleNamespace(dp_mode="ddp")
    )
    assert not resumed.state
    _step(model, optimizer, 1.0)
    _step(reference_model, reference, 1.0)
    _step(resumed_model, resumed, 1.0)
    for first, second, restored in zip(
        model.parameters(), reference_model.parameters(), resumed_model.parameters(), strict=True
    ):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(first, restored, rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state[first]["step"], reference.state[second]["step"], rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state[first]["step"], resumed.state[restored]["step"], rtol=0, atol=0)


def test_zero_step_checkpoint_keeps_group_lrs_and_scheduler(tmp_path):
    model, optimizer = _model_and_optimizer(split=True)
    optimizer.param_groups[0]["lr"] = 0.01
    optimizer.param_groups[1]["lr"] = 0.02
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lambda step: 0.5**step, lambda step: 0.8**step])
    DistributedCheckpointer.save(
        str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
    )
    assert not optimizer.state
    target_model, target = _model_and_optimizer(split=True)
    target_scheduler = torch.optim.lr_scheduler.LambdaLR(target, [lambda step: 0.5**step, lambda step: 0.8**step])
    DistributedCheckpointer.load(
        str(tmp_path), {"model": target_model, "optimizer": target}, parallel_state=SimpleNamespace(dp_mode="ddp")
    )
    target_scheduler.load_state_dict(scheduler.state_dict())
    assert not target.state
    assert [group["lr"] for group in target.param_groups] == [0.01, 0.02]
    _step(model, optimizer, 1.0)
    _step(target_model, target, 1.0)
    scheduler.step()
    target_scheduler.step()
    assert scheduler.get_last_lr() == target_scheduler.get_last_lr()
    for source, restored in zip(model.parameters(), target_model.parameters(), strict=True):
        torch.testing.assert_close(source, restored, rtol=0, atol=0)


def test_zero_step_resume_preserves_mixed_moment_precision_across_updates(tmp_path):
    def build():
        model = torch.nn.Linear(4, 3, dtype=torch.bfloat16)
        optimizer = CPUOffloadAdamW(
            [{"params": [model.weight], "cpu_offload": True}, {"params": [model.bias]}],
            lr=0.01,
            cpu_moment_dtype=torch.bfloat16,
            resident_moment_dtype=torch.float32,
        )
        return model, optimizer

    model, optimizer = build()
    reference_model, reference = build()
    reference_model.load_state_dict(model.state_dict())
    DistributedCheckpointer.save(
        str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
    )
    target_model, target = build()
    DistributedCheckpointer.load(
        str(tmp_path), {"model": target_model, "optimizer": target}, parallel_state=SimpleNamespace(dp_mode="ddp")
    )
    for gradient in (1.0, 0.3):
        for current_model, current_optimizer in (
            (model, optimizer),
            (reference_model, reference),
            (target_model, target),
        ):
            _step(current_model, current_optimizer, gradient)
        for source, control, restored in zip(
            model.parameters(), reference_model.parameters(), target_model.parameters(), strict=True
        ):
            torch.testing.assert_close(source, control, rtol=0, atol=0)
            torch.testing.assert_close(source, restored, rtol=0, atol=0)
            for key in ("master_weight", "exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(optimizer.state[source][key], reference.state[control][key], rtol=0, atol=0)
                torch.testing.assert_close(optimizer.state[source][key], target.state[restored][key], rtol=0, atol=0)


@pytest.mark.parametrize("state_kind", ["dict", "ordered", "defaultdict"])
def test_zero_step_with_preexisting_empty_parameter_state_resumes(tmp_path, state_kind):
    from collections import OrderedDict, defaultdict

    model, optimizer = _model_and_optimizer()
    optimizer.state[model.weight] = {"dict": dict, "ordered": OrderedDict, "defaultdict": lambda: defaultdict(dict)}[
        state_kind
    ]()
    assert optimizer.state[model.weight] == {}
    DistributedCheckpointer.save(
        str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
    )
    assert optimizer.state[model.weight] == {}
    target_model, target = _model_and_optimizer()
    assert target.state[target_model.weight] == {}
    DistributedCheckpointer.load(
        str(tmp_path), {"model": target_model, "optimizer": target}, parallel_state=SimpleNamespace(dp_mode="ddp")
    )
    assert not target.state
    _step(model, optimizer, 1.0)
    _step(target_model, target, 1.0)
    for source, restored in zip(model.parameters(), target_model.parameters(), strict=True):
        torch.testing.assert_close(source, restored, rtol=0, atol=0)


def test_zero_step_resume_clears_an_already_initialized_target_optimizer(tmp_path):
    model, optimizer = _model_and_optimizer()
    DistributedCheckpointer.save(
        str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
    )
    target_model, target = _model_and_optimizer()
    _step(target_model, target, 0.5)
    DistributedCheckpointer.load(
        str(tmp_path), {"model": target_model, "optimizer": target}, parallel_state=SimpleNamespace(dp_mode="ddp")
    )
    assert not target.state
    _step(model, optimizer, 1.0)
    _step(target_model, target, 1.0)
    for source, restored in zip(model.parameters(), target_model.parameters(), strict=True):
        torch.testing.assert_close(source, restored, rtol=0, atol=0)


@pytest.mark.parametrize("change", ["initialize", "replace"])
def test_model_save_hook_cannot_silently_change_optimizer_state(tmp_path, change):
    model, optimizer = _model_and_optimizer()
    _, replacement = _model_and_optimizer()
    state = {"model": model, "optimizer": optimizer}

    def mutate(_model, _state_dict, _prefix, _metadata):
        if change == "initialize":
            optimizer.state[model.weight]["step"] = torch.tensor(1)
        else:
            state["optimizer"] = replacement

    model.register_state_dict_post_hook(mutate)
    with pytest.raises(RuntimeError, match="Optimizer (state presence|object) changed"):
        DistributedCheckpointer.save(str(tmp_path), state, parallel_state=SimpleNamespace(dp_mode="ddp"))
    assert not (tmp_path / "model" / "optimizer" / ".metadata").exists()


def test_model_save_hook_cannot_freeze_parameter_with_optimizer_history(tmp_path):
    model, optimizer = _model_and_optimizer()
    _step(model, optimizer, 1.0)

    def freeze(_model, _state_dict, _prefix, _metadata):
        model.weight.requires_grad_(False)

    model.register_state_dict_post_hook(freeze)
    with pytest.raises(RuntimeError, match="frozen precision-sensitive optimizer state"):
        DistributedCheckpointer.save(
            str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
        )
    assert not (tmp_path / "model" / "optimizer" / ".metadata").exists()


def _model_only_overwrite_failure_worker(rank, root, save_async):
    import torch.distributed as dist

    from veomni.checkpoint import layout

    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2)
    try:
        model, optimizer = _model_and_optimizer()
        checkpoint = f"{root}/checkpoint"
        parallel = SimpleNamespace(dp_mode="ddp")
        DistributedCheckpointer.save(checkpoint, {"model": model}, parallel_state=parallel)
        if rank == 0:
            layout.write_manifest(checkpoint, global_step=1, world_size=2, model_state=True)
        dist.barrier()
        assert layout.checkpoint_is_complete(checkpoint)
        if rank == 1:

            def mutate(_model, _state_dict, _prefix, _metadata):
                optimizer.state[model.weight]["step"] = torch.tensor(1)

            model.register_state_dict_post_hook(mutate)
        with pytest.raises(RuntimeError, match="state presence|ranks disagree"):
            DistributedCheckpointer.save(
                checkpoint, {"model": model, "optimizer": optimizer}, parallel_state=parallel, save_async=save_async
            )
        if save_async:
            DistributedCheckpointer.wait_for_pending_save()
        dist.barrier()
        assert not layout.checkpoint_is_complete(checkpoint)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("save_async", [False, True])
def test_failed_overwrite_never_looks_like_a_complete_checkpoint(tmp_path, save_async):
    torch.multiprocessing.spawn(
        _model_only_overwrite_failure_worker, args=(str(tmp_path), save_async), nprocs=2, join=True
    )


def _pending_optimizer_overwrite_worker(rank, root):
    import threading
    from unittest.mock import patch

    import torch.distributed as dist

    from veomni.checkpoint import layout

    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2)
    try:
        model, optimizer = _model_and_optimizer()
        checkpoint = f"{root}/checkpoint"
        parallel = SimpleNamespace(dp_mode="ddp")
        entered = threading.Event()
        release = threading.Event()
        real_writer = DistributedCheckpointer._create_storage_writer
        optimizer_writers = 0

        class GateWriter(dcp.FileSystemWriter):
            def write_data(self, plan, planner):
                entered.set()
                if not release.wait(20):
                    raise RuntimeError("timed out waiting to release optimizer writer")
                return super().write_data(plan, planner)

        def writer(path):
            nonlocal optimizer_writers
            if str(path).endswith("/optimizer"):
                optimizer_writers += 1
                if optimizer_writers == 1:
                    return GateWriter(path)
            return real_writer(path)

        with patch.object(DistributedCheckpointer, "_create_storage_writer", side_effect=writer):
            DistributedCheckpointer.save(
                checkpoint, {"model": model, "optimizer": optimizer}, parallel_state=parallel, save_async=True
            )
            assert entered.wait(20)
            DistributedCheckpointer._drain_slot("ckpt")
            if rank == 0:
                layout.write_manifest(checkpoint, global_step=1, world_size=2, model_state=True)
            dist.barrier()
            with torch.no_grad():
                model.weight.add_(1)
            if rank == 1:

                def mutate(_model, _state_dict, _prefix, _metadata):
                    optimizer.state[model.weight]["step"] = torch.tensor(1)

                model.register_state_dict_post_hook(mutate)
            timer = threading.Timer(2, release.set)
            timer.start()
            try:
                with pytest.raises(RuntimeError, match="state presence|ranks disagree"):
                    DistributedCheckpointer.save(
                        checkpoint, {"model": model, "optimizer": optimizer}, parallel_state=parallel
                    )
            finally:
                release.set()
                timer.join()
                DistributedCheckpointer.wait_for_pending_save()
            dist.barrier()
            assert not layout.checkpoint_is_complete(checkpoint)
    finally:
        dist.destroy_process_group()


def test_old_async_optimizer_write_cannot_revive_marker_during_failed_overwrite(tmp_path):
    torch.multiprocessing.spawn(_pending_optimizer_overwrite_worker, args=(str(tmp_path),), nprocs=2, join=True)


def test_all_pending_slots_are_drained_before_propagating_save_error(tmp_path, monkeypatch):
    from concurrent.futures import Future

    weight_future, optimizer_future = Future(), Future()
    weight_future.set_exception(RuntimeError("previous weights failed"))
    optimizer_future.set_result(None)
    monkeypatch.setattr(
        DistributedCheckpointer, "_save_futures", {"ckpt": weight_future, "optimizer": optimizer_future}
    )
    model, optimizer = _model_and_optimizer()
    with pytest.raises(RuntimeError, match="previous weights failed"):
        DistributedCheckpointer.save(
            str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
        )
    assert not DistributedCheckpointer._save_futures
    assert not (tmp_path / "model").exists()


def _rank_pending_save_error_worker(rank, root):
    from concurrent.futures import Future

    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2)
    try:
        weight_future, optimizer_future = Future(), Future()
        if rank == 1:
            weight_future.set_exception(RuntimeError("rank-local previous weights failed"))
        else:
            weight_future.set_result(None)
        optimizer_future.set_result(None)
        DistributedCheckpointer._save_futures = {"ckpt": weight_future, "optimizer": optimizer_future}
        DistributedCheckpointer._async_process_groups = {"ckpt": dist.group.WORLD, "optimizer": dist.group.WORLD}
        model, optimizer = _model_and_optimizer()
        with pytest.raises(RuntimeError, match="previous weights failed|another rank"):
            DistributedCheckpointer.save(
                f"{root}/checkpoint",
                {"model": model, "optimizer": optimizer},
                parallel_state=SimpleNamespace(dp_mode="ddp"),
            )
        assert not DistributedCheckpointer._save_futures
    finally:
        dist.destroy_process_group()


def test_rank_local_pending_error_drains_both_slots_before_save(tmp_path):
    torch.multiprocessing.spawn(_rank_pending_save_error_worker, args=(str(tmp_path),), nprocs=2, join=True)
    assert not (tmp_path / "checkpoint").exists()


def test_cursor_only_manifest_cannot_be_overwritten_in_place_with_optimizer(tmp_path):
    from veomni.checkpoint import layout

    layout.write_manifest(str(tmp_path), global_step=1, world_size=1, model_state=False)
    assert layout.checkpoint_is_complete(str(tmp_path))
    model, optimizer = _model_and_optimizer()
    with pytest.raises(RuntimeError, match="cursor-only checkpoint"):
        DistributedCheckpointer.save(
            str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
        )
    assert layout.read_manifest(str(tmp_path))["model_state"] is False
    assert not (tmp_path / "model").exists()
    assert not optimizer.state


@pytest.mark.parametrize("content", ["{invalid-json", "[]", "null"])
def test_invalid_existing_manifest_rejects_checkpoint_overwrite(tmp_path, content):
    from veomni.checkpoint.layout import manifest_path

    path = manifest_path(str(tmp_path))
    with open(path, "w") as handle:
        handle.write(content)
    model, optimizer = _model_and_optimizer()
    with pytest.raises((ValueError, RuntimeError)):
        DistributedCheckpointer.save(
            str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
        )
    with open(path) as handle:
        assert handle.read() == content
    assert not (tmp_path / "model").exists()


def test_unreadable_existing_manifest_rejects_checkpoint_overwrite(tmp_path, monkeypatch):
    import builtins

    from veomni.checkpoint.layout import manifest_path

    path = manifest_path(str(tmp_path))
    with open(path, "w") as handle:
        handle.write('{"model_state": false}')
    real_open = builtins.open

    def deny_manifest(file, *args, **kwargs):
        if str(file) == path:
            raise PermissionError("manifest unavailable")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", deny_manifest)
    model, optimizer = _model_and_optimizer()
    with pytest.raises(PermissionError, match="manifest unavailable"):
        DistributedCheckpointer.save(
            str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
        )
    assert not (tmp_path / "model").exists()


def _zero_step_checkpoint_worker(rank, root, save_async):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2)
    try:
        model, optimizer = _model_and_optimizer()
        model = DistributedDataParallel(model)
        if rank == 1:
            assert optimizer.state[model.module.weight] == {}
        checkpoint = f"{root}/checkpoint"
        parallel = SimpleNamespace(dp_mode="ddp")
        DistributedCheckpointer.save(
            checkpoint, {"model": model, "optimizer": optimizer}, parallel_state=parallel, save_async=save_async
        )
        if save_async:
            DistributedCheckpointer.wait_for_pending_save()
        assert not any(optimizer.state.values())
        target_model, target = _model_and_optimizer()
        target_model = DistributedDataParallel(target_model)
        DistributedCheckpointer.load(
            checkpoint,
            {"model": target_model, "optimizer": target},
            parallel_state=parallel,
            process_group=dist.group.WORLD,
        )
        assert not target.state
        _step(model, optimizer, 1.0)
        _step(target_model, target, 1.0)
        for source, restored in zip(model.parameters(), target_model.parameters(), strict=True):
            torch.testing.assert_close(source, restored, rtol=0, atol=0)
            torch.testing.assert_close(optimizer.state[source]["step"], target.state[restored]["step"], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("save_async", [False, True])
def test_zero_step_checkpoint_is_consistent_across_ranks(tmp_path, save_async):
    torch.multiprocessing.spawn(_zero_step_checkpoint_worker, args=(str(tmp_path), save_async), nprocs=2, join=True)


def _asymmetric_zero_step_worker(rank, root):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2)
    try:
        model, optimizer = _model_and_optimizer()
        if rank == 1:
            _step(model, optimizer, 1.0)
        with pytest.raises(RuntimeError, match="ranks disagree"):
            DistributedCheckpointer.save(
                f"{root}/checkpoint",
                {"model": model, "optimizer": optimizer},
                parallel_state=SimpleNamespace(dp_mode="ddp"),
            )
        assert bool(optimizer.state) == (rank == 1)
    finally:
        dist.destroy_process_group()


def test_asymmetric_zero_step_state_is_rejected_before_save(tmp_path):
    torch.multiprocessing.spawn(_asymmetric_zero_step_worker, args=(str(tmp_path),), nprocs=2, join=True)
    assert not (tmp_path / "checkpoint").exists()


def _rank_foreign_optimizer_state_worker(rank, root, wrapper_only):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2)
    try:
        model, optimizer = _model_and_optimizer()
        _step(model, optimizer, 1.0)
        if rank == 1:
            optimizer.state["foreign"] = {"step": torch.tensor(1)}
        with pytest.raises(RuntimeError, match="non-parameter|another rank"):
            if wrapper_only:
                OptimizerState(
                    model,
                    optimizer,
                    parallel_state=SimpleNamespace(dp_mode="ddp"),
                    coordinated_group_validation=True,
                    coordinated_group=dist.group.WORLD,
                ).state_dict()
            else:
                DistributedCheckpointer.save(
                    f"{root}/checkpoint",
                    {"model": model, "optimizer": optimizer},
                    parallel_state=SimpleNamespace(dp_mode="ddp"),
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("wrapper_only", [False, True])
def test_rank_local_foreign_optimizer_state_fails_before_save(tmp_path, wrapper_only):
    torch.multiprocessing.spawn(
        _rank_foreign_optimizer_state_worker, args=(str(tmp_path), wrapper_only), nprocs=2, join=True
    )
    assert not (tmp_path / "checkpoint").exists()


def _rank_unsupported_optimizer_state_value_worker(rank, root):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2)
    try:
        model, optimizer = _model_and_optimizer()
        if rank == 1:
            optimizer.state[model.weight] = []
        with pytest.raises(RuntimeError, match="state value type|another rank"):
            DistributedCheckpointer.save(
                f"{root}/checkpoint",
                {"model": model, "optimizer": optimizer},
                parallel_state=SimpleNamespace(dp_mode="ddp"),
            )
    finally:
        dist.destroy_process_group()


def test_rank_local_unsupported_optimizer_state_value_fails_before_save(tmp_path):
    torch.multiprocessing.spawn(
        _rank_unsupported_optimizer_state_value_worker, args=(str(tmp_path),), nprocs=2, join=True
    )
    assert not (tmp_path / "checkpoint").exists()


def _rank_optimizer_kind_mismatch_worker(rank, root):
    import torch.distributed as dist

    from veomni.optim.master_fp32_adamw import FullFP32AdamW

    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2)
    try:
        model, optimizer = _model_and_optimizer()
        if rank == 1:
            optimizer = FullFP32AdamW(model.parameters())
        _step(model, optimizer, 1.0)
        with pytest.raises(RuntimeError, match="ranks disagree"):
            OptimizerState(
                model,
                optimizer,
                parallel_state=SimpleNamespace(dp_mode="ddp"),
                coordinated_group_validation=True,
                coordinated_group=dist.group.WORLD,
            ).state_dict()
    finally:
        dist.destroy_process_group()


def test_rank_local_optimizer_kind_mismatch_is_coordinated(tmp_path):
    torch.multiprocessing.spawn(_rank_optimizer_kind_mismatch_worker, args=(str(tmp_path),), nprocs=2, join=True)


def _rank_model_save_hook_failure_worker(rank, root, hook_phase, save_async):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2)
    try:
        model, optimizer = _model_and_optimizer()
        if rank == 1:

            def fail(*_args):
                raise RuntimeError("rank-local model save hook failed")

            if hook_phase == "pre":
                model.register_state_dict_pre_hook(fail)
            else:
                model.register_state_dict_post_hook(fail)
        with pytest.raises(RuntimeError, match="model save hook failed|another rank"):
            DistributedCheckpointer.save(
                f"{root}/checkpoint",
                {"model": model, "optimizer": optimizer},
                parallel_state=SimpleNamespace(dp_mode="ddp"),
                save_async=save_async,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("hook_phase", ["pre", "post"])
@pytest.mark.parametrize("save_async", [False, True])
def test_rank_local_model_save_hook_failure_is_coordinated(tmp_path, hook_phase, save_async):
    torch.multiprocessing.spawn(
        _rank_model_save_hook_failure_worker, args=(str(tmp_path), hook_phase, save_async), nprocs=2, join=True
    )


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
