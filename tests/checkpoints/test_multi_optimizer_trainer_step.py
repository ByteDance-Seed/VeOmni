# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Trainer state must match the optimizer step restored from DCP."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.checkpoints.test_checkpoint_callback import _make_mock_trainer
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _trainer(tmp_path, *, optimizer_step=3, strict=True):
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.args.train.checkpoint.load_path = str(tmp_path)
    trainer.state = SimpleNamespace(global_step=0)
    trainer.train_dataloader = None
    trainer.data_iterator = SimpleNamespace(state_dict=lambda: {})
    trainer.environ_meter.state_dict.return_value = {}
    trainer.channel_loss_callback = None
    trainer.offline_cache_writer = None
    runtime = SimpleNamespace(optimizer=SimpleNamespace(_is_multi_optimizer=strict))
    runtime.checkpoint = SimpleNamespace(
        runtime=runtime,
        _loaded_optimizer_global_step=optimizer_step if strict else None,
        _loaded_optimizer_generation="generation" if strict else None,
    )
    trainer.model = runtime
    return trainer


@pytest.mark.parametrize(
    "mode",
    [
        "missing",
        "wrong_step",
        "invalid_step",
        "optimizer_step_missing",
        "generation_missing",
        "cursor_generation_missing",
        "cursor_generation_wrong",
        "matching",
        "legacy_missing",
    ],
)
def test_trainer_state_cannot_silently_diverge_from_restored_optimizer(tmp_path, mode):
    trainer = _trainer(tmp_path, strict=mode != "legacy_missing")
    if mode == "optimizer_step_missing":
        trainer.model.checkpoint._loaded_optimizer_global_step = None
    if mode == "generation_missing":
        trainer.model.checkpoint._loaded_optimizer_generation = None
    callback = GlobalStateCallback(trainer)
    state = (
        None
        if mode in ("missing", "legacy_missing")
        else {
            "global_step": 4 if mode == "wrong_step" else 3,
            "_veomni_optimizer_generations": {"model": "generation"},
        }
    )
    if mode == "invalid_step":
        state["global_step"] = True
    if mode == "cursor_generation_missing":
        state.pop("_veomni_optimizer_generations")
    if mode == "cursor_generation_wrong":
        state["_veomni_optimizer_generations"]["model"] = "older-generation"
    with (
        patch.object(callback, "_read_current", return_value=state),
        patch("veomni.checkpoint.legacy_v0_1_12.apply_legacy_global_state", return_value=None),
        patch("veomni.trainer.callbacks.global_state_callback.set_device_rng_state"),
    ):
        if mode in (
            "missing",
            "wrong_step",
            "invalid_step",
            "optimizer_step_missing",
            "generation_missing",
            "cursor_generation_missing",
            "cursor_generation_wrong",
        ):
            with pytest.raises(
                RuntimeError,
                match="Trainer state is missing|checkpoint steps differ|step is missing|step or generation|generations differ",
            ):
                callback.load_global_state()
            assert trainer.state.global_step == 0
        else:
            callback.load_global_state()
            assert trainer.state.global_step == (3 if mode == "matching" else 0)


@pytest.mark.parametrize("mode", ["missing", "stale_generation", "matching"])
@pytest.mark.parametrize("instrumented", ["none", "source", "target", "both"])
def test_real_dcp_with_scheduler_requires_matching_trainer_generation(tmp_path, mode, instrumented):
    import torch

    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime
    from veomni.checkpoint import layout
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.models.checkpoint.manager import ModelCheckpointManager
    from veomni.optim.lr_scheduler import build_lr_scheduler
    from veomni.optim.optimizer import MultiOptimizer
    from veomni.trainer.callbacks.base import TrainerState

    def build(with_instrumentation):
        runtime = _make_mock_runtime(save_path=str(tmp_path))
        runtime.model = torch.nn.Linear(2, 2)
        leaf = torch.optim.AdamW(runtime.model.parameters(), lr=0.01)
        runtime.optimizer = MultiOptimizer(runtime.model, {"model": leaf}, ["model"])
        runtime.lr_scheduler = build_lr_scheduler(runtime.optimizer, train_steps=100, lr=0.01)
        if with_instrumentation:
            scheduler = runtime.lr_scheduler["model"]
            original_state_dict = scheduler.state_dict

            def observed_state_dict():
                # A profiling wrapper delegates to the scheduler and keeps the
                # callable itself out of the serializable scheduler snapshot.
                return {key: value for key, value in original_state_dict().items() if key != "state_dict"}

            scheduler.state_dict = observed_state_dict
        runtime.parallel_state = SimpleNamespace(global_rank=0, dp_mode="ddp", extra_parallel_names=[])
        return runtime

    with (
        patch("veomni.models.checkpoint.manager.build_checkpointer", return_value=DistributedCheckpointer),
        patch("torch.distributed.barrier"),
    ):
        source = build(instrumented in ("source", "both"))
        manager = ModelCheckpointManager(source)
        for _ in range(3):
            source.model(torch.ones(1, 2)).sum().backward()
            source.optimizer.step()
            source.optimizer.zero_grad()
            source.lr_scheduler.step()
        manager.save_dcp(TrainerState(global_step=3))
        source.checkpoint = manager
        if mode == "matching":
            source_trainer = _make_mock_trainer(save_path=str(tmp_path))
            source_trainer.model = source
            source_trainer.train_dataloader = None
            source_trainer.data_iterator = SimpleNamespace(state_dict=lambda: {})
            source_trainer.environ_meter.state_dict.return_value = {}
            source_trainer.channel_loss_callback = None
            source_trainer.offline_cache_writer = None
            GlobalStateCallback(source_trainer).save_global_state(TrainerState(global_step=3))
        elif mode == "stale_generation":
            sidecar = layout.extra_state_path(str(tmp_path / "global_step_3"), 0)
            import os

            os.makedirs(os.path.dirname(sidecar), exist_ok=True)
            torch.save({"global_step": 3, "_veomni_optimizer_generations": {"model": "stale"}}, sidecar)
        target = build(instrumented in ("target", "both"))
        target.train_args.checkpoint.load_path = str(tmp_path / "global_step_3")
        target.checkpoint = ModelCheckpointManager(target)
        target.checkpoint.load()
    assert target.checkpoint._loaded_optimizer_global_step == 3
    assert target.lr_scheduler["model"].last_epoch == 3
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.args.train.checkpoint.load_path = target.train_args.checkpoint.load_path
    trainer.model = target
    trainer.state = SimpleNamespace(global_step=0)
    trainer.train_dataloader = None
    if mode == "matching":
        GlobalStateCallback(trainer).load_global_state()
        assert trainer.state.global_step == 3
        for runtime in (source, target):
            runtime.optimizer.zero_grad()
            runtime.model(torch.ones(1, 2)).sum().backward()
            runtime.optimizer.step()
            runtime.lr_scheduler.step()
        for actual, expected in zip(target.model.parameters(), source.model.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert target.lr_scheduler["model"].get_last_lr() == source.lr_scheduler["model"].get_last_lr()
    else:
        with pytest.raises(RuntimeError, match="Trainer state is missing|generations differ"):
            GlobalStateCallback(trainer).load_global_state()
        assert trainer.state.global_step == 0
    assert target.lr_scheduler["model"].last_epoch == (4 if mode == "matching" else 3)


@pytest.mark.parametrize("fail_once", [False, True])
def test_hf_step_and_epoch_dcp_share_multi_optimizer_checkpoint(tmp_path, fail_once):
    import torch

    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime, _make_mock_trainer
    from veomni.checkpoint import layout
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.models.checkpoint.manager import ModelCheckpointManager
    from veomni.optim.lr_scheduler import build_lr_scheduler
    from veomni.optim.optimizer import MultiOptimizer
    from veomni.trainer.callbacks.base import TrainerState
    from veomni.trainer.callbacks.checkpoint_callback import CheckpointCallback

    trainer = _make_mock_trainer(save_path=str(tmp_path))
    ckpt = trainer.args.train.checkpoint
    ckpt.save_steps, ckpt.save_epochs = 0, 1
    ckpt.save_hf_weights, ckpt.hf_save_steps, ckpt.hf_save_epochs = True, 5, 0
    trainer.train_dataloader = None
    trainer.data_iterator = SimpleNamespace(state_dict=lambda: {})
    trainer.environ_meter.state_dict.return_value = {}
    trainer.channel_loss_callback = None
    trainer.offline_cache_writer = None

    runtime = _make_mock_runtime(save_path=str(tmp_path))
    runtime.model = torch.nn.Linear(2, 2)
    leaf = torch.optim.AdamW(runtime.model.parameters(), lr=0.01)
    runtime.optimizer = MultiOptimizer(runtime.model, {"model": leaf}, ["model"])
    runtime.lr_scheduler = build_lr_scheduler(runtime.optimizer, train_steps=100, lr=0.01)
    runtime.parallel_state = SimpleNamespace(global_rank=0, dp_mode="ddp", extra_parallel_names=[])
    with (
        patch("veomni.models.checkpoint.manager.build_checkpointer", return_value=DistributedCheckpointer),
        patch("torch.distributed.barrier"),
    ):
        manager = ModelCheckpointManager(runtime)
        runtime.checkpoint = manager
        trainer.model = runtime
        trainer.save_dcp.side_effect = manager.save_dcp
        attempts = []

        def export(state, stage):
            manager.prepare_export(state, stage)
            attempts.append(True)
            if fail_once and len(attempts) == 1:
                raise RuntimeError("HF conversion failed after DCP")

        trainer.save_hf_or_lora.side_effect = export
        for _ in range(5):
            runtime.model(torch.ones(1, 2)).sum().backward()
            runtime.optimizer.step()
            runtime.optimizer.zero_grad(set_to_none=True)
            runtime.lr_scheduler.step()
        checkpoint_cb = CheckpointCallback(trainer)
        global_cb = GlobalStateCallback(trainer)
        state = TrainerState(global_step=5, epoch=0)

        if fail_once:
            with pytest.raises(RuntimeError, match="HF conversion failed"):
                checkpoint_cb.on_step_end(state)
        checkpoint_cb.on_step_end(state)
        global_cb.on_step_end(state)
        assert layout.checkpoint_is_complete(layout.step_dir(str(tmp_path), 5))
        checkpoint_cb.on_epoch_end(state)
        global_cb.on_epoch_end(state)
        trainer.save_dcp.assert_called_once_with(state)
        assert manager.last_saved_step == checkpoint_cb._last_dcp_step == global_cb._last_saved_step == 5

        restored = _make_mock_runtime(save_path=str(tmp_path))
        restored.model = torch.nn.Linear(2, 2)
        restored_leaf = torch.optim.AdamW(restored.model.parameters(), lr=0.01)
        restored.optimizer = MultiOptimizer(restored.model, {"model": restored_leaf}, ["model"])
        restored.lr_scheduler = build_lr_scheduler(restored.optimizer, train_steps=100, lr=0.01)
        restored.parallel_state = SimpleNamespace(global_rank=0, dp_mode="ddp", extra_parallel_names=[])
        restored.train_args.checkpoint.load_path = layout.step_dir(str(tmp_path), 5)
        restored.checkpoint = ModelCheckpointManager(restored)
        restored.checkpoint.load()
        restored_trainer = _make_mock_trainer(save_path=str(tmp_path))
        restored_trainer.args.train.checkpoint.load_path = restored.train_args.checkpoint.load_path
        restored_trainer.model = restored
        restored_trainer.train_dataloader = None
        restored_trainer.data_iterator = SimpleNamespace(load_state_dict=lambda state: None)
        restored_trainer.channel_loss_callback = None
        restored_trainer.offline_cache_writer = None
        GlobalStateCallback(restored_trainer).load_global_state()
        assert restored_trainer.state.global_step == 5
        assert restored.lr_scheduler["model"].last_epoch == 5
        torch.testing.assert_close(restored.optimizer.state_dict(), runtime.optimizer.state_dict(), rtol=0, atol=0)
        for model_runtime in (runtime, restored):
            model_runtime.optimizer.zero_grad()
            model_runtime.model(torch.full((1, 2), 2.0)).sum().backward()
            model_runtime.optimizer.step()
            model_runtime.lr_scheduler.step()
        for actual, expected in zip(restored.model.parameters(), runtime.model.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_train_end_sidecar_keeps_generation_after_optimizer_release(tmp_path):
    import torch

    from veomni.checkpoint import layout
    from veomni.trainer.callbacks.base import TrainerState

    trainer = _trainer(tmp_path)
    trainer.model.optimizer = None
    trainer.model.checkpoint._last_saved_step = 3
    trainer.model.checkpoint._last_saved_optimizer_generation = "saved-generation"
    trainer.environ_meter.state_dict.return_value = {}
    callback = GlobalStateCallback(trainer)
    callback.save_global_state(TrainerState(global_step=3))
    sidecar = layout.extra_state_path(str(tmp_path / "global_step_3"), 0)
    saved = torch.load(sidecar, map_location="cpu", weights_only=False)
    assert saved["_veomni_optimizer_generations"] == {"model": "saved-generation"}


def _resume_worker(rank, tmp_path, mode, endpoint):
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        _init_parallel_state(dp_size=2, dp_shard_size=2, device_type="cpu")
        step = 4 if mode == "optimizer_step" and rank == 1 else 3
        trainer = _trainer(tmp_path, optimizer_step=step)
        trainer.args.train.global_rank = rank
        trainer.args.train.world_size = 2
        callback = GlobalStateCallback(trainer)
        state = (
            None
            if mode == "missing" and rank == 1
            else {
                "global_step": 4 if mode == "trainer_step" and rank == 1 else 3,
                "_veomni_optimizer_generations": {
                    "model": "older" if mode == "trainer_generation" and rank == 1 else "generation"
                },
            }
        )
        with (
            patch.object(callback, "_read_current", return_value=state),
            patch("veomni.checkpoint.legacy_v0_1_12.apply_legacy_global_state", return_value=None),
            patch("veomni.trainer.callbacks.global_state_callback.set_device_rng_state"),
        ):
            if mode == "matching":
                callback.load_global_state()
                assert trainer.state.global_step == 3
            else:
                with pytest.raises(
                    RuntimeError,
                    match="Trainer state is missing|checkpoint steps differ|steps differ across ranks|"
                    "generations differ|checking the trainer checkpoint step failed on another rank",
                ):
                    callback.load_global_state()
                assert trainer.state.global_step == 0
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
@pytest.mark.parametrize("mode", ["missing", "optimizer_step", "trainer_step", "trainer_generation", "matching"])
def test_multi_optimizer_trainer_resume_is_rank_coordinated(tmp_path, mode):
    spawn_with_rendezvous(_resume_worker, args=(str(tmp_path), mode), nprocs=2)


def _save_failure_worker(rank, tmp_path, mode, endpoint):
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
    from veomni.trainer.callbacks.base import TrainerState

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        _init_parallel_state(dp_size=2, dp_shard_size=2, device_type="cpu")
        trainer = _trainer(tmp_path)
        trainer.args.train.global_rank = rank
        trainer.args.train.world_size = 2
        trainer.model.checkpoint._last_saved_step = 3
        trainer.model.checkpoint._last_saved_optimizer_generation = (
            None if mode == "generation" and rank == 1 else "gen"
        )
        callback = GlobalStateCallback(trainer)
        if mode == "cursor" and rank == 1:
            callback.state_dict = lambda _: (_ for _ in ()).throw(ValueError("injected cursor failure"))
        with pytest.raises(
            (RuntimeError, ValueError), match="trainer checkpoint state|injected cursor failure|generation"
        ):
            callback.save_global_state(TrainerState(global_step=3))
        dist.barrier()
        assert not (tmp_path / "global_step_3").exists()
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
@pytest.mark.parametrize("mode", ["cursor", "generation"])
def test_trainer_save_failure_is_coordinated_before_write(tmp_path, mode):
    spawn_with_rendezvous(_save_failure_worker, args=(tmp_path, mode), nprocs=2)
