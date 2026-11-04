# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""The trainer cursor must belong to the same optimizer checkpoint generation."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.checkpoints.test_checkpoint_callback import _make_mock_trainer
from veomni.checkpoint import layout
from veomni.trainer.callbacks.base import TrainerState
from veomni.trainer.callbacks.checkpoint_callback import CheckpointCallback
from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _ready_to_load(tmp_path, monkeypatch, generation="source"):
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.args.train.checkpoint.load_path = str(tmp_path)
    trainer.model.optimizer = SimpleNamespace(_is_multi_optimizer=True)
    trainer.model.checkpoint = SimpleNamespace(
        _loaded_optimizer_global_step=3, _loaded_optimizer_generation=generation
    )
    trainer.train_dataloader = None
    monkeypatch.setattr(layout, "read_manifest", lambda _: {"model_state": True, "global_step": 3, "world_size": 1})
    monkeypatch.setattr(layout, "checkpoint_is_complete", lambda _: True)
    return trainer


def test_fresh_multi_optimizer_training_needs_no_restore_receipt(tmp_path):
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.model.optimizer = SimpleNamespace(_is_multi_optimizer=True)
    trainer.model.checkpoint = SimpleNamespace(_loaded_optimizer_global_step=None, _loaded_optimizer_generation=None)
    assert GlobalStateCallback(trainer).load_global_state() is None


@pytest.mark.parametrize("same_path", [True, False])
def test_multi_optimizer_reuse_requires_the_owner_load_path(tmp_path, same_path):
    class Owner:
        def __init__(self, loaded_path):
            self._loaded_checkpoint_path = loaded_path
            self.confirmed = False

        def confirm_loaded_checkpoint_for_reuse(self, step, generation):
            assert (step, generation) == (3, "source")
            self.confirmed = True

    trainer = _make_mock_trainer(save_path=str(tmp_path))
    callback = GlobalStateCallback(trainer)
    path = layout.step_dir(str(tmp_path), 3)
    owner = Owner(path if same_path else layout.step_dir(str(tmp_path / "other"), 3))
    if same_path:
        callback._confirm_same_output_checkpoint(3, path, [("model", owner)], {"model": "source"})
        assert owner.confirmed
    else:
        with pytest.raises(RuntimeError, match="different checkpoint path"):
            callback._confirm_same_output_checkpoint(3, path, [("model", owner)], {"model": "source"})
        assert not owner.confirmed


def test_legacy_checkpoint_manager_does_not_opt_in_to_generation_guard(tmp_path):
    from veomni.trainer.callbacks.global_state_callback import _multi_optimizer_checkpoints

    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.model.optimizer = SimpleNamespace(_is_multi_optimizer=True)
    trainer.model.checkpoint = SimpleNamespace(_last_saved_step=1)
    assert _multi_optimizer_checkpoints(trainer) == []


def test_same_step_cursor_from_another_generation_fails_before_position_or_rng(tmp_path, monkeypatch):
    trainer = _ready_to_load(tmp_path, monkeypatch)
    callback = GlobalStateCallback(trainer)
    data = {
        "global_step": 3,
        "torch_rng_state": torch.get_rng_state(),
        "_veomni_optimizer_generations": {"model": "different"},
    }
    monkeypatch.setattr(callback, "_read_current", lambda _: data)
    with pytest.raises(ValueError, match="generations differ"):
        callback.load_global_state()
    assert trainer.state.global_step == 0


def test_matching_cursor_generation_restores(tmp_path, monkeypatch):
    trainer = _ready_to_load(tmp_path, monkeypatch)
    callback = GlobalStateCallback(trainer)
    data = {
        "global_step": 3,
        "torch_rng_state": torch.get_rng_state(),
        "_veomni_optimizer_generations": {"model": "source"},
    }
    monkeypatch.setattr(callback, "_read_current", lambda _: data)
    monkeypatch.setattr(callback, "_restore_position", lambda _: None)
    callback.load_global_state()
    assert trainer.state.global_step == 3


def test_save_requires_same_step_generation_before_writing_cursor(tmp_path):
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.model.optimizer = SimpleNamespace(_is_multi_optimizer=True)
    trainer.model.checkpoint = SimpleNamespace(
        _last_saved_step=2, _last_saved_optimizer_generation="source", _loaded_optimizer_generation=None
    )
    callback = GlobalStateCallback(trainer)
    with pytest.raises(RuntimeError, match="same-step"):
        callback.save_global_state(TrainerState(global_step=3), require_optimizer_checkpoint=True)
    assert not list(tmp_path.iterdir())


def test_hf_only_export_does_not_require_a_same_step_optimizer_checkpoint(tmp_path):
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.model.optimizer = SimpleNamespace(_is_multi_optimizer=True)
    trainer.model.checkpoint = SimpleNamespace(
        _last_saved_step=2, _last_saved_optimizer_generation="source", _loaded_optimizer_generation=None
    )
    trainer.train_dataloader = None
    trainer.data_iterator = None
    trainer.environ_meter.state_dict.return_value = {}
    GlobalStateCallback(trainer).save_global_state(TrainerState(global_step=3))
    path = layout.extra_state_path(layout.step_dir(str(tmp_path), 3), trainer.args.train.global_rank)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "_veomni_optimizer_generations" not in payload


def test_save_records_generation_in_the_trainer_sidecar(tmp_path):
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.model.optimizer = SimpleNamespace(_is_multi_optimizer=True)
    trainer.model.checkpoint = SimpleNamespace(
        _last_saved_step=3, _last_saved_optimizer_generation="source", _loaded_optimizer_generation=None
    )
    trainer.train_dataloader = None
    trainer.data_iterator = None
    trainer.environ_meter.state_dict.return_value = {}
    callback = GlobalStateCallback(trainer)
    callback.save_global_state(TrainerState(global_step=3))
    path = layout.extra_state_path(layout.step_dir(str(tmp_path), 3), trainer.args.train.global_rank)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["_veomni_optimizer_generations"] == {"model": "source"}


def test_train_end_export_retains_receipt_after_optimizer_is_released(tmp_path):
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.model.optimizer = None
    trainer.model.checkpoint = SimpleNamespace(
        _last_saved_step=3, _last_saved_optimizer_generation="source", _loaded_optimizer_generation=None
    )
    trainer.train_dataloader = None
    trainer.data_iterator = None
    trainer.environ_meter.state_dict.return_value = {}
    callback = GlobalStateCallback(trainer)
    callback.save_global_state(TrainerState(global_step=3))
    payload = torch.load(
        layout.extra_state_path(layout.step_dir(str(tmp_path), 3), trainer.args.train.global_rank),
        map_location="cpu",
        weights_only=False,
    )
    assert payload["_veomni_optimizer_generations"] == {"model": "source"}


def test_same_step_new_generation_rebinds_existing_cursor(tmp_path):
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.model.optimizer = SimpleNamespace(_is_multi_optimizer=True)
    checkpoint = trainer.model.checkpoint = SimpleNamespace(
        _last_saved_step=3, _last_saved_optimizer_generation="first", _loaded_optimizer_generation=None
    )
    trainer.train_dataloader = None
    trainer.data_iterator = None
    trainer.environ_meter.state_dict.return_value = {}
    callback = GlobalStateCallback(trainer)
    callback.dcp_every_n_epochs = 1
    state = TrainerState(global_step=3, epoch=0)
    callback.save_global_state(state, require_optimizer_checkpoint=True)
    path = layout.extra_state_path(layout.step_dir(str(tmp_path), 3), trainer.args.train.global_rank)
    before = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint._last_saved_optimizer_generation = "second"
    callback.on_epoch_end(state)
    after = torch.load(path, map_location="cpu", weights_only=False)
    assert after["_veomni_optimizer_generations"] == {"model": "second"}
    assert torch.equal(after["torch_rng_state"], before["torch_rng_state"])


def test_hf_step_dcp_epoch_do_not_rewrite_a_completed_checkpoint(tmp_path):
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.args.train.checkpoint.save_steps = 0
    trainer.args.train.checkpoint.save_epochs = 1
    trainer.args.train.checkpoint.save_hf_weights = True
    trainer.args.train.checkpoint.hf_save_steps = 3
    trainer.args.train.checkpoint.hf_save_epochs = 0
    trainer.model.optimizer = SimpleNamespace(_is_multi_optimizer=True)
    checkpoint = trainer.model.checkpoint = SimpleNamespace(
        _last_saved_step=-1, _last_saved_optimizer_generation=None, _loaded_optimizer_generation=None
    )
    trainer.train_dataloader = None
    trainer.data_iterator = None
    trainer.environ_meter.state_dict.return_value = {}

    def export(state, stage="step_end"):
        checkpoint._last_saved_step = state.global_step
        checkpoint._last_saved_optimizer_generation = "export-generation"

    trainer.save_dcp.side_effect = export
    model_callback = CheckpointCallback(trainer)
    global_callback = GlobalStateCallback(trainer)
    state = TrainerState(global_step=3, epoch=0)
    model_callback.on_step_end(state)
    global_callback.on_step_end(state)
    model_callback.on_epoch_end(state)
    global_callback.on_epoch_end(state)
    trainer.save_dcp.assert_called_once_with(state)
    payload = torch.load(
        layout.extra_state_path(layout.step_dir(str(tmp_path), 3), trainer.args.train.global_rank),
        map_location="cpu",
        weights_only=False,
    )
    assert payload["_veomni_optimizer_generations"] == {"model": "export-generation"}


def test_hf_export_without_dcp_receipt_does_not_suppress_epoch_dcp(tmp_path):
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.args.train.checkpoint.save_steps = 0
    trainer.args.train.checkpoint.save_epochs = 1
    trainer.args.train.checkpoint.save_hf_weights = True
    trainer.args.train.checkpoint.hf_save_steps = 3
    trainer.args.train.checkpoint.hf_save_epochs = 0
    trainer.model.checkpoint = SimpleNamespace(_last_saved_step=3)
    callback = CheckpointCallback(trainer)
    state = TrainerState(global_step=3, epoch=0)
    callback.on_step_end(state)
    callback.on_epoch_end(state)
    trainer.save_dcp.assert_called_once_with(state)


def test_legacy_export_confirms_dcp_and_suppresses_epoch_rewrite(tmp_path):
    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime
    from veomni.models.checkpoint.manager import ModelCheckpointManager

    runtime = _make_mock_runtime(save_path=str(tmp_path))
    runtime.optimizer = MagicMock(spec=[])
    with (
        patch("veomni.models.checkpoint.manager.build_checkpointer"),
        patch("veomni.models.checkpoint.manager.dist.barrier"),
    ):
        manager = ModelCheckpointManager(runtime)
        manager.wait_for_pending_save = MagicMock()
        manager.save_dcp = MagicMock(wraps=manager.save_dcp)
        trainer = _make_mock_trainer(save_path=str(tmp_path))
        trainer.args.train.checkpoint.save_steps = 0
        trainer.args.train.checkpoint.save_epochs = 1
        trainer.args.train.checkpoint.hf_save_steps = 3
        trainer.args.train.checkpoint.hf_save_epochs = 0
        trainer.model.checkpoint = manager
        trainer.save_dcp.side_effect = manager.save_dcp
        trainer.save_hf_or_lora.side_effect = lambda state, **kwargs: manager.prepare_export(state, "step_end")
        callback = CheckpointCallback(trainer)
        state = TrainerState(global_step=3, epoch=0)
        callback.on_step_end(state)
        callback.on_epoch_end(state)
        manager.save_dcp.assert_called_once_with(state)
        trainer.save_dcp.assert_called_once_with(state)
        assert manager._last_legacy_export_dcp_step == 3


def test_hf_only_epoch_then_empty_epoch_preserves_real_dcp(tmp_path):
    import hashlib

    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.models.checkpoint.manager import ModelCheckpointManager

    runtime = _make_mock_runtime(save_path=str(tmp_path))
    runtime.model = torch.nn.Linear(2, 2)
    runtime.optimizer = torch.optim.AdamW(runtime.model.parameters())
    runtime.lr_scheduler = None
    runtime.parallel_state = SimpleNamespace(global_rank=0, dp_mode="ddp", extra_parallel_names=[])
    runtime.model(torch.ones(1, 2)).sum().backward()
    runtime.optimizer.step()
    with (
        patch("veomni.models.checkpoint.manager.build_checkpointer", return_value=DistributedCheckpointer),
        patch("veomni.models.checkpoint.manager.dist.barrier"),
    ):
        manager = ModelCheckpointManager(runtime)
        runtime.checkpoint = manager
        trainer = _make_mock_trainer(save_path=str(tmp_path))
        trainer.model = runtime
        trainer.train_dataloader = None
        trainer.data_iterator = None
        trainer.environ_meter.state_dict.return_value = {}
        trainer.channel_loss_callback = None
        trainer.args.train.checkpoint.save_epochs = 2
        trainer.args.train.checkpoint.hf_save_epochs = 1
        trainer.save_dcp.side_effect = manager.save_dcp
        trainer.save_hf_or_lora.side_effect = lambda state, stage: manager.prepare_export(state, stage)
        callback = CheckpointCallback(trainer)
        global_callback = GlobalStateCallback(trainer)
        state = TrainerState(global_step=1, epoch=0)
        callback.on_epoch_end(state)
        global_callback.on_epoch_end(state)
        snapshot = {
            p.relative_to(tmp_path): hashlib.sha256(p.read_bytes()).digest()
            for p in tmp_path.rglob("*")
            if p.is_file()
        }
        callback.on_epoch_end(TrainerState(global_step=1, epoch=1))
        global_callback.on_epoch_end(TrainerState(global_step=1, epoch=1))
        trainer.save_dcp.assert_not_called()
        assert snapshot == {
            p.relative_to(tmp_path): hashlib.sha256(p.read_bytes()).digest()
            for p in tmp_path.rglob("*")
            if p.is_file()
        }
        restored = _make_mock_runtime(save_path=str(tmp_path))
        restored.model = torch.nn.Linear(2, 2)
        restored.optimizer = torch.optim.AdamW(restored.model.parameters())
        restored.lr_scheduler = None
        restored.parallel_state = runtime.parallel_state
        restored.train_args.checkpoint.load_path = layout.step_dir(str(tmp_path), 1)
        ModelCheckpointManager(restored).load()
        for expected, actual in zip(runtime.model.parameters(), restored.model.parameters()):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    runtime.optimizer.state[expected][key], restored.optimizer.state[actual][key], rtol=0, atol=0
                )


def test_unconfirmed_or_multi_optimizer_export_does_not_create_legacy_receipt(tmp_path):
    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime
    from veomni.models.checkpoint.manager import ModelCheckpointManager

    for mode in ("failed_wait", "multi"):
        runtime = _make_mock_runtime(save_path=str(tmp_path))
        runtime.optimizer = MagicMock(spec=[])
        if mode == "multi":
            runtime.optimizer = SimpleNamespace(_is_multi_optimizer=True)
        with (
            patch("veomni.models.checkpoint.manager.build_checkpointer"),
            patch("veomni.models.checkpoint.manager.dist.barrier"),
        ):
            manager = ModelCheckpointManager(runtime)
            manager.checkpointer.save.return_value = "new-generation"
            manager.save_dcp = MagicMock(wraps=manager.save_dcp)
            manager.wait_for_pending_save = MagicMock(
                side_effect=RuntimeError("injected wait failure") if mode == "failed_wait" else None
            )
            if mode == "failed_wait":
                with pytest.raises(RuntimeError, match="injected wait failure"):
                    manager.prepare_export(TrainerState(global_step=3), "step_end")
            else:
                manager.prepare_export(TrainerState(global_step=3), "step_end")
            assert manager._last_legacy_export_dcp_step is None


@pytest.mark.parametrize("method", ["wait", "slot"])
def test_shared_async_failure_never_certifies_another_manager(tmp_path, method):
    from concurrent.futures import Future

    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.models.checkpoint.manager import ModelCheckpointManager

    failed = Future()
    failed.set_exception(OSError("injected shared async failure"))
    runtime_a = _make_mock_runtime(save_path=str(tmp_path))
    runtime_b = _make_mock_runtime(save_path=str(tmp_path))
    runtime_a.optimizer = MagicMock(spec=[])
    runtime_b.optimizer = MagicMock(spec=[])
    with (
        patch.object(DistributedCheckpointer, "_async_save_failed", False),
        patch.object(DistributedCheckpointer, "_save_futures", {"weights": failed}),
        patch.object(DistributedCheckpointer, "_async_process_groups", {}),
        patch("veomni.models.checkpoint.manager.build_checkpointer", return_value=DistributedCheckpointer),
        patch("veomni.models.checkpoint.manager.dist.is_initialized", return_value=False),
    ):
        a, b = ModelCheckpointManager(runtime_a), ModelCheckpointManager(runtime_b)
        with pytest.raises(OSError, match="injected shared async failure"):
            if method == "wait":
                a.wait_for_pending_save()
            else:
                DistributedCheckpointer._drain_slot("weights")
        assert DistributedCheckpointer._async_save_failed is True
        with pytest.raises(RuntimeError, match="restart from a complete checkpoint"):
            b.wait_for_pending_save()


def test_manager_accepts_only_proven_loaded_optimizer_receipt(tmp_path):
    import weakref

    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime
    from veomni.models.checkpoint.manager import ModelCheckpointManager

    runtime = _make_mock_runtime(save_path=str(tmp_path))
    runtime.optimizer = MagicMock()
    runtime.optimizer._is_multi_optimizer = True
    with patch("veomni.models.checkpoint.manager.build_checkpointer"):
        manager = ModelCheckpointManager(runtime)
        manager._loaded_optimizer_global_step = 3
        manager._loaded_optimizer_generation = "saved-generation"
        manager._loaded_optimizer_ref = weakref.ref(runtime.optimizer)
        with pytest.raises(RuntimeError, match="unproven"):
            manager.confirm_loaded_checkpoint_for_reuse(3, "different-generation")
        assert manager._last_saved_step == -1
        manager.confirm_loaded_checkpoint_for_reuse(3, "saved-generation")
        assert manager._last_saved_step == 3
        assert manager._last_saved_optimizer_ref() is runtime.optimizer
        assert manager._last_saved_optimizer_generation == "saved-generation"


def test_same_output_strict_sft_resume_binds_optimizer_for_train_end(tmp_path, monkeypatch):
    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime
    from veomni.models.checkpoint.manager import ModelCheckpointManager

    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.args.train.checkpoint.load_path = layout.step_dir(str(tmp_path), 3)
    trainer.args.train.checkpoint.save_hf_weights = True
    trainer.train_dataloader = None
    runtime = _make_mock_runtime(save_path=str(tmp_path))
    runtime.optimizer = MagicMock()
    runtime.optimizer._is_multi_optimizer = True
    with (
        patch("veomni.models.checkpoint.manager.build_checkpointer"),
        patch("veomni.models.checkpoint.manager.dist.barrier"),
    ):
        manager = ModelCheckpointManager(runtime)
        manager._loaded_optimizer_global_step = 3
        manager._loaded_optimizer_generation = "saved-generation"
        manager._loaded_checkpoint_path = trainer.args.train.checkpoint.load_path
        import weakref

        manager._loaded_optimizer_ref = weakref.ref(runtime.optimizer)
        runtime.checkpoint = manager
        trainer.model = runtime
        trainer.checkpoint_callback = CheckpointCallback(trainer)
        callback = GlobalStateCallback(trainer)
        monkeypatch.setattr(layout, "checkpoint_is_complete", lambda _: True)
        monkeypatch.setattr(
            layout, "read_manifest", lambda _: {"model_state": True, "global_step": 3, "world_size": 1}
        )
        monkeypatch.setattr(
            callback,
            "_read_current",
            lambda _: {
                "global_step": 3,
                "torch_rng_state": torch.get_rng_state(),
                "_veomni_optimizer_generations": {"model": "saved-generation"},
            },
        )
        monkeypatch.setattr(callback, "_restore_position", lambda _: None)
        callback.load_global_state()
        assert trainer.checkpoint_callback._last_dcp_step == 3
        assert manager._last_saved_step == 3
        assert manager._last_saved_optimizer_ref() is runtime.optimizer
        manager.prepare_export(TrainerState(global_step=3), "train_end")
        manager.checkpointer.save.assert_not_called()
        assert manager._last_saved_optimizer_generation == "saved-generation"
        assert runtime.optimizer is None


def test_same_output_path_comparison_covers_uri_and_symlink(tmp_path):
    source = tmp_path / "actual"
    source.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    assert layout.same_checkpoint_path(str(alias) + "/", str(source))
    assert layout.same_checkpoint_path("hdfs://cluster/a/./b/", "hdfs://cluster/a/b")


@pytest.mark.parametrize("stage", ["epoch", "train_end"])
@pytest.mark.parametrize("incomplete", ["none", "manifest", "rng", "rng_conflict"])
def test_legacy_same_output_resume_never_rewrites_completed_dcp(tmp_path, stage, incomplete):
    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime, _make_mock_trainer
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.models.checkpoint.manager import ModelCheckpointManager

    def runtime(load_path=None):
        obj = _make_mock_runtime(save_path=str(tmp_path))
        obj.model = torch.nn.Linear(2, 2)
        obj.optimizer = torch.optim.AdamW(obj.model.parameters(), lr=0.01)
        obj.lr_scheduler = None
        obj.parallel_state = SimpleNamespace(global_rank=0, dp_mode="ddp")
        obj.train_args.checkpoint.load_path = load_path
        return obj

    def trainer(obj, manager, load_path=None):
        result = _make_mock_trainer(save_path=str(tmp_path))
        result.args.train.checkpoint.load_path = load_path
        result.args.train.checkpoint.save_epochs = 1
        result.args.train.checkpoint.save_hf_weights = False
        result.args.train_steps = 2
        result.train_dataloader = None
        result.data_iterator = None
        result.environ_meter.state_dict.return_value = {}
        result.channel_loss_callback = None
        result.offline_cache_writer = None
        obj.checkpoint = manager
        result.model = obj
        result.save_dcp.side_effect = manager.save_dcp
        return result

    step = TrainerState(global_step=2, epoch=0)
    with (
        patch("veomni.models.checkpoint.manager.build_checkpointer", return_value=DistributedCheckpointer),
        patch("torch.distributed.barrier"),
    ):
        source = runtime()
        for param in source.model.parameters():
            param.grad = torch.ones_like(param)
        source.optimizer.step()
        manager = ModelCheckpointManager(source)
        manager.save_dcp(step)
        GlobalStateCallback(trainer(source, manager)).save_global_state(step)
        load_path = layout.step_dir(str(tmp_path), 2)
        assert layout.checkpoint_is_complete(load_path)
        if incomplete == "manifest":
            os.remove(layout.manifest_path(load_path))
        elif incomplete in ("rng", "rng_conflict"):
            state_path = layout.extra_state_path(load_path, 0)
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            if incomplete == "rng":
                state.pop("torch_rng_state")
            else:
                alternate = torch.Generator().manual_seed(0).get_state()
                if torch.equal(alternate, state["torch_rng_state"]):
                    alternate = torch.Generator().manual_seed(1).get_state()
                state["device_rng_state"] = alternate
            torch.save(state, state_path)
        target = runtime(load_path)
        restored = ModelCheckpointManager(target)
        restored.load()
        resumed = trainer(target, restored, load_path)
        resumed.checkpoint_callback = CheckpointCallback(resumed)
        cursor = GlobalStateCallback(resumed)
        if incomplete != "none":
            with pytest.raises(RuntimeError, match="Cannot safely reuse"):
                cursor.load_global_state()
            assert restored._last_saved_step == -1
            assert resumed.checkpoint_callback._last_dcp_step == -1
            assert cursor._last_saved_step == -1
            return
        cursor.load_global_state()
        with patch.object(DistributedCheckpointer, "_create_storage_writer", side_effect=OSError("must not rewrite")):
            if stage == "epoch":
                resumed.checkpoint_callback.on_epoch_end(step)
            else:
                restored.prepare_export(step, "train_end")
        assert layout.checkpoint_is_complete(load_path)
        resumed.save_dcp.assert_not_called()


def test_reloading_manager_cannot_reuse_or_overwrite_previous_step(tmp_path):
    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.models.checkpoint.manager import ModelCheckpointManager

    def build(path, value):
        runtime = _make_mock_runtime(save_path=str(path))
        runtime.model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            runtime.model.weight.fill_(value)
        runtime.optimizer = torch.optim.AdamW(runtime.model.parameters())
        runtime.lr_scheduler = None
        runtime.parallel_state = SimpleNamespace(global_rank=0, dp_mode="ddp", extra_parallel_names=[])
        manager = ModelCheckpointManager(runtime)
        manager.checkpointer = DistributedCheckpointer
        return runtime, manager

    with patch("veomni.models.checkpoint.manager.dist.barrier"):
        step = TrainerState(global_step=1)
        source, manager = build(tmp_path / "a", 3.0)
        other, other_manager = build(tmp_path / "b", 7.0)
        manager.save_dcp(step)
        manager.save_dcp(TrainerState(global_step=2))
        other_manager.save_dcp(step)
        a_dir = layout.step_dir(str(tmp_path / "a"), 1)
        b_dir = layout.step_dir(str(tmp_path / "b"), 1)
        trainer_state = layout.extra_state_path(a_dir, 0)
        os.makedirs(os.path.dirname(trainer_state), exist_ok=True)
        torch.save({"global_step": 1, "source": "a"}, trainer_state)
        with open(trainer_state, "rb") as handle:
            original_sidecar = handle.read()
        manager.config.load_path = b_dir
        manager.load()
        assert source.model.weight.item() == 7.0
        assert manager._last_saved_step == -1
        # A later valid save must not accidentally reauthorize the old path.
        manager.save_dcp(TrainerState(global_step=3))
        with pytest.raises(RuntimeError, match="Cannot overwrite a checkpoint after loading"):
            manager.prepare_export(step, "step_end")
        with open(trainer_state, "rb") as handle:
            assert handle.read() == original_sidecar
        restored, restored_manager = build(tmp_path / "other", -1.0)
        restored_manager.config.load_path = a_dir
        restored_manager.load()
        assert restored.model.weight.item() == 3.0


@pytest.mark.parametrize("mode", ["no_path", "load_failure", "wait_failure"])
def test_failed_or_absent_reload_invalidates_old_manager_receipt(tmp_path, mode):
    import weakref

    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.models.checkpoint.manager import ModelCheckpointManager

    runtime = _make_mock_runtime(save_path=str(tmp_path))
    runtime.optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    runtime.train_args.checkpoint.load_path = None if mode == "no_path" else str(tmp_path / "source")
    with patch("veomni.models.checkpoint.manager.build_checkpointer", return_value=DistributedCheckpointer):
        manager = ModelCheckpointManager(runtime)
    manager._last_saved_step = 3
    manager._last_saved_step_path = layout.step_dir(str(tmp_path), 3)
    manager._saved_step_paths = [layout.step_dir(str(tmp_path), 3)]
    manager._last_saved_optimizer_ref = weakref.ref(runtime.optimizer)
    manager._last_export_released_optimizer = True
    manager._loaded_optimizer_global_step = 3
    manager._loaded_optimizer_generation = "old"
    with (
        patch.object(
            DistributedCheckpointer,
            "wait_for_pending_save",
            side_effect=OSError("wait failed") if mode == "wait_failure" else None,
        ),
        patch.object(
            DistributedCheckpointer, "load", side_effect=OSError("load failed") if mode == "load_failure" else None
        ),
    ):
        if mode == "no_path":
            manager.load()
        else:
            with pytest.raises(OSError, match="wait failed|load failed"):
                manager.load()
    assert manager._last_saved_step == -1
    assert manager._last_saved_optimizer_ref is None
    assert not manager._last_export_released_optimizer
    assert manager._loaded_optimizer_global_step is None
    assert manager._loaded_optimizer_generation is None
    assert manager._invalidated_saved_step_paths == [layout.step_dir(str(tmp_path), 3)]


def test_trainer_load_discards_old_callback_receipts(tmp_path):
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.args.train.checkpoint.load_path = None
    trainer.checkpoint_callback = CheckpointCallback(trainer)
    trainer.checkpoint_callback._last_dcp_step = 3
    trainer.checkpoint_callback._last_hf_step = 3
    callback = GlobalStateCallback(trainer)
    callback._last_saved_step = 3
    assert callback.load_global_state() is None
    assert callback._last_saved_step == -1
    assert trainer.checkpoint_callback._last_dcp_step == -1
    assert trainer.checkpoint_callback._last_hf_step == -1


def test_legacy_receipt_rejects_unknown_checkpoint_provider(tmp_path):
    import weakref

    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime
    from veomni.models.checkpoint.manager import ModelCheckpointManager

    class NoOpCheckpointer:
        pass

    runtime = _make_mock_runtime(save_path=str(tmp_path))
    runtime.optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    load_path = layout.step_dir(str(tmp_path), 2)
    with patch("veomni.models.checkpoint.manager.build_checkpointer", return_value=NoOpCheckpointer):
        manager = ModelCheckpointManager(runtime)
    manager._loaded_optimizer_ref = weakref.ref(runtime.optimizer)
    manager._loaded_checkpoint_path = load_path
    manager._loaded_checkpointer = NoOpCheckpointer
    with pytest.raises(RuntimeError, match="unproven legacy"):
        manager.confirm_legacy_loaded_checkpoint_for_reuse(2, load_path)
    assert manager._last_saved_step == -1


def test_unproven_legacy_can_use_a_different_output_directory(tmp_path):
    trainer = _make_mock_trainer(save_path=str(tmp_path / "new_output"))
    callback = GlobalStateCallback(trainer)
    checkpoint = SimpleNamespace(_last_saved_step=-1)
    callback._confirm_same_output_checkpoint(
        2,
        str(tmp_path / "old_output" / "global_step_2"),
        [],
        {},
        [("model", checkpoint)],
        [],
        False,
    )
    assert checkpoint._last_saved_step == -1
    assert callback._last_saved_step == -1


def test_same_output_unknown_provider_cannot_be_certified(tmp_path, monkeypatch):
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.args.train.checkpoint.load_path = layout.step_dir(str(tmp_path), 2)
    trainer.train_dataloader = None
    trainer.model = SimpleNamespace(optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))]), checkpoint=None)
    callback = GlobalStateCallback(trainer)
    monkeypatch.setattr(
        callback, "_read_current", lambda _: {"global_step": 2, "torch_rng_state": torch.get_rng_state()}
    )
    monkeypatch.setattr(callback, "_restore_position", lambda _: None)
    with pytest.raises(RuntimeError, match="cannot confirm optimizer identity"):
        callback.load_global_state()


@pytest.mark.parametrize("mode", ["different", "same", "symlink", "symlink_dotdot"])
def test_weights_only_resume_does_not_overwrite_source_step_zero(tmp_path, monkeypatch, mode):
    save_root = tmp_path / ("new" if mode == "different" else "old")
    trainer = _make_mock_trainer(save_path=str(save_root))
    source = tmp_path / "old" / "global_step_0"
    if mode == "symlink":
        source.mkdir(parents=True)
        alias = tmp_path / "alias"
        alias.symlink_to(source, target_is_directory=True)
        source = alias
    elif mode == "symlink_dotdot":
        nested = tmp_path / "old" / "nested"
        nested.mkdir(parents=True)
        alias = tmp_path / "alias"
        alias.symlink_to(nested, target_is_directory=True)
        source = alias / ".." / "global_step_0"
    trainer.args.train.checkpoint.load_path = str(source)
    trainer.model = SimpleNamespace(optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))]), checkpoint=None)
    callback = GlobalStateCallback(trainer)
    monkeypatch.setattr(callback, "_read_current", lambda _: None)
    with patch("veomni.checkpoint.legacy_v0_1_12.apply_legacy_global_state", return_value=None):
        if mode != "different":
            with pytest.raises(RuntimeError, match="without trainer state"):
                callback.load_global_state()
        else:
            assert callback.load_global_state() is None


def test_failed_load_barrier_cannot_certify_legacy_receipt(tmp_path):
    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.models.checkpoint.manager import ModelCheckpointManager

    runtime = _make_mock_runtime(save_path=str(tmp_path))
    runtime.optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    load_path = layout.step_dir(str(tmp_path), 2)
    runtime.train_args.checkpoint.load_path = load_path
    with (
        patch("veomni.models.checkpoint.manager.build_checkpointer", return_value=DistributedCheckpointer),
        patch.object(DistributedCheckpointer, "load"),
        patch("torch.distributed.barrier", side_effect=RuntimeError("barrier failed")),
    ):
        manager = ModelCheckpointManager(runtime)
        with pytest.raises(RuntimeError, match="barrier failed"):
            manager.load()
    assert manager._loaded_optimizer_ref is None
    assert manager._loaded_checkpoint_path is None
    with pytest.raises(RuntimeError, match="unproven legacy"):
        manager.confirm_legacy_loaded_checkpoint_for_reuse(2, load_path)


@pytest.mark.parametrize("aggregate_case", ["exact", "missing", "extra", "different", "untrusted_proxy"])
def test_legacy_omni_aggregate_uses_real_module_owners(tmp_path, monkeypatch, aggregate_case):
    import weakref

    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.models.checkpoint.manager import ModelCheckpointManager
    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import (
        MultiOptimizer as OmniMultiOptimizer,
    )
    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import (
        OmniModelRuntime,
    )
    from veomni.trainer.callbacks.global_state_callback import _legacy_optimizer_checkpoints

    runtime = _make_mock_runtime(save_path=str(tmp_path))
    runtime.optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    load_path = layout.step_dir(str(tmp_path), 2)
    with patch("veomni.models.checkpoint.manager.build_checkpointer", return_value=DistributedCheckpointer):
        manager = ModelCheckpointManager(runtime)
    manager._loaded_optimizer_ref = weakref.ref(runtime.optimizer)
    manager._loaded_checkpoint_path = load_path
    manager._loaded_checkpointer = DistributedCheckpointer
    runtime.checkpoint = manager
    runtime.has_trainable_parameters = True
    present = runtime
    frozen = SimpleNamespace(optimizer=None, checkpoint=None, has_trainable_parameters=False)
    optimizers = {"trainable": runtime.optimizer}
    if aggregate_case == "missing":
        optimizers = {}
    elif aggregate_case == "extra":
        optimizers["unowned"] = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    elif aggregate_case == "different":
        optimizers["trainable"] = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    modules = {"trainable": present, "frozen": frozen}
    if aggregate_case == "untrusted_proxy":
        root = SimpleNamespace(
            optimizer=SimpleNamespace(optimizers=optimizers), checkpoint=None, module_runtimes=modules
        )
    else:
        root = OmniModelRuntime(SimpleNamespace(), module_runtimes=modules)
        root.optimizer = OmniMultiOptimizer(optimizers)
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.model = root
    trainer.checkpoint_callback = CheckpointCallback(trainer)
    known, unknown = _legacy_optimizer_checkpoints(trainer)
    assert [name for name, _ in known] == ["module:trainable"]
    assert unknown == ([] if aggregate_case == "exact" else ["model"])
    callback = GlobalStateCallback(trainer)
    monkeypatch.setattr(layout, "checkpoint_is_complete", lambda _: True)
    monkeypatch.setattr(
        layout,
        "read_manifest",
        lambda _: {"model_state": True, "global_step": 2, "world_size": trainer.args.train.world_size},
    )
    if aggregate_case == "exact":
        callback._confirm_same_output_checkpoint(2, load_path, [], {}, known, unknown)
        assert manager._last_saved_step == 2
        assert manager._last_saved_optimizer_ref() is runtime.optimizer
    else:
        with pytest.raises(RuntimeError, match="cannot confirm optimizer identity"):
            callback._confirm_same_output_checkpoint(2, load_path, [], {}, known, unknown)
        assert manager._last_saved_step == -1


@pytest.mark.parametrize("shared_optimizer", [False, True])
def test_legacy_omni_shared_checkpoint_manager_does_not_prove_other_owner(tmp_path, shared_optimizer):
    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.models.checkpoint.manager import ModelCheckpointManager
    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import (
        MultiOptimizer as OmniMultiOptimizer,
    )
    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import (
        OmniModelRuntime,
    )
    from veomni.trainer.callbacks.global_state_callback import _legacy_optimizer_checkpoints

    first = _make_mock_runtime(save_path=str(tmp_path))
    second = _make_mock_runtime(save_path=str(tmp_path))
    first.optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    second.optimizer = first.optimizer if shared_optimizer else torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    with patch("veomni.models.checkpoint.manager.build_checkpointer", return_value=DistributedCheckpointer):
        manager = ModelCheckpointManager(first)
    first.checkpoint = second.checkpoint = manager
    root = OmniModelRuntime(SimpleNamespace(), module_runtimes={"a": first, "b": second})
    root.optimizer = OmniMultiOptimizer({"a": first.optimizer, "b": second.optimizer})
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.model = root
    known, unknown = _legacy_optimizer_checkpoints(trainer)
    assert [name for name, _ in known] == ["module:a"]
    assert unknown == ["module:b"]


def test_legacy_omni_missing_trainable_provider_is_not_certified(tmp_path, monkeypatch):
    import weakref

    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.models.checkpoint.manager import ModelCheckpointManager
    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import (
        MultiOptimizer as OmniMultiOptimizer,
    )
    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import (
        OmniModelRuntime,
    )
    from veomni.trainer.callbacks.global_state_callback import _legacy_optimizer_checkpoints

    runtime = _make_mock_runtime(save_path=str(tmp_path))
    runtime.optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    load_path = layout.step_dir(str(tmp_path), 2)
    with patch("veomni.models.checkpoint.manager.build_checkpointer", return_value=DistributedCheckpointer):
        manager = ModelCheckpointManager(runtime)
    manager._loaded_optimizer_ref = weakref.ref(runtime.optimizer)
    manager._loaded_checkpoint_path = load_path
    manager._loaded_checkpointer = DistributedCheckpointer
    runtime.checkpoint = manager
    runtime.has_trainable_parameters = True
    present = runtime
    missing = SimpleNamespace(optimizer=None, checkpoint=None, has_trainable_parameters=True)
    frozen = SimpleNamespace(optimizer=None, checkpoint=None, has_trainable_parameters=False)
    root = OmniModelRuntime(SimpleNamespace(), module_runtimes={"yes": present, "missing": missing, "frozen": frozen})
    root.optimizer = OmniMultiOptimizer({"yes": runtime.optimizer})
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.model = root
    known, unknown = _legacy_optimizer_checkpoints(trainer)
    assert [name for name, _ in known] == ["module:yes"]
    assert unknown == ["module:missing"]
    callback = GlobalStateCallback(trainer)
    monkeypatch.setattr(layout, "checkpoint_is_complete", lambda _: True)
    monkeypatch.setattr(
        layout,
        "read_manifest",
        lambda _: {"model_state": True, "global_step": 2, "world_size": trainer.args.train.world_size},
    )
    with pytest.raises(RuntimeError, match="cannot confirm optimizer identity"):
        callback._confirm_same_output_checkpoint(2, load_path, [], {}, known, unknown)
    assert manager._last_saved_step == -1


def test_legacy_omni_aggregate_optimizer_uses_child_checkpoint_managers(tmp_path, monkeypatch):
    import weakref

    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.models.checkpoint.manager import ModelCheckpointManager
    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import OmniModelRuntime
    from veomni.trainer.callbacks.global_state_callback import _legacy_optimizer_checkpoints

    child = _make_mock_runtime(save_path=str(tmp_path))
    child.optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    with patch("veomni.models.checkpoint.manager.build_checkpointer", return_value=DistributedCheckpointer):
        child.checkpoint = ModelCheckpointManager(child)
    root = OmniModelRuntime(model=None, module_runtimes={"child": child})
    root._build_optimizer()
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.model = root

    known, unknown = _legacy_optimizer_checkpoints(trainer)
    assert known == [("module:child", child.checkpoint)]
    assert unknown == []
    load_path = layout.step_dir(str(tmp_path), 2)
    child.checkpoint._loaded_optimizer_ref = weakref.ref(child.optimizer)
    child.checkpoint._loaded_checkpoint_path = load_path
    child.checkpoint._loaded_checkpointer = DistributedCheckpointer
    trainer.checkpoint_callback = CheckpointCallback(trainer)
    monkeypatch.setattr(layout, "checkpoint_is_complete", lambda _: True)
    monkeypatch.setattr(layout, "read_manifest", lambda _: {"model_state": True, "global_step": 2, "world_size": 1})
    GlobalStateCallback(trainer)._confirm_same_output_checkpoint(2, load_path, [], {}, known, unknown)
    assert child.checkpoint._last_saved_step == 2

    # Do not certify an aggregate whose members are not exactly the child
    # optimizers: its additional state would have no checkpoint manager.
    root.optimizer.optimizers["unexpected"] = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    known, unknown = _legacy_optimizer_checkpoints(trainer)
    assert known == [("module:child", child.checkpoint)]
    assert unknown == ["model"]

    other = _make_mock_runtime(save_path=str(tmp_path))
    other.optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    other.checkpoint = child.checkpoint
    root.module_runtimes["other"] = other
    root._build_optimizer()
    known, unknown = _legacy_optimizer_checkpoints(trainer)
    assert known == [("module:child", child.checkpoint)]
    assert unknown == ["module:other"]

    # Sharing both objects still does not prove the other model's state.
    other.optimizer = child.optimizer
    root._build_optimizer()
    known, unknown = _legacy_optimizer_checkpoints(trainer)
    assert known == [("module:child", child.checkpoint)]
    assert unknown == ["module:other"]


def test_mixed_omni_receipt_confirms_every_optimizer(tmp_path, monkeypatch):
    import weakref

    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.models.checkpoint.manager import ModelCheckpointManager

    load_path = layout.step_dir(str(tmp_path), 2)
    strict_runtime = _make_mock_runtime(save_path=str(tmp_path))
    strict_runtime.optimizer = MagicMock()
    strict_runtime.optimizer._is_multi_optimizer = True
    legacy_runtime = _make_mock_runtime(save_path=str(tmp_path))
    legacy_runtime.optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    with patch("veomni.models.checkpoint.manager.build_checkpointer", return_value=DistributedCheckpointer):
        strict = ModelCheckpointManager(strict_runtime)
        legacy = ModelCheckpointManager(legacy_runtime)
    strict._loaded_optimizer_ref = weakref.ref(strict_runtime.optimizer)
    strict._loaded_optimizer_global_step = 2
    strict._loaded_optimizer_generation = "test-generation"
    strict._loaded_checkpoint_path = load_path
    legacy._loaded_optimizer_ref = weakref.ref(legacy_runtime.optimizer)
    legacy._loaded_checkpoint_path = load_path
    legacy._loaded_checkpointer = DistributedCheckpointer
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.checkpoint_callback = CheckpointCallback(trainer)
    callback = GlobalStateCallback(trainer)
    monkeypatch.setattr(layout, "checkpoint_is_complete", lambda _: True)
    monkeypatch.setattr(
        layout,
        "read_manifest",
        lambda _: {"model_state": True, "global_step": 2, "world_size": trainer.args.train.world_size},
    )
    callback._confirm_same_output_checkpoint(
        2, load_path, [("module:strict", strict)], {"module:strict": "test-generation"}, [("module:legacy", legacy)]
    )
    assert strict._last_saved_step == legacy._last_saved_step == 2
    assert strict._last_saved_optimizer_ref() is strict_runtime.optimizer
    assert legacy._last_saved_optimizer_ref() is legacy_runtime.optimizer
    assert trainer.checkpoint_callback._last_dcp_step == 2
    assert not layout.same_checkpoint_path("hdfs://one/a", "hdfs://two/a")


def test_same_output_path_resolves_symlink_before_dotdot(tmp_path):
    output = tmp_path / "output"
    source = tmp_path / "source"
    (source / "inner").mkdir(parents=True)
    output.mkdir()
    (output / "alias").symlink_to(source / "inner", target_is_directory=True)
    (output / "global_step_3").mkdir()
    (source / "global_step_3").mkdir()
    assert not layout.same_checkpoint_path(
        str(output / "alias" / ".." / "global_step_3"), str(output / "global_step_3")
    )
