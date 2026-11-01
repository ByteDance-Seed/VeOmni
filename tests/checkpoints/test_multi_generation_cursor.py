# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""The trainer cursor must belong to the same optimizer checkpoint generation."""

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

    trainer.save_hf_or_lora.side_effect = export
    model_callback = CheckpointCallback(trainer)
    global_callback = GlobalStateCallback(trainer)
    state = TrainerState(global_step=3, epoch=0)
    model_callback.on_step_end(state)
    global_callback.on_step_end(state)
    model_callback.on_epoch_end(state)
    global_callback.on_epoch_end(state)
    trainer.save_dcp.assert_not_called()
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
        trainer.save_hf_or_lora.side_effect = lambda state, **kwargs: manager.prepare_export(state, "step_end")
        callback = CheckpointCallback(trainer)
        state = TrainerState(global_step=3, epoch=0)
        callback.on_step_end(state)
        callback.on_epoch_end(state)
        manager.save_dcp.assert_called_once_with(state)
        trainer.save_dcp.assert_not_called()
        assert manager._last_legacy_export_dcp_step == 3


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


def test_failed_export_wait_requires_new_dcp_before_retry(tmp_path):
    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime
    from veomni.models.checkpoint.manager import ModelCheckpointManager

    runtime = _make_mock_runtime(save_path=str(tmp_path), save_async=True)
    runtime.optimizer = MagicMock(spec=[])
    with (
        patch("veomni.models.checkpoint.manager.build_checkpointer"),
        patch("veomni.models.checkpoint.manager.dist.barrier"),
    ):
        manager = ModelCheckpointManager(runtime)
        manager.save_dcp = MagicMock(wraps=manager.save_dcp)
        manager.wait_for_pending_save = MagicMock(side_effect=[RuntimeError("injected wait failure"), None])
        state = TrainerState(global_step=3)
        with pytest.raises(RuntimeError, match="injected wait failure"):
            manager.prepare_export(state, "step_end")
        assert manager._last_saved_step == -1
        assert manager._last_legacy_export_dcp_step is None
        manager.prepare_export(state, "step_end")
        assert manager.save_dcp.call_count == 2
        assert manager._last_legacy_export_dcp_step == 3


@pytest.mark.parametrize("before,after", [("none", "optimizer"), ("optimizer", "other"), ("optimizer", "none")])
def test_legacy_export_does_not_reuse_dcp_with_changed_optimizer(tmp_path, before, after):
    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime
    from veomni.models.checkpoint.manager import ModelCheckpointManager
    from veomni.trainer.callbacks.checkpoint_callback import _export_wrote_dcp

    runtime = _make_mock_runtime(save_path=str(tmp_path))
    original = MagicMock(spec=[])
    replacement = MagicMock(spec=[])
    runtime.optimizer = None if before == "none" else original
    with (
        patch("veomni.models.checkpoint.manager.build_checkpointer"),
        patch("veomni.models.checkpoint.manager.dist.barrier"),
    ):
        manager = ModelCheckpointManager(runtime)
        manager.save_dcp(TrainerState(global_step=3))
        runtime.optimizer = {"optimizer": original, "other": replacement, "none": None}[after]
        manager.prepare_export(TrainerState(global_step=3), "step_end")
        if after == "none":
            assert manager._last_legacy_export_dcp_step is None
            assert not _export_wrote_dcp(SimpleNamespace(model=SimpleNamespace(checkpoint=manager)), 3)
        else:
            assert manager.checkpointer.save.call_count == 2
            assert manager._last_saved_optimizer_ref() is runtime.optimizer
            assert _export_wrote_dcp(SimpleNamespace(model=SimpleNamespace(checkpoint=manager)), 3)


def test_omni_frozen_runtimes_do_not_hide_or_fabricate_receipts():
    from veomni.trainer.callbacks.checkpoint_callback import _export_wrote_dcp

    confirmed = SimpleNamespace(checkpoint=SimpleNamespace(_last_saved_step=3, _last_legacy_export_dcp_step=3))
    frozen = SimpleNamespace(checkpoint=None, has_trainable_parameters=False)
    unknown = SimpleNamespace(checkpoint=None, has_trainable_parameters=True)
    failed = SimpleNamespace(checkpoint=SimpleNamespace(_last_saved_step=3, _last_legacy_export_dcp_step=None))

    def trainer(*modules):
        return SimpleNamespace(model=SimpleNamespace(module_runtimes=dict(enumerate(modules))))

    assert _export_wrote_dcp(trainer(confirmed, frozen), 3)
    assert not _export_wrote_dcp(trainer(confirmed, unknown), 3)
    assert not _export_wrote_dcp(trainer(confirmed, failed), 3)
    assert not _export_wrote_dcp(trainer(frozen), 3)


@pytest.mark.parametrize("method", ["wait", "slot"])
def test_shared_async_failure_never_certifies_another_manager(tmp_path, method):
    from concurrent.futures import Future

    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.models.checkpoint.manager import ModelCheckpointManager
    from veomni.trainer.callbacks.checkpoint_callback import _export_wrote_dcp

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
        b._last_saved_step = 3
        b._last_legacy_export_dcp_step = 3
        with pytest.raises(OSError, match="injected shared async failure"):
            if method == "wait":
                a.wait_for_pending_save()
            else:
                DistributedCheckpointer._drain_slot("weights")
        assert DistributedCheckpointer._async_save_failed is True
        assert not _export_wrote_dcp(SimpleNamespace(model=SimpleNamespace(checkpoint=b)), 3)
        with pytest.raises(RuntimeError, match="restart from a complete checkpoint"):
            b.wait_for_pending_save()


def test_strict_generation_not_reused_after_optimizer_replacement(tmp_path):
    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime
    from veomni.models.checkpoint.manager import ModelCheckpointManager
    from veomni.trainer.callbacks.checkpoint_callback import _export_wrote_dcp

    runtime = _make_mock_runtime(save_path=str(tmp_path))
    first, second = MagicMock(), MagicMock()
    first._is_multi_optimizer = second._is_multi_optimizer = True
    runtime.optimizer = first
    with (
        patch("veomni.models.checkpoint.manager.build_checkpointer"),
        patch("veomni.models.checkpoint.manager.dist.barrier"),
    ):
        manager = ModelCheckpointManager(runtime)
        manager.save_dcp(TrainerState(global_step=3))
        manager._last_saved_optimizer_generation = "first-generation"
        runtime.optimizer = second
        manager.prepare_export(TrainerState(global_step=3), "step_end")
        assert manager.checkpointer.save.call_count == 2
        assert manager._last_saved_optimizer_ref() is second
        assert manager._last_saved_optimizer_generation is None
        assert not _export_wrote_dcp(SimpleNamespace(model=SimpleNamespace(checkpoint=manager)), 3)


def test_legacy_export_does_not_certify_ambiguous_multi_flag(tmp_path):
    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime
    from veomni.models.checkpoint.manager import ModelCheckpointManager
    from veomni.trainer.callbacks.checkpoint_callback import _export_wrote_dcp

    runtime = _make_mock_runtime(save_path=str(tmp_path))
    runtime.optimizer = MagicMock()
    runtime.optimizer._is_multi_optimizer = 1
    with (
        patch("veomni.models.checkpoint.manager.build_checkpointer"),
        patch("veomni.models.checkpoint.manager.dist.barrier"),
    ):
        manager = ModelCheckpointManager(runtime)
        manager.prepare_export(TrainerState(global_step=3), "step_end")
        assert manager._last_legacy_export_dcp_step is None
        assert not _export_wrote_dcp(SimpleNamespace(model=SimpleNamespace(checkpoint=manager)), 3)
