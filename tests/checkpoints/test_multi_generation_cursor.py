# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""The trainer cursor must belong to the same optimizer checkpoint generation."""

from types import SimpleNamespace

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
