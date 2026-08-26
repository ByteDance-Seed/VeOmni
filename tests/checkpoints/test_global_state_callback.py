"""Unit tests for the job-level checkpoint callbacks.

Covers what :class:`GlobalStateCallback` persists, that it round-trips, and that
it keeps each rank's own copy — the dataloader cursor it carries is rank-local.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from veomni.trainer.callbacks.base import TrainerState
from veomni.trainer.callbacks.global_state_callback import (
    GlobalStateCallback,
    RootAssetsCallback,
    global_state_path,
)


def _make_mock_trainer(save_path, global_rank=0):
    checkpoint_cfg = SimpleNamespace(
        save_path=save_path,
        save_steps=5,
        save_epochs=1,
        load_path=None,
    )
    args = SimpleNamespace(
        train=SimpleNamespace(checkpoint=checkpoint_cfg, global_rank=global_rank),
        train_steps=100,
    )

    trainer = MagicMock()
    trainer.args = args
    trainer.state = TrainerState()
    trainer.start_epoch = 0
    trainer.start_step = 0
    trainer.train_dataloader.state_dict.return_value = {"cursor": global_rank}
    trainer.environ_meter.state_dict.return_value = {"tokens": 1}
    trainer.channel_loss_callback.state_dict.return_value = {}
    # A MagicMock would answer to ``data_iterator`` and shadow the dataloader.
    del trainer.data_iterator
    return trainer


def test_the_saved_state_carries_the_dataloader_cursor_and_rng(tmp_path):
    trainer = _make_mock_trainer(str(tmp_path))
    trainer.channel_loss_callback.state_dict.return_value = {"source_registry": [(1, "train/a")]}
    cb = GlobalStateCallback(trainer)

    global_state = cb.state_dict(TrainerState(global_step=10))

    assert global_state["global_step"] == 10
    assert global_state["train_dataloader"] == {"cursor": 0}
    assert global_state["environ_meter"] == {"tokens": 1}
    assert global_state["channel_loss_callback"] == {"source_registry": [(1, "train/a")]}
    assert "torch_rng_state" in global_state


def test_each_rank_writes_its_own_file(tmp_path):
    """Iterable datasets are ``dp_rank``-sharded, so one rank's cursor does not
    describe another's; a single shared file would resume every rank on rank 0's
    slice, replaying it and skipping the rest."""
    for rank in (0, 1):
        cb = GlobalStateCallback(_make_mock_trainer(str(tmp_path), global_rank=rank))
        cb.save_global_state(TrainerState(global_step=10))

    step_dir = tmp_path / "global_step_10"
    written = sorted(p.name for p in step_dir.iterdir())
    assert written == ["trainer_state_rank_0.pt", "trainer_state_rank_1.pt"]

    for rank in (0, 1):
        blob = torch.load(global_state_path(str(step_dir), rank), weights_only=False)
        assert blob["train_dataloader"] == {"cursor": rank}


def test_save_then_load_restores_the_job_state(tmp_path):
    saver = _make_mock_trainer(str(tmp_path))
    saver.channel_loss_callback.state_dict.return_value = {"source_registry": [(1, "train/a")]}
    GlobalStateCallback(saver).save_global_state(TrainerState(global_step=7))

    loader = _make_mock_trainer(str(tmp_path))
    loader.args.train.checkpoint.load_path = os.path.join(str(tmp_path), "global_step_7")
    cb = GlobalStateCallback(loader)

    cb.load_global_state()

    assert loader.state.global_step == 7
    loader.channel_loss_callback.load_state_dict.assert_called_once_with({"source_registry": [(1, "train/a")]})
    loader.train_dataloader.load_state_dict.assert_called_once_with({"cursor": 0})
    loader.environ_meter.load_state_dict.assert_called_once_with({"tokens": 1})


def test_a_missing_state_file_resumes_weights_only(tmp_path):
    """A crash between the model write and this one leaves the step without a
    trainer state; the weights are still worth resuming from."""
    trainer = _make_mock_trainer(str(tmp_path))
    trainer.args.train.checkpoint.load_path = str(tmp_path / "global_step_7")
    cb = GlobalStateCallback(trainer)

    assert cb.load_global_state() is None
    trainer.train_dataloader.load_state_dict.assert_not_called()


def test_position_is_derived_from_the_step_grid(tmp_path):
    trainer = _make_mock_trainer(str(tmp_path))
    trainer.args.train_steps = 30
    cb = GlobalStateCallback(trainer)

    cb._restore_position({"global_step": 70})

    assert (trainer.start_epoch, trainer.start_step) == (2, 10)


def test_root_assets_are_exported_once_at_train_begin(tmp_path):
    trainer = _make_mock_trainer(str(tmp_path))

    RootAssetsCallback(trainer).on_train_begin(TrainerState())

    trainer.model.save_model_assets.assert_called_once_with()
