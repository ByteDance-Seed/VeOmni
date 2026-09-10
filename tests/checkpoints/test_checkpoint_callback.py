"""Unit tests for checkpoint cadence, manager save contract, and job-level state.

Validates that ``_last_saved_step`` is only updated AFTER the save succeeds, that
DCP save keys staging on the run-root path plus ``global_steps``, that extra_state
is model-bound only, and that job-level state lives on ``GlobalStateCallback``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from veomni.models.checkpoint_manager import ModelCheckpointManager
from veomni.trainer.callbacks.base import TrainerState
from veomni.trainer.callbacks.checkpoint_callback import (
    CheckpointCallback,
)
from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback


def _make_mock_trainer(save_path="/tmp/test_ckpt", save_async=False):
    """Build a minimal mock trainer for CheckpointCallback / manager tests."""
    checkpoint_cfg = SimpleNamespace(
        save_path=save_path,
        save_steps=5,
        save_epochs=1,
        save_async=save_async,
        load_path=None,
        manager="dcp",
        dcp_save_to_lowest_rank=False,
        stage_dir=None,
        save_hf_weights=True,
        hf_save_steps=5,
        hf_save_epochs=1,
        model_assets_dir="/tmp/assets",
        output_dir="/tmp/output",
    )
    fsdp_config = SimpleNamespace(fsdp_mode="fsdp2")
    accelerator = SimpleNamespace(fsdp_config=fsdp_config)
    train_cfg = SimpleNamespace(
        checkpoint=checkpoint_cfg,
        global_rank=0,
    )
    model_cfg = SimpleNamespace(fqn_to_index_mapping={}, accelerator=accelerator, lora_config=None)
    args = SimpleNamespace(train=train_cfg, model=model_cfg, train_steps=100)

    trainer = MagicMock()
    trainer.args = args
    trainer.model = MagicMock()
    trainer.optimizer = MagicMock()
    trainer.lr_scheduler = MagicMock()
    trainer.lr_scheduler.state_dict.return_value = {"lr": 1e-4}
    trainer.train_dataloader = MagicMock()
    trainer.environ_meter = MagicMock()
    trainer.channel_loss_callback = MagicMock()
    trainer.channel_loss_callback.state_dict.return_value = {}
    trainer.model_assets = []
    trainer.state = TrainerState()
    trainer.start_epoch = 0
    trainer.start_step = 0
    trainer.checkpoint = MagicMock()

    return trainer


@patch("veomni.trainer.callbacks.checkpoint_callback.helper")
class TestCheckpointCallbackDcpLastSavedStep:
    """Tests for CheckpointCallback DCP _last_dcp_step placement."""

    def test_last_saved_step_updated_after_successful_save(self, mock_helper):
        trainer = _make_mock_trainer()
        cb = CheckpointCallback(trainer)
        state = TrainerState(global_step=10)

        assert cb._last_dcp_step == -1
        cb._save_dcp(state)
        assert cb._last_dcp_step == 10

    def test_last_saved_step_not_updated_on_save_failure(self, mock_helper):
        trainer = _make_mock_trainer()
        trainer.save_dcp.side_effect = RuntimeError("disk full")
        cb = CheckpointCallback(trainer)
        state = TrainerState(global_step=10)

        with pytest.raises(RuntimeError, match="disk full"):
            cb._save_dcp(state)
        assert cb._last_dcp_step == -1

    def test_the_dcp_save_carries_no_job_level_state(self, mock_helper):
        """Job state has its own writer; a model checkpoint only holds the model."""
        trainer = _make_mock_trainer()
        cb = CheckpointCallback(trainer)

        cb._save_dcp(TrainerState(global_step=10))

        trainer.save_dcp.assert_called_once()
        assert trainer.save_dcp.call_args.args == (TrainerState(global_step=10),)
        assert not trainer.save_dcp.call_args.kwargs

    def test_epoch_end_retries_after_failed_save(self, mock_helper):
        """If save fails at step_end, epoch_end should still attempt to save (not skip)."""
        trainer = _make_mock_trainer()
        trainer.args.train.checkpoint.save_hf_weights = False
        cb = CheckpointCallback(trainer)
        cb.dcp_every_n_steps = 5
        cb.dcp_every_n_epochs = 1

        state = TrainerState(global_step=5, epoch=0)

        trainer.save_dcp.side_effect = RuntimeError("disk full")
        with pytest.raises(RuntimeError):
            cb.on_step_end(state)
        assert cb._last_dcp_step == -1

        trainer.save_dcp.side_effect = None
        trainer.save_dcp.reset_mock()

        cb.on_epoch_end(state)
        assert trainer.save_dcp.call_count == 1
        assert cb._last_dcp_step == 5

    def test_epoch_end_skips_after_successful_step_save(self, mock_helper):
        """If save succeeds at step_end, epoch_end should skip duplicate save."""
        trainer = _make_mock_trainer()
        trainer.args.train.checkpoint.save_hf_weights = False
        cb = CheckpointCallback(trainer)
        cb.dcp_every_n_steps = 5
        cb.dcp_every_n_epochs = 1

        state = TrainerState(global_step=5, epoch=0)

        cb.on_step_end(state)
        assert cb._last_dcp_step == 5

        trainer.save_dcp.reset_mock()
        cb.on_epoch_end(state)
        trainer.save_dcp.assert_not_called()


@patch("veomni.trainer.callbacks.checkpoint_callback.helper")
class TestCheckpointCallbackHfLastSavedStep:
    """Tests for CheckpointCallback HF _last_hf_step placement."""

    def test_last_saved_step_updated_after_successful_hf_save(self, mock_helper):
        trainer = _make_mock_trainer()
        cb = CheckpointCallback(trainer)
        state = TrainerState(global_step=10)

        assert cb._last_hf_step == -1
        cb._save_hf(state)
        assert cb._last_hf_step == 10

    def test_last_saved_step_not_updated_on_hf_save_failure(self, mock_helper):
        trainer = _make_mock_trainer()
        trainer.save_hf_or_lora.side_effect = RuntimeError("conversion failed")
        cb = CheckpointCallback(trainer)
        state = TrainerState(global_step=10)

        with pytest.raises(RuntimeError, match="conversion failed"):
            cb._save_hf(state)
        assert cb._last_hf_step == -1

    def test_train_end_retries_after_failed_hf_save(self, mock_helper):
        """If HF save fails at step_end, train_end should still attempt to save."""
        trainer = _make_mock_trainer()
        trainer.args.train.checkpoint.save_steps = 0
        cb = CheckpointCallback(trainer)
        cb.dcp_every_n_steps = 0
        cb.hf_every_n_steps = 5

        state = TrainerState(global_step=5, epoch=0)

        trainer.save_hf_or_lora.side_effect = RuntimeError("conversion failed")
        with pytest.raises(RuntimeError):
            cb.on_step_end(state)
        assert cb._last_hf_step == -1

        trainer.save_hf_or_lora.side_effect = None
        trainer.save_hf_or_lora.reset_mock()

        cb.on_train_end(state)
        assert trainer.save_hf_or_lora.call_count == 1
        assert cb._last_hf_step == 5

    def test_train_end_skips_after_successful_step_save(self, mock_helper):
        """If HF save succeeds at step_end, train_end should skip."""
        trainer = _make_mock_trainer()
        trainer.args.train.checkpoint.save_steps = 0
        cb = CheckpointCallback(trainer)
        cb.dcp_every_n_steps = 0
        cb.hf_every_n_steps = 5

        state = TrainerState(global_step=5, epoch=0)

        cb.on_step_end(state)
        assert cb._last_hf_step == 5

        trainer.save_hf_or_lora.reset_mock()
        cb.on_train_end(state)
        trainer.save_hf_or_lora.assert_not_called()


@patch("veomni.trainer.callbacks.checkpoint_callback.helper")
class TestCheckpointCallbackTrainEndWait:
    """CheckpointCallback.on_train_end must consume a pending async save."""

    def test_train_end_waits_for_pending_async_save(self, mock_helper):
        trainer = _make_mock_trainer(save_async=True)
        trainer.args.train.checkpoint.save_hf_weights = False
        cb = CheckpointCallback(trainer)

        cb.on_train_end(TrainerState(global_step=60))

        trainer.checkpoint.wait_for_pending_save.assert_called_once_with()

    def test_train_end_propagates_async_save_failure(self, mock_helper):
        trainer = _make_mock_trainer(save_async=True)
        trainer.args.train.checkpoint.save_hf_weights = False
        trainer.checkpoint.wait_for_pending_save.side_effect = RuntimeError("HDFS write failed")
        cb = CheckpointCallback(trainer)

        with pytest.raises(RuntimeError, match="HDFS write failed"):
            cb.on_train_end(TrainerState(global_step=60))

    def test_train_end_waits_even_without_async(self, mock_helper):
        """The call is unconditional; wait_for_pending_save is a no-op when nothing is pending."""
        trainer = _make_mock_trainer(save_async=False)
        trainer.args.train.checkpoint.save_hf_weights = False
        cb = CheckpointCallback(trainer)

        cb.on_train_end(TrainerState(global_step=60))

        trainer.checkpoint.wait_for_pending_save.assert_called_once_with()


@patch("veomni.models.checkpoint_manager.get_parallel_state")
@patch("veomni.models.checkpoint_manager.build_checkpointer")
@patch("veomni.models.checkpoint_manager.dist")
@patch("veomni.models.checkpoint_manager.helper")
class TestModelCheckpointManagerSaveContract:
    """``stage_dir`` keys its staging directory on the ``path`` given to ``save``.

    That path must name the run, not the step. A caller that folds the step in
    gets a fresh staging directory per step, and a save killed part-way then
    strands a model-plus-optimizer-sized copy that no later save clears.

    extra_state is model-bound only: the dataloader cursor, rng, and meters
    belong to ``GlobalStateCallback``.
    """

    def test_the_step_reaches_save_instead_of_being_folded_into_the_path(
        self, mock_helper, mock_dist, mock_build_ckpt, mock_get_ps, tmp_path
    ):
        from veomni.checkpoint.dcp_checkpointer import _prepare_stage_dir

        trainer = _make_mock_trainer(save_path=str(tmp_path / "run"))
        trainer.args.train.checkpoint.stage_dir = str(tmp_path / "stage")
        mock_build_ckpt.return_value = MagicMock()
        manager = ModelCheckpointManager(trainer)
        trainer.checkpoint = manager

        staged = []
        with patch("veomni.checkpoint.dcp_checkpointer._any_rank_failed", return_value=False):
            for step in (10, 20):
                manager.save_dcp(TrainerState(global_step=step))
                call = manager.checkpointer.save.call_args
                assert call.kwargs["global_steps"] == step
                assert call.kwargs["stage_dir"] == str(tmp_path / "stage")
                staged.append(_prepare_stage_dir(call.kwargs["stage_dir"], call.args[0]))

        assert staged[0] == staged[1], "each step staged somewhere different"

    def test_the_logged_destination_is_the_one_save_writes(self, mock_helper, mock_dist, mock_build_ckpt, mock_get_ps):
        """The manager names the step directory for its log and its HF export, while
        ``save`` builds the same directory from ``path`` and ``global_steps``."""
        from veomni.checkpoint.dcp_checkpointer import _GLOBAL_STEP_PREFIX

        trainer = _make_mock_trainer(save_path="/remote/run")
        mock_build_ckpt.return_value = MagicMock()
        manager = ModelCheckpointManager(trainer)

        manager.save_dcp(TrainerState(global_step=10))

        call = manager.checkpointer.save.call_args
        assert f"{call.args[0]}/{_GLOBAL_STEP_PREFIX}{call.kwargs['global_steps']}" == "/remote/run/global_step_10"

    def test_extra_state_is_only_the_scheduler(self, mock_helper, mock_dist, mock_build_ckpt, mock_get_ps):
        trainer = _make_mock_trainer()
        mock_build_ckpt.return_value = MagicMock()
        manager = ModelCheckpointManager(trainer)

        manager.save_dcp(TrainerState(global_step=10))

        extra_state = manager.checkpointer.save.call_args.args[1]["extra_state"]
        assert set(extra_state) == {"lr_scheduler"}
        assert extra_state["lr_scheduler"] == {"lr": 1e-4}

    def test_legacy_extra_state_restores_the_job_cursor(self, mock_helper, mock_dist, mock_build_ckpt, mock_get_ps):
        """Checkpoints written by CheckpointerCallback still resume the cursor."""
        trainer = _make_mock_trainer()
        trainer.args.train.checkpoint.load_path = "/tmp/old_ckpt"
        trainer.train_dataloader = MagicMock()
        mock_checkpointer = MagicMock()
        mock_build_ckpt.return_value = mock_checkpointer

        def load_checkpoint(path, state, **kwargs):
            state["extra_state"] = {
                "global_step": 7,
                "lr_scheduler": {"lr": 1e-5},
                "train_dataloader": {"cursor": 3},
                "environ_meter": {"tokens": 1},
                "channel_loss_callback": {"source_registry": [(1, "train/a")]},
                "torch_rng_state": torch.get_rng_state(),
            }

        mock_checkpointer.load.side_effect = load_checkpoint
        manager = ModelCheckpointManager(trainer)
        manager.load()

        assert trainer.state.global_step == 7
        assert trainer.start_epoch == 0
        assert trainer.start_step == 7
        trainer.lr_scheduler.load_state_dict.assert_called_once_with({"lr": 1e-5})
        trainer.train_dataloader.load_state_dict.assert_called_once_with({"cursor": 3})
        trainer.channel_loss_callback.load_state_dict.assert_called_once_with({"source_registry": [(1, "train/a")]})
        assert mock_checkpointer.load.call_args.kwargs["parallel_state"] is mock_get_ps.return_value


@patch("veomni.trainer.callbacks.global_state_callback.dist")
class TestGlobalStateCallbackJobState:
    """Job-level state — dataloader, rng, meters, channel-loss — is not in DCP extra_state."""

    def test_state_dict_includes_channel_loss_callback_state(self, mock_dist):
        trainer = _make_mock_trainer()
        trainer.channel_loss_callback.state_dict.return_value = {
            "source_registry": [(1, "train/a")],
        }
        cb = GlobalStateCallback(trainer)

        global_state = cb.state_dict(TrainerState(global_step=10))

        assert global_state["channel_loss_callback"] == {"source_registry": [(1, "train/a")]}
        assert "global_step" in global_state
        assert "train_dataloader" in global_state
        assert "environ_meter" in global_state
        assert "torch_rng_state" in global_state

    def test_load_restores_channel_loss_callback_state(self, mock_dist, tmp_path):
        trainer = _make_mock_trainer()
        trainer.args.train.checkpoint.load_path = str(tmp_path)
        trainer.args.train.global_rank = 0
        trainer.train_dataloader = None
        callback_state = {"source_registry": [(1, "train/a")]}
        payload = {
            "global_step": 7,
            "train_dataloader": None,
            "environ_meter": {},
            "channel_loss_callback": callback_state,
            "torch_rng_state": torch.get_rng_state(),
        }
        torch.save(payload, tmp_path / "trainer_state_rank_0.pt")

        cb = GlobalStateCallback(trainer)
        cb.load_global_state()

        trainer.channel_loss_callback.load_state_dict.assert_called_once_with(callback_state)
        assert trainer.state.global_step == 7
