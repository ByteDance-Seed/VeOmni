"""Unit tests for checkpoint callback _last_saved_step correctness.

Validates that _last_saved_step is only updated AFTER the save operation
succeeds, so that a failed save does not suppress future retry attempts.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from veomni.trainer.callbacks.base import TrainerState
from veomni.trainer.callbacks.checkpoint_callback import (
    ModelDcpCallback,
    ModelHfCallback,
)


def _make_mock_trainer(save_path="/tmp/test_ckpt", save_async=False):
    """Build a minimal mock trainer for ModelDcpCallback tests."""
    checkpoint_cfg = SimpleNamespace(
        save_path=save_path,
        save_steps=5,
        save_epochs=1,
        save_async=save_async,
        load_path=None,
        manager="dcp",
        dcp_save_to_lowest_rank=False,
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
    model_cfg = SimpleNamespace(fqn_to_index_mapping={}, accelerator=accelerator)
    args = SimpleNamespace(train=train_cfg, model=model_cfg)

    trainer = MagicMock()
    trainer.args = args
    # The trainer fans out to its model handles; these callbacks only decide when.
    trainer.model = MagicMock()
    trainer.train_dataloader = MagicMock()
    trainer.environ_meter = MagicMock()
    trainer.channel_loss_callback = MagicMock()
    trainer.channel_loss_callback.state_dict.return_value = {}

    return trainer


@patch("veomni.trainer.callbacks.checkpoint_callback.helper")
class TestModelDcpCallbackLastSavedStep:
    """Tests for ModelDcpCallback._last_saved_step placement."""

    def test_last_saved_step_updated_after_successful_save(self, mock_helper):
        trainer = _make_mock_trainer()
        cb = ModelDcpCallback(trainer)
        state = TrainerState(global_step=10)

        assert cb._last_saved_step == -1
        cb._save_checkpoint(state)
        assert cb._last_saved_step == 10

    def test_last_saved_step_not_updated_on_save_failure(self, mock_helper):
        trainer = _make_mock_trainer()
        trainer.save_dcp.side_effect = RuntimeError("disk full")
        cb = ModelDcpCallback(trainer)
        state = TrainerState(global_step=10)

        with pytest.raises(RuntimeError, match="disk full"):
            cb._save_checkpoint(state)
        assert cb._last_saved_step == -1

    def test_the_dcp_save_carries_no_job_level_state(self, mock_helper):
        """Job state has its own writer; a model checkpoint only holds the model.

        With several models in one job there is one dataloader cursor but N of
        these checkpoints, so a cursor in here would be written N times over.
        """
        trainer = _make_mock_trainer()
        cb = ModelDcpCallback(trainer)

        cb._save_checkpoint(TrainerState(global_step=10))

        trainer.save_dcp.assert_called_once()
        assert trainer.save_dcp.call_args.args == (TrainerState(global_step=10),)
        assert not trainer.save_dcp.call_args.kwargs

    def test_epoch_end_retries_after_failed_save(self, mock_helper):
        """If save fails at step_end, epoch_end should still attempt to save (not skip)."""
        trainer = _make_mock_trainer()
        cb = ModelDcpCallback(trainer)
        cb.every_n_steps = 5
        cb.every_n_epochs = 1

        state = TrainerState(global_step=5, epoch=0)

        # Simulate save failure at step_end
        trainer.save_dcp.side_effect = RuntimeError("disk full")
        with pytest.raises(RuntimeError):
            cb.on_step_end(state)
        assert cb._last_saved_step == -1

        # Now the disk is available again
        trainer.save_dcp.side_effect = None
        trainer.save_dcp.reset_mock()

        # epoch_end should NOT skip because _last_saved_step was not updated
        cb.on_epoch_end(state)
        assert trainer.save_dcp.call_count == 1
        assert cb._last_saved_step == 5

    def test_epoch_end_skips_after_successful_step_save(self, mock_helper):
        """If save succeeds at step_end, epoch_end should skip duplicate save."""
        trainer = _make_mock_trainer()
        cb = ModelDcpCallback(trainer)
        cb.every_n_steps = 5
        cb.every_n_epochs = 1

        state = TrainerState(global_step=5, epoch=0)

        cb.on_step_end(state)
        assert cb._last_saved_step == 5

        trainer.save_dcp.reset_mock()
        cb.on_epoch_end(state)
        # Should skip — no new save call
        trainer.save_dcp.assert_not_called()


@patch("veomni.trainer.callbacks.checkpoint_callback.helper")
class TestModelHfCallbackLastSavedStep:
    """Tests for ModelHfCallback._last_saved_step placement."""

    def test_last_saved_step_updated_after_successful_hf_save(self, mock_helper):
        trainer = _make_mock_trainer()
        cb = ModelHfCallback(trainer)
        state = TrainerState(global_step=10)

        assert cb._last_saved_step == -1
        cb._save_checkpoint(state)
        assert cb._last_saved_step == 10

    def test_last_saved_step_not_updated_on_hf_save_failure(self, mock_helper):
        trainer = _make_mock_trainer()
        trainer.save_hf_or_lora.side_effect = RuntimeError("conversion failed")
        cb = ModelHfCallback(trainer)
        state = TrainerState(global_step=10)

        with pytest.raises(RuntimeError, match="conversion failed"):
            cb._save_checkpoint(state)
        assert cb._last_saved_step == -1

    def test_train_end_retries_after_failed_hf_save(self, mock_helper):
        """If HF save fails at step_end, train_end should still attempt to save."""
        trainer = _make_mock_trainer()
        cb = ModelHfCallback(trainer)
        cb.every_n_steps = 5

        state = TrainerState(global_step=5, epoch=0)

        # Simulate HF save failure at step_end
        trainer.save_hf_or_lora.side_effect = RuntimeError("conversion failed")
        with pytest.raises(RuntimeError):
            cb.on_step_end(state)
        assert cb._last_saved_step == -1

        # Now the save works
        trainer.save_hf_or_lora.side_effect = None
        trainer.save_hf_or_lora.reset_mock()

        # train_end should NOT skip because _last_saved_step was not updated
        cb.on_train_end(state)
        assert trainer.save_hf_or_lora.call_count == 1
        assert cb._last_saved_step == 5

    def test_train_end_skips_after_successful_step_save(self, mock_helper):
        """If HF save succeeds at step_end, train_end should skip."""
        trainer = _make_mock_trainer()
        cb = ModelHfCallback(trainer)
        cb.every_n_steps = 5

        state = TrainerState(global_step=5, epoch=0)

        cb.on_step_end(state)
        assert cb._last_saved_step == 5

        trainer.save_hf_or_lora.reset_mock()
        cb.on_train_end(state)
        trainer.save_hf_or_lora.assert_not_called()
