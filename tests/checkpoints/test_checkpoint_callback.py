"""Unit tests for checkpoint callback _last_saved_step correctness.

Validates that _last_saved_step is only updated AFTER the save operation
succeeds, so that a failed save does not suppress future retry attempts.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from veomni.trainer.callbacks.base import TrainerState
from veomni.trainer.callbacks.checkpoint_callback import (
    CheckpointerCallback,
    HuggingfaceCkptCallback,
)


pytestmark = pytest.mark.unit


def _make_mock_trainer(save_path="/tmp/test_ckpt", save_async=False):
    """Build a minimal mock trainer for CheckpointerCallback tests."""
    checkpoint_cfg = SimpleNamespace(
        save_path=save_path,
        save_steps=5,
        save_epochs=1,
        save_async=save_async,
        load_path=None,
        manager="dcp",
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
        accelerator=accelerator,
        global_rank=0,
        train_architecture="text",
    )
    model_cfg = SimpleNamespace(fqn_to_index_mapping={})
    args = SimpleNamespace(train=train_cfg, model=model_cfg)

    trainer = MagicMock()
    trainer.args = args
    trainer.model = MagicMock()
    trainer.optimizer = MagicMock()
    trainer.lr_scheduler = MagicMock()
    trainer.train_dataloader = MagicMock()
    trainer.environ_meter = MagicMock()
    trainer.checkpointer = MagicMock()
    trainer.checkpointer.save_future = None
    trainer.model_assets = []

    return trainer


@patch("veomni.trainer.callbacks.checkpoint_callback.build_checkpointer")
@patch("veomni.trainer.callbacks.checkpoint_callback.dist")
@patch("veomni.trainer.callbacks.checkpoint_callback.helper")
class TestCheckpointerCallbackLastSavedStep:
    """Tests for CheckpointerCallback._last_saved_step placement."""

    def test_last_saved_step_updated_after_successful_save(self, mock_helper, mock_dist, mock_build_ckpt):
        trainer = _make_mock_trainer()
        mock_build_ckpt.return_value = trainer.checkpointer
        cb = CheckpointerCallback(trainer)
        state = TrainerState(global_step=10)

        assert cb._last_saved_step == -1
        cb._save_checkpoint(state)
        assert cb._last_saved_step == 10

    def test_last_saved_step_not_updated_on_save_failure(self, mock_helper, mock_dist, mock_build_ckpt):
        trainer = _make_mock_trainer()
        mock_build_ckpt.return_value = trainer.checkpointer
        trainer.checkpointer.save.side_effect = RuntimeError("disk full")
        cb = CheckpointerCallback(trainer)
        state = TrainerState(global_step=10)

        with pytest.raises(RuntimeError, match="disk full"):
            cb._save_checkpoint(state)
        assert cb._last_saved_step == -1

    def test_epoch_end_retries_after_failed_save(self, mock_helper, mock_dist, mock_build_ckpt):
        """If save fails at step_end, epoch_end should still attempt to save (not skip)."""
        trainer = _make_mock_trainer()
        mock_build_ckpt.return_value = trainer.checkpointer
        cb = CheckpointerCallback(trainer)
        cb.every_n_steps = 5
        cb.every_n_epochs = 1

        state = TrainerState(global_step=5, epoch=0)

        # Simulate save failure at step_end
        trainer.checkpointer.save.side_effect = RuntimeError("disk full")
        with pytest.raises(RuntimeError):
            cb.on_step_end(state)
        assert cb._last_saved_step == -1

        # Now the disk is available again
        trainer.checkpointer.save.side_effect = None
        trainer.checkpointer.save.reset_mock()

        # epoch_end should NOT skip because _last_saved_step was not updated
        cb.on_epoch_end(state)
        assert trainer.checkpointer.save.call_count == 1
        assert cb._last_saved_step == 5

    def test_epoch_end_skips_after_successful_step_save(self, mock_helper, mock_dist, mock_build_ckpt):
        """If save succeeds at step_end, epoch_end should skip duplicate save."""
        trainer = _make_mock_trainer()
        mock_build_ckpt.return_value = trainer.checkpointer
        cb = CheckpointerCallback(trainer)
        cb.every_n_steps = 5
        cb.every_n_epochs = 1

        state = TrainerState(global_step=5, epoch=0)

        cb.on_step_end(state)
        assert cb._last_saved_step == 5

        trainer.checkpointer.save.reset_mock()
        cb.on_epoch_end(state)
        # Should skip — no new save call
        trainer.checkpointer.save.assert_not_called()


@patch("veomni.trainer.callbacks.checkpoint_callback.save_hf_safetensor")
@patch("veomni.trainer.callbacks.checkpoint_callback.build_checkpointer")
@patch("veomni.trainer.callbacks.checkpoint_callback.dist")
@patch("veomni.trainer.callbacks.checkpoint_callback.helper")
@patch("os.path.exists", return_value=True)
class TestHuggingfaceCkptCallbackLastSavedStep:
    """Tests for HuggingfaceCkptCallback._last_saved_step placement."""

    def test_last_saved_step_updated_after_successful_hf_save(
        self, mock_exists, mock_helper, mock_dist, mock_build_ckpt, mock_save_hf
    ):
        trainer = _make_mock_trainer()
        mock_build_ckpt.return_value = trainer.checkpointer
        cb = HuggingfaceCkptCallback(trainer)
        state = TrainerState(global_step=10)

        assert cb._last_saved_step == -1
        cb._save_checkpoint(state)
        assert cb._last_saved_step == 10

    def test_last_saved_step_not_updated_on_hf_save_failure(
        self, mock_exists, mock_helper, mock_dist, mock_build_ckpt, mock_save_hf
    ):
        trainer = _make_mock_trainer()
        mock_build_ckpt.return_value = trainer.checkpointer
        mock_save_hf.side_effect = RuntimeError("conversion failed")
        cb = HuggingfaceCkptCallback(trainer)
        state = TrainerState(global_step=10)

        with pytest.raises(RuntimeError, match="conversion failed"):
            cb._save_checkpoint(state)
        assert cb._last_saved_step == -1

    def test_train_end_retries_after_failed_hf_save(
        self, mock_exists, mock_helper, mock_dist, mock_build_ckpt, mock_save_hf
    ):
        """If HF save fails at step_end, train_end should still attempt to save."""
        trainer = _make_mock_trainer()
        mock_build_ckpt.return_value = trainer.checkpointer
        cb = HuggingfaceCkptCallback(trainer)
        cb.every_n_steps = 5

        state = TrainerState(global_step=5, epoch=0)

        # Simulate HF save failure at step_end
        mock_save_hf.side_effect = RuntimeError("conversion failed")
        with pytest.raises(RuntimeError):
            cb.on_step_end(state)
        assert cb._last_saved_step == -1

        # Now the save works
        mock_save_hf.side_effect = None
        mock_save_hf.reset_mock()

        # train_end should NOT skip because _last_saved_step was not updated
        cb.on_train_end(state)
        assert mock_save_hf.call_count == 1
        assert cb._last_saved_step == 5

    def test_train_end_skips_after_successful_step_save(
        self, mock_exists, mock_helper, mock_dist, mock_build_ckpt, mock_save_hf
    ):
        """If HF save succeeds at step_end, train_end should skip."""
        trainer = _make_mock_trainer()
        mock_build_ckpt.return_value = trainer.checkpointer
        cb = HuggingfaceCkptCallback(trainer)
        cb.every_n_steps = 5

        state = TrainerState(global_step=5, epoch=0)

        cb.on_step_end(state)
        assert cb._last_saved_step == 5

        mock_save_hf.reset_mock()
        cb.on_train_end(state)
        mock_save_hf.assert_not_called()
