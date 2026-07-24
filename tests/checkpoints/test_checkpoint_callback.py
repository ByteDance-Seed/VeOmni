"""Unit tests for checkpoint callback _last_saved_step correctness.

Validates that _last_saved_step is only updated AFTER the save operation
succeeds, so that a failed save does not suppress future retry attempts.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from veomni.trainer.callbacks.base import TrainerState
from veomni.trainer.callbacks.checkpoint_callback import (
    CheckpointerCallback,
    HuggingfaceCkptCallback,
)


def _make_mock_trainer(save_path="/tmp/test_ckpt", save_async=False):
    """Build a minimal mock trainer for CheckpointerCallback tests."""
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
        accelerator=accelerator,
        global_rank=0,
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
    trainer.channel_loss_callback = MagicMock()
    trainer.channel_loss_callback.state_dict.return_value = {}
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

    def test_save_includes_channel_loss_callback_state(self, mock_helper, mock_dist, mock_build_ckpt):
        trainer = _make_mock_trainer()
        trainer.channel_loss_callback.state_dict.return_value = {
            "source_registry": [(1, "train/a")],
        }
        mock_build_ckpt.return_value = trainer.checkpointer
        cb = CheckpointerCallback(trainer)

        cb._save_checkpoint(TrainerState(global_step=10))

        checkpoint_state = trainer.checkpointer.save.call_args.args[1]
        assert checkpoint_state["extra_state"]["channel_loss_callback"] == {"source_registry": [(1, "train/a")]}

    def test_load_restores_channel_loss_callback_state(self, mock_helper, mock_dist, mock_build_ckpt):
        trainer = _make_mock_trainer()
        trainer.args.train.checkpoint.load_path = "/tmp/test_ckpt/global_step_7"
        trainer.args.train_steps = 100
        trainer.state = TrainerState()
        mock_build_ckpt.return_value = trainer.checkpointer
        callback_state = {"source_registry": [(1, "train/a")]}

        def load_checkpoint(path, state, **kwargs):
            state["extra_state"] = {
                "global_step": 7,
                "lr_scheduler": {},
                "train_dataloader": None,
                "environ_meter": {},
                "channel_loss_callback": callback_state,
                "torch_rng_state": torch.get_rng_state(),
            }

        trainer.checkpointer.load.side_effect = load_checkpoint
        cb = CheckpointerCallback(trainer)

        cb._load_checkpoint()

        trainer.channel_loss_callback.load_state_dict.assert_called_once_with(callback_state)

    def test_save_includes_bucket_batch_sampler_state(self, mock_helper, mock_dist, mock_build_ckpt):
        """P4: BucketBatchSampler cursors + epochs land in extra_state on save."""
        trainer = _make_mock_trainer()
        sampler_state = {
            "version": 1,
            "seed": 42,
            "dp_rank": 0,
            "dp_size": 4,
            "micro_batch_size": 4,
            "num_micro_batch": 2,
            "global_step": 17,
            "epochs": {0: 1, 1: 0},
            "cursors": {0: 8, 1: 12},
        }
        trainer.bucket_batch_sampler = MagicMock()
        trainer.bucket_batch_sampler.state_dict.return_value = sampler_state
        mock_build_ckpt.return_value = trainer.checkpointer
        cb = CheckpointerCallback(trainer)

        cb._save_checkpoint(TrainerState(global_step=17))

        checkpoint_state = trainer.checkpointer.save.call_args.args[1]
        assert checkpoint_state["extra_state"]["bucket_batch_sampler"] == sampler_state

    def test_load_restores_bucket_batch_sampler_state(self, mock_helper, mock_dist, mock_build_ckpt):
        """P4: on resume the callback calls sampler.load_state_dict with the saved state."""
        trainer = _make_mock_trainer()
        trainer.args.train.checkpoint.load_path = "/tmp/test_ckpt/global_step_17"
        trainer.args.train_steps = 100
        trainer.state = TrainerState()

        sampler_state = {
            "version": 1,
            "seed": 42,
            "dp_rank": 0,
            "dp_size": 4,
            "micro_batch_size": 4,
            "num_micro_batch": 2,
            "global_step": 17,
            "epochs": {0: 1, 1: 0},
            "cursors": {0: 8, 1: 12},
        }
        trainer.bucket_batch_sampler = MagicMock()
        trainer.bucket_batch_sampler.global_step = 17

        def load_checkpoint(path, state, **kwargs):
            state["extra_state"] = {
                "global_step": 17,
                "lr_scheduler": {},
                "train_dataloader": None,
                "environ_meter": {},
                "channel_loss_callback": None,
                "bucket_batch_sampler": sampler_state,
                "torch_rng_state": torch.get_rng_state(),
            }

        trainer.checkpointer.load.side_effect = load_checkpoint
        mock_build_ckpt.return_value = trainer.checkpointer
        cb = CheckpointerCallback(trainer)

        cb._load_checkpoint()

        trainer.bucket_batch_sampler.load_state_dict.assert_called_once_with(sampler_state)

    def test_manifest_identity_includes_bucket_indexer_fingerprint(self, mock_helper, mock_dist, mock_build_ckpt):
        """P4: ``bucket_indexer_fingerprint`` gets hard-gated via extra_hashes."""
        trainer = _make_mock_trainer()
        trainer.bucket_indexer_fingerprint = "deadbeef1234"
        # No bucket_scheduler on this trainer to keep the assertion clean.
        del trainer.bucket_scheduler
        mock_build_ckpt.return_value = trainer.checkpointer
        cb = CheckpointerCallback(trainer)

        extra, _soft = cb._manifest_identity()
        assert extra.get("bucket_indexer_fingerprint") == "deadbeef1234"

    def test_manifest_identity_omits_bucket_indexer_fingerprint_when_absent(
        self, mock_helper, mock_dist, mock_build_ckpt
    ):
        """Non-HI3 trainers (or HI3 with ``same_bucket_batching=False``) never
        set ``bucket_indexer_fingerprint``; the manifest must stay generic."""
        trainer = _make_mock_trainer()
        # Ensure the getattr fallback in _manifest_identity returns None.
        if hasattr(trainer, "bucket_indexer_fingerprint"):
            del trainer.bucket_indexer_fingerprint
        del trainer.bucket_scheduler  # keep extra dict minimal
        mock_build_ckpt.return_value = trainer.checkpointer
        cb = CheckpointerCallback(trainer)

        extra, _soft = cb._manifest_identity()
        assert "bucket_indexer_fingerprint" not in extra

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
