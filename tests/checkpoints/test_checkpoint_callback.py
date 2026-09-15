"""Unit tests for checkpoint cadence, manager save contract, and job-level state.

Validates that ``_last_saved_step`` is only updated AFTER the save succeeds, that
DCP save keys staging on the run-root path plus ``global_steps``, that the
manager forwards ``lr_scheduler`` like the optimizer, and that job-level state
lives on ``GlobalStateCallback``.
"""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from veomni.checkpoint import layout
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
        world_size=1,
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
    # Single-model job: the real manager's class default, which keeps the
    # manifest's module list empty.
    trainer.checkpoint.module_name = ""

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
class TestCheckpointCallbackTrainBegin:
    """Sidecar export and DCP resume share on_train_begin; assets go first."""

    def test_on_train_begin_exports_assets_then_loads(self, mock_helper):
        trainer = _make_mock_trainer()
        order = []
        trainer.save_model_assets.side_effect = lambda: order.append("assets")
        trainer.load.side_effect = lambda: order.append("load")
        cb = CheckpointCallback(trainer)

        cb.on_train_begin(TrainerState())

        assert order == ["assets", "load"]
        mock_helper.empty_cache.assert_called_once_with()


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

    The manager forwards this model's lr_scheduler like the optimizer; the
    dataloader cursor, rng, and meters belong to ``GlobalStateCallback``.
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
        with patch("veomni.checkpoint.dcp_checkpointer.any_rank_failed", return_value=False):
            for step in (10, 20):
                manager.save_dcp(TrainerState(global_step=step))
                call = manager.checkpointer.save.call_args
                assert call.kwargs["global_steps"] == step
                assert call.kwargs["stage_dir"] == str(tmp_path / "stage")
                staged.append(_prepare_stage_dir(call.kwargs["stage_dir"], call.args[0]))

        assert staged[0] == staged[1], "each step staged somewhere different"

    def test_the_logged_destination_is_the_one_save_writes(self, mock_helper, mock_dist, mock_build_ckpt, mock_get_ps):
        """The manager names the model directory for its log, and ``save`` rebuilds
        the same one from ``path``, ``global_steps`` and ``module``."""
        from veomni.checkpoint.layout import model_dir, step_dir

        trainer = _make_mock_trainer(save_path="/remote/run")
        mock_build_ckpt.return_value = MagicMock()
        manager = ModelCheckpointManager(trainer)
        state = TrainerState(global_step=10)

        manager.save_dcp(state)

        call = manager.checkpointer.save.call_args
        rebuilt = model_dir(step_dir(call.args[0], call.kwargs["global_steps"]), call.kwargs["module"])
        assert rebuilt == "/remote/run/global_step_10/model"
        assert rebuilt == manager.save_dir(state)

    def test_save_forwards_lr_scheduler_like_optimizer(self, mock_helper, mock_dist, mock_build_ckpt, mock_get_ps):
        trainer = _make_mock_trainer()
        mock_build_ckpt.return_value = MagicMock()
        manager = ModelCheckpointManager(trainer)

        manager.save_dcp(TrainerState(global_step=10))

        saved = manager.checkpointer.save.call_args.args[1]
        assert saved["lr_scheduler"] is trainer.lr_scheduler
        assert saved["optimizer"] is trainer.optimizer
        assert "extra_state" not in saved

    def test_load_forwards_lr_scheduler_like_optimizer(self, mock_helper, mock_dist, mock_build_ckpt, mock_get_ps):
        trainer = _make_mock_trainer()
        trainer.args.train.checkpoint.load_path = "/tmp/ckpt"
        mock_checkpointer = MagicMock()
        mock_build_ckpt.return_value = mock_checkpointer

        manager = ModelCheckpointManager(trainer)
        manager.load()

        loaded = mock_checkpointer.load.call_args.args[1]
        assert loaded["lr_scheduler"] is trainer.lr_scheduler
        assert loaded["optimizer"] is trainer.optimizer
        assert trainer.state.global_step == 0
        assert mock_checkpointer.load.call_args.kwargs["parallel_state"] is mock_get_ps.return_value

    def test_save_lora_writes_the_adapter_to_its_own_export_dir(
        self, mock_helper, mock_dist, mock_build_ckpt, mock_get_ps, tmp_path
    ):
        """The adapter is an export: it goes to lora_ckpt/, not in with the shards.

        Separate from hf_ckpt/ as well, so a future LoRA merge can write both for
        one step without either landing on the other.
        """
        trainer = _make_mock_trainer(save_path=str(tmp_path / "checkpoints"))
        mock_build_ckpt.return_value = MagicMock()
        manager = ModelCheckpointManager(trainer)
        state = TrainerState(global_step=10)

        with patch("veomni.utils.save_safetensor_utils.save_lora_adapter_with_dcp") as save_adapter:
            manager.save_lora(state)

        save_path = save_adapter.call_args.kwargs["save_path"]
        assert save_path == str(tmp_path / "checkpoints" / "global_step_10" / "lora_ckpt")
        assert save_path == manager.lora_export_dir(state)
        assert save_path != manager.hf_export_dir(state)
        assert save_path != manager.save_dir(state)

    def test_export_rewrites_a_step_this_run_did_not_save(
        self, mock_helper, mock_dist, mock_build_ckpt, mock_get_ps, tmp_path
    ):
        """A ``model/`` an interrupted save left behind is indistinguishable from
        a finished one, and its weights belong to a trajectory this run
        abandoned. Skipping the save on the strength of that directory would
        export stale weights and leave the step's record set without anything
        having finished writing it. This run's own record is what decides."""
        trainer = _make_mock_trainer(save_path=str(tmp_path / "checkpoints"))
        mock_build_ckpt.return_value = MagicMock()
        manager = ModelCheckpointManager(trainer)
        state = TrainerState(global_step=10)
        # What an interrupted save leaves: the tree, and no completion marker.
        (tmp_path / "checkpoints" / "global_step_10" / "model" / "ckpt").mkdir(parents=True)

        with patch("veomni.utils.save_safetensor_utils.save_lora_adapter_with_dcp"):
            manager.save_lora(state)

        assert manager.checkpointer.save.call_count == 1
        assert manager.last_saved_step == 10

    def test_export_does_not_repeat_the_save_this_run_just_made(
        self, mock_helper, mock_dist, mock_build_ckpt, mock_get_ps, tmp_path
    ):
        """The other half of the same rule: the cadence already wrote this step
        in this process, so the export must not write it a second time."""
        trainer = _make_mock_trainer(save_path=str(tmp_path / "checkpoints"))
        mock_build_ckpt.return_value = MagicMock()
        manager = ModelCheckpointManager(trainer)
        state = TrainerState(global_step=10)

        manager.save_dcp(state)
        with patch("veomni.utils.save_safetensor_utils.save_lora_adapter_with_dcp"):
            manager.save_lora(state)

        assert manager.checkpointer.save.call_count == 1

    def test_an_export_alone_does_not_advance_the_record(
        self, mock_helper, mock_dist, mock_build_ckpt, mock_get_ps, tmp_path
    ):
        """``last_saved_step`` is read as 'this step's DCP is on disk', so only a
        DCP save may move it. Here the save is stubbed out, standing in for any
        path that exports without writing one: the export must not leave the
        record claiming a step it did not write."""
        trainer = _make_mock_trainer(save_path=str(tmp_path / "checkpoints"))
        mock_build_ckpt.return_value = MagicMock()
        manager = ModelCheckpointManager(trainer)

        with patch.object(manager, "save_dcp") as save_dcp:
            with patch("veomni.utils.save_safetensor_utils.save_lora_adapter_with_dcp"):
                manager.save_lora(TrainerState(global_step=10))

        save_dcp.assert_called_once()
        assert manager.last_saved_step == -1


@patch("veomni.trainer.callbacks.global_state_callback.dist")
class TestGlobalStateCallbackJobState:
    """Job-level state — dataloader, rng, meters, channel-loss — is not in the DCP sidecar."""

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

    def test_save_does_not_wait_for_a_pending_dcp(self, mock_dist, tmp_path):
        """The cursor does not depend on the shards, and the manifest beside it
        claims only the cursor — the model state answers for itself through DCP's
        own markers. Waiting here is what used to leave ``save_async`` with
        nothing to overlap: the drain landed in the same ``on_step_end`` that
        issued the write."""
        mock_dist.is_initialized.return_value = False
        trainer = _make_mock_trainer(save_path=str(tmp_path))
        trainer.train_dataloader = None
        trainer.data_iterator = None
        trainer.environ_meter.state_dict.return_value = {}
        cb = GlobalStateCallback(trainer)

        cb.save_global_state(TrainerState(global_step=10))

        trainer.checkpoint.wait_for_pending_save.assert_not_called()
        step = tmp_path / "global_step_10"
        assert (step / "extra_state" / "rank_0.pt").is_file()
        assert (step / "loader" / "rank_0.pt").is_file()
        # Last, after both per-rank files: it is what says they are down.
        assert (step / "checkpoint_manifest.json").is_file()

    def test_train_end_writes_the_cursor_for_the_export_that_follows(self, mock_dist, tmp_path):
        """A run ending off the cadence still exports, and that export writes the
        step's DCP (``ModelCheckpointManager._prepare_export``). Model state at a
        step with no cursor beside it is a step nothing can resume from, so the
        cursor follows the export."""
        mock_dist.is_initialized.return_value = False
        trainer = _make_mock_trainer(save_path=str(tmp_path))
        trainer.train_dataloader = None
        trainer.data_iterator = None
        trainer.environ_meter.state_dict.return_value = {}
        cb = GlobalStateCallback(trainer)
        cb._last_saved_step = 100

        cb.on_train_end(TrainerState(global_step=150))

        step = tmp_path / "global_step_150"
        assert (step / "extra_state" / "rank_0.pt").is_file()
        assert (step / "checkpoint_manifest.json").is_file()

    def test_an_export_cadence_of_its_own_pulls_the_cursor_along(self, mock_dist, tmp_path):
        """Same reasoning mid-run: ``hf_save_steps`` shorter than ``save_steps``
        puts a DCP at steps the DCP cadence never reaches."""
        mock_dist.is_initialized.return_value = False
        trainer = _make_mock_trainer(save_path=str(tmp_path))
        trainer.train_dataloader = None
        trainer.data_iterator = None
        trainer.environ_meter.state_dict.return_value = {}
        trainer.args.train.checkpoint.hf_save_steps = 3
        cb = GlobalStateCallback(trainer)

        # Not a multiple of save_steps=5, so only the export puts anything here.
        cb.on_step_end(TrainerState(global_step=3))

        assert (tmp_path / "global_step_3" / "checkpoint_manifest.json").is_file()

    def test_train_end_leaves_an_already_written_step_alone(self, mock_dist, tmp_path):
        """Training that ends on a cadence step already has its cursor; rewriting
        it would cost a second dataloader-state dump for nothing."""
        mock_dist.is_initialized.return_value = False
        trainer = _make_mock_trainer(save_path=str(tmp_path))
        cb = GlobalStateCallback(trainer)
        cb._last_saved_step = 200

        cb.on_train_end(TrainerState(global_step=200))

        assert not (tmp_path / "global_step_200").exists()

    def test_train_end_writes_nothing_when_the_run_does_not_export(self, mock_dist, tmp_path):
        """With ``save_hf_weights`` off nothing writes a DCP at train end, so a
        cursor there would describe a step that has no model state."""
        mock_dist.is_initialized.return_value = False
        trainer = _make_mock_trainer(save_path=str(tmp_path))
        trainer.args.train.checkpoint.save_hf_weights = False
        cb = GlobalStateCallback(trainer)
        cb._last_saved_step = 100

        cb.on_train_end(TrainerState(global_step=150))

        assert not (tmp_path / "global_step_150").exists()

    def test_a_peers_write_failure_stops_the_manifest(self, mock_dist, tmp_path):
        """Each rank writes its own state files, so a full disk is visible to one
        rank. A rank whose own write succeeded must not record the step, and must
        raise rather than walk into the next collective alone."""
        mock_dist.is_initialized.return_value = False
        trainer = _make_mock_trainer(save_path=str(tmp_path))
        trainer.train_dataloader = None
        trainer.data_iterator = None
        trainer.environ_meter.state_dict.return_value = {}
        cb = GlobalStateCallback(trainer)

        def fail_the_write(_error, context):
            if "trainer state" in context:
                raise RuntimeError("writing the trainer state failed on another rank")

        with patch(
            "veomni.trainer.callbacks.global_state_callback.raise_if_any_rank_failed",
            side_effect=fail_the_write,
        ):
            with pytest.raises(RuntimeError, match="failed on another rank"):
                cb.save_global_state(TrainerState(global_step=10))

        assert not (tmp_path / "global_step_10" / "checkpoint_manifest.json").exists()
        assert cb._last_saved_step == -1

    def test_rewriting_a_step_removes_its_manifest_first(self, mock_dist, tmp_path):
        """A restarted run reaching this step again overwrites the cursor files.
        Until that finishes, what is on disk is neither the old state nor the new
        one, so the manifest the earlier attempt left has to go before the first
        of them lands — otherwise a rewrite that dies half-way leaves a step that
        still reads as complete.

        This file and no other. ``DistributedCheckpointer`` drops the ``.metadata``
        files when it rewrites a module, because it is the one that writes them
        back."""
        mock_dist.is_initialized.return_value = False
        trainer = _make_mock_trainer(save_path=str(tmp_path))
        trainer.train_dataloader = None
        trainer.data_iterator = None
        trainer.environ_meter.state_dict.return_value = {}
        cb = GlobalStateCallback(trainer)

        step_root = str(tmp_path / "global_step_10")
        layout.write_manifest(step_root, global_step=10, world_size=1)
        weights = Path(layout.weights_dir(step_root))
        weights.mkdir(parents=True)
        marker = weights / layout.DCP_MARKER_FILENAME
        marker.write_text("written by the module's own save")

        seen = []
        real_save = torch.save
        with patch.object(
            torch,
            "save",
            side_effect=lambda *a, **kw: (
                seen.append(os.path.exists(layout.manifest_path(step_root))),
                real_save(*a, **kw),
            )[-1],
        ):
            cb.save_global_state(TrainerState(global_step=10))

        # Gone before the first cursor file, back once every rank's is down.
        assert seen == [False, False]
        assert os.path.exists(layout.manifest_path(step_root))
        # Untouched: rewriting the cursor says nothing about the module's shards.
        assert marker.read_text() == "written by the module's own save"

    def test_load_restores_channel_loss_callback_state(self, mock_dist, tmp_path):
        mock_dist.is_initialized.return_value = False
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

    @patch("veomni.trainer.callbacks.global_state_callback.get_device_type", return_value="cpu")
    def test_load_skips_when_any_rank_is_missing_state(self, mock_device, mock_dist, tmp_path):
        """A missing cursor on one rank must not leave the others at a different step."""
        mock_dist.is_initialized.return_value = True

        def drop_presence(flag, op=None):
            flag.zero_()

        mock_dist.all_reduce.side_effect = drop_presence
        trainer = _make_mock_trainer()
        trainer.args.train.checkpoint.load_path = str(tmp_path)
        trainer.args.train.global_rank = 0
        trainer.train_dataloader = None
        torch.save(
            {
                "global_step": 7,
                "train_dataloader": None,
                "environ_meter": {},
                "channel_loss_callback": {},
                "torch_rng_state": torch.get_rng_state(),
            },
            tmp_path / "trainer_state_rank_0.pt",
        )

        cb = GlobalStateCallback(trainer)
        assert cb.load_global_state() is None
        trainer.channel_loss_callback.load_state_dict.assert_not_called()
        assert trainer.state.global_step == 0
        mock_dist.all_reduce.assert_called_once()
        assert mock_dist.all_reduce.call_args.kwargs["op"] is torch.distributed.ReduceOp.MIN

    def test_load_restores_0_1_12_extra_state_job_cursor(self, mock_dist, tmp_path):
        mock_dist.is_initialized.return_value = False
        trainer = _make_mock_trainer()
        trainer.args.train.checkpoint.load_path = str(tmp_path)
        trainer.args.train.global_rank = 0
        trainer.args.train_steps = 100
        rng = torch.get_rng_state()
        extra = tmp_path / "extra_state"
        extra.mkdir()
        torch.save(
            {
                "global_step": 7,
                "lr_scheduler": {"last_epoch": 7},
                "train_dataloader": {"cursor": 3},
                "environ_meter": {"tokens": 1},
                "channel_loss_callback": {"source_registry": [(1, "train/a")]},
                "torch_rng_state": rng,
            },
            extra / "extra_state_rank_0.pt",
        )

        cb = GlobalStateCallback(trainer)
        cb.load_global_state()

        assert trainer.state.global_step == 7
        assert trainer.start_epoch == 0
        assert trainer.start_step == 7
        trainer.train_dataloader.load_state_dict.assert_called_once_with({"cursor": 3})
        trainer.environ_meter.load_state_dict.assert_called_once_with({"tokens": 1})
        trainer.channel_loss_callback.load_state_dict.assert_called_once_with({"source_registry": [(1, "train/a")]})

    def test_trainer_state_wins_over_extra_state_job_cursor(self, mock_dist, tmp_path):
        mock_dist.is_initialized.return_value = False
        trainer = _make_mock_trainer()
        trainer.args.train.checkpoint.load_path = str(tmp_path)
        trainer.args.train.global_rank = 0
        trainer.train_dataloader = None
        torch.save(
            {
                "global_step": 7,
                "train_dataloader": None,
                "environ_meter": {},
                "channel_loss_callback": {},
                "torch_rng_state": torch.get_rng_state(),
            },
            tmp_path / "trainer_state_rank_0.pt",
        )
        extra = tmp_path / "extra_state"
        extra.mkdir()
        torch.save(
            {"global_step": 99, "lr_scheduler": {}, "torch_rng_state": torch.get_rng_state()},
            extra / "extra_state_rank_0.pt",
        )

        cb = GlobalStateCallback(trainer)
        cb.load_global_state()
        assert trainer.state.global_step == 7

    def test_scheduler_only_extra_state_does_not_restore_job_cursor(self, mock_dist, tmp_path):
        mock_dist.is_initialized.return_value = False
        trainer = _make_mock_trainer()
        trainer.args.train.checkpoint.load_path = str(tmp_path)
        trainer.args.train.global_rank = 0
        extra = tmp_path / "extra_state"
        extra.mkdir()
        torch.save({"lr_scheduler": {"last_epoch": 3}}, extra / "extra_state_rank_0.pt")

        cb = GlobalStateCallback(trainer)
        assert cb.load_global_state() is None
        assert trainer.state.global_step == 0
