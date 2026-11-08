"""Reject legacy external streaming requests without changing native loaders."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.checkpoints.test_checkpoint_callback import _make_mock_trainer
from veomni.data.dynamic_batching import DynamicBatchSizeDataLoader
from veomni.trainer.callbacks.base import TrainerState
from veomni.trainer.callbacks.checkpoint_callback import CheckpointCallback
from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.mark.parametrize("source", ["trainer", "loader", "wrapped"])
def test_retired_provider_request_fails_before_model_load(source):
    trainer = _make_mock_trainer()
    trainer.train_dataloader = SimpleNamespace()
    if source == "trainer":
        trainer._veomni_require_exact_data_resume = True
    elif source == "loader":
        trainer.train_dataloader._veomni_exact_resume_requested = True
    else:
        inner = SimpleNamespace(_veomni_exact_resume_requested=True)
        trainer.train_dataloader = DynamicBatchSizeDataLoader.__new__(DynamicBatchSizeDataLoader)
        trainer.train_dataloader._dataloader = inner
    with pytest.raises(ValueError, match="unsupported"):
        CheckpointCallback(trainer).on_train_begin(TrainerState())
    trainer.save_model_assets.assert_not_called()
    trainer.load.assert_not_called()


@pytest.mark.parametrize("marker", ["external", "cursor", "wrapped_cursor"])
def test_retired_provider_checkpoint_fails_before_cursor_or_rng_restore(tmp_path, marker):
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.args.train.checkpoint.load_path = str(tmp_path)
    trainer.train_dataloader = MagicMock()
    callback = GlobalStateCallback(trainer)
    payload = {"global_step": 3, "train_dataloader": {"index": 1}}
    if marker == "external":
        payload["_veomni_external_data"] = {}
    elif marker == "cursor":
        payload["train_dataloader"]["_veomni_streaming_cursor_version"] = 1
    else:
        payload["train_dataloader"]["dataloader_state"] = {"_veomni_streaming_cursor_version": 1}
    callback._read_current = lambda _: payload
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="External data checkpoint"):
        callback.load_global_state()
    assert trainer.state.global_step == 0
    assert torch.equal(torch.get_rng_state(), rng_before)
    trainer.train_dataloader.load_state_dict.assert_not_called()


def _rank_local_legacy_rejection(rank, store, checkpoint):
    dist.init_process_group(
        "gloo", init_method="file://" + store, rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        trainer = _make_mock_trainer(save_path=checkpoint)
        trainer.train_dataloader = SimpleNamespace(_veomni_exact_resume_requested=rank == 0)
        checkpoint_callback = CheckpointCallback.__new__(CheckpointCallback)
        checkpoint_callback.trainer = trainer
        with pytest.raises((ValueError, RuntimeError)):
            checkpoint_callback.on_train_begin(TrainerState())
        trainer.load.assert_not_called()
        alive = torch.ones(1)
        dist.all_reduce(alive)
        assert alive.item() == 2

        trainer = _make_mock_trainer(save_path=checkpoint)
        trainer.args.train.checkpoint.load_path = checkpoint
        trainer.args.train.global_rank = rank
        trainer.args.train.world_size = 2
        trainer.train_dataloader = None
        callback = GlobalStateCallback.__new__(GlobalStateCallback)
        callback.trainer = trainer
        callback._read_current = lambda _: {"global_step": 3, "_veomni_external_data": {}} if rank == 0 else None
        with patch("veomni.checkpoint.legacy_v0_1_12.apply_legacy_global_state", return_value=None):
            with pytest.raises((ValueError, RuntimeError)):
                callback.load_global_state()
        assert trainer.state.global_step == 0
        alive = torch.ones(1)
        dist.all_reduce(alive)
        assert alive.item() == 2
    finally:
        dist.destroy_process_group()


def test_legacy_request_and_missing_peer_state_are_coordinated(tmp_path):
    mp.spawn(_rank_local_legacy_rejection, args=(str(tmp_path / "store"), str(tmp_path)), nprocs=2, join=True)
