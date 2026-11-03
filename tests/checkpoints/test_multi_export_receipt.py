"""Same-step MultiOptimizer DCP and HF export must share a proven receipt."""

import gc
import hashlib
import weakref
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime, _make_mock_trainer
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.models.checkpoint.manager import ModelCheckpointManager
from veomni.optim.optimizer import MultiOptimizer
from veomni.trainer.callbacks.base import TrainerState
from veomni.trainer.callbacks.checkpoint_callback import CheckpointCallback
from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _multi_optimizer(model):
    return MultiOptimizer(model, {"model": torch.optim.AdamW(model.parameters())}, ["model"])


def _runtime(tmp_path):
    runtime = _make_mock_runtime(save_path=str(tmp_path))
    runtime.model = torch.nn.Linear(2, 2)
    runtime.optimizer = _multi_optimizer(runtime.model)
    runtime.lr_scheduler = None
    runtime.parallel_state = SimpleNamespace(global_rank=0, dp_mode="ddp", extra_parallel_names=[])
    return runtime


@pytest.mark.parametrize("replacement", ["multi", "ordinary", "none"])
def test_replaced_optimizer_cannot_reuse_multi_receipt(tmp_path, replacement):
    runtime = _runtime(tmp_path)
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    state = TrainerState(global_step=1, epoch=0)
    with (
        patch("veomni.models.checkpoint.manager.build_checkpointer", return_value=DistributedCheckpointer),
        patch("torch.distributed.barrier"),
    ):
        manager = ModelCheckpointManager(runtime)
        runtime.checkpoint = manager
        trainer.model = runtime
        trainer.save_dcp.side_effect = manager.save_dcp
        trainer.save_hf_or_lora.side_effect = lambda state, stage="step_end": manager.prepare_export(state, stage)
        callback = CheckpointCallback(trainer)
        callback._save_hf(state)
        runtime.optimizer = {
            "multi": lambda: _multi_optimizer(runtime.model),
            "ordinary": lambda: torch.optim.AdamW(runtime.model.parameters()),
            "none": lambda: None,
        }[replacement]()
        with pytest.raises(RuntimeError, match="receipt changed"):
            callback.on_epoch_end(state)
        trainer.save_dcp.assert_called_once_with(state)


def test_train_end_does_not_retain_optimizer_during_hf_export(tmp_path):
    runtime = _runtime(tmp_path)
    optimizer_ref = weakref.ref(runtime.optimizer)
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    state = TrainerState(global_step=1, epoch=0)
    with (
        patch("veomni.models.checkpoint.manager.build_checkpointer", return_value=DistributedCheckpointer),
        patch("torch.distributed.barrier"),
    ):
        manager = ModelCheckpointManager(runtime)
        runtime.checkpoint = manager
        trainer.model = runtime
        trainer.save_dcp.side_effect = manager.save_dcp

        def export(state, stage="step_end"):
            manager.prepare_export(state, stage)
            gc.collect()
            assert optimizer_ref() is None

        trainer.save_hf_or_lora.side_effect = export
        CheckpointCallback(trainer)._save_hf(state, stage="train_end")


def test_unproven_same_step_receipt_is_not_overwritten(tmp_path):
    runtime = _runtime(tmp_path)
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    with patch("veomni.models.checkpoint.manager.build_checkpointer", return_value=DistributedCheckpointer):
        manager = ModelCheckpointManager(runtime)
    runtime.checkpoint = manager
    manager._last_saved_step = 1
    manager._last_saved_optimizer_generation = "unproven"
    trainer.model = runtime
    callback = CheckpointCallback(trainer)
    with pytest.raises(RuntimeError, match="no matching same-step checkpoint receipt"):
        callback._save_hf(TrainerState(global_step=1))
    trainer.save_dcp.assert_not_called()
    trainer.save_hf_or_lora.assert_not_called()


def _real_multi_save_worker(rank, directory, endpoint):
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        _init_parallel_state(dp_size=2, dp_shard_size=2, device_type="cpu")
        torch.manual_seed(123)
        runtime = _runtime(directory)
        runtime.parallel_state = SimpleNamespace(global_rank=rank, dp_mode="ddp", extra_parallel_names=[])
        runtime.model(torch.ones(1, 2)).sum().backward()
        runtime.optimizer.step()
        runtime.optimizer.zero_grad()
        trainer = _make_mock_trainer(save_path=directory)
        trainer.args.train.global_rank = rank
        trainer.args.train.world_size = 2
        trainer.args.train.checkpoint.save_steps = 0
        trainer.args.train.checkpoint.save_epochs = 1
        trainer.args.train.checkpoint.hf_save_steps = 1
        trainer.args.train.checkpoint.hf_save_epochs = 0
        trainer.train_dataloader = None
        trainer.data_iterator = None
        trainer.environ_meter.state_dict.return_value = {}
        trainer.channel_loss_callback = None
        trainer.offline_cache_writer = None
        with patch("veomni.models.checkpoint.manager.build_checkpointer", return_value=DistributedCheckpointer):
            manager = ModelCheckpointManager(runtime)
        runtime.checkpoint = manager
        trainer.model = runtime
        trainer.save_dcp.side_effect = manager.save_dcp
        trainer.save_hf_or_lora.side_effect = lambda state, stage="step_end": manager.prepare_export(state, stage)
        callback = CheckpointCallback(trainer)
        global_callback = GlobalStateCallback(trainer)
        state = TrainerState(global_step=1, epoch=0)
        callback.on_step_end(state)
        global_callback.on_step_end(state)
        generation = manager._last_saved_optimizer_generation
        assert generation and callback._last_multi_receipt == generation
        snapshot = None
        if rank == 0:
            root = Path(directory)
            snapshot = {
                p.relative_to(root): hashlib.sha256(p.read_bytes()).digest() for p in root.rglob("*") if p.is_file()
            }
            assert snapshot
        callback.on_epoch_end(state)
        global_callback.on_epoch_end(state)
        trainer.save_dcp.assert_called_once_with(state)
        assert manager._last_saved_optimizer_generation == generation
        if rank == 0:
            assert snapshot == {
                p.relative_to(root): hashlib.sha256(p.read_bytes()).digest() for p in root.rglob("*") if p.is_file()
            }
        if rank == 1:
            manager._last_saved_optimizer_generation = None
        with pytest.raises(RuntimeError, match="receipt changed"):
            callback.on_epoch_end(state)
        dist.barrier()
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_real_multi_dcp_hf_epoch_on_two_ranks(tmp_path):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_real_multi_save_worker, args=(str(tmp_path),), nprocs=2)
