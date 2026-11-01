# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""One-rank checkpoint identity changes must fail on every participant."""

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime, _make_mock_trainer
from veomni.models.checkpoint.manager import ModelCheckpointManager
from veomni.trainer.callbacks.base import TrainerState


def _worker(rank, rendezvous, root, mode):
    from veomni.distributed.parallel_state import _init_parallel_state

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=15)
    )
    try:
        _init_parallel_state(dp_size=2, dp_shard_size=2, device_type="cpu")
        runtime = _make_mock_runtime(save_path=root)
        runtime.optimizer = MagicMock(spec=[])
        with patch("veomni.models.checkpoint.manager.build_checkpointer"):
            manager = ModelCheckpointManager(runtime)
            if mode in ("dcp_reader", "model_state", "optimizer_state"):
                from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer, ModelState, OptimizerState

                manager.config.load_path = root
                runtime.model = torch.nn.Linear(2, 2)
                runtime.optimizer = (
                    torch.optim.AdamW(runtime.model.parameters()) if mode == "optimizer_state" else None
                )
                manager.checkpointer = DistributedCheckpointer

                def reader(_path):
                    if rank == 1 and mode == "dcp_reader":
                        raise RuntimeError("injected reader creation failure")
                    return MagicMock()

                error = None

                def model_snapshot(owner):
                    if rank == 1 and mode == "model_state":
                        raise RuntimeError("injected model state failure")
                    return original_model_snapshot(owner)

                def optimizer_snapshot(owner):
                    if rank == 1 and mode == "optimizer_state":
                        raise RuntimeError("injected optimizer state failure")
                    return original_optimizer_snapshot(owner)

                original_model_snapshot = ModelState.state_dict
                original_optimizer_snapshot = OptimizerState.state_dict
                with (
                    patch.object(DistributedCheckpointer, "_create_storage_reader", side_effect=reader),
                    patch.object(ModelState, "state_dict", model_snapshot),
                    patch.object(OptimizerState, "state_dict", optimizer_snapshot),
                    patch("veomni.checkpoint.dcp_checkpointer.dcp.load"),
                ):
                    try:
                        manager.load()
                    except RuntimeError as exc:
                        error = str(exc)
                result = torch.tensor(int(error is not None))
                dist.all_reduce(result)
                assert result.item() == 2, (rank, error)
                return
            if mode in ("load_kwargs", "save_kwargs"):
                manager.config.load_path = root

                def kwargs():
                    if rank == 1:
                        raise RuntimeError("injected load argument failure")
                    return {}

                manager._checkpointer_kwargs = kwargs
                error = None
                try:
                    if mode == "load_kwargs":
                        manager.load()
                    else:
                        manager.save_dcp(TrainerState(global_step=3))
                except RuntimeError as exc:
                    error = str(exc)
                result = torch.tensor(int(error is not None))
                dist.all_reduce(result)
                assert result.item() == 2, (rank, error)
                manager.checkpointer.load.assert_not_called()
                manager.checkpointer.save.assert_not_called()
                return
            manager.save_dcp(TrainerState(global_step=3))
            if mode in ("before", "epoch") and rank == 1:
                runtime.optimizer = MagicMock(spec=[])
            if mode == "during":

                def wait():
                    if rank == 1:
                        runtime.optimizer = MagicMock(spec=[])

                manager.wait_for_pending_save = wait
            error = None
            try:
                if mode in ("epoch", "cadence"):
                    from veomni.trainer.callbacks.checkpoint_callback import CheckpointCallback

                    trainer = _make_mock_trainer(save_path=root)
                    trainer.args.train.checkpoint.save_epochs = 0 if mode == "cadence" and rank == 1 else 1
                    trainer.args.train.checkpoint.save_hf_weights = False
                    trainer.model.checkpoint = manager
                    callback = CheckpointCallback(trainer)
                    callback._last_dcp_step = 3
                    callback.on_epoch_end(TrainerState(global_step=3, epoch=0))
                else:
                    manager.prepare_export(TrainerState(global_step=3), "step_end")
            except RuntimeError as exc:
                error = str(exc)
            result = torch.tensor(int(error is not None))
            dist.all_reduce(result)
            assert result.item() == 2, (rank, mode, error)
            if mode not in ("epoch", "cadence"):
                assert manager._last_saved_step == -1
                assert manager._last_saved_optimizer_ref is None
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize(
    "mode",
    [
        "before",
        "during",
        "epoch",
        "cadence",
        "load_kwargs",
        "save_kwargs",
        "dcp_reader",
        "model_state",
        "optimizer_state",
    ],
)
def test_optimizer_identity_change_is_rank_coordinated(tmp_path, mode):
    mp.start_processes(
        _worker, args=(str(tmp_path / "gloo"), str(tmp_path), mode), nprocs=2, join=True, start_method="fork"
    )
