# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""Rank-local checkpoint I/O failures must reach every participant."""

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
            if mode == "invalidated_output":
                state = TrainerState(global_step=3)
                if rank == 0:
                    manager._invalidated_saved_step_paths.append(manager.step_dir(state))
                error = None
                try:
                    manager.save_dcp(state)
                except RuntimeError as exc:
                    error = str(exc)
                result = torch.tensor(int(error is not None))
                dist.all_reduce(result)
                assert result.item() == 2, (rank, error)
                manager.checkpointer.save.assert_not_called()
                return
            if mode in ("same_output", "legacy_incomplete"):
                from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback

                trainer = _make_mock_trainer(save_path=root)
                trainer.args.train.world_size = 2
                callback = GlobalStateCallback(trainer)
                checkpoint = MagicMock()
                manifest = {"model_state": True, "global_step": 3, "world_size": 2}
                same = rank == 0 if mode == "same_output" else True
                complete = rank != 0
                with (
                    patch(
                        "veomni.trainer.callbacks.global_state_callback.layout.same_checkpoint_path", return_value=same
                    ),
                    patch(
                        "veomni.trainer.callbacks.global_state_callback.layout.read_manifest", return_value=manifest
                    ),
                    patch(
                        "veomni.trainer.callbacks.global_state_callback.layout.checkpoint_is_complete",
                        return_value=complete,
                    ),
                    pytest.raises(RuntimeError),
                ):
                    callback._confirm_same_output_checkpoint(
                        3,
                        root,
                        (),
                        {},
                        legacy_checkpoints=(("model", checkpoint),) if mode == "legacy_incomplete" else (),
                    )
                checkpoint.confirm_loaded_checkpoint_for_reuse.assert_not_called()
                checkpoint.confirm_legacy_loaded_checkpoint_for_reuse.assert_not_called()
                dist.barrier()
                return
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
            raise AssertionError(f"Unknown test mode: {mode}")
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize(
    "mode",
    [
        "same_output",
        "legacy_incomplete",
        "load_kwargs",
        "save_kwargs",
        "dcp_reader",
        "model_state",
        "optimizer_state",
        "invalidated_output",
    ],
)
def test_rank_local_checkpoint_failures_are_coordinated(tmp_path, mode):
    mp.start_processes(
        _worker, args=(str(tmp_path / "gloo"), str(tmp_path), mode), nprocs=2, join=True, start_method="fork"
    )
