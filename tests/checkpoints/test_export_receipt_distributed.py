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
from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime
from veomni.models.checkpoint.manager import ModelCheckpointManager
from veomni.trainer.callbacks.base import TrainerState


def _worker(rank, rendezvous, root, mode):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=15)
    )
    try:
        runtime = _make_mock_runtime(save_path=root)
        runtime.optimizer = MagicMock(spec=[])
        with patch("veomni.models.checkpoint.manager.build_checkpointer"):
            manager = ModelCheckpointManager(runtime)
            manager.save_dcp(TrainerState(global_step=3))
            if mode == "before" and rank == 1:
                runtime.optimizer = MagicMock(spec=[])
            if mode == "during":

                def wait():
                    if rank == 1:
                        runtime.optimizer = MagicMock(spec=[])

                manager.wait_for_pending_save = wait
            error = None
            try:
                manager.prepare_export(TrainerState(global_step=3), "step_end")
            except RuntimeError as exc:
                error = str(exc)
            result = torch.tensor(int(error is not None))
            dist.all_reduce(result)
            assert result.item() == 2, (rank, mode, error)
            assert manager._last_saved_step == -1
            assert manager._last_saved_optimizer_ref is None
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("mode", ["before", "during"])
def test_optimizer_identity_change_is_rank_coordinated(tmp_path, mode):
    mp.start_processes(
        _worker, args=(str(tmp_path / "gloo"), str(tmp_path), mode), nprocs=2, join=True, start_method="fork"
    )
