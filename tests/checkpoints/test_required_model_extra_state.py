"""Missing rank-local model state must not silently reset a scheduler."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.checkpoint import layout
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.models.checkpoint.manager import ModelCheckpointManager


def _missing_extra_state_worker(rank, root, failure, endpoint):
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(123)
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 0.9)
        state = {"model": model, "optimizer": optimizer, "extra_state": {"lr_scheduler": scheduler.state_dict()}}
        parallel = SimpleNamespace(dp_mode="ddp")
        DistributedCheckpointer.save(root, state, parallel_state=parallel)
        dist.barrier()
        if rank == 0:
            path = Path(layout.model_dir(root)) / "extra_state" / "extra_state_rank_1.pt"
            if failure == "missing":
                path.unlink()
            elif failure in {"scheduler_missing", "scheduler_none", "scheduler_empty", "scheduler_partial"}:
                saved = torch.load(path, weights_only=False)
                if failure == "scheduler_missing":
                    saved.pop("lr_scheduler")
                elif failure == "scheduler_none":
                    saved["lr_scheduler"] = None
                elif failure == "scheduler_empty":
                    saved["lr_scheduler"] = {}
                else:
                    saved["lr_scheduler"].pop("last_epoch")
                torch.save(saved, path)
        dist.barrier()
        target = torch.nn.Linear(2, 2)
        target_optimizer = torch.optim.AdamW(target.parameters(), lr=0.01)
        target_scheduler = torch.optim.lr_scheduler.LambdaLR(target_optimizer, lambda _: 0.9)
        runtime = SimpleNamespace(
            model=target,
            optimizer=target_optimizer,
            lr_scheduler=target_scheduler,
            parallel_state=parallel,
            args=SimpleNamespace(lora_config=None),
            load_extra_state=lambda _: None,
        )
        manager = object.__new__(ModelCheckpointManager)
        manager.runtime = runtime
        manager.config = SimpleNamespace(load_path=root)
        manager.checkpointer = DistributedCheckpointer
        original_load = torch.load

        def load_with_io_error(path, *args, **kwargs):
            if failure == "io_error" and rank == 1 and str(path).endswith("extra_state_rank_1.pt"):
                raise OSError("injected sidecar read failure")
            return original_load(path, *args, **kwargs)

        with patch("torch.load", side_effect=load_with_io_error):
            with pytest.raises((FileNotFoundError, OSError, RuntimeError)):
                manager.load()
        weights_only = torch.nn.Linear(2, 2)
        DistributedCheckpointer.load(root, {"model": weights_only}, parallel_state=parallel)
        for expected, actual in zip(model.parameters(), weights_only.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        value = torch.ones(1)
        dist.all_reduce(value)
        assert value.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
@pytest.mark.parametrize(
    "failure", ["missing", "io_error", "scheduler_missing", "scheduler_none", "scheduler_empty", "scheduler_partial"]
)
def test_missing_rank_extra_state_fails_on_all_ranks(tmp_path, failure):
    spawn_with_rendezvous(_missing_extra_state_worker, args=(str(tmp_path / "checkpoint"), failure), nprocs=2)


@pytest.mark.parametrize("state", [{}, {"lr_scheduler": None}, {"lr_scheduler": {}}])
def test_missing_scheduler_state_is_rejected(state):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 0.9)
    runtime = SimpleNamespace(lr_scheduler=scheduler, load_extra_state=lambda _: None)
    manager = object.__new__(ModelCheckpointManager)
    manager.runtime = runtime
    with pytest.raises(RuntimeError, match="scheduler state is missing"):
        manager._load_extra_state(state)


def test_no_scheduler_accepts_none():
    runtime = SimpleNamespace(lr_scheduler=None, load_extra_state=lambda _: None)
    manager = object.__new__(ModelCheckpointManager)
    manager.runtime = runtime
    manager._load_extra_state({"lr_scheduler": None})


def test_scheduler_state_is_restored():
    source_optimizer = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters(), lr=0.01)
    source = torch.optim.lr_scheduler.LambdaLR(source_optimizer, lambda _: 0.9)
    source_optimizer.step()
    source.step()
    target_optimizer = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters(), lr=0.01)
    target = torch.optim.lr_scheduler.LambdaLR(target_optimizer, lambda _: 0.9)
    runtime = SimpleNamespace(lr_scheduler=target, load_extra_state=lambda _: None)
    manager = object.__new__(ModelCheckpointManager)
    manager.runtime = runtime
    manager._load_extra_state({"lr_scheduler": source.state_dict()})
    assert target.state_dict() == source.state_dict()


def test_custom_scheduler_retains_its_own_state_contract():
    class CustomScheduler:
        def load_state_dict(self, state):
            self.state = state

    scheduler = CustomScheduler()
    runtime = SimpleNamespace(lr_scheduler=scheduler, load_extra_state=lambda _: None)
    manager = object.__new__(ModelCheckpointManager)
    manager.runtime = runtime
    manager._load_extra_state({"lr_scheduler": {"custom_step": 3}})
    assert scheduler.state == {"custom_step": 3}
