"""A partial trainer cursor must not resume with an already restored optimizer."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.data.data_loader import DistributedDataloader
from veomni.models.checkpoint.manager import ModelCheckpointManager
from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback


class _Meter:
    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


def _loader(rank):
    data = torch.utils.data.TensorDataset(torch.arange(100))
    sampler = StatefulDistributedSampler(data, num_replicas=2, rank=rank, shuffle=False)
    return DistributedDataloader(data, sampler=sampler, batch_size=1, num_workers=0)


def _callback(trainer):
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    return callback


def _legacy_resume_worker(rank, root, missing, endpoint):
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        parallel = SimpleNamespace(dp_mode="ddp", global_rank=rank)
        torch.manual_seed(123)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: 0.9**n)
        for _ in range(5):
            optimizer.zero_grad()
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            scheduler.step()
        source = str(Path(root) / "source")
        step_dir = Path(source) / "global_step_5"
        DistributedCheckpointer.save(
            source,
            {"model": model, "optimizer": optimizer, "extra_state": {"lr_scheduler": scheduler.state_dict()}},
            global_steps=5,
            parallel_state=parallel,
        )
        loader = _loader(rank)
        iterator = iter(loader)
        for _ in range(5):
            next(iterator)
        checkpoint = SimpleNamespace(save_path=source)
        trainer = SimpleNamespace(
            model=SimpleNamespace(model=model, optimizer=optimizer, checkpoint=None),
            args=SimpleNamespace(train=SimpleNamespace(checkpoint=checkpoint, global_rank=rank, world_size=2)),
            train_dataloader=loader,
            environ_meter=_Meter(),
            channel_loss_callback=None,
            offline_cache_writer=None,
        )
        _callback(trainer).save_global_state(SimpleNamespace(global_step=5, epoch=0))
        dist.barrier()
        if rank == 0 and missing in {"extra_state", "loader"}:
            (step_dir / missing / "rank_1.pt").unlink()
        elif rank == 0 and missing in {"rng_missing", "rng_invalid"}:
            path = step_dir / "extra_state" / "rank_1.pt"
            state = torch.load(path, weights_only=False)
            if missing == "rng_missing":
                state.pop("torch_rng_state", None)
                state.pop("device_rng_state", None)
            else:
                state["torch_rng_state"] = torch.ones(1, dtype=torch.uint8)
            torch.save(state, path)
        dist.barrier()

        target = torch.nn.Linear(2, 1)
        target_optimizer = torch.optim.AdamW(target.parameters(), lr=0.01)
        target_scheduler = torch.optim.lr_scheduler.LambdaLR(target_optimizer, lambda n: 0.9**n)
        config = SimpleNamespace(load_path=str(step_dir), save_path=str(Path(root) / "new-output"), manager="dcp")
        runtime = SimpleNamespace(
            model=target,
            optimizer=target_optimizer,
            lr_scheduler=target_scheduler,
            args=SimpleNamespace(lora_config=None),
            parallel_state=parallel,
            load_extra_state=lambda _: None,
        )
        manager = ModelCheckpointManager.__new__(ModelCheckpointManager)
        manager.runtime, manager.config, manager.checkpointer = runtime, config, DistributedCheckpointer
        runtime.checkpoint = manager
        resumed = _loader(rank)
        target_trainer = SimpleNamespace(
            model=runtime,
            args=SimpleNamespace(train=SimpleNamespace(checkpoint=config, global_rank=rank), train_steps=10),
            state=SimpleNamespace(global_step=0),
            train_dataloader=resumed,
            environ_meter=_Meter(),
            channel_loss_callback=None,
            offline_cache_writer=None,
        )
        manager.load()
        callback = _callback(target_trainer)
        if missing == "none":
            callback.load_global_state()
            assert target_trainer.state.global_step == 5
            assert int(next(iter(resumed))[0].item()) == 10 + rank
            assert target_scheduler.last_epoch == 5
        else:
            rng_before = torch.get_rng_state().clone()
            with pytest.raises((RuntimeError, ValueError), match="missing|invalid|failed on another rank"):
                callback.load_global_state()
            assert target_trainer.state.global_step == 0
            assert torch.equal(torch.get_rng_state(), rng_before)
        alive = torch.ones(1)
        dist.all_reduce(alive)
        assert alive.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
@pytest.mark.parametrize("missing", ["none", "extra_state", "loader", "rng_missing", "rng_invalid"])
def test_legacy_optimizer_resume_requires_rank_local_cursor(tmp_path, missing):
    spawn_with_rendezvous(_legacy_resume_worker, args=(str(tmp_path), missing), nprocs=2)
