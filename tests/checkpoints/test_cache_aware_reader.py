"""DCP cache advice must preserve restored values and follow completed reads."""

import os

import pytest
import torch
import torch.distributed.checkpoint as dcp

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.cache_aware_reader import CacheAwareFileSystemReader


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
_TWO_RANK = LaunchSpec(nproc=2, device="cpu")


@pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="requires POSIX fadvise")
def test_dcp_read_advises_only_after_copy_and_preserves_values(tmp_path, monkeypatch):
    source = {"weight": torch.arange(8192, dtype=torch.float32), "step": torch.tensor(7)}
    dcp.save(source, checkpoint_id=tmp_path)
    target = {"weight": torch.empty_like(source["weight"]), "step": torch.empty_like(source["step"])}
    calls = []

    def observe(fd, offset, length, advice):
        assert torch.equal(target["weight"], source["weight"])
        assert torch.equal(target["step"], source["step"])
        assert os.fstat(fd).st_size >= offset + length
        calls.append((offset, length, advice))

    monkeypatch.setattr(os, "posix_fadvise", observe)
    dcp.load(target, storage_reader=CacheAwareFileSystemReader(tmp_path))
    assert calls and all(length > 0 and advice == os.POSIX_FADV_DONTNEED for _, length, advice in calls)
    assert all(torch.equal(source[key], target[key]) for key in source)


@pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="requires POSIX fadvise")
def test_dcp_advice_error_propagates(tmp_path, monkeypatch):
    dcp.save({"weight": torch.ones(8)}, checkpoint_id=tmp_path)

    def fail(*args):
        raise OSError("injected advice error")

    monkeypatch.setattr(os, "posix_fadvise", fail)
    with pytest.raises(BaseException, match="injected advice error"):
        dcp.load({"weight": torch.empty(8)}, storage_reader=CacheAwareFileSystemReader(tmp_path))


def _two_rank_worker(rank, rendezvous, checkpoint):
    import datetime

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=2, timeout=datetime.timedelta(seconds=60)
    )
    try:
        source = {
            "model": {"weight": torch.arange(1024, dtype=torch.float32)},
            "optimizer": {"moment": torch.arange(1024, dtype=torch.float32) / 8},
        }
        dcp.save(source, checkpoint_id=checkpoint)
        target = {"model": {"weight": torch.empty(1024)}, "optimizer": {"moment": torch.empty(1024)}}
        dcp.load(target, storage_reader=CacheAwareFileSystemReader(checkpoint))
        assert torch.equal(target["model"]["weight"], source["model"]["weight"])
        assert torch.equal(target["optimizer"]["moment"], source["optimizer"]["moment"])
        advice = os.posix_fadvise
        if rank == 0:
            del os.posix_fadvise
        try:
            with pytest.raises(BaseException, match="POSIX fadvise"):
                dcp.load(target, storage_reader=CacheAwareFileSystemReader(checkpoint))
        finally:
            if rank == 0:
                os.posix_fadvise = advice
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="requires POSIX fadvise")
@pytest.mark.launch_plan(provider=_TWO_RANK)
def test_two_rank_dcp_cache_advice(tmp_path):
    torch.multiprocessing.spawn(
        _two_rank_worker,
        args=(str(tmp_path / "rdzv"), str(tmp_path / "checkpoint")),
        nprocs=_TWO_RANK.nproc,
        join=True,
    )


def test_checkpointer_split_resume_restores_model_optimizer_and_sidecar(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from veomni.checkpoint import dcp_checkpointer as module
    from veomni.models.checkpoint.manager import ModelCheckpointManager

    monkeypatch.setattr(module, "get_parallel_state", lambda: SimpleNamespace(dp_mode="fsdp2"))
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model(torch.ones(2, 4)).sum().backward()
    optimizer.step()
    state = {"model": model, "optimizer": optimizer, "extra_state": {"epoch": 3}}
    module.DistributedCheckpointer.save(str(tmp_path), state, global_steps=7)
    restored_model = torch.nn.Linear(4, 4)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.001)
    restored_extra = {}
    runtime = SimpleNamespace(
        train_args=SimpleNamespace(
            checkpoint=SimpleNamespace(
                manager="dcp", load_path=str(tmp_path / "global_step_7"), dcp_evict_source_cache=True
            )
        ),
        args=SimpleNamespace(
            accelerator=SimpleNamespace(fsdp_config=SimpleNamespace(fsdp_mode="fsdp2")), lora_config=None
        ),
        model=restored_model,
        optimizer=restored_optimizer,
        lr_scheduler=None,
        parallel_state=None,
        load_extra_state=lambda state: restored_extra.update(state),
    )
    monkeypatch.setattr(module.dist, "barrier", lambda: None)
    calls = []
    monkeypatch.setattr(os, "posix_fadvise", lambda *args: calls.append(args))
    ModelCheckpointManager(runtime).load()
    assert calls
    assert torch.equal(model.weight, runtime.model.weight)
    old_moment = optimizer.state_dict()["state"][0]["exp_avg"]
    assert torch.equal(old_moment, runtime.optimizer.state_dict()["state"][0]["exp_avg"])
    calls.clear()
    runtime.model = torch.nn.Linear(4, 4)
    runtime.optimizer = torch.optim.AdamW(runtime.model.parameters(), lr=0.001)
    runtime.train_args.checkpoint.dcp_evict_source_cache = False
    ModelCheckpointManager(runtime).load()
    assert not calls
    assert torch.equal(model.weight, runtime.model.weight)
    assert restored_extra["epoch"] == 3
    new_moment = runtime.optimizer.state_dict()["state"][0]["exp_avg"]
    assert torch.equal(old_moment, new_moment)


def test_symlink_shard_still_loads_and_skips_advice(tmp_path):
    import warnings
    from pathlib import Path

    dcp.save({"weight": torch.arange(4)}, checkpoint_id=tmp_path)
    shard = next(Path(tmp_path).glob("*.distcp"))
    actual = shard.with_suffix(".saved")
    shard.rename(actual)
    shard.symlink_to(actual.name)
    target = {"weight": torch.empty(4, dtype=torch.int64)}
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        dcp.load(target, storage_reader=CacheAwareFileSystemReader(tmp_path))
    assert torch.equal(target["weight"], torch.arange(4))
    assert any("symlink shard" in str(item.message) for item in recorded)


@pytest.mark.parametrize("layout", ["fused", "module", "trainable_only"])
def test_enabled_reader_for_legacy_module_and_partial_model(tmp_path, monkeypatch, layout):
    from types import SimpleNamespace

    from veomni.checkpoint import dcp_checkpointer as module

    ps = SimpleNamespace(dp_mode="fsdp2")
    monkeypatch.setattr(module, "get_parallel_state", lambda: ps)
    model = torch.nn.Linear(4, 4)
    if layout == "trainable_only":
        model.bias.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    model(torch.ones(2, 4)).sum().backward()
    optimizer.step()
    checkpoint = tmp_path / "global_step_7"
    if layout == "fused":
        dcp.save(
            {
                "model": module.ModelState(model, parallel_state=ps),
                "optimizer": module.OptimizerState(model, optimizer, parallel_state=ps),
            },
            checkpoint_id=checkpoint,
        )
    else:
        module.DistributedCheckpointer.save(
            str(tmp_path),
            {"model": model, "optimizer": optimizer},
            global_steps=7,
            module="vision" if layout == "module" else "",
            trainable_only=layout == "trainable_only",
        )
    restored = torch.nn.Linear(4, 4)
    if layout == "trainable_only":
        restored.bias.requires_grad_(False)
    new_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=0.001)
    module.DistributedCheckpointer.load(
        str(checkpoint),
        {"model": restored, "optimizer": new_optimizer},
        module="vision" if layout == "module" else "",
        trainable_only=layout == "trainable_only",
        evict_source_cache=True,
    )
    assert torch.equal(model.weight, restored.weight)
    assert torch.equal(
        optimizer.state_dict()["state"][0]["exp_avg"], new_optimizer.state_dict()["state"][0]["exp_avg"]
    )
