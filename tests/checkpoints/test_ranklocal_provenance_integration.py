from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.distributed.parallel_plan import SpecInfo
from veomni.optim.optimizer import MultiOptimizer


def _worker(rank, path, endpoint):
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        mesh = init_device_mesh("cpu", (1, 2), mesh_dim_names=("ep_fsdp", "ep"))
        parallel = SimpleNamespace(
            dp_mode="fsdp2", extra_parallel_names=("ep",), extra_parallel_fsdp_device_mesh={"ep": mesh}
        )

        def build(seed):
            torch.manual_seed(seed)
            model = torch.nn.Module()
            model.experts = torch.nn.Module()
            values = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16
            model.experts.weight = torch.nn.Parameter(distribute_tensor(values, mesh, [Shard(1), Shard(0)]))
            model._fqn2spec_info = {"experts.weight": SpecInfo("ep", Shard(0), "experts.weight", mesh, 1, 1)}
            leaf = torch.optim.AdamW(model.parameters(), lr=0.01)
            return model, MultiOptimizer(model, {"leaf": leaf}, ["leaf"])

        source, optimizer = build(1)
        for param in source.parameters():
            param.grad = torch.ones_like(param)
        optimizer.step()
        optimizer.zero_grad()
        state = {"model": source, "optimizer": optimizer, "_veomni_scheduler_object": None}
        DistributedCheckpointer.save(path, state, parallel_state=parallel)
        target, restored = build(2)
        recovered = {"model": target, "optimizer": restored, "_veomni_scheduler_object": None}
        DistributedCheckpointer.load(path, recovered, parallel_state=parallel)
        for left, right in zip(source.parameters(), target.parameters(), strict=True):
            torch.testing.assert_close(left.to_local(), right.to_local(), rtol=0, atol=0)
            left.grad = torch.ones_like(left)
            right.grad = torch.ones_like(right)
        optimizer.step()
        restored.step()
        for left, right in zip(source.parameters(), target.parameters(), strict=True):
            torch.testing.assert_close(left.to_local(), right.to_local(), rtol=0, atol=0)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_ranklocal_expert_optimizer_checkpoint_roundtrip(tmp_path):
    spawn_with_rendezvous(_worker, args=(str(tmp_path / "ckpt"),), nprocs=2)


def test_rank_prefix_or_fake_adapter_does_not_prove_ownership(tmp_path):
    import torch.distributed.checkpoint as dcp

    from veomni.checkpoint.dcp_checkpointer import _ExtraParallelSavePlanner, _ModelStrictLoadPlanner

    key = "__veomni_rank_0__state.experts.weight.exp_avg"
    full_key = f"optimizer.{key}"
    fake = SimpleNamespace(
        _verified_rank_local_keys=frozenset({full_key}), _load=False, should_extra_parallel_aware=True
    )
    source = {"optimizer": {key: torch.ones(2, 2)}}
    dcp.save(source, checkpoint_id=tmp_path, planner=_ExtraParallelSavePlanner(rank_local_adapter=fake))
    fake._load = True
    target = {"optimizer": {key: torch.zeros(2, 2)}}
    with pytest.raises(dcp.CheckpointException, match="noncanonical expert order"):
        dcp.load(
            target,
            checkpoint_id=tmp_path,
            planner=_ModelStrictLoadPlanner(strict_model=False, rank_local_adapter=fake),
        )
    assert target["optimizer"][key].count_nonzero() == 0
