"""A lazy empty FSDP shard must not falsely mark a populated checkpoint as empty."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.optim.qwen38_muon import Qwen38MuonAdamW, add_qwen38_plans


def _worker(rank, world_size, rendezvous, root):
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    from veomni.checkpoint import dcp_checkpointer
    from veomni.optim import file_backed_state, qwen38_muon
    from veomni.optim.ple_world_profile import _FactoryPermit, establish
    from veomni.utils import dist_utils

    dist_utils.get_device_type = lambda: "cpu"
    # This is a CPU/Gloo test even when the surrounding image has NPU support.
    dcp_checkpointer.synchronize = lambda: None
    file_backed_state._filesystem_type = lambda root: "ext4"
    qwen38_muon.FileBackedFP32State = lambda root: file_backed_state.FileBackedFP32State(root, minimum_free_bytes=0)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size, timeout=timedelta(seconds=20)
    )
    try:
        receipt = establish(
            SimpleNamespace(
                type="qwen38_muon", qwen38_ple_moments_world_profile=True, qwen38_ple_moments_scratch_dir=root
            ),
            SimpleNamespace(),
            SimpleNamespace(save_async=False),
        )
        mesh = init_device_mesh("cpu", (world_size,))

        def create():
            model = torch.nn.Module()
            model.lm_head = torch.nn.Module()
            full = torch.ones(1, 4, dtype=torch.bfloat16)
            model.lm_head.weight = torch.nn.Parameter(distribute_tensor(full, mesh, [Shard(0)]))
            groups = add_qwen38_plans(
                model, [{"params": list(model.parameters()), "weight_decay": 0.0, "cpu_offload": True}]
            )
            optimizer = Qwen38MuonAdamW(
                groups,
                lr=0.1,
                ple_moments_scratch_dir=root,
                ple_moments_world_profile=True,
                ple_world_receipt=receipt,
                ple_factory_permit=_FactoryPermit(receipt),
            )
            return model, optimizer

        model, optimizer = create()
        parameter = model.lm_head.weight
        distributed_gradient = distribute_tensor(torch.ones(1, 4, dtype=torch.bfloat16), mesh, [Shard(0)])
        parameter.grad = distributed_gradient if rank == 0 else None
        optimizer.step()
        assert bool(optimizer.state.get(parameter)) is (rank == 0)
        path = root + "/checkpoint"
        parallel = SimpleNamespace(dp_mode="ddp")
        DistributedCheckpointer.save(
            path, {"model": model, "optimizer": optimizer}, parallel_state=parallel, file_profile_receipt=receipt
        )
        metadata = FileSystemReader(path + "/model/optimizer").read_metadata().state_dict_metadata
        assert not any("__veomni_zero_optimizer_state__" in key for key in metadata)
        assert any(key.endswith(".master_weight") for key in metadata)

        restored_model, restored = create()
        DistributedCheckpointer.load(
            path,
            {"model": restored_model, "optimizer": restored},
            parallel_state=parallel,
            file_profile_receipt=receipt,
        )
        assert torch.equal(restored_model.lm_head.weight.to_local(), parameter.to_local())
        distributed_gradient = distribute_tensor(torch.ones(1, 4, dtype=torch.bfloat16), mesh, [Shard(0)])
        gradient = distributed_gradient if rank == 0 else None
        parameter.grad = gradient
        restored_model.lm_head.weight.grad = gradient
        optimizer.step()
        restored.step()
        assert torch.equal(restored_model.lm_head.weight.to_local(), parameter.to_local())
        if rank == 0:
            for key in ("step", "master_weight", "exp_avg", "exp_avg_sq"):
                left = optimizer.state[parameter][key]
                right = restored.state[restored_model.lm_head.weight][key]
                left = left.to_local() if hasattr(left, "to_local") else left
                right = right.to_local() if hasattr(right, "to_local") else right
                assert torch.equal(left, right), key
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_file_checkpoint_with_one_lazy_empty_shard_restores_next_update(tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(
        _worker,
        args=(2, str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=2,
        join=True,
        start_method="fork",
    )
