"""Native FSDP2 checkpoint preflight coverage; requires two accelerators."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.optim.test_cpu_offload_adamw_device import _require_native_device
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.optim.cpu_offload_adamw import CPUOffloadAdamW
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


_TWO_ACCELERATORS = LaunchSpec(nproc=2, device="accelerator")


def _local(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _fsdp2_worker(rank, root, endpoint):
    device_type = get_device_type()
    device = torch.device(f"{device_type}:{rank}")
    get_torch_device().set_device(device)
    dist.init_process_group(
        get_dist_comm_backend(), store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=120)
    )
    try:
        mesh = init_device_mesh(device_type, (2,), mesh_dim_names=("fsdp",))

        def build(reverse=False):
            model = torch.nn.Linear(8, 4, device=device, dtype=torch.bfloat16)
            fully_shard(model, mesh=mesh)
            params = list(model.parameters())
            if reverse:
                params.reverse()
            optimizer = CPUOffloadAdamW([{"params": params, "cpu_offload": True}], lr=0.01)
            # Exercise the genuine PyTorch scheduler wrapper, not a copied
            # `_wrapped_by_lr_sched` marker on an arbitrary Python wrapper.
            torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            return model, optimizer

        def step(model, optimizer, value):
            for param in model.parameters():
                param.grad = torch.ones_like(param) * value
            optimizer.step()
            optimizer.zero_grad()

        torch.manual_seed(123)
        source_model, source = build()
        step(source_model, source, 1.0)
        parallel = SimpleNamespace(dp_mode="fsdp2")
        checkpoint = f"{root}/checkpoint"
        DistributedCheckpointer.save(checkpoint, {"model": source_model, "optimizer": source}, parallel_state=parallel)

        target_model, target = build(reverse=True)
        step(target_model, target, 0.25)
        before = [_local(param).clone() for param in target_model.parameters()]
        before_state = [
            {key: _local(value).clone() for key, value in target.state[param].items()}
            for param in target_model.parameters()
        ]
        with pytest.raises(RuntimeError, match="parameter groups|another rank"):
            DistributedCheckpointer.load(
                checkpoint, {"model": target_model, "optimizer": target}, parallel_state=parallel
            )
        for index, param in enumerate(target_model.parameters()):
            torch.testing.assert_close(_local(param), before[index], rtol=0, atol=0)
            for key, value in before_state[index].items():
                torch.testing.assert_close(_local(target.state[param][key]), value, rtol=0, atol=0)

        hook_model, hook_optimizer = build()
        if rank == 1:

            def fail_load(_model, _incompatible):
                raise RuntimeError("rank-local FSDP2 model hook failed")

            hook_model.register_load_state_dict_post_hook(fail_load)
        with pytest.raises(RuntimeError, match="model hook failed|another rank"):
            DistributedCheckpointer.load(
                checkpoint, {"model": hook_model, "optimizer": hook_optimizer}, parallel_state=parallel
            )
        assert not hook_optimizer.state

        for hook_phase in ("pre", "post"):
            hook_model, hook_optimizer = build()
            if rank == 1:

                def fail_optimizer(*_args):
                    raise RuntimeError("rank-local FSDP2 optimizer hook failed")

                if hook_phase == "pre":
                    hook_optimizer.register_load_state_dict_pre_hook(fail_optimizer)
                else:
                    hook_optimizer.register_load_state_dict_post_hook(fail_optimizer)
            with pytest.raises(RuntimeError, match="optimizer hook failed|another rank"):
                DistributedCheckpointer.load(
                    checkpoint, {"model": hook_model, "optimizer": hook_optimizer}, parallel_state=parallel
                )

        resumed_model, resumed = build()
        DistributedCheckpointer.load(
            checkpoint, {"model": resumed_model, "optimizer": resumed}, parallel_state=parallel
        )
        step(source_model, source, 0.5)
        step(resumed_model, resumed, 0.5)
        for original, restored in zip(source_model.parameters(), resumed_model.parameters(), strict=True):
            torch.testing.assert_close(_local(original), _local(restored), rtol=0, atol=0)
            for key in source.state[original]:
                torch.testing.assert_close(
                    _local(source.state[original][key]), _local(resumed.state[restored][key]), rtol=0, atol=0
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=_TWO_ACCELERATORS)
def test_accelerator_fsdp2_cpu_offload_group_preflight(tmp_path):
    _require_native_device()
    if get_torch_device().device_count() < 2:
        pytest.fail("FSDP2 preflight test requires at least two native accelerators")
    spawn_with_rendezvous(_fsdp2_worker, args=(str(tmp_path),), nprocs=2)
