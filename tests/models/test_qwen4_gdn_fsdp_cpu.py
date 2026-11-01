"""CPU FSDP2 smoke test for the separate Qwen GDN FP32 parameter group."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import checkpoint

from scripts.ci.accelerator.launch_plan import LaunchSpec


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))


class _GDN(nn.Module):
    def __init__(self):
        super().__init__()
        self.A_log = nn.Parameter(torch.ones(4))
        self.dt_bias = nn.Parameter(torch.ones(4))
        self.lin = nn.Linear(4, 4)
        self.dtypes = None

    def forward(self, x):
        self.dtypes = (x.dtype, self.lin.weight.dtype, self.A_log.dtype, self.dt_bias.dtype)
        return self.lin(x) * self.A_log * self.dt_bias


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.gdn = _GDN()

    def forward(self, x):
        return self.gdn(x)


class _Root(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.layer = _Layer()
        self.mode = mode

    def forward(self, x):
        if self.mode == "plain":
            return self.layer(x)
        return checkpoint(self.layer, x, use_reentrant=self.mode == "reentrant")


def _fsdp_gdn_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))
        for mode in ("plain", "non_reentrant", "reentrant"):
            torch.manual_seed(11)
            root = _Root(mode)
            reference = _Root("plain")
            reference.load_state_dict(root.state_dict())
            original_fp32 = {root.layer.gdn.A_log, root.layer.gdn.dt_bias}
            options = dict(
                mesh=mesh, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
            )
            fully_shard(root.layer, **options, ignored_params=original_fp32)
            gdn = root.layer.gdn
            fp32_params = {gdn.A_log, gdn.dt_bias}
            fully_shard(
                gdn,
                mesh=mesh,
                reshard_after_forward=False,
                ignored_params={param for param in gdn.parameters() if param not in fp32_params},
            )
            fully_shard(root, **options)
            assert isinstance(root.layer, FSDPModule) and isinstance(gdn, FSDPModule)
            assert gdn.A_log.dtype == torch.float32 and gdn.dt_bias.dtype == torch.float32
            x = torch.ones(2, 4, requires_grad=True)
            out = root(x)
            assert gdn.dtypes == (torch.bfloat16, torch.bfloat16, torch.float32, torch.float32)
            out.float().sum().backward()
            reference.layer.gdn.lin.to(torch.bfloat16)
            expected = reference(x.detach().to(torch.bfloat16).requires_grad_())
            expected.float().sum().backward()
            torch.testing.assert_close(out.float(), expected.float(), rtol=0, atol=0)
            for name in ("A_log", "dt_bias"):
                param = getattr(gdn, name)
                assert param.grad is not None and param.grad.dtype == torch.float32
                torch.testing.assert_close(
                    param.grad.full_tensor(), getattr(reference.layer.gdn, name).grad, rtol=0, atol=0
                )
    finally:
        dist.destroy_process_group()


def test_gdn_fp32_nested_fsdp2_cpu_forward_backward(tmp_path):
    mp.spawn(_fsdp_gdn_worker, args=(str(tmp_path / "gloo"),), nprocs=2, join=True)
