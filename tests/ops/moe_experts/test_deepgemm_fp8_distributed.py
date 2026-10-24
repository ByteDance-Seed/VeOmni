# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Real NCCL EP dispatch and FSDP2 resharding for Hopper/Blackwell FP8 experts."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from scripts.ci.accelerator.launch_plan import from_parameter, runner_device_count
from tests.ops.moe_experts.reference import standard_ep_reference
from tests.ops.moe_experts.test_deepgemm_fp8 import _relative_error
from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
from veomni.ops.kernels.moe_experts.standard.deepgemm import wrapper


class _Experts(nn.Module):
    def __init__(self, w1, w2, experts, use_checkpoint=False):
        super().__init__()
        self.fc1 = nn.Parameter(w1.clone())
        self.fc2 = nn.Parameter(w2.clone())
        self.experts = experts
        self.use_checkpoint = use_checkpoint

    def forward(self, x, routing, indices):
        from torch.utils.checkpoint import checkpoint

        # Checkpoint inside FSDP hooks so original and recomputed inputs share
        # the same mixed-precision casts (including routing weights).
        if self.use_checkpoint:
            return checkpoint(self._forward, x, routing, indices, use_reentrant=False)
        return self._forward(x, routing, indices)

    def _forward(self, x, routing, indices):
        empty = x.new_empty(0)
        return wrapper(
            x,
            routing,
            indices,
            empty,
            empty,
            self.fc2.to(x.dtype),
            self.fc1.to(x.dtype),
            num_experts=self.experts,
            swiglu_limit=10.0,
            assume_distinct_experts=False,
        )


def _worker(rank, world, ep_size, use_fsdp, repeated, use_checkpoint, rendezvous):
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import Shard

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=world, timeout=timedelta(seconds=180)
    )
    try:
        state = _init_parallel_state(dp_size=world, dp_shard_size=world, extra_parallel_sizes=(ep_size,))
        torch.manual_seed(104)
        e, t, h, intermediate = max(4, ep_size), 131, 256, 128
        x = torch.randn(world, t, h, device="cuda", dtype=torch.bfloat16)
        routing = torch.rand(world, t, 2, device="cuda")
        indices = torch.randn(world, t, e - 1, device="cuda").topk(2, dim=-1).indices + 1
        if repeated:
            # Hash routers can select the same expert twice. EP must dispatch
            # one token row and combine both slot weights and their gradients.
            indices[:, ::2, 1] = indices[:, ::2, 0]
        grad = torch.randn_like(x) * 0.1
        w1 = (torch.randn(e, 2 * intermediate, h, device="cuda") * 0.03).bfloat16().float()
        w2 = (torch.randn(e, h, intermediate, device="cuda") * 0.03).bfloat16().float()
        expert_rank = state.ep_rank if ep_size > 1 else 0
        begin, end = expert_rank * e // ep_size, (expert_rank + 1) * e // ep_size
        module = _Experts(w1[begin:end], w2[begin:end], e, use_checkpoint)
        if use_fsdp:
            mesh = (
                state.extra_parallel_fsdp_device_mesh["ep"]["ep_fsdp"]
                if ep_size > 1
                else state.device_mesh["dp_shard_sp"]
            )
            fully_shard(
                module,
                mesh=mesh,
                reshard_after_forward=True,
                mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
                shard_placement_fn=lambda param: Shard(1 if ep_size > 1 else 0),
            )
            if ep_size > 1:
                module.set_gradient_divide_factor(state.ep_gradient_divide_factor)
        opt = torch.optim.SGD(module.parameters(), lr=0.01)
        divisor = world if use_fsdp else 1
        for step in range(2):
            xi, ri = x[rank].clone().requires_grad_(), routing[rank].clone().requires_grad_()
            opt.zero_grad(set_to_none=True)
            y = module(xi, ri, indices[rank])
            assert y.dtype == xi.dtype
            y.backward(grad[rank])
            xr, rr, wr1, wr2 = [a.detach().clone().requires_grad_() for a in (x, routing, w1, w2)]
            gate, up = wr1.bfloat16().chunk(2, dim=1)
            yr = standard_ep_reference(
                xr.flatten(0, 1),
                rr.flatten(0, 1),
                indices.flatten(0, 1),
                gate,
                up,
                wr2.bfloat16(),
                num_experts=e,
                swiglu_limit=10.0,
            )
            yr.backward(grad.flatten(0, 1))
            _relative_error(y, yr.reshape_as(x)[rank], 0.085, f"step {step} output")
            _relative_error(xi.grad, xr.grad[rank], 0.085, f"step {step} input gradient")
            _relative_error(ri.grad, rr.grad[rank], 0.085, f"step {step} routing gradient")
            for param, reference, label in ((module.fc1, wr1, "gate_up"), (module.fc2, wr2, "down")):
                actual = param.grad.full_tensor() if use_fsdp else param.grad
                _relative_error(actual, reference.grad[begin:end] / divisor, 0.085, f"step {step} {label}")
                if begin == 0:
                    assert torch.count_nonzero(actual[0]) == 0
            opt.step()
            # Use the actual updated master weights for the next reference.
            # This isolates stale quantized-weight caches from optimizer drift.
            local = [p.full_tensor().detach() if use_fsdp else p.detach() for p in (module.fc1, module.fc2)]
            if ep_size > 1:
                for full, part in zip((w1, w2), local):
                    dist.all_gather_into_tensor(full, part.contiguous(), group=state.ep_group)
            else:
                w1, w2 = local
        dist.barrier()
    finally:
        dist.destroy_process_group()
        clear_parallel_state()


@pytest.mark.launch_plan(provider=from_parameter("world"))
@pytest.mark.parametrize(
    "world,ep_size,use_fsdp,repeated,use_checkpoint",
    [
        pytest.param(
            *case,
            marks=pytest.mark.skipif(runner_device_count() < case[0], reason=f"Requires {case[0]} GPUs."),
        )
        for case in (
            (2, 2, False, False, False),
            (2, 1, True, False, False),
            (4, 2, True, False, False),
            (4, 2, True, True, False),
            (4, 4, False, True, False),
            (8, 8, False, True, False),
            (2, 1, True, False, True),
            (4, 2, True, True, True),
        )
    ],
)
def test_distributed_fp8_experts(world, ep_size, use_fsdp, repeated, use_checkpoint, tmp_path):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVIDIA Hopper or Blackwell is required.")
    if torch.cuda.device_count() < world or any(
        torch.cuda.get_device_capability(i) not in ((9, 0), (10, 0)) for i in range(world)
    ):
        pytest.skip(f"Requires {world} Hopper/Blackwell GPUs.")
    pytest.importorskip("deep_gemm")
    mp.spawn(
        _worker,
        args=(world, ep_size, use_fsdp, repeated, use_checkpoint, f"file://{tmp_path / 'rdzv'}"),
        nprocs=world,
        join=True,
    )
