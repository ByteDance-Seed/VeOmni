# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Real NCCL EP dispatch and FSDP2 resharding for Hopper/Blackwell FP8 experts."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.tensor import DTensor

from scripts.ci.accelerator.launch_plan import from_parameter, runner_device_count
from tests.ops.moe_experts.reference import standard_ep_reference
from tests.ops.moe_experts.test_deepgemm_fp8 import _relative_error
from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
from veomni.ops import VeomniOp


class _Experts(nn.Module):
    def __init__(self, w1, w2, experts, use_checkpoint=False):
        super().__init__()
        self.fc1 = nn.Parameter(w1.clone())
        self.fc2 = nn.Parameter(w2.clone())
        self.veomni_moe = VeomniOp("moe_experts", "standard", "deepgemm_fp8")
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
        return self.veomni_moe(
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


def _worker(rank, world, ep_size, use_fsdp, repeated, use_checkpoint, fp8_all_gather, use_offload, rendezvous):
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import Shard

    from veomni.distributed.offloading import build_activation_offloading_context

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
        w1 = torch.randn(e, 2 * intermediate, h, device="cuda") * 0.03
        w2 = torch.randn(e, h, intermediate, device="cuda") * 0.03
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
        if fp8_all_gather:
            from veomni.distributed.fsdp2.fp8_all_gather import register_deepgemm_fp8_all_gather

            assert register_deepgemm_fp8_all_gather(module) == 2
        expected_weights = []
        if fp8_all_gather:
            from veomni.ops.kernels.moe_experts.shared.deepgemm_fp8 import quantize_weights
            from veomni.ops.kernels.moe_experts.shared.fp8_weight import FP8ExpertWeight

            def check_gathered_weights(mod, args):
                for weight, (qdata, scale, _, _) in zip((mod.fc1, mod.fc2), expected_weights):
                    assert isinstance(weight, FP8ExpertWeight)
                    torch.testing.assert_close(weight.qdata.float(), qdata.float(), rtol=0, atol=0)
                    torch.testing.assert_close(weight.scale, scale, rtol=0, atol=0)

            module.register_forward_pre_hook(check_gathered_weights)
        opt = torch.optim.SGD(module.parameters(), lr=0.01)
        divisor = world if use_fsdp else 1
        forward_context, backward_context = build_activation_offloading_context(use_offload, use_checkpoint)
        for step in range(2):
            xi, ri = x[rank].clone().requires_grad_(), routing[rank].clone().requires_grad_()
            opt.zero_grad(set_to_none=True)
            if fp8_all_gather:
                expected_weights[:] = [
                    quantize_weights(p.full_tensor().bfloat16(), store_transpose=False)
                    for p in (module.fc1, module.fc2)
                ]
            with forward_context:
                y = module(xi, ri, indices[rank])
            assert y.dtype == xi.dtype
            with backward_context:
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
            if fp8_all_gather:
                # Normal state-dict copy/load must retain the extensions and
                # store only high precision shards, never the transport wrapper.
                saved = {name: value.detach().clone() for name, value in module.state_dict().items()}
                assert all(value.dtype == torch.float32 for value in saved.values())
                module.load_state_dict(saved)

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
    "world,ep_size,use_fsdp,repeated,use_checkpoint,fp8_all_gather,use_offload",
    [
        pytest.param(
            *case,
            marks=pytest.mark.skipif(runner_device_count() < case[0], reason=f"Requires {case[0]} GPUs."),
        )
        for case in (
            (2, 2, False, False, False, False, False),
            (2, 1, True, False, False, False, False),
            (4, 2, True, False, False, False, False),
            (4, 2, True, True, False, False, False),
            (4, 4, False, True, False, False, False),
            (8, 8, False, True, False, False, False),
            (2, 1, True, False, True, False, False),
            (4, 2, True, True, True, False, False),
            (2, 1, True, False, False, True, False),
            (4, 2, True, True, False, True, False),
            (4, 2, True, True, True, True, False),
            (4, 2, True, True, False, True, True),
            (4, 2, True, True, True, True, True),
        )
    ],
)
def test_distributed_fp8_experts(
    world, ep_size, use_fsdp, repeated, use_checkpoint, fp8_all_gather, use_offload, tmp_path
):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVIDIA Hopper or Blackwell is required.")
    if torch.cuda.device_count() < world or any(
        torch.cuda.get_device_capability(i) not in ((9, 0), (10, 0)) for i in range(world)
    ):
        pytest.skip(f"Requires {world} Hopper/Blackwell GPUs.")
    pytest.importorskip("deep_gemm")
    mp.spawn(
        _worker,
        args=(
            world,
            ep_size,
            use_fsdp,
            repeated,
            use_checkpoint,
            fp8_all_gather,
            use_offload,
            f"file://{tmp_path / 'rdzv'}",
        ),
        nprocs=world,
        join=True,
    )


class _MetaModel(nn.Module):
    _no_split_modules = ["_Experts"]

    def __init__(self):
        super().__init__()
        self.experts = _Experts(torch.empty(4, 256, 256), torch.empty(4, 256, 128), 4, use_checkpoint=True)

    def get_parallel_plan(self):
        from torch.distributed.tensor import Shard

        from veomni.distributed.parallel_plan import ParallelPlan

        return ParallelPlan({"ep": {"experts.fc1": Shard(0), "experts.fc2": Shard(0)}})

    def init_weights(self):
        with torch.no_grad():
            for p in self.parameters():
                p.to_local().normal_(std=0.03)

    def forward(self, x, routing, indices):
        return self.experts(x, routing, indices)


def _builder_worker(rank, world, shard_dim, directory, rendezvous):
    import torch.distributed.checkpoint as dcp

    from veomni.checkpoint.dcp_checkpointer import (
        ModelState,
        OptimizerState,
        _ExtraParallelSavePlanner,
        _ModelStrictLoadPlanner,
    )
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.ops.kernels.moe_experts.shared.fp8_weight import FP8ExpertWeight
    from veomni.optim import build_optimizer

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=world, timeout=timedelta(seconds=180)
    )
    try:
        _init_parallel_state(dp_size=world, dp_shard_size=world, extra_parallel_sizes=(2,))
        torch.manual_seed(401)
        with torch.device("meta"):
            model = _MetaModel()
        model = build_parallelize_model(
            model,
            init_device="meta",
            fp8_all_gather=True,
            expert_fsdp_shard_dim=shard_dim,
            enable_gradient_checkpointing=False,
        )
        assert all(p.placements[0].dim == shard_dim for p in model.parameters())
        assert all(spec.fsdp_shard_dim == shard_dim for spec in model._fqn2spec_info.values())
        opt = build_optimizer(model, lr=1e-3)
        x = torch.randn(127, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        routing = torch.rand(127, 2, device="cuda", requires_grad=True)
        indices = torch.randint(0, 4, (127, 2), device="cuda")
        expected = None
        for step in range(3):
            opt.zero_grad()
            # Two accumulated microbatches, synchronizing only the last.
            model.set_requires_gradient_sync(False)
            model(x, routing, indices).float().square().mean().backward()
            model.set_requires_gradient_sync(True)
            y = model(x, routing, indices)
            y.float().square().mean().backward()
            for param in model.parameters():
                assert param.dtype == torch.float32 and torch.isfinite(param.grad.to_local()).all()
                assert not isinstance(param.to_local(), FP8ExpertWeight)
            if expected is not None:
                torch.testing.assert_close(y, expected, rtol=0, atol=0)
            if step == 0:
                opt.step()
                saved_optim = {
                    (group, param, key): value.detach().clone()
                    for group, sub in opt.optimizers_dict.items()
                    for param, state in sub.state.items()
                    for key, value in state.items()
                }
                dcp.save(
                    {"model": ModelState(model), "optimizer": OptimizerState(model, opt)},
                    checkpoint_id=directory,
                    planner=_ExtraParallelSavePlanner(),
                )
            elif step == 1:
                expected = y.detach().clone()
                with torch.no_grad():
                    for p in model.parameters():
                        p.zero_()
                for sub in opt.optimizers_dict.values():
                    for values in sub.state.values():
                        for value in values.values():
                            value.add_(1)
                dcp.load(
                    {"model": ModelState(model), "optimizer": OptimizerState(model, opt, load=True)},
                    checkpoint_id=directory,
                    planner=_ModelStrictLoadPlanner(strict_model=True),
                )
                for (group, param, key), expected_value in saved_optim.items():
                    actual_value = opt.optimizers_dict[group].state[param][key]
                    torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
        # DCP must restore the same global experts when the FSDP layout changes.
        with torch.device("meta"):
            restored = _MetaModel()
        restored = build_parallelize_model(
            restored,
            init_device="meta",
            fp8_all_gather=True,
            expert_fsdp_shard_dim=1 - shard_dim,
            enable_gradient_checkpointing=False,
        )
        restored_opt = build_optimizer(restored, lr=1e-3)
        dcp.load(
            {"model": ModelState(restored), "optimizer": OptimizerState(restored, restored_opt, load=True)},
            checkpoint_id=directory,
            planner=_ModelStrictLoadPlanner(strict_model=True),
        )
        parameter_map = dict(zip(model.parameters(), restored.parameters()))
        for (group, param, key), expected_value in saved_optim.items():
            actual_value = restored_opt.optimizers_dict[group].state[parameter_map[param]][key]
            if isinstance(actual_value, DTensor):
                actual_value, expected_value = actual_value.full_tensor(), expected_value.full_tensor()
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
        for original, loaded in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(original.full_tensor(), loaded.full_tensor(), rtol=0, atol=0)
        restored_y = restored(x, routing, indices)
        torch.testing.assert_close(restored_y, expected, rtol=0, atol=0)
        restored_y.float().square().mean().backward()
        for original, loaded in zip(model.parameters(), restored.parameters()):
            # The original accumulated two identical microbatches.
            torch.testing.assert_close(
                original.grad.full_tensor() / 2, loaded.grad.full_tensor(), rtol=1e-5, atol=1e-7
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()
        clear_parallel_state()


@pytest.mark.launch_plan(provider=from_parameter("world"))
@pytest.mark.parametrize("world", [4])
@pytest.mark.parametrize("shard_dim", [0, 1])
def test_fp8_all_gather_public_builder_checkpoint_and_accumulation(world, shard_dim, tmp_path):
    """Cover meta materialization, EP/FSDP wiring, AdamW and real DCP reload."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < world or torch.version.hip is not None:
        pytest.skip("Requires four NVIDIA GPUs.")
    if any(torch.cuda.get_device_capability(i) not in ((9, 0), (10, 0)) for i in range(world)):
        pytest.skip("Requires Hopper/Blackwell.")
    pytest.importorskip("deep_gemm")
    mp.spawn(
        _builder_worker,
        args=(world, shard_dim, str(tmp_path / "dcp"), f"file://{tmp_path / 'rdzv'}"),
        nprocs=world,
        join=True,
    )
