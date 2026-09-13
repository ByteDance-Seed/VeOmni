# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import weakref

import pytest
import torch
import torch.distributed as dist

from tests.tools.launch_utils import torchrun
from veomni.distributed.moe.comm import AsyncCollectiveHandle
from veomni.utils.device import IS_NPU_AVAILABLE


pytestmark = pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="NPU cached MoE backward requires torch_npu")


def test_async_collective_handle_releases_completed_work() -> None:
    class FakeWork:
        def wait(self) -> None:
            pass

    handle = AsyncCollectiveHandle()
    work = FakeWork()
    work_ref = weakref.ref(work)
    handle.set_work(work)
    del work

    handle.wait()

    assert work_ref() is None


def _clone_leaf(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_(True)


def _cached_backward_worker() -> None:
    from veomni.distributed.moe.comm import (
        all_to_all,
        moe_backward_collective_callback,
        moe_backward_phase_callback,
        register_expert_backward_boundary,
    )
    from veomni.ops.kernels.moe.npu_group_gemm import (
        npu_ep_combine_async,
        npu_ep_combine_wait,
        npu_ep_dispatch_async,
        npu_ep_dispatch_input_backward,
        npu_ep_dispatch_input_prepare,
        npu_ep_dispatch_prepare,
        npu_ep_dispatch_wait,
        npu_ep_expert_forward,
    )

    rank = dist.get_rank()
    group = dist.group.WORLD
    torch.manual_seed(1234 + rank)
    sync_phases = []
    sync_collectives = []
    sync_input = torch.randn(4, 2, device="npu", dtype=torch.bfloat16, requires_grad=True)
    with (
        moe_backward_phase_callback(sync_phases.append),
        moe_backward_collective_callback(lambda phase, boundary: sync_collectives.append(f"{phase}:{boundary}")),
    ):
        sync_output = all_to_all(group, sync_input, phase="dispatch")
    sync_output.sum().backward()
    assert sync_phases == ["dispatch"], sync_phases
    assert sync_collectives == ["dispatch:before", "dispatch:after"], sync_collectives

    hidden = torch.randn(6, 8, device="npu", dtype=torch.bfloat16)
    selected = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]],
        device="npu",
        dtype=torch.int64,
    )
    if rank:
        selected = (selected + 1) % 4
    routing = torch.rand(6, 2, device="npu", dtype=torch.float32)
    routing = (routing / routing.sum(dim=-1, keepdim=True)).to(torch.bfloat16)
    down = torch.randn(2, 8, 4, device="npu", dtype=torch.bfloat16)
    gate_up = torch.randn(2, 8, 8, device="npu", dtype=torch.bfloat16)
    output_grad = torch.randn(6, 8, device="npu", dtype=torch.bfloat16)

    hidden_ref, routing_ref, down_ref, gate_up_ref = map(_clone_leaf, (hidden, routing, down, gate_up))
    dispatch_ref = npu_ep_dispatch_async(hidden_ref, selected, 4, group)
    dispatched_ref = npu_ep_dispatch_wait(dispatch_ref)
    expert_ref = npu_ep_expert_forward(dispatched_ref, dispatch_ref, down_ref, gate_up_ref)
    combine_ref = npu_ep_combine_async(expert_ref, routing_ref, dispatch_ref)
    output_ref = npu_ep_combine_wait(combine_ref)
    (output_ref * output_grad).sum().backward()

    phases = []
    with moe_backward_phase_callback(phases.append):
        routing_cached, down_cached, gate_up_cached = map(_clone_leaf, (routing, down, gate_up))
        dispatch_plan = npu_ep_dispatch_prepare(selected, 4, group)
        prepared_dispatch = npu_ep_dispatch_input_prepare(hidden.detach(), selected, dispatch_plan)
        dispatch_cached = npu_ep_dispatch_async(
            hidden.detach(),
            selected,
            4,
            group,
            plan=dispatch_plan,
            prepared_input=prepared_dispatch,
        )
        expert_input = register_expert_backward_boundary(
            npu_ep_dispatch_wait(dispatch_cached, synchronize=False).detach().requires_grad_(True),
            callback=phases.append,
        )
        expert_cached = npu_ep_expert_forward(expert_input, dispatch_cached, down_cached, gate_up_cached)
        combine_cached = npu_ep_combine_async(
            expert_cached,
            routing_cached,
            dispatch_cached,
            backward_phase_callback=phases.append,
        )
        output_cached = npu_ep_combine_wait(combine_cached)
        (output_cached * output_grad).sum().backward()
    hidden_grad_cached = npu_ep_dispatch_input_backward(expert_input.grad, dispatch_cached)

    assert phases == ["combine", "experts"], phases

    for actual, expected in (
        (output_cached, output_ref),
        (hidden_grad_cached, hidden_ref.grad),
        (routing_cached.grad, routing_ref.grad),
        (down_cached.grad, down_ref.grad),
        (gate_up_cached.grad, gate_up_ref.grad),
    ):
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)


def test_npu_ep_cached_backward_matches_autograd() -> None:
    torchrun(_cached_backward_worker, world_size=2)
