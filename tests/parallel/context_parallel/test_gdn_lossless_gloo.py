import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from veomni.distributed.context_parallel.gdn_lossless import (
    GdnLosslessRuntimePlan,
    attach_state_dependency,
    compile_gdn_lossless_runtime_plan,
    exchange_conv_halo,
    make_state_participation,
    make_state_template,
    owned_to_physical,
    physical_to_owned,
    receive_initial_state,
    send_final_state,
    trim_conv_halo,
)
from veomni.distributed.context_parallel.gdn_ownership import build_gdn_lossless_plan
from veomni.distributed.context_parallel.gdn_runtime import make_gdn_cp_runtime_observer


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _physical_values(plan, rank: int) -> torch.Tensor:
    rows: list[float] = []
    valid_offset = 0
    for valid_length, ring_length in zip(plan.valid_lengths, plan.ring_physical_lengths):
        half = ring_length // (2 * plan.cp_size)
        physical = [float(valid_offset + index + 1) for index in range(valid_length)]
        physical.extend([0.0] * (ring_length - valid_length))
        rows.extend(physical[rank * half : (rank + 1) * half])
        second = 2 * plan.cp_size - 1 - rank
        rows.extend(physical[second * half : (second + 1) * half])
        valid_offset += valid_length
    return torch.tensor(rows, dtype=torch.float64).unsqueeze(-1)


def test_state_receive_preserves_zero_participation_gradient_for_bos_owner():
    global_plan = build_gdn_lossless_plan([64], cp_size=2)
    plan = GdnLosslessRuntimePlan(global_plan=global_plan, local=global_plan.rank_plan(0))
    physical_input = torch.ones(2, dtype=torch.float64, requires_grad=True)

    initial_state = receive_initial_state(
        plan=plan,
        cp_group=dist.group.WORLD,
        state_template=torch.zeros(1, 1, 1, 1, dtype=torch.float64),
        participation=physical_input,
    )
    initial_state.sum().backward()

    assert physical_input.grad is not None
    torch.testing.assert_close(physical_input.grad, torch.zeros_like(physical_input))


def _run_gloo_contract(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        plan = compile_gdn_lossless_runtime_plan([65, 128], cp_group=dist.group.WORLD, ulysses_size=2)
        observer = make_gdn_cp_runtime_observer("state_passing_lossless", plan=plan)
        physical = _physical_values(plan.global_plan, rank).requires_grad_()
        owned = physical_to_owned(physical, plan=plan, cp_group=dist.group.WORLD, sequence_dim=0, observer=observer)
        restored = owned_to_physical(
            owned * 3, plan=plan, cp_group=dist.group.WORLD, sequence_dim=0, observer=observer
        )
        expected = physical.detach() * 3
        torch.testing.assert_close(restored, expected, rtol=0, atol=0)
        restored.sum().backward()
        expected_grad = torch.where(
            physical.detach() == 0,
            torch.zeros_like(physical),
            torch.full_like(physical, 3),
        )
        torch.testing.assert_close(physical.grad, expected_grad, rtol=0, atol=0)

        dist.barrier()
        local_parameter = torch.full((2, 1, 1, 1), float(rank + 1), dtype=torch.float64, requires_grad=True)
        template = torch.zeros_like(local_parameter)
        initial = receive_initial_state(
            plan=plan,
            cp_group=dist.group.WORLD,
            state_template=template,
            participation=local_parameter.sum(),
            observer=observer,
        )
        final = initial + local_parameter
        sent = send_final_state(final, plan=plan, cp_group=dist.group.WORLD, observer=observer)
        if rank == world_size - 1:
            loss = final.square().sum()
        else:
            loss = attach_state_dependency(final.new_zeros(()), sent)
        loss.backward()
        torch.testing.assert_close(local_parameter.grad, torch.full_like(local_parameter, 6.0), rtol=0, atol=0)

        dist.barrier()
        local = torch.arange(
            plan.local.owned_token_count,
            dtype=torch.float64,
            requires_grad=True,
        ).unsqueeze(-1)
        local.retain_grad()
        with_halo, _ = exchange_conv_halo(
            local,
            plan=plan,
            cp_group=dist.group.WORLD,
            kernel_size=4,
            sequence_dim=0,
            observer=observer,
        )
        trimmed = trim_conv_halo(
            with_halo,
            plan=plan,
            kernel_size=4,
            sequence_dim=0,
        )
        torch.testing.assert_close(trimmed, local, rtol=0, atol=0)
        if rank == world_size - 1:
            halo_loss = with_halo[:3].sum()
        else:
            halo_loss = with_halo.sum() * 0
        halo_loss.backward()
        expected_halo_grad = torch.zeros_like(local)
        if rank == 0:
            first_sample_end = plan.local.samples[0].local_end
            expected_halo_grad[first_sample_end - 3 : first_sample_end] = 1
        torch.testing.assert_close(local.grad, expected_halo_grad, rtol=0, atol=0)
        snapshot = observer.snapshot()
        assert snapshot.identity.layout == "lossless_sparse_packed"
        assert snapshot.balanced
        operations = {event.operation for event in snapshot.events}
        assert "ownership_a2a" in operations
        if rank == 0:
            assert "state_p2p_send" in operations
            assert "halo_p2p_send" in operations
        else:
            assert "state_p2p_recv" in operations
            assert "halo_p2p_recv" in operations
    finally:
        dist.destroy_process_group()


def _run_empty_owner_state_backward(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        plan = compile_gdn_lossless_runtime_plan([64], cp_group=dist.group.WORLD)
        source_tokens = plan.local.source_token_count
        physical_inputs = (
            torch.randn(1, source_tokens, 1, 2, requires_grad=True),
            torch.randn(1, source_tokens, 1, 2, requires_grad=True),
            torch.randn(1, source_tokens, 1, 2, requires_grad=True),
            torch.randn(1, source_tokens, 1, requires_grad=True),
            torch.randn(1, source_tokens, 1, requires_grad=True),
        )
        query, key, value, g, beta = (
            physical_to_owned(tensor, plan=plan, cp_group=dist.group.WORLD) for tensor in physical_inputs
        )
        cu_seqlens = torch.tensor(plan.local.owned_cu_seqlens, dtype=torch.int32)
        initial_state = receive_initial_state(
            plan=plan,
            cp_group=dist.group.WORLD,
            state_template=make_state_template(query, value, cu_seqlens),
            participation=make_state_participation(query, key, value, g, beta),
        )
        core_output = attach_state_dependency(value + query, initial_state)
        physical_output = owned_to_physical(core_output, plan=plan, cp_group=dist.group.WORLD)
        physical_output.sum().backward()

        for tensor in physical_inputs:
            assert tensor.grad is not None
    finally:
        dist.destroy_process_group()


def test_gloo_repartition_state_and_halo_autograd():
    world_size = 2
    mp.spawn(_run_gloo_contract, args=(world_size, _free_port()), nprocs=world_size, join=True)


def test_state_passing_empty_owner_preserves_all_input_a2a_backwards():
    world_size = 2
    mp.spawn(_run_empty_owner_state_backward, args=(world_size, _free_port()), nprocs=world_size, join=True)
