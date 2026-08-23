"""Packed mixed-length state-passing parity gate for lossless GDN CP.

The production route keeps one recurrent-state row per packed sample while
physical tokens are routed to native-chunk owners.  Existing Gloo coverage
checks ownership and P2P contracts, but its toy scan treats every packed
sample as one concatenated stream.  This gate deliberately uses distinct
sample boundaries and initial states so a missing per-sample reset is visible
in both forward values and VJP gradients.
"""

from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from veomni.distributed.context_parallel.gdn_lossless import (
    GdnLosslessRuntimePlan,
    compile_gdn_lossless_runtime_plan,
    make_state_participation,
    owned_to_physical,
    physical_to_owned,
    receive_initial_state,
    send_final_state,
)


_DECAY = 0.73
_DIMS = 3


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _physical_indices(plan, rank: int) -> list[int]:
    """Map this rank's zigzag physical rows to global valid-token ordinals."""

    indices: list[int] = []
    valid_offset = 0
    for valid_length, ring_length in zip(
        plan.global_plan.valid_lengths,
        plan.global_plan.ring_physical_lengths,
    ):
        padded = list(range(valid_offset, valid_offset + valid_length))
        padded.extend([-1] * (ring_length - valid_length))
        half = ring_length // (2 * plan.cp_size)
        indices.extend(padded[rank * half : (rank + 1) * half])
        second = 2 * plan.cp_size - 1 - rank
        indices.extend(padded[second * half : (second + 1) * half])
        valid_offset += valid_length
    if len(indices) != plan.local.source_token_count:
        raise AssertionError("physical index count does not match the compiled plan")
    return indices


def _physical_shard(global_tokens: torch.Tensor, plan, rank: int) -> torch.Tensor:
    """Materialize a rank's physical shard, preserving ring padding as zero."""

    rows = []
    for index in _physical_indices(plan, rank):
        rows.append(global_tokens.new_zeros((_DIMS,)) if index < 0 else global_tokens[index])
    if not rows:
        return global_tokens.new_empty((1, 0, _DIMS))
    return torch.stack(rows, dim=0).unsqueeze(0)


def _scan_packed(tokens: torch.Tensor, lengths: tuple[int, ...], initial: torch.Tensor):
    """Reference recurrence with an independent state reset for each sample."""

    outputs: list[torch.Tensor] = []
    finals: list[torch.Tensor] = []
    offset = 0
    for sample, length in enumerate(lengths):
        state = initial[sample]
        for token in range(offset, offset + length):
            state = state * _DECAY + tokens[token]
            outputs.append(state)
        finals.append(state)
        offset += length
    output = torch.stack(outputs, dim=0).unsqueeze(0) if outputs else tokens.new_empty((1, 0, _DIMS))
    return output, torch.stack(finals, dim=0)


def _scan_owned(tokens: torch.Tensor, plan: GdnLosslessRuntimePlan, initial: torch.Tensor):
    """Run the same recurrence over this rank's per-sample owned ranges."""

    outputs: list[torch.Tensor] = []
    finals: list[torch.Tensor] = []
    cu = plan.local.owned_cu_seqlens
    for sample, (start, end) in enumerate(zip(cu, cu[1:])):
        state = initial[sample]
        for token in range(start, end):
            state = state * _DECAY + tokens[0, token]
            outputs.append(state)
        finals.append(state)
    output = torch.stack(outputs, dim=0).unsqueeze(0) if outputs else tokens.new_empty((1, 0, _DIMS))
    return output, torch.stack(finals, dim=0)


def _run_case(rank: int, lengths: tuple[int, ...], case_seed: int) -> None:
    world_size = dist.get_world_size()
    plan = compile_gdn_lossless_runtime_plan(
        lengths,
        cp_group=dist.group.WORLD,
        ulysses_size=1,
    )
    total_tokens = sum(lengths)
    generator = torch.Generator(device="cpu").manual_seed(case_seed)
    global_tokens = torch.randn(total_tokens, _DIMS, generator=generator, dtype=torch.float64)
    global_initial = torch.randn(len(lengths), _DIMS, generator=generator, dtype=torch.float64)
    output_weight = torch.randn(total_tokens, _DIMS, generator=generator, dtype=torch.float64)
    final_weight = torch.randn(len(lengths), _DIMS, generator=generator, dtype=torch.float64)
    nonempty_mask = torch.tensor([length > 0 for length in lengths], dtype=final_weight.dtype).unsqueeze(-1)
    final_weight = final_weight * nonempty_mask

    physical = _physical_shard(global_tokens, plan, rank).detach().requires_grad_(True)
    physical_weight = _physical_shard(output_weight, plan, rank)
    local_initial = global_initial.detach().clone().requires_grad_(True)
    owned = physical_to_owned(
        physical,
        plan=plan,
        cp_group=dist.group.WORLD,
        sequence_dim=1,
    )
    state_template = torch.zeros(len(plan.local.samples), _DIMS, dtype=physical.dtype)
    received = receive_initial_state(
        plan=plan,
        cp_group=dist.group.WORLD,
        state_template=state_template,
        participation=make_state_participation(owned),
    )
    bos_mask = torch.tensor(
        [sample.is_bos_owner for sample in plan.local.samples],
        dtype=local_initial.dtype,
    ).unsqueeze(-1)
    initial = received + local_initial * bos_mask
    owned_output, local_final = _scan_owned(owned, plan, initial)
    sent_final = send_final_state(local_final, plan=plan, cp_group=dist.group.WORLD)
    physical_output = owned_to_physical(
        owned_output,
        plan=plan,
        cp_group=dist.group.WORLD,
        sequence_dim=1,
    )
    terminal_mask = torch.tensor(
        [sample.is_active and sample.successor_rank is None for sample in plan.local.samples],
        dtype=sent_final.dtype,
    ).unsqueeze(-1)
    local_loss = (physical_output * physical_weight).sum() + (sent_final * final_weight * terminal_mask).sum()
    local_loss.backward()

    oracle_tokens = global_tokens.detach().clone().requires_grad_(True)
    oracle_initial = global_initial.detach().clone().requires_grad_(True)
    oracle_output, oracle_final = _scan_packed(oracle_tokens, lengths, oracle_initial)
    oracle_loss = (oracle_output * output_weight.unsqueeze(0)).sum() + (oracle_final * final_weight).sum()
    oracle_loss.backward()

    expected_output = _physical_shard(oracle_output[0].detach(), plan, rank)
    torch.testing.assert_close(physical_output.detach(), expected_output, rtol=0, atol=1e-12)
    expected_grad = _physical_shard(oracle_tokens.grad.detach(), plan, rank)
    assert physical.grad is not None
    torch.testing.assert_close(physical.grad, expected_grad, rtol=0, atol=1e-12)

    gathered_final = sent_final.detach() * terminal_mask
    dist.all_reduce(gathered_final, op=dist.ReduceOp.SUM)
    torch.testing.assert_close(
        gathered_final * nonempty_mask,
        oracle_final.detach() * nonempty_mask,
        rtol=0,
        atol=1e-12,
    )

    assert local_initial.grad is not None
    dist.all_reduce(local_initial.grad, op=dist.ReduceOp.SUM)
    torch.testing.assert_close(local_initial.grad, oracle_initial.grad, rtol=0, atol=1e-12)

    # Every rank has exercised the same state P2P ordinals, including ranks
    # that are inactive for one or more packed samples.
    assert plan.cp_size == world_size


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        if world_size == 2:
            # The leading empty segment validates sample-index preservation and
            # the two non-empty lengths validate reset/partial-tail behavior.
            # A second shape covers the no-empty metadata path.
            cases = (
                ((0, 65, 128), 2026082301),
                ((65, 128), 2026082302),
            )
        elif world_size == 4:
            # Four ranks expose different chain depths: samples of 65/128
            # tokens use ranks 0->1, while 193 tokens use 0->1->2->3.
            cases = (((65, 128, 193), 2026082303),)
        else:
            raise AssertionError(f"unsupported Gloo gate world size {world_size}")
        for lengths, case_seed in cases:
            _run_case(rank, lengths, case_seed=case_seed)
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_gloo_packed_mixed_length_state_reset_and_full_vjp():
    """Catch cross-sample state leakage in ownership + state P2P + inverse A2A."""

    for world_size in (2, 4):
        mp.spawn(_worker, args=(world_size, _free_port()), nprocs=world_size, join=True)
