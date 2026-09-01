from __future__ import annotations

import os
import socket
import time
import traceback
from queue import Empty

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from veomni.distributed.context_parallel.gdn_headwise import (
    prepare_gdn_headwise_inputs,
    restore_gdn_headwise_output,
)
from veomni.distributed.context_parallel.packed_sharding import (
    PackedContextParallelPartition,
    apply_packed_context_parallel_partition,
    build_packed_context_parallel_partition,
    pad_packed_samples,
)


_WORLD_SIZE = 2
_KEY_HEADS = 4
_VALUE_HEADS = 8
_KEY_DIM = 3
_VALUE_DIM = 2


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _valid_inputs(
    token_count: int,
    *,
    key_heads: int = _KEY_HEADS,
    value_heads: int = _VALUE_HEADS,
) -> tuple[Tensor, ...]:
    generator = torch.Generator().manual_seed(20260825)
    return (
        torch.randn(1, token_count, key_heads, _KEY_DIM, dtype=torch.float64, generator=generator) * 0.1,
        torch.randn(1, token_count, key_heads, _KEY_DIM, dtype=torch.float64, generator=generator) * 0.1,
        torch.randn(1, token_count, value_heads, _VALUE_DIM, dtype=torch.float64, generator=generator) * 0.1,
        torch.randn(1, token_count, value_heads, dtype=torch.float64, generator=generator) * 0.05,
        torch.randn(1, token_count, value_heads, dtype=torch.float64, generator=generator) * 0.05,
    )


def _padded_inputs(inputs: tuple[Tensor, ...], cu_seqlens: Tensor, world_size: int) -> tuple[Tensor, ...]:
    padded: list[Tensor] = []
    expected_cu: Tensor | None = None
    for index, tensor in enumerate(inputs):
        padded_tensor, padded_cu = pad_packed_samples(
            tensor,
            cu_seqlens,
            multiple=2 * world_size,
            dim=1,
            pad_value=100.0 + index,
        )
        if expected_cu is None:
            expected_cu = padded_cu
        else:
            torch.testing.assert_close(padded_cu, expected_cu, rtol=0, atol=0)
        padded.append(padded_tensor)
    return tuple(padded)


def _padded_cu_seqlens(cu_seqlens: Tensor, world_size: int) -> Tensor:
    probe = torch.empty(1, int(cu_seqlens[-1]), 1)
    _, padded_cu = pad_packed_samples(probe, cu_seqlens, multiple=2 * world_size, dim=1)
    return padded_cu


def _physical_partition(
    cu_seqlens: Tensor,
    rank: int,
    world_size: int,
    *,
    cp_size: int | None = None,
) -> PackedContextParallelPartition:
    cp_size = world_size if cp_size is None else cp_size
    assert world_size % cp_size == 0
    ulysses_size = world_size // cp_size
    cp_rank, ulysses_rank = divmod(rank, ulysses_size)
    return build_packed_context_parallel_partition(
        _padded_cu_seqlens(cu_seqlens, world_size),
        cp_size=cp_size,
        cp_rank=cp_rank,
        ulysses_size=ulysses_size,
        ulysses_rank=ulysses_rank,
    )


def _physical_inputs(
    inputs: tuple[Tensor, ...],
    cu_seqlens: Tensor,
    *,
    rank: int,
    world_size: int,
    requires_grad: bool,
    cp_size: int | None = None,
) -> tuple[tuple[Tensor, ...], PackedContextParallelPartition]:
    partition = _physical_partition(cu_seqlens, rank, world_size, cp_size=cp_size)
    physical = tuple(
        apply_packed_context_parallel_partition(tensor, partition, dim=1)
        .detach()
        .clone()
        .requires_grad_(requires_grad)
        for tensor in _padded_inputs(inputs, cu_seqlens, world_size)
    )
    return physical, partition


def _head_shard(tensor: Tensor, rank: int, world_size: int) -> Tensor:
    heads = tensor.size(2)
    assert heads % world_size == 0
    return tensor.narrow(2, rank * (heads // world_size), heads // world_size).contiguous()


def _pad_with_zeros(tensor: Tensor, cu_seqlens: Tensor, world_size: int) -> Tensor:
    padded, _ = pad_packed_samples(tensor, cu_seqlens, multiple=2 * world_size, dim=1, pad_value=0)
    return padded


def _toy_gdn(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    b: Tensor,
    a: Tensor,
    cu_seqlens: Tensor,
) -> Tensor:
    if value.size(1) == 0:
        dependency = query.sum() + key.sum() + b.sum() + a.sum()
        return value + dependency * 0

    value_heads = value.size(2)
    key_heads = key.size(2)
    assert value_heads % key_heads == 0
    repeat = value_heads // key_heads
    query = query.repeat_interleave(repeat, dim=2)
    key = key.repeat_interleave(repeat, dim=2)

    outputs: list[Tensor] = []
    points = [int(point) for point in cu_seqlens.tolist()]
    for start, end in zip(points, points[1:]):
        state = value.new_zeros(value.size(0), value_heads, key.size(-1), value.size(-1))
        for token_index in range(start, end):
            key_token = key[:, token_index]
            value_token = value[:, token_index]
            state = torch.sigmoid(b[:, token_index]).unsqueeze(-1).unsqueeze(-1) * state
            state = state + torch.einsum("bhd,bhe->bhde", key_token, value_token) * torch.sigmoid(
                a[:, token_index]
            ).unsqueeze(-1).unsqueeze(-1)
            outputs.append(torch.einsum("bhd,bhde->bhe", query[:, token_index], state))
    return torch.stack(outputs, dim=1)


class _CollectiveCounter:
    def __init__(self) -> None:
        self.single = 0
        self.list = 0
        self._real_single = dist.all_to_all_single
        self._real_list = dist.all_to_all

    def __enter__(self) -> _CollectiveCounter:
        def counted_single(*args, **kwargs):
            self.single += 1
            return self._real_single(*args, **kwargs)

        def forbidden_list(*args, **kwargs):
            self.list += 1
            raise AssertionError("headwise GDN must not use list all_to_all")

        dist.all_to_all_single = counted_single
        dist.all_to_all = forbidden_list
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        dist.all_to_all_single = self._real_single
        dist.all_to_all = self._real_list


def _prepare(
    inputs: tuple[Tensor, ...],
    cu_seqlens: Tensor,
    world_size: int,
    *,
    cp_size: int | None = None,
):
    return prepare_gdn_headwise_inputs(
        inputs,
        cu_seqlens=cu_seqlens,
        group=dist.group.WORLD,
        cp_size=world_size if cp_size is None else cp_size,
        sequence_dim=1,
        head_dim=2,
    )


def _run_layout_contract(rank: int, world_size: int) -> None:
    cu_seqlens = torch.tensor([0, 5, 14], dtype=torch.int32)
    valid_inputs = _valid_inputs(int(cu_seqlens[-1]))
    physical_inputs, partition = _physical_inputs(
        valid_inputs,
        cu_seqlens,
        rank=rank,
        world_size=world_size,
        requires_grad=False,
    )

    with _CollectiveCounter() as collectives:
        prepared_inputs, layout = _prepare(physical_inputs, cu_seqlens, world_size)
        for prepared, expected in zip(prepared_inputs, valid_inputs):
            torch.testing.assert_close(prepared, _head_shard(expected, rank, world_size), rtol=0, atol=0)

        restored = restore_gdn_headwise_output(prepared_inputs[2], layout=layout, group=dist.group.WORLD)

    expected_restored = apply_packed_context_parallel_partition(
        _pad_with_zeros(valid_inputs[2], cu_seqlens, world_size),
        partition,
        dim=1,
    )
    torch.testing.assert_close(restored, expected_restored, rtol=0, atol=0)
    assert collectives.single == 2
    assert collectives.list == 0


def _run_transport_vjp_contract(rank: int, world_size: int) -> None:
    cu_seqlens = torch.tensor([0, 5, 14], dtype=torch.int32)
    valid_inputs = _valid_inputs(int(cu_seqlens[-1]))
    canonical_gradient = torch.arange(valid_inputs[2].numel(), dtype=torch.float64).reshape_as(valid_inputs[2]) + 1

    prepare_inputs, partition = _physical_inputs(
        valid_inputs,
        cu_seqlens,
        rank=rank,
        world_size=world_size,
        requires_grad=True,
    )
    with _CollectiveCounter() as prepare_collectives:
        prepared_inputs, layout = _prepare(prepare_inputs, cu_seqlens, world_size)
        (prepared_inputs[2] * _head_shard(canonical_gradient, rank, world_size)).sum().backward()
    expected_physical_gradient = apply_packed_context_parallel_partition(
        _pad_with_zeros(canonical_gradient, cu_seqlens, world_size),
        partition,
        dim=1,
    )
    assert prepare_inputs[2].grad is not None
    torch.testing.assert_close(prepare_inputs[2].grad, expected_physical_gradient, rtol=0, atol=0)
    assert prepare_collectives.single == 2
    assert prepare_collectives.list == 0

    canonical_output = _head_shard(valid_inputs[2], rank, world_size).detach().clone().requires_grad_()
    local_output_gradient = apply_packed_context_parallel_partition(
        _pad_with_zeros(canonical_gradient, cu_seqlens, world_size),
        partition,
        dim=1,
    )
    with _CollectiveCounter() as restore_collectives:
        physical_output = restore_gdn_headwise_output(
            canonical_output,
            layout=layout,
            group=dist.group.WORLD,
        )
        (physical_output * local_output_gradient).sum().backward()
    assert canonical_output.grad is not None
    torch.testing.assert_close(
        canonical_output.grad,
        _head_shard(canonical_gradient, rank, world_size),
        rtol=0,
        atol=0,
    )
    assert restore_collectives.single == 2
    assert restore_collectives.list == 0

    transport_inputs, _ = _physical_inputs(
        valid_inputs,
        cu_seqlens,
        rank=rank,
        world_size=world_size,
        requires_grad=True,
    )
    with _CollectiveCounter() as transport_collectives:
        transported_inputs, transport_layout = _prepare(transport_inputs, cu_seqlens, world_size)
        transported_output = restore_gdn_headwise_output(
            transported_inputs[2],
            layout=transport_layout,
            group=dist.group.WORLD,
        )
        (transported_output * local_output_gradient).sum().backward()
    assert transport_inputs[2].grad is not None
    torch.testing.assert_close(transport_inputs[2].grad, local_output_gradient, rtol=0, atol=0)
    assert transport_collectives.single == 4
    assert transport_collectives.list == 0


def _run_vjp_contract(rank: int, world_size: int) -> None:
    cu_seqlens = torch.tensor([0, 0, 5, 14], dtype=torch.int32)
    valid_inputs = _valid_inputs(int(cu_seqlens[-1]))
    physical_inputs, partition = _physical_inputs(
        valid_inputs,
        cu_seqlens,
        rank=rank,
        world_size=world_size,
        requires_grad=True,
    )
    generator = torch.Generator().manual_seed(17)
    output_gradient = torch.randn(
        1,
        int(cu_seqlens[-1]),
        _VALUE_HEADS,
        _VALUE_DIM,
        dtype=torch.float64,
        generator=generator,
    )
    local_output_gradient = apply_packed_context_parallel_partition(
        _pad_with_zeros(output_gradient, cu_seqlens, world_size),
        partition,
        dim=1,
    )

    with _CollectiveCounter() as collectives:
        prepared_inputs, layout = _prepare(physical_inputs, cu_seqlens, world_size)
        prepared_output = _toy_gdn(*prepared_inputs, cu_seqlens)
        physical_output = restore_gdn_headwise_output(
            prepared_output,
            layout=layout,
            group=dist.group.WORLD,
        )
        (physical_output * local_output_gradient).sum().backward()

    oracle_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in valid_inputs)
    oracle_output = _toy_gdn(*oracle_inputs, cu_seqlens)
    (oracle_output * output_gradient).sum().backward()
    expected_output = apply_packed_context_parallel_partition(
        _pad_with_zeros(oracle_output.detach(), cu_seqlens, world_size),
        partition,
        dim=1,
    )
    torch.testing.assert_close(physical_output, expected_output, rtol=1e-10, atol=1e-12)
    for physical_input, oracle_input in zip(physical_inputs, oracle_inputs):
        assert physical_input.grad is not None
        assert oracle_input.grad is not None
        expected_gradient = apply_packed_context_parallel_partition(
            _pad_with_zeros(oracle_input.grad, cu_seqlens, world_size),
            partition,
            dim=1,
        )
        torch.testing.assert_close(physical_input.grad, expected_gradient, rtol=1e-10, atol=1e-12)
    assert collectives.single == 4
    assert collectives.list == 0


def _run_hybrid_u2_cp4_contract(rank: int, world_size: int) -> None:
    cp_size = 4
    ulysses_size = 2
    assert world_size == cp_size * ulysses_size
    cp_rank, ulysses_rank = divmod(rank, ulysses_size)
    assert rank == cp_rank * ulysses_size + ulysses_rank

    cu_seqlens = torch.tensor([0, 0, 13, 38], dtype=torch.int32)
    key_heads = world_size
    value_heads = 2 * world_size
    valid_inputs = _valid_inputs(
        int(cu_seqlens[-1]),
        key_heads=key_heads,
        value_heads=value_heads,
    )
    physical_inputs, partition = _physical_inputs(
        valid_inputs,
        cu_seqlens,
        rank=rank,
        world_size=world_size,
        requires_grad=True,
        cp_size=cp_size,
    )
    generator = torch.Generator().manual_seed(811)
    output_gradient = torch.randn(
        1,
        int(cu_seqlens[-1]),
        value_heads,
        _VALUE_DIM,
        dtype=torch.float64,
        generator=generator,
    )
    local_output_gradient = apply_packed_context_parallel_partition(
        _pad_with_zeros(output_gradient, cu_seqlens, world_size),
        partition,
        dim=1,
    )

    with _CollectiveCounter() as collectives:
        prepared_inputs, layout = _prepare(
            physical_inputs,
            cu_seqlens,
            world_size,
            cp_size=cp_size,
        )
        assert layout.cp_size == cp_size
        assert layout.ulysses_size == ulysses_size
        assert layout.rank == rank
        assert layout.world_size == world_size
        for prepared, expected in zip(prepared_inputs, valid_inputs):
            torch.testing.assert_close(prepared, _head_shard(expected, rank, world_size), rtol=0, atol=0)
        prepared_output = _toy_gdn(*prepared_inputs, cu_seqlens)
        physical_output = restore_gdn_headwise_output(
            prepared_output,
            layout=layout,
            group=dist.group.WORLD,
        )
        (physical_output * local_output_gradient).sum().backward()

    oracle_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in valid_inputs)
    oracle_output = _toy_gdn(*oracle_inputs, cu_seqlens)
    (oracle_output * output_gradient).sum().backward()
    expected_output = apply_packed_context_parallel_partition(
        _pad_with_zeros(oracle_output.detach(), cu_seqlens, world_size),
        partition,
        dim=1,
    )
    torch.testing.assert_close(physical_output, expected_output, rtol=1e-10, atol=1e-12)
    for physical_input, oracle_input in zip(physical_inputs, oracle_inputs):
        assert physical_input.grad is not None
        assert oracle_input.grad is not None
        expected_gradient = apply_packed_context_parallel_partition(
            _pad_with_zeros(oracle_input.grad, cu_seqlens, world_size),
            partition,
            dim=1,
        )
        torch.testing.assert_close(physical_input.grad, expected_gradient, rtol=1e-10, atol=1e-12)
    assert collectives.single == 4
    assert collectives.list == 0


def _run_all_empty_contract(rank: int, world_size: int) -> None:
    cu_seqlens = torch.tensor([0, 0, 0], dtype=torch.int32)
    valid_inputs = _valid_inputs(0)
    physical_inputs, _ = _physical_inputs(
        valid_inputs,
        cu_seqlens,
        rank=rank,
        world_size=world_size,
        requires_grad=True,
    )

    with _CollectiveCounter() as collectives:
        prepared_inputs, layout = _prepare(physical_inputs, cu_seqlens, world_size)
        prepared_output = _toy_gdn(*prepared_inputs, cu_seqlens)
        physical_output = restore_gdn_headwise_output(
            prepared_output,
            layout=layout,
            group=dist.group.WORLD,
        )
        loss = physical_output.sum() + sum(tensor.sum() * 0 for tensor in prepared_inputs)
        loss.backward()

    assert physical_output.shape == (1, 0, _VALUE_HEADS, _VALUE_DIM)
    for tensor in physical_inputs:
        assert tensor.grad is not None
        torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor), rtol=0, atol=0)
    assert collectives.list == 0
    counts = [None] * world_size
    dist.all_gather_object(counts, collectives.single)
    assert len(set(counts)) == 1


def _headwise_path(inputs: tuple[Tensor, ...], cu_seqlens: Tensor, world_size: int) -> Tensor:
    prepared_inputs, layout = _prepare(inputs, cu_seqlens, world_size)
    prepared_output = _toy_gdn(*prepared_inputs, cu_seqlens)
    return restore_gdn_headwise_output(prepared_output, layout=layout, group=dist.group.WORLD)


def _run_checkpoint_contract(rank: int, world_size: int) -> None:
    cu_seqlens = torch.tensor([0, 5, 14], dtype=torch.int32)
    valid_inputs = _valid_inputs(int(cu_seqlens[-1]))
    eager_inputs, partition = _physical_inputs(
        valid_inputs,
        cu_seqlens,
        rank=rank,
        world_size=world_size,
        requires_grad=True,
    )
    checkpoint_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in eager_inputs)
    generator = torch.Generator().manual_seed(91)
    output_gradient = torch.randn(
        1,
        int(cu_seqlens[-1]),
        _VALUE_HEADS,
        _VALUE_DIM,
        dtype=torch.float64,
        generator=generator,
    )
    local_output_gradient = apply_packed_context_parallel_partition(
        _pad_with_zeros(output_gradient, cu_seqlens, world_size),
        partition,
        dim=1,
    )

    with _CollectiveCounter() as eager_collectives:
        eager_output = _headwise_path(eager_inputs, cu_seqlens, world_size)
        (eager_output * local_output_gradient).sum().backward()
    with _CollectiveCounter() as checkpoint_collectives:
        checkpoint_output = checkpoint(
            lambda *inputs: _headwise_path(inputs, cu_seqlens, world_size),
            *checkpoint_inputs,
            use_reentrant=False,
        )
        (checkpoint_output * local_output_gradient).sum().backward()

    torch.testing.assert_close(checkpoint_output, eager_output, rtol=1e-10, atol=1e-12)
    for checkpoint_input, eager_input in zip(checkpoint_inputs, eager_inputs):
        assert checkpoint_input.grad is not None
        assert eager_input.grad is not None
        torch.testing.assert_close(checkpoint_input.grad, eager_input.grad, rtol=1e-10, atol=1e-12)
    assert eager_collectives.single == 4
    assert eager_collectives.list == 0
    assert checkpoint_collectives.list == 0
    local_counts = (eager_collectives.single, checkpoint_collectives.single)
    all_counts = [None] * world_size
    dist.all_gather_object(all_counts, local_counts)
    assert len(set(all_counts)) == 1


def _run_asymmetric_metadata_contract(rank: int, world_size: int) -> None:
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32) if rank == 0 else torch.tensor([0, 8], dtype=torch.int32)
    valid_inputs = _valid_inputs(8)
    physical_inputs, _ = _physical_inputs(
        valid_inputs,
        cu_seqlens,
        rank=rank,
        world_size=world_size,
        requires_grad=False,
    )

    with _CollectiveCounter() as collectives:
        try:
            _prepare(physical_inputs, cu_seqlens, world_size)
        except RuntimeError as error:
            message = str(error).lower()
            assert "headwise" in message
            assert "cu_seqlens" in message or "metadata" in message or "signature" in message
        else:
            raise AssertionError("asymmetric packed metadata must fail on every rank")

    assert collectives.single == 0
    assert collectives.list == 0


def _worker_entry(rank: int, world_size: int, port: int, case_name: str, errors) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        cases = {
            "layout": _run_layout_contract,
            "transport_vjp": _run_transport_vjp_contract,
            "vjp": _run_vjp_contract,
            "hybrid_u2_cp4": _run_hybrid_u2_cp4_contract,
            "all_empty": _run_all_empty_contract,
            "checkpoint": _run_checkpoint_contract,
            "asymmetric_metadata": _run_asymmetric_metadata_contract,
        }
        cases[case_name](rank, world_size)
        errors.put((rank, None))
    except BaseException:  # noqa: BLE001
        errors.put((rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_gloo_case(case_name: str, *, world_size: int = _WORLD_SIZE, timeout: float = 60.0) -> None:
    ctx = mp.get_context("spawn")
    errors = ctx.Queue()
    port = _free_port()
    processes = [
        ctx.Process(target=_worker_entry, args=(rank, world_size, port, case_name, errors))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()

    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    hanging = [process.pid for process in processes if process.is_alive()]
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    reports: dict[int, str | None] = {}
    for _ in range(world_size):
        try:
            rank, error = errors.get(timeout=1)
        except Empty:
            break
        reports[rank] = error
    failures = [f"rank {rank}:\n{error}" for rank, error in sorted(reports.items()) if error is not None]
    exit_codes = [process.exitcode for process in processes]
    assert not hanging, f"{case_name} hung on worker pids {hanging}"
    assert len(reports) == world_size, f"{case_name} missing worker reports; exit_codes={exit_codes}"
    assert all(code == 0 for code in exit_codes), f"{case_name} worker exit_codes={exit_codes}"
    assert not failures, "\n".join(failures)


def test_headwise_packs_all_inputs_into_one_a2a_and_restores_physical_layout():
    _run_gloo_case("layout")


def test_headwise_transport_prepare_restore_and_composition_vjp():
    _run_gloo_case("transport_vjp")


def test_headwise_leading_empty_gqa_forward_and_vjp_match_monolithic():
    _run_gloo_case("vjp")


def test_headwise_u2_cp4_flat_rank_forward_and_vjp_match_monolithic():
    _run_gloo_case("hybrid_u2_cp4", world_size=8, timeout=60)


def test_headwise_all_empty_keeps_autograd_and_collective_symmetry():
    _run_gloo_case("all_empty")


def test_headwise_non_reentrant_checkpoint_matches_eager_and_stays_symmetric():
    _run_gloo_case("checkpoint", timeout=45)


def test_headwise_asymmetric_metadata_fails_on_all_ranks_before_payload_a2a():
    _run_gloo_case("asymmetric_metadata")
