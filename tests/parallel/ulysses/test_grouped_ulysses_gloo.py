from __future__ import annotations

import socket
import traceback
from queue import Empty

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from veomni.distributed.sequence_parallel.ulysses import (
    _SeqAllToAll,
    gather_seq_scatter_heads_grouped,
)


_WORLD_SIZE = 2


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _inputs(rank: int, *, requires_grad: bool) -> tuple[Tensor, ...]:
    generator = torch.Generator().manual_seed(20260825 + rank)
    shapes = (
        (1, 3, 4, 3),
        (1, 3, 4, 3),
        (1, 3, 8, 2),
        (1, 3, 8),
        (1, 3, 8),
    )
    return tuple(
        torch.randn(shape, dtype=torch.float64, generator=generator).requires_grad_(requires_grad) for shape in shapes
    )


def _loss(outputs: tuple[Tensor, ...], rank: int) -> Tensor:
    terms: list[Tensor] = []
    for index, output in enumerate(outputs):
        weight = torch.arange(output.numel(), dtype=output.dtype).reshape_as(output)
        terms.append((output * (weight + 1 + rank * 1000 + index * 100)).sum())
    return torch.stack(terms).sum()


def _single_reference(tensor: Tensor) -> Tensor:
    """Independent single-projection Ulysses exchange using A2A-single."""

    world_size = dist.get_world_size()
    batch_size, sequence_length, total_heads = tensor.shape[:3]
    local_heads = total_heads // world_size
    tail_shape = tensor.shape[3:]
    destination_major = tensor.reshape(
        batch_size,
        sequence_length,
        world_size,
        local_heads,
        *tail_shape,
    ).permute(2, 0, 1, 3, *range(4, 4 + len(tail_shape)))
    send = destination_major.reshape(world_size * batch_size, sequence_length, -1).contiguous()
    received = _SeqAllToAll.apply(dist.group.WORLD, send, 0, 0)
    return (
        received.reshape(world_size, batch_size, sequence_length, local_heads, *tail_shape)
        .permute(1, 0, 2, 3, *range(4, 4 + len(tail_shape)))
        .reshape(batch_size, world_size * sequence_length, local_heads, *tail_shape)
        .contiguous()
    )


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
            raise AssertionError("grouped Ulysses must not use list all_to_all")

        dist.all_to_all_single = counted_single
        dist.all_to_all = forbidden_list
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        dist.all_to_all_single = self._real_single
        dist.all_to_all = self._real_list


def _run_parity(rank: int) -> None:
    legacy_inputs = _inputs(rank, requires_grad=True)
    legacy_outputs = tuple(_single_reference(tensor) for tensor in legacy_inputs)
    _loss(legacy_outputs, rank).backward()

    grouped_inputs = tuple(tensor.detach().clone().requires_grad_(True) for tensor in legacy_inputs)
    with _CollectiveCounter() as collectives:
        grouped_outputs = gather_seq_scatter_heads_grouped(
            grouped_inputs,
            seq_dim=1,
            head_dim=2,
            group=dist.group.WORLD,
        )
        _loss(grouped_outputs, rank).backward()

    assert collectives.single == 2
    assert collectives.list == 0
    for grouped, legacy in zip(grouped_outputs, legacy_outputs):
        torch.testing.assert_close(grouped, legacy, rtol=0, atol=0)
    for grouped, legacy in zip(grouped_inputs, legacy_inputs):
        torch.testing.assert_close(grouped.grad, legacy.grad, rtol=0, atol=0)


def _run_checkpoint(rank: int) -> None:
    eager_inputs = _inputs(rank, requires_grad=True)
    eager_outputs = gather_seq_scatter_heads_grouped(
        eager_inputs,
        seq_dim=1,
        head_dim=2,
        group=dist.group.WORLD,
    )
    _loss(eager_outputs, rank).backward()

    checkpoint_inputs = tuple(tensor.detach().clone().requires_grad_(True) for tensor in eager_inputs)

    def grouped(*inputs: Tensor) -> tuple[Tensor, ...]:
        return gather_seq_scatter_heads_grouped(
            inputs,
            seq_dim=1,
            head_dim=2,
            group=dist.group.WORLD,
        )

    checkpoint_outputs = checkpoint(grouped, *checkpoint_inputs, use_reentrant=False)
    _loss(checkpoint_outputs, rank).backward()
    for actual, expected in zip(checkpoint_outputs, eager_outputs):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(checkpoint_inputs, eager_inputs):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


def _run_validation(rank: int) -> None:
    inputs = _inputs(rank, requires_grad=False)
    bad_sequence = (*inputs[:-1], inputs[-1][:, :-1])
    try:
        gather_seq_scatter_heads_grouped(
            bad_sequence,
            seq_dim=1,
            head_dim=2,
            group=dist.group.WORLD,
        )
    except ValueError as error:
        assert "sequence length" in str(error)
    else:
        raise AssertionError("mismatched grouped sequence length must fail closed")

    bad_heads = (*inputs[:-1], inputs[-1][:, :, :-1])
    try:
        gather_seq_scatter_heads_grouped(
            bad_heads,
            seq_dim=1,
            head_dim=2,
            group=dist.group.WORLD,
        )
    except ValueError as error:
        assert "head count" in str(error)
    else:
        raise AssertionError("non-divisible grouped head count must fail closed")


def _worker(rank: int, world_size: int, port: int, case: str, errors) -> None:
    error: str | None = None
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
        )
        if case == "parity":
            _run_parity(rank)
        elif case == "checkpoint":
            _run_checkpoint(rank)
        elif case == "validation":
            _run_validation(rank)
        else:
            raise AssertionError(f"unknown case: {case}")
    except Exception:  # noqa: BLE001 - worker tracebacks are reported to the parent
        error = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        errors.put((rank, error))


def _run_gloo_case(case: str, *, timeout: float = 30) -> None:
    context = mp.get_context("spawn")
    errors = context.Queue()
    port = _free_port()
    processes = [
        context.Process(target=_worker, args=(rank, _WORLD_SIZE, port, case, errors)) for rank in range(_WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=timeout)
    hanging = [process.pid for process in processes if process.is_alive()]
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    reports: dict[int, str | None] = {}
    for _ in range(_WORLD_SIZE):
        try:
            rank, error = errors.get(timeout=1)
        except Empty:
            break
        reports[rank] = error
    failures = [f"rank {rank}:\n{error}" for rank, error in sorted(reports.items()) if error is not None]
    assert not hanging, f"{case} hung on worker pids {hanging}"
    assert len(reports) == _WORLD_SIZE, f"{case} missing worker reports"
    assert all(process.exitcode == 0 for process in processes)
    assert not failures, "\n".join(failures)


def test_grouped_ulysses_matches_five_independent_exchanges_and_vjp():
    _run_gloo_case("parity")


def test_grouped_ulysses_matches_non_reentrant_checkpoint():
    _run_gloo_case("checkpoint", timeout=45)


def test_grouped_ulysses_rejects_invalid_layouts_before_payload_collective():
    _run_gloo_case("validation")
