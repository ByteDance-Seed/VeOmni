import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from veomni.distributed.context_parallel.ring_attention import (
    dense_causal_attention,
    ringattn_context_parallel,
)
from veomni.distributed.context_parallel.ring_p2p import RingP2P
from veomni.distributed.context_parallel.sharding import balanced_cp_slice


def _init_gloo(rank: int, world_size: int, file_name: str):
    store = dist.FileStore(file_name, world_size)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
    return dist.group.WORLD


def _ring_p2p_worker(rank: int, world_size: int, file_name: str, errors: mp.Queue):
    try:
        group = _init_gloo(rank, world_size, file_name)
        ranks = list(range(world_size))
        ring = RingP2P(ranks, group)
        send = torch.full((4,), float(rank + 1))
        recv = torch.zeros_like(send)
        ring.async_send_recv(send, recv)
        ring.wait()
        expected = float((rank - 1 + world_size) % world_size + 1)
        torch.testing.assert_close(recv, torch.full_like(recv, expected))

        # A strided destination must retain its identity and receive through a
        # temporary contiguous payload.
        noncontiguous_storage = torch.zeros(4, 2)
        noncontiguous_recv = noncontiguous_storage[:, 0]
        assert not noncontiguous_recv.is_contiguous()
        ring.async_send_recv(send + 10, noncontiguous_recv)
        ring.wait()
        torch.testing.assert_close(noncontiguous_recv, torch.full_like(noncontiguous_recv, expected + 10))

        # Packed receive must copy into the caller's original buffers rather
        # than replacing list entries with views into an internal payload.
        packed_send = [send[:2] + 20, send.reshape(2, 2) + 30]
        packed_recv = [torch.zeros_like(packed_send[0]), torch.zeros_like(packed_send[1])]
        original_recv = tuple(packed_recv)
        ring.async_send_recv(packed_send, packed_recv)
        ring.wait()
        assert all(actual is original for actual, original in zip(packed_recv, original_recv))
        torch.testing.assert_close(packed_recv[0], torch.full_like(packed_recv[0], expected + 20))
        torch.testing.assert_close(packed_recv[1], torch.full_like(packed_recv[1], expected + 30))

        dist.destroy_process_group()
        errors.put(None)
    except Exception as exc:  # noqa: BLE001 - surface worker failures to parent
        errors.put(repr(exc))


def _attention_worker(rank: int, world_size: int, file_name: str, errors: mp.Queue):
    try:
        group = _init_gloo(rank, world_size, file_name)
        ranks = list(range(world_size))
        torch.manual_seed(0)
        batch, hq, hkv, seq_len, head_dim = 1, 16, 2, 64, 256
        scale = head_dim**-0.5

        query = torch.empty(batch, hq, seq_len, head_dim, dtype=torch.float32)
        key = torch.empty(batch, hkv, seq_len, head_dim, dtype=torch.float32)
        value = torch.empty(batch, hkv, seq_len, head_dim, dtype=torch.float32)
        if rank == 0:
            torch.manual_seed(0)
            query = torch.randn_like(query)
            key = torch.randn_like(key)
            value = torch.randn_like(value)
        dist.broadcast(query, src=0)
        dist.broadcast(key, src=0)
        dist.broadcast(value, src=0)

        local_q = balanced_cp_slice(query, cp_size=world_size, cp_rank=rank, dim=2).detach().requires_grad_(True)
        local_k = balanced_cp_slice(key, cp_size=world_size, cp_rank=rank, dim=2).detach().requires_grad_(True)
        local_v = balanced_cp_slice(value, cp_size=world_size, cp_rank=rank, dim=2).detach().requires_grad_(True)

        local_out = ringattn_context_parallel(
            local_q,
            local_k,
            local_v,
            hq,
            group,
            ranks,
            softmax_scale=scale,
            backend="torch",
        )
        loss = local_out.sum()
        loss.backward()

        # Every rank validates its own shard. A rank-local assertion failure
        # therefore cannot strand a peer in a trailing barrier, and rank-specific
        # gradient regressions are covered directly.
        query_r = query.detach().requires_grad_(True)
        key_r = key.detach().requires_grad_(True)
        value_r = value.detach().requires_grad_(True)
        dense_out = dense_causal_attention(query_r, key_r, value_r, softmax_scale=scale)
        dense_out.sum().backward()
        torch.testing.assert_close(
            local_out,
            balanced_cp_slice(dense_out, cp_size=world_size, cp_rank=rank, dim=2),
            atol=2e-4,
            rtol=2e-4,
        )
        for local_grad, dense_grad in (
            (local_q.grad, query_r.grad),
            (local_k.grad, key_r.grad),
            (local_v.grad, value_r.grad),
        ):
            torch.testing.assert_close(
                local_grad,
                balanced_cp_slice(dense_grad, cp_size=world_size, cp_rank=rank, dim=2),
                atol=2e-4,
                rtol=2e-4,
            )

        dist.destroy_process_group()
        errors.put(None)
    except Exception as exc:  # noqa: BLE001
        errors.put(repr(exc))


def _run_mp(worker, world_size: int = 2):
    with tempfile.TemporaryDirectory() as tmp:
        file_name = os.path.join(tmp, "pg")
        ctx = mp.get_context("spawn")
        errors = ctx.Queue()
        processes = [
            ctx.Process(target=worker, args=(rank, world_size, file_name, errors)) for rank in range(world_size)
        ]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=120)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
        for process in processes:
            assert process.exitcode == 0, f"worker exited with {process.exitcode}"
        for _ in range(world_size):
            err = errors.get(timeout=5)
            assert err is None, err


def test_ring_p2p_exchanges_tensor_gloo():
    _run_mp(_ring_p2p_worker)


def test_attention_with_cp_matches_dense_gloo():
    _run_mp(_attention_worker)
