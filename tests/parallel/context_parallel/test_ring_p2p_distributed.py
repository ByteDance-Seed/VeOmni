import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from veomni.distributed.context_parallel.ring_attention import (
    dense_causal_attention,
    ringattn_context_parallel,
)
from veomni.distributed.context_parallel.ring_p2p import RingP2P
from veomni.distributed.context_parallel.sharding import balanced_cp_restore, balanced_cp_slice


def _init_gloo(rank: int, world_size: int, file_name: str):
    store = dist.FileStore(file_name, world_size)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
    return dist.distributed_c10d._get_default_group()


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

        local_q = balanced_cp_slice(query, cp_size=2, cp_rank=rank, dim=2).detach().requires_grad_(True)
        local_k = balanced_cp_slice(key, cp_size=2, cp_rank=rank, dim=2).detach().requires_grad_(True)
        local_v = balanced_cp_slice(value, cp_size=2, cp_rank=rank, dim=2).detach().requires_grad_(True)

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

        gathered = [torch.empty_like(local_out) for _ in range(world_size)]
        dist.all_gather(gathered, local_out.detach())
        if rank == 0:
            ring_full = balanced_cp_restore(torch.cat(gathered, dim=2), cp_size=2, dim=2)
            dense = dense_causal_attention(query, key, value, softmax_scale=scale)
            torch.testing.assert_close(ring_full, dense, atol=2e-4, rtol=2e-4)

            # Gradient parity on local shards vs dense sliced grads.
            query_r = query.detach().requires_grad_(True)
            key_r = key.detach().requires_grad_(True)
            value_r = value.detach().requires_grad_(True)
            dense_out = dense_causal_attention(query_r, key_r, value_r, softmax_scale=scale)
            dense_out.sum().backward()
            torch.testing.assert_close(
                local_q.grad,
                balanced_cp_slice(query_r.grad, cp_size=2, cp_rank=0, dim=2),
                atol=2e-4,
                rtol=2e-4,
            )
            torch.testing.assert_close(
                local_k.grad,
                balanced_cp_slice(key_r.grad, cp_size=2, cp_rank=0, dim=2),
                atol=2e-4,
                rtol=2e-4,
            )
            torch.testing.assert_close(
                local_v.grad,
                balanced_cp_slice(value_r.grad, cp_size=2, cp_rank=0, dim=2),
                atol=2e-4,
                rtol=2e-4,
            )

        dist.barrier()
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
            ctx.Process(target=worker, args=(rank, world_size, file_name, errors))
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=120)
            assert process.exitcode == 0, f"worker exited with {process.exitcode}"
        for _ in range(world_size):
            err = errors.get(timeout=5)
            assert err is None, err


def test_ring_p2p_exchanges_tensor_gloo():
    _run_mp(_ring_p2p_worker)


def test_attention_with_cp_matches_dense_gloo():
    _run_mp(_attention_worker)
