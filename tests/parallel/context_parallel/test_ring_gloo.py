import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from veomni.distributed.context_parallel.attention_backend import torch_packed_causal_attention
from veomni.distributed.context_parallel.packed_sharding import (
    apply_packed_context_parallel_partition,
    build_packed_context_parallel_partition,
)
from veomni.distributed.context_parallel.ring_attention import dense_causal_attention, ringattn_context_parallel
from veomni.distributed.context_parallel.sharding import balanced_cp_restore, balanced_cp_slice


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_ring_oracle(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(17)
        batch, num_q_heads, num_kv_heads, seq_len, head_dim = 1, 4, 2, 16, 4
        scale = head_dim**-0.5
        global_query = torch.randn(batch, num_q_heads, seq_len, head_dim, dtype=torch.float64)
        global_key = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.float64)
        global_value = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.float64)
        global_dout = torch.randn(batch, num_q_heads, seq_len, head_dim, dtype=torch.float64)

        query = balanced_cp_slice(global_query, world_size, rank, dim=2).detach().requires_grad_()
        key = balanced_cp_slice(global_key, world_size, rank, dim=2).detach().requires_grad_()
        value = balanced_cp_slice(global_value, world_size, rank, dim=2).detach().requires_grad_()
        dout = balanced_cp_slice(global_dout, world_size, rank, dim=2)

        output = ringattn_context_parallel(
            query,
            key,
            value,
            num_q_heads,
            dist.group.WORLD,
            list(range(world_size)),
            softmax_scale=scale,
            backend="torch",
        )
        saved = output.grad_fn.saved_tensors
        assert len(saved) == 6
        assert all(tensor.ndim < 6 for tensor in saved), "Ring backward must not retain a CP-stacked global KV cache"
        gathered = [torch.empty_like(output) for _ in range(world_size)]
        dist.all_gather(gathered, output.detach())
        restored = balanced_cp_restore(torch.cat(gathered, dim=2), world_size, dim=2)

        oracle_query = global_query.detach().requires_grad_()
        oracle_key = global_key.detach().requires_grad_()
        oracle_value = global_value.detach().requires_grad_()
        oracle_output = dense_causal_attention(
            oracle_query,
            oracle_key,
            oracle_value,
            softmax_scale=scale,
        )
        # The reference backend intentionally evaluates attention scores in
        # FP32, so block-wise online-softmax merging is not bit-exact with one
        # monolithic score matrix even when inputs are FP64.
        torch.testing.assert_close(restored, oracle_output, atol=1e-6, rtol=1e-4)

        (output * dout).sum().backward()
        (oracle_output * global_dout).sum().backward()
        for actual, expected in (
            (query.grad, balanced_cp_slice(oracle_query.grad, world_size, rank, dim=2)),
            (key.grad, balanced_cp_slice(oracle_key.grad, world_size, rank, dim=2)),
            (value.grad, balanced_cp_slice(oracle_value.grad, world_size, rank, dim=2)),
        ):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

        # Packed batches must run a separate causal ring per sample so tokens
        # cannot attend across CU boundaries. Exercise the actual distributed
        # entrypoint and the physical packed partition used by the collator.
        packed_cu = torch.tensor([0, 8, 16], dtype=torch.int32)
        partition = build_packed_context_parallel_partition(
            packed_cu,
            cp_size=world_size,
            cp_rank=rank,
        )
        packed_query = (
            apply_packed_context_parallel_partition(global_query, partition, dim=2).detach().requires_grad_()
        )
        packed_key = apply_packed_context_parallel_partition(global_key, partition, dim=2).detach().requires_grad_()
        packed_value = (
            apply_packed_context_parallel_partition(global_value, partition, dim=2).detach().requires_grad_()
        )
        packed_dout = apply_packed_context_parallel_partition(global_dout, partition, dim=2)
        packed_output = ringattn_context_parallel(
            packed_query,
            packed_key,
            packed_value,
            num_q_heads,
            dist.group.WORLD,
            list(range(world_size)),
            softmax_scale=scale,
            backend="torch",
            cu_seqlens=partition.local_cu_seqlens,
        )
        gathered_packed = [torch.empty_like(packed_output) for _ in range(world_size)]
        dist.all_gather(gathered_packed, packed_output.detach())
        restored_packed = torch.empty_like(global_query)
        for cp_rank, shard in enumerate(gathered_packed):
            cp_partition = build_packed_context_parallel_partition(
                packed_cu,
                cp_size=world_size,
                cp_rank=cp_rank,
            )
            restored_packed.index_copy_(2, cp_partition.token_indices, shard)

        packed_oracle_query = global_query.detach().requires_grad_()
        packed_oracle_key = global_key.detach().requires_grad_()
        packed_oracle_value = global_value.detach().requires_grad_()
        packed_oracle = torch_packed_causal_attention(
            packed_oracle_query,
            packed_oracle_key,
            packed_oracle_value,
            packed_cu,
            softmax_scale=scale,
        )
        torch.testing.assert_close(restored_packed, packed_oracle, atol=1e-6, rtol=1e-4)

        (packed_output * packed_dout).sum().backward()
        (packed_oracle * global_dout).sum().backward()
        for actual, expected_global in (
            (packed_query.grad, packed_oracle_query.grad),
            (packed_key.grad, packed_oracle_key.grad),
            (packed_value.grad, packed_oracle_value.grad),
        ):
            expected = apply_packed_context_parallel_partition(expected_global, partition, dim=2)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_gloo_ring_forward_and_full_grad_match_dense_oracle(world_size: int):
    mp.spawn(_run_ring_oracle, args=(world_size, _free_port()), nprocs=world_size, join=True)
