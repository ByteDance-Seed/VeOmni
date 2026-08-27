import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from veomni.distributed.context_parallel.attention_backend import torch_packed_causal_attention
from veomni.distributed.context_parallel.packed_sharding import (
    apply_packed_context_parallel_partition,
    build_packed_context_parallel_partition,
    reorder_sample_major_to_ulysses_rank_major,
    reorder_ulysses_rank_major_to_sample_major,
)
from veomni.distributed.context_parallel.ring_attention import ringattn_context_parallel
from veomni.distributed.parallel_state import init_parallel_state
from veomni.ops.kernels.attention.ulysses import prepare_ulysses_qkv, restore_ulysses_output


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_hybrid_oracle(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        cp_size, ulysses_size = 2, 2
        state = init_parallel_state(
            dp_size=1,
            cp_size=cp_size,
            gdn_context_parallel_implementation="headwise_lossless",
            ulysses_size=ulysses_size,
            device_type="cpu",
            name="hybrid_cp_test",
        )
        assert state.cp_rank == rank // ulysses_size
        assert state.ulysses_rank == rank % ulysses_size
        assert state.sp_rank == state.cp_rank * ulysses_size + state.ulysses_rank

        torch.manual_seed(29)
        batch, num_q_heads, num_kv_heads, seq_len, head_dim = 1, 4, 2, 16, 4
        scale = head_dim**-0.5
        packed_cu = torch.tensor([0, 8, 16], dtype=torch.int32)
        global_query = torch.randn(batch, seq_len, num_q_heads, head_dim, dtype=torch.float64)
        global_key = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=torch.float64)
        global_value = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=torch.float64)
        global_dout = torch.randn(batch, seq_len, num_q_heads, head_dim, dtype=torch.float64)
        partition = build_packed_context_parallel_partition(
            packed_cu,
            cp_size=cp_size,
            cp_rank=state.cp_rank,
            ulysses_size=ulysses_size,
            ulysses_rank=state.ulysses_rank,
        )
        query = apply_packed_context_parallel_partition(global_query, partition, dim=1).detach().requires_grad_()
        key = apply_packed_context_parallel_partition(global_key, partition, dim=1).detach().requires_grad_()
        value = apply_packed_context_parallel_partition(global_value, partition, dim=1).detach().requires_grad_()
        dout = apply_packed_context_parallel_partition(global_dout, partition, dim=1)

        query_u, key_u, value_u, _ = prepare_ulysses_qkv(
            query,
            key,
            value,
            group=state.ulysses_group,
            ulysses_size=ulysses_size,
        )
        for tensor in (query_u, key_u, value_u):
            assert tensor.size(1) == query.size(1) * ulysses_size
        query_u = reorder_ulysses_rank_major_to_sample_major(
            query_u, partition.local_cu_seqlens, ulysses_size=ulysses_size, sequence_dim=1
        )
        key_u = reorder_ulysses_rank_major_to_sample_major(
            key_u, partition.local_cu_seqlens, ulysses_size=ulysses_size, sequence_dim=1
        )
        value_u = reorder_ulysses_rank_major_to_sample_major(
            value_u, partition.local_cu_seqlens, ulysses_size=ulysses_size, sequence_dim=1
        )
        cp_local_cu = partition.local_cu_seqlens * ulysses_size
        output_u = ringattn_context_parallel(
            query_u.transpose(1, 2).contiguous(),
            key_u.transpose(1, 2).contiguous(),
            value_u.transpose(1, 2).contiguous(),
            query_u.size(2),
            state.cp_group,
            dist.get_process_group_ranks(state.cp_group),
            softmax_scale=scale,
            backend="torch",
            cu_seqlens=cp_local_cu,
        ).transpose(1, 2)
        output_u = reorder_sample_major_to_ulysses_rank_major(
            output_u, partition.local_cu_seqlens, ulysses_size=ulysses_size, sequence_dim=1
        )
        output = restore_ulysses_output(output_u, group=state.ulysses_group)

        oracle_query = global_query.transpose(1, 2).detach().requires_grad_()
        oracle_key = global_key.transpose(1, 2).detach().requires_grad_()
        oracle_value = global_value.transpose(1, 2).detach().requires_grad_()
        oracle = torch_packed_causal_attention(
            oracle_query,
            oracle_key,
            oracle_value,
            packed_cu,
            softmax_scale=scale,
        ).transpose(1, 2)
        expected = apply_packed_context_parallel_partition(oracle, partition, dim=1)
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-4)

        (output * dout).sum().backward()
        (oracle * global_dout).sum().backward()
        for actual, expected_global in (
            (query.grad, oracle_query.grad.transpose(1, 2)),
            (key.grad, oracle_key.grad.transpose(1, 2)),
            (value.grad, oracle_value.grad.transpose(1, 2)),
        ):
            expected_grad = apply_packed_context_parallel_partition(expected_global, partition, dim=1)
            torch.testing.assert_close(actual, expected_grad, atol=1e-5, rtol=1e-4)
    finally:
        dist.destroy_process_group()


def test_gloo_hybrid_cp2_u2_ring_matches_packed_dense_full_grad():
    world_size = 4
    mp.spawn(_run_hybrid_oracle, args=(world_size, _free_port()), nprocs=world_size, join=True)
