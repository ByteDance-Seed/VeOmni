import pytest
import torch

from veomni.distributed.context_parallel.ring_attention import (
    dense_causal_attention,
    simulate_ring_causal_attention,
)
from veomni.distributed.context_parallel.sharding import balanced_cp_slice


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,head_dim,seq_len",
    [
        (4, 2, 8, 32),
        (16, 2, 256, 64),  # Qwen3.6-35B-A3B exact head geometry, short seq
    ],
)
def test_simulate_ring_matches_dense_causal_forward(
    cp_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
):
    torch.manual_seed(0)
    batch = 1
    query = torch.randn(batch, num_q_heads, seq_len, head_dim, dtype=torch.float32)
    key = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.float32)
    value = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.float32)
    scale = head_dim**-0.5

    expected = dense_causal_attention(query, key, value, softmax_scale=scale)
    actual = simulate_ring_causal_attention(
        query, key, value, cp_size=cp_size, softmax_scale=scale, backend="torch"
    )

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_simulate_ring_matches_dense_causal_gradients_gqa():
    torch.manual_seed(1)
    cp_size = 2
    batch, hq, hkv, seq_len, head_dim = 1, 16, 2, 64, 256
    scale = head_dim**-0.5

    query = torch.randn(batch, hq, seq_len, head_dim, dtype=torch.float64, requires_grad=True)
    key = torch.randn(batch, hkv, seq_len, head_dim, dtype=torch.float64, requires_grad=True)
    value = torch.randn(batch, hkv, seq_len, head_dim, dtype=torch.float64, requires_grad=True)

    ring_out = simulate_ring_causal_attention(
        query, key, value, cp_size=cp_size, softmax_scale=scale, backend="torch"
    )
    ring_out.sum().backward()
    ring_grads = [query.grad.detach().clone(), key.grad.detach().clone(), value.grad.detach().clone()]

    for tensor in (query, key, value):
        tensor.grad = None

    dense_out = dense_causal_attention(query, key, value, softmax_scale=scale)
    dense_out.sum().backward()
    dense_grads = [query.grad, key.grad, value.grad]

    torch.testing.assert_close(ring_out.detach(), dense_out.detach(), atol=1e-5, rtol=1e-5)
    for actual, expected in zip(ring_grads, dense_grads):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_balanced_local_shards_cover_full_sequence_without_overlap():
    seq = torch.arange(64).view(1, 1, 64, 1)
    shards = [balanced_cp_slice(seq, cp_size=2, cp_rank=rank, dim=2) for rank in range(2)]
    assert shards[0].shape[2] == 32
    assert shards[1].shape[2] == 32
    # Rank0 = c0||c3, Rank1 = c1||c2
    torch.testing.assert_close(shards[0][0, 0, :, 0], torch.cat((torch.arange(0, 16), torch.arange(48, 64))))
    torch.testing.assert_close(shards[1][0, 0, :, 0], torch.cat((torch.arange(16, 32), torch.arange(32, 48))))
