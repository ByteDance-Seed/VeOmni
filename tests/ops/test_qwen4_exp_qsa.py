import pytest
import torch


try:
    from veomni.ops.kernels.attention.qwen4_exp_qsa import qsa_is_supported, qsa_sparse_attention
except ImportError:
    qsa_is_supported = None
    qsa_sparse_attention = None


def _sparse_eager_qsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_indices: torch.Tensor,
    selected_counts: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    outputs = []
    for batch_idx in range(q.shape[0]):
        head_outputs = []
        for head_idx in range(q.shape[1]):
            kv_head_idx = head_idx // (q.shape[1] // k.shape[1])
            query_outputs = []
            for query_idx in range(q.shape[2]):
                count = int(selected_counts[batch_idx, query_idx])
                indices = selected_indices[batch_idx, query_idx, :count].long()
                keys = k[batch_idx, kv_head_idx].index_select(0, indices)
                values = v[batch_idx, kv_head_idx].index_select(0, indices)
                probs = torch.softmax(q[batch_idx, head_idx][query_idx] @ keys.T * sm_scale, dim=-1)
                query_outputs.append(probs @ values)
            head_outputs.append(torch.stack(query_outputs))
        outputs.append(torch.stack(head_outputs))
    return torch.stack(outputs)


@pytest.mark.parametrize("heads,kv_heads", [(4, 4), (4, 2)])
@pytest.mark.parametrize("seq_len,selector_capacity", [(64, 67), (128, 130)])
def test_qwen4_exp_qsa_matches_sparse_eager_forward_and_backward(
    heads: int,
    kv_heads: int,
    seq_len: int,
    selector_capacity: int,
):
    if not torch.cuda.is_available() or qsa_sparse_attention is None:
        pytest.skip("Qwen4-Exp QSA Triton requires CUDA and Triton.")

    torch.manual_seed(0)
    batch_size, dim = 1, 16
    q = torch.randn(batch_size, heads, seq_len, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch_size, kv_heads, seq_len, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(batch_size, kv_heads, seq_len, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    counts = torch.randint(
        1,
        min(seq_len, selector_capacity) + 1,
        (batch_size, seq_len),
        device="cuda",
        dtype=torch.int32,
    )
    indices = torch.zeros(batch_size, seq_len, selector_capacity, device="cuda", dtype=torch.int32)
    for query_idx in range(seq_len):
        count = int(counts[0, query_idx])
        indices[0, query_idx, :count] = torch.randperm(seq_len, device="cuda", dtype=torch.int64)[:count].to(
            torch.int32
        )

    assert qsa_is_supported(q, k, v, indices, counts)
    grad = torch.randn_like(q)
    output = qsa_sparse_attention(q, k, v, indices, counts)
    (output * grad).sum().backward()
    q_grad, k_grad, v_grad = q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone()

    q_ref = q.detach().float().requires_grad_()
    k_ref = k.detach().float().requires_grad_()
    v_ref = v.detach().float().requires_grad_()
    reference = _sparse_eager_qsa(q_ref, k_ref, v_ref, indices, counts, dim**-0.5)
    (reference * grad.float()).sum().backward()

    torch.testing.assert_close(output.float(), reference.detach(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(q_grad.float(), q_ref.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_grad.float(), k_ref.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_grad.float(), v_ref.grad, rtol=1e-2, atol=1e-2)


def test_qwen4_exp_qsa_rejects_cpu_inputs():
    if qsa_is_supported is None:
        pytest.skip("Triton is unavailable in this environment.")
    q = torch.randn(1, 2, 4, 16, dtype=torch.float16)
    k = torch.randn(1, 1, 4, 16, dtype=torch.float16)
    v = torch.randn(1, 1, 4, 16, dtype=torch.float16)
    indices = torch.zeros(1, 4, 3, dtype=torch.int32)
    counts = torch.ones(1, 4, dtype=torch.int32)
    assert not qsa_is_supported(q, k, v, indices, counts)


@pytest.mark.parametrize("selector_capacity", [3, 67])
def test_qwen4_exp_qsa_kernel_padding_does_not_change_sparse_result(selector_capacity: int):
    if not torch.cuda.is_available() or qsa_sparse_attention is None:
        pytest.skip("Qwen4-Exp QSA Triton requires CUDA and Triton.")

    torch.manual_seed(1)
    q = torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 1, 8, 16, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 1, 8, 16, device="cuda", dtype=torch.bfloat16)
    counts = torch.arange(1, 9, device="cuda", dtype=torch.int32).view(1, -1)
    indices = torch.zeros(1, 8, selector_capacity, device="cuda", dtype=torch.int32)
    for query_idx in range(8):
        count = min(int(counts[0, query_idx]), selector_capacity)
        counts[0, query_idx] = count
        indices[0, query_idx, :count] = torch.arange(count, device="cuda", dtype=torch.int32)

    output = qsa_sparse_attention(q, k, v, indices, counts)
    reference = _sparse_eager_qsa(q.float(), k.float(), v.float(), indices, counts, 16**-0.5)
    torch.testing.assert_close(output.float(), reference, rtol=1e-2, atol=1e-2)
