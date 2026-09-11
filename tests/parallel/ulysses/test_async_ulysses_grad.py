"""Single-process regression for dense async Ulysses projection gradients.

These tests mock collectives so they can run without NCCL. Keep them out of
``test_async_ulysses.py``, which is a 4-GPU ``SequenceParallelTest`` and exits
when the distributed backend is missing.

They lock the contracts that GitHub PR 1080 found on the dense path: weight
and bias grads must match parameter shapes after reducing leading dims, and
``needs_input_grad`` must read the bias index rather than the weight index.
Helper math itself lives in ``test_backward.py``.
"""

from __future__ import annotations

import torch

import veomni.distributed.sequence_parallel.async_ulysses as au


def _mock_identity_comm(monkeypatch) -> None:
    monkeypatch.setattr(au, "padding_tensor_for_seqeunce_parallel", lambda tensor, dim, **kwargs: tensor)
    monkeypatch.setattr(
        au,
        "unpadding_tensor_for_seqeunce_parallel",
        lambda tensor, dim, size, **kwargs: tensor,
    )
    monkeypatch.setattr(au, "get_ulysses_sequence_parallel_world_size", lambda: 1)

    def fake_all_to_all(tensor, **kwargs):
        return (lambda: tensor) if kwargs.get("async_op") else tensor

    monkeypatch.setattr(au, "all_to_all_tensor", fake_all_to_all)


def test_output_projection_weight_bias_grad_shapes(monkeypatch) -> None:
    _mock_identity_comm(monkeypatch)

    batch, seq, heads, head_dim = 2, 3, 4, 5
    out_dim = 7
    # attn_output layout: [batch, seq, num_heads, head_dim]; seq_dimension=1, head_dimension=2.
    hidden_states = torch.randn(batch, seq, heads, head_dim, requires_grad=True)
    proj_weight = torch.randn(out_dim, heads * head_dim, requires_grad=True)
    proj_bias = torch.randn(out_dim, requires_grad=True)

    output = au.AsyncUlyssesOutputProjection.apply(hidden_states, 1, 2, proj_weight, proj_bias, seq, object())
    output.sum().backward()

    assert proj_weight.grad is not None
    assert proj_weight.grad.shape == proj_weight.shape
    assert proj_bias.grad is not None
    assert proj_bias.grad.shape == proj_bias.shape


def test_output_projection_bias_grad_when_weight_frozen(monkeypatch) -> None:
    _mock_identity_comm(monkeypatch)

    batch, seq, heads, head_dim = 2, 3, 4, 5
    out_dim = 7
    hidden_states = torch.randn(batch, seq, heads, head_dim, requires_grad=True)
    proj_weight = torch.randn(out_dim, heads * head_dim)
    proj_bias = torch.randn(out_dim, requires_grad=True)

    output = au.AsyncUlyssesOutputProjection.apply(hidden_states, 1, 2, proj_weight, proj_bias, seq, object())
    output.sum().backward()

    # needs_input_grad must be read at the bias index (4), not the weight index (3).
    assert proj_bias.grad is not None
    assert proj_bias.grad.shape == proj_bias.shape


def test_qkv_projection_bias_grad_when_weights_frozen(monkeypatch) -> None:
    _mock_identity_comm(monkeypatch)

    batch, seq, hidden_size = 2, 3, 20
    head_dim = 5
    query_size = 20
    key_value_size = 10
    hidden_states = torch.randn(batch, seq, hidden_size, requires_grad=True)
    q_weight = torch.randn(query_size, hidden_size)
    k_weight = torch.randn(key_value_size, hidden_size)
    v_weight = torch.randn(key_value_size, hidden_size)
    q_bias = torch.randn(query_size, requires_grad=True)
    k_bias = torch.randn(key_value_size, requires_grad=True)
    v_bias = torch.randn(key_value_size, requires_grad=True)

    query, key, value = au.AsyncUlyssesQKVProjection.apply(
        hidden_states,
        1,
        2,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        seq,
        head_dim,
        object(),
    )
    (query.sum() + key.sum() + value.sum()).backward()

    # Q/K/V bias indices are 4/6/8. Frozen weights must not skip those grads.
    for bias in (q_bias, k_bias, v_bias):
        assert bias.grad is not None
        assert bias.grad.shape == bias.shape
