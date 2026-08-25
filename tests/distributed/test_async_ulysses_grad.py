import torch

import veomni.distributed.sequence_parallel.async_ulysses as au


def _mock_identity_comm(monkeypatch):
    monkeypatch.setattr(au, "padding_tensor_for_seqeunce_parallel", lambda t, dim: t)
    monkeypatch.setattr(au, "unpadding_tensor_for_seqeunce_parallel", lambda t, dim, size: t)

    def fake_all_to_all(t, **kwargs):
        return (lambda: t) if kwargs.get("async_op") else t

    monkeypatch.setattr(au, "all_to_all_tensor", fake_all_to_all)


def test_output_projection_weight_bias_grad_shapes(monkeypatch):
    _mock_identity_comm(monkeypatch)

    batch, seq, heads, head_dim = 2, 3, 4, 5
    out_dim = 7
    # attn_output layout: [batch, seq, num_heads, head_dim]; seq_dimension=1, head_dimension=2.
    hidden_states = torch.randn(batch, seq, heads, head_dim, requires_grad=True)
    proj_weight = torch.randn(out_dim, heads * head_dim, requires_grad=True)
    proj_bias = torch.randn(out_dim, requires_grad=True)

    out = au.AsyncUlyssesOutputProjection.apply(hidden_states, 1, 2, proj_weight, proj_bias, seq, object())
    out.sum().backward()

    # The weight grad must be reduced over the batch dim, and the bias grad over
    # the batch and sequence dims, to match the parameter shapes.
    assert proj_weight.grad is not None
    assert proj_weight.grad.shape == proj_weight.shape
    assert proj_bias.grad is not None
    assert proj_bias.grad.shape == proj_bias.shape


def test_output_projection_bias_grad_when_weight_frozen(monkeypatch):
    _mock_identity_comm(monkeypatch)

    batch, seq, heads, head_dim = 2, 3, 4, 5
    out_dim = 7
    hidden_states = torch.randn(batch, seq, heads, head_dim, requires_grad=True)
    proj_weight = torch.randn(out_dim, heads * head_dim)  # frozen weight
    proj_bias = torch.randn(out_dim, requires_grad=True)  # trainable bias

    out = au.AsyncUlyssesOutputProjection.apply(hidden_states, 1, 2, proj_weight, proj_bias, seq, object())
    out.sum().backward()

    # needs_input_grad must be read at the bias index (4), not the weight index
    # (3), so a trainable bias still receives its gradient.
    assert proj_bias.grad is not None
    assert proj_bias.grad.shape == proj_bias.shape
