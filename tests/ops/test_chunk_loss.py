from types import SimpleNamespace

import torch

import veomni.ops.kernels.cross_entropy.chunk_loss as chunk_loss_module


def test_chunk_loss_reuses_valid_token_denominator(monkeypatch):
    monkeypatch.setattr(chunk_loss_module, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))

    original_sum = torch.Tensor.sum
    denominator_sum_calls = 0

    def counting_sum(self, *args, **kwargs):
        nonlocal denominator_sum_calls
        if self.dtype == torch.bool and self.shape == (1, 5):
            denominator_sum_calls += 1
        return original_sum(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "sum", counting_sum)

    hidden_states = torch.randn(1, 6, 4, requires_grad=True)
    weights = torch.randn(8, 4, requires_grad=True)
    labels = torch.tensor([[1, 2, -100, 3, 4, 5]])

    loss, _ = chunk_loss_module.chunk_loss_function(
        hidden_states,
        weights,
        labels,
        chunk_size=2,
    )
    loss.backward()

    assert denominator_sum_calls == 1
    assert hidden_states.grad is not None
    assert weights.grad is not None
