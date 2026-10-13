from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import veomni.ops.kernels.cross_entropy as cross_entropy
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


def test_chunk_loss_honors_explicit_shift_labels(monkeypatch):
    monkeypatch.setattr(chunk_loss_module, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))

    hidden_states = torch.randn(1, 3, 4)
    weights = torch.randn(8, 4)
    labels = torch.tensor([[0, 1, 2]])
    shift_labels = torch.tensor([[7, -100, 3]])

    actual, _ = chunk_loss_module.chunk_loss_function(
        hidden_states,
        weights,
        labels,
        shift_labels=shift_labels,
        chunk_size=2,
    )
    expected = F.cross_entropy(
        F.linear(hidden_states, weights).float().reshape(-1, weights.size(0)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )

    torch.testing.assert_close(actual, expected)


def _reference_loss_and_grads(hidden_states, weights, logits_bias, targets):
    hidden_states = hidden_states.detach().clone().requires_grad_(True)
    weights = weights.detach().clone().requires_grad_(True)
    logits_bias = logits_bias.detach().clone().requires_grad_(True)
    logits = F.linear(hidden_states, weights).float() + logits_bias.float()
    loss = F.cross_entropy(logits.reshape(-1, weights.size(0)), targets.reshape(-1), ignore_index=-100)
    loss.backward()
    return loss, hidden_states.grad, weights.grad, logits_bias.grad


@pytest.mark.parametrize("layout", ["shift_labels", "implicit_shift", "flat_rows"])
def test_chunk_loss_logits_bias_matches_reference(monkeypatch, layout):
    monkeypatch.setattr(chunk_loss_module, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    torch.manual_seed(0)
    vocab = 8
    hidden_states = torch.randn(1, 7, 4, requires_grad=True)
    weights = torch.randn(vocab, 4, requires_grad=True)
    logits_bias = torch.randn(1, 7, vocab, requires_grad=True)
    labels = torch.tensor([[1, 2, -100, 3, 4, 5, 6]])

    if layout == "shift_labels":
        call = {"hidden_states": hidden_states, "logits_bias": logits_bias, "shift_labels": labels}
        expected = _reference_loss_and_grads(hidden_states, weights, logits_bias, labels)
    elif layout == "implicit_shift":
        call = {"hidden_states": hidden_states, "logits_bias": logits_bias}
        reference_bias = logits_bias.detach().clone().requires_grad_(True)
        loss, grad_h, grad_w, grad_b = _reference_loss_and_grads(
            hidden_states[:, :-1], weights, reference_bias[:, :-1], labels[:, 1:]
        )
        pad = torch.zeros(1, 1, grad_h.size(-1))
        expected = (
            loss,
            torch.cat([grad_h, pad], dim=1),
            grad_w,
            torch.cat([grad_b, torch.zeros(1, 1, vocab)], dim=1),
        )
    else:
        call = {"hidden_states": hidden_states[0], "logits_bias": logits_bias[0], "shift_labels": labels[0]}
        loss, grad_h, grad_w, grad_b = _reference_loss_and_grads(hidden_states[0], weights, logits_bias[0], labels[0])
        expected = (loss, grad_h[None], grad_w, grad_b[None])

    loss, _ = chunk_loss_module.chunk_loss_function(weights=weights, labels=labels, chunk_size=3, **call)
    loss.backward()

    torch.testing.assert_close(loss, expected[0])
    torch.testing.assert_close(hidden_states.grad, expected[1])
    torch.testing.assert_close(weights.grad, expected[2])
    torch.testing.assert_close(logits_bias.grad, expected[3])


def test_chunk_loss_logits_bias_without_grad(monkeypatch):
    monkeypatch.setattr(chunk_loss_module, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    hidden_states = torch.randn(1, 5, 4, requires_grad=True)
    weights = torch.randn(6, 4, requires_grad=True)
    logits_bias = torch.randn(1, 5, 6)
    labels = torch.tensor([[1, 2, -100, 3, 4]])

    loss, _ = chunk_loss_module.chunk_loss_function(
        hidden_states, weights, labels, chunk_size=2, shift_labels=labels, logits_bias=logits_bias
    )
    loss.backward()

    expected, grad_h, grad_w, _ = _reference_loss_and_grads(hidden_states, weights, logits_bias, labels)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(hidden_states.grad, grad_h)
    torch.testing.assert_close(weights.grad, grad_w)


def test_chunk_loss_logits_bias_projects_only_supervised_rows(monkeypatch):
    monkeypatch.setattr(chunk_loss_module, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    projected_rows = []
    linear = F.linear

    def counting_linear(input, weight, bias=None):
        projected_rows.append(input.shape[0])
        return linear(input, weight, bias)

    torch.manual_seed(0)
    hidden_states = torch.randn(2, 5, 4, requires_grad=True)
    weights = torch.randn(6, 4, requires_grad=True)
    logits_bias = torch.randn(2, 5, 6, requires_grad=True)
    labels = torch.tensor([[-100, -100, -100, 1, 2], [-100, 3, -100, -100, -100]])

    monkeypatch.setattr(F, "linear", counting_linear)
    loss, _ = chunk_loss_module.chunk_loss_function(
        hidden_states, weights, labels, chunk_size=2, shift_labels=labels, logits_bias=logits_bias
    )
    monkeypatch.setattr(F, "linear", linear)
    loss.backward()

    assert projected_rows == [2, 1]
    expected = _reference_loss_and_grads(hidden_states, weights, logits_bias, labels)
    torch.testing.assert_close(loss, expected[0])
    torch.testing.assert_close(hidden_states.grad, expected[1])
    torch.testing.assert_close(weights.grad, expected[2])
    torch.testing.assert_close(logits_bias.grad, expected[3])


def test_chunk_loss_logits_bias_with_nothing_supervised_scores_zero(monkeypatch):
    monkeypatch.setattr(chunk_loss_module, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    hidden_states = torch.randn(1, 3, 4, requires_grad=True)
    weights = torch.randn(6, 4, requires_grad=True)
    logits_bias = torch.randn(1, 3, 6, requires_grad=True)
    labels = torch.full((1, 3), -100)

    loss, _ = chunk_loss_module.chunk_loss_function(
        hidden_states, weights, labels, chunk_size=2, shift_labels=labels, logits_bias=logits_bias
    )
    loss.backward()

    assert loss.item() == 0.0
    assert not hidden_states.grad.any() and not weights.grad.any() and not logits_bias.grad.any()


def test_chunk_loss_rejects_logits_bias_with_mismatched_rows(monkeypatch):
    monkeypatch.setattr(chunk_loss_module, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    with pytest.raises(ValueError, match="logits_bias rows"):
        chunk_loss_module.chunk_loss_function(
            torch.randn(1, 5, 4),
            torch.randn(6, 4),
            torch.zeros(1, 5, dtype=torch.long),
            logits_bias=torch.randn(1, 4, 6),
        )


def test_causal_lm_loss_refuses_logits_bias():
    with pytest.raises(NotImplementedError, match="logits_bias is only supported"):
        cross_entropy.ForCausalLMLoss(
            logits=torch.randn(1, 3, 4),
            labels=torch.tensor([[1, 2, 3]]),
            vocab_size=4,
            logits_bias=torch.randn(1, 3, 4),
        )
