from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import veomni.ops.kernels.cross_entropy as cross_entropy
import veomni.ops.kernels.cross_entropy.chunk_loss as chunk_loss_module


def test_causal_lm_loss_honors_explicit_shift_labels_and_reduction_group(monkeypatch):
    explicit_group = object()
    labels = torch.tensor([[0, 1, 2]])
    shift_labels = torch.tensor([[2, -100, 1]])
    captured = {}

    monkeypatch.setattr(cross_entropy, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=True))

    def fake_cross_entropy(logits, target, *args, **kwargs):
        captured["target"] = target.clone()
        return torch.tensor(3.0), logits

    def fake_reduce(loss, num_valid_tokens, group=None):
        captured["num_valid_tokens"] = num_valid_tokens.clone()
        captured["group"] = group
        return loss

    monkeypatch.setattr(cross_entropy, "reduce_sequence_parallel_loss", fake_reduce)

    loss, _, _ = cross_entropy.ForCausalLMLoss(
        logits=torch.randn(1, 3, 4),
        labels=labels,
        shift_labels=shift_labels,
        vocab_size=4,
        cross_entropy_fn=fake_cross_entropy,
        loss_reduction_group=explicit_group,
    )

    torch.testing.assert_close(loss, torch.tensor(3.0))
    torch.testing.assert_close(captured["target"], shift_labels.view(-1))
    torch.testing.assert_close(captured["num_valid_tokens"], torch.tensor(2))
    assert captured["group"] is explicit_group


def test_causal_lm_loss_reduces_explicit_group_without_sequence_parallel(monkeypatch):
    explicit_group = object()
    captured = {}

    monkeypatch.setattr(cross_entropy, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))

    def fake_cross_entropy(logits, target, *args, **kwargs):
        return torch.tensor(2.0), logits

    def fake_reduce(loss, num_valid_tokens, group=None):
        captured["num_valid_tokens"] = num_valid_tokens.clone()
        captured["group"] = group
        return loss

    monkeypatch.setattr(cross_entropy, "reduce_sequence_parallel_loss", fake_reduce)

    shift_labels = torch.tensor([[1, -100, 3]])
    loss, _, _ = cross_entropy.ForCausalLMLoss(
        logits=torch.randn(1, 3, 4),
        labels=shift_labels,
        shift_labels=shift_labels,
        vocab_size=4,
        cross_entropy_fn=fake_cross_entropy,
        loss_reduction_group=explicit_group,
    )

    torch.testing.assert_close(loss, torch.tensor(2.0))
    torch.testing.assert_close(captured["num_valid_tokens"], torch.tensor(2))
    assert captured["group"] is explicit_group


def test_chunk_loss_scores_flat_pre_shifted_rows(monkeypatch):
    monkeypatch.setattr(chunk_loss_module, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    hidden_states = torch.randn(5, 3)
    weights = torch.randn(4, 3)
    shift_labels = torch.tensor([2, -100, 1, 3, 0])

    loss, _, _ = cross_entropy._chunk_loss_dispatch(
        logits=None,
        labels=shift_labels,
        shift_labels=shift_labels,
        vocab_size=4,
        hidden_states=hidden_states,
        weights=weights,
        chunk_size=2,
    )

    expected = F.cross_entropy(F.linear(hidden_states, weights), shift_labels, ignore_index=-100)
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize(
    ("sp_enabled", "reduction_group", "reduced"),
    [
        pytest.param(False, object(), True, id="explicit-group"),
        pytest.param(True, None, True, id="sp-default-group"),
        pytest.param(False, None, False, id="no-reduction"),
    ],
)
def test_chunk_loss_reduces_like_causal_lm_loss(monkeypatch, sp_enabled, reduction_group, reduced):
    monkeypatch.setattr(chunk_loss_module, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=sp_enabled))
    captured = {}

    def fake_reduce(loss, num_valid_tokens, group=None):
        captured["num_valid_tokens"] = num_valid_tokens.clone()
        captured["group"] = group
        return loss

    monkeypatch.setattr(chunk_loss_module, "reduce_sequence_parallel_loss", fake_reduce)

    shift_labels = torch.tensor([[2, -100, 1]])
    cross_entropy._chunk_loss_dispatch(
        logits=None,
        labels=shift_labels,
        shift_labels=shift_labels,
        vocab_size=4,
        hidden_states=torch.randn(1, 3, 3),
        weights=torch.randn(4, 3),
        loss_reduction_group=reduction_group,
    )

    if not reduced:
        assert captured == {}
        return
    torch.testing.assert_close(captured["num_valid_tokens"], torch.tensor(2))
    assert captured["group"] is reduction_group


def test_chunk_loss_scores_zero_without_supervised_rows(monkeypatch):
    monkeypatch.setattr(chunk_loss_module, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    hidden_states = torch.randn(4, 3, requires_grad=True)
    weights = torch.randn(5, 3, requires_grad=True)
    shift_labels = torch.full((4,), -100)

    loss, _, _ = cross_entropy._chunk_loss_dispatch(
        logits=None,
        labels=shift_labels,
        shift_labels=shift_labels,
        vocab_size=5,
        hidden_states=hidden_states,
        weights=weights,
    )
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(0.0))
    assert torch.isfinite(hidden_states.grad).all() and torch.isfinite(weights.grad).all()


def test_chunk_loss_dispatch_forwards_chunk_size(monkeypatch):
    captured = {}

    def fake_chunk_loss_function(*args, chunk_size=1024, **kwargs):
        captured["chunk_size"] = chunk_size
        return torch.tensor(0.0), None

    monkeypatch.setattr(cross_entropy, "chunk_loss_function", fake_chunk_loss_function)

    cross_entropy._chunk_loss_dispatch(
        logits=None,
        labels=torch.tensor([[1, 2]]),
        vocab_size=4,
        hidden_states=torch.randn(1, 2, 3),
        weights=torch.randn(4, 3),
        chunk_size=7,
    )

    assert captured["chunk_size"] == 7
