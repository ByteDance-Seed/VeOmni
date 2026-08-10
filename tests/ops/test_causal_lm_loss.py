from types import SimpleNamespace

import pytest
import torch

import veomni.ops.kernels.cross_entropy as cross_entropy


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


@pytest.mark.parametrize(
    "reduction_group",
    [pytest.param(object(), id="group"), pytest.param(None, id="none")],
)
def test_chunk_loss_rejects_pre_shifted_contract_for_any_reduction_group(reduction_group):
    with pytest.raises(NotImplementedError, match="does not support explicitly pre-shifted targets"):
        cross_entropy._chunk_loss_dispatch(
            logits=None,
            labels=torch.tensor([1, 2]),
            shift_labels=torch.tensor([2, -100]),
            vocab_size=4,
            hidden_states=torch.randn(2, 3),
            weights=torch.randn(4, 3),
            loss_reduction_group=reduction_group,
        )


def test_chunk_loss_rejects_pre_shifted_contract_without_reduction_group():
    with pytest.raises(NotImplementedError, match="does not support explicitly pre-shifted targets"):
        cross_entropy._chunk_loss_dispatch(
            logits=None,
            labels=torch.tensor([1, 2]),
            shift_labels=torch.tensor([2, -100]),
            vocab_size=4,
            hidden_states=torch.randn(2, 3),
            weights=torch.randn(4, 3),
        )
