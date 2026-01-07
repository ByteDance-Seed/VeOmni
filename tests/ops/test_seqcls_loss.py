import math

import pytest
import torch

import veomni.ops.fused_cross_entropy as m


def _manual_ce_one_token(logits_1d: torch.Tensor, target: int) -> float:
    """
    calculate cross-entropy manually： -log softmax[target]
    """
    logp = torch.log_softmax(logits_1d, dim=-1)
    return float(-logp[target].item())


class _FakePS:
    def __init__(self, sp_enabled: bool):
        self.sp_enabled = sp_enabled


def test_seqcls_loss_logits_path_manual_handcalc(monkeypatch):
    """
    logits provided
    hidden_states/weights = None
    sp_enabled = False
    Manually calculate the cross-entropy for a single effective token, and verify that it matches the function output.
    """
    monkeypatch.setattr(m, "get_parallel_state", lambda: _FakePS(sp_enabled=False))

    ignore = -100
    num_labels = 3

    logits = torch.tensor(
        [
            [
                [1.0, 0.0, -1.0],  # ignored
                [0.0, 0.0, 0.0],  # ignored
                [2.0, 1.0, 0.0],  # supervised, target=2
            ]
        ]
    )

    labels = torch.tensor([[ignore, ignore, 2]])

    expected = _manual_ce_one_token(logits[0, 2], target=2)
    loss, out_logits = m.ForSequenceClassificationLoss(
        logits=logits,
        labels=labels,
        num_labels=num_labels,
        ignore_index=ignore,
    )

    assert out_logits is not None
    assert out_logits.shape == (1 * 3, 3)
    assert torch.allclose(out_logits, logits.view(-1, num_labels).float())
    assert torch.isfinite(loss)
    assert abs(loss.item() - expected) < 1e-6


def test_seqcls_loss_hidden_states_weights_path_build_logits_and_loss(monkeypatch):
    """
    logits = None
    hidden_states and weights provided
    sp_enabled = False
    """
    monkeypatch.setattr(m, "get_parallel_state", lambda: _FakePS(sp_enabled=False))
    monkeypatch.setattr(m, "_cross_entropy", m.eager_cross_entropy)

    ignore = -100
    num_labels = 4
    B, L = 1, 3
    hidden_states = torch.tensor(
        [
            [
                [1.0, 0.0],  # ignored
                [0.0, 1.0],  # ignored
                [1.0, 1.0],  # supervised
            ]
        ]
    )
    weights = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    labels = torch.tensor([[ignore, ignore, 2]])
    supervised_logits = torch.tensor([1.0, 1.0, 2.0, -1.0])
    expected = _manual_ce_one_token(supervised_logits, target=2)

    loss, out_logits = m.ForSequenceClassificationLoss(
        logits=None,
        labels=labels,
        num_labels=num_labels,
        ignore_index=ignore,
        hidden_states=hidden_states,
        weights=weights,
    )

    assert torch.isfinite(loss)
    assert abs(loss.item() - expected) < 1e-6

    assert out_logits.shape == (B * L, num_labels)
    assert out_logits.dtype == torch.float32
    sup_row = out_logits.view(B, L, num_labels)[0, 2]
    assert torch.allclose(sup_row.cpu(), supervised_logits.float(), atol=1e-6)


def test_seqcls_loss_prefers_cross_entropy_when_hidden_states_and_weights_present(monkeypatch):
    """
    logits provided
    hidden_states+weights present
    sp_enabled = False
    """
    device = torch.device("cuda")
    monkeypatch.setattr(m, "get_parallel_state", lambda: _FakePS(sp_enabled=False))
    ignore = -100
    logits = torch.zeros((1, 2, 3), device=device)
    labels = torch.tensor([[ignore, 1]], device=device)
    hidden_states = torch.zeros((1, 2, 5), device=device)
    weights = torch.zeros((3, 5), device=device)

    loss, out_logits = m.ForSequenceClassificationLoss(
        logits=logits,
        labels=labels,
        num_labels=3,
        ignore_index=ignore,
        hidden_states=hidden_states,
        weights=weights,
    )
    expected = math.log(float(3))
    assert torch.isfinite(loss)
    assert abs(loss.item() - expected) < 1e-6

    assert out_logits is not None
    assert out_logits.shape == (1 * 2, 3)
    assert out_logits.dtype == torch.float32
    assert out_logits.device == logits.device
    assert torch.allclose(out_logits, logits.view(-1, 3).float())


def test_seqcls_loss_sp_enabled_calls_reduce_with_correct_num_valid_tokens(monkeypatch):
    """
    sp_enabled=True
    """
    seen = {"called": False}
    monkeypatch.setattr(m, "get_parallel_state", lambda: _FakePS(sp_enabled=True))

    def _fake_reduce(loss, num_valid_tokens):
        # there are 4 tokens, 2 are ignore_index, 2 are valid
        assert int(num_valid_tokens.item()) == 2
        seen["called"] = True
        return loss  # identity

    monkeypatch.setattr(m, "reduce_sequence_parallel_loss", _fake_reduce)
    ignore = -100
    num_labels = 3

    logits = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],  # valid (target=0)
                [1.0, 0.0, 0.0],  # ignored
                [0.0, 1.0, 0.0],  # valid (target=1)
                [0.0, 0.0, 1.0],  # ignored
            ]
        ]
    )
    labels = torch.tensor([[0, ignore, 1, ignore]])

    e0 = _manual_ce_one_token(logits[0, 0], target=0)
    e1 = _manual_ce_one_token(logits[0, 2], target=1)
    expected = (e0 + e1) / 2.0

    loss, _ = m.ForSequenceClassificationLoss(
        logits=logits,
        labels=labels,
        num_labels=num_labels,
        ignore_index=ignore,
    )

    assert seen["called"] is True
    assert abs(loss.item() - expected) < 1e-6


def test_seqcls_loss_assertions(monkeypatch):
    """
    labels = None
    num_labels = None
    logits = None and hidden_states = None
    """
    monkeypatch.setattr(m, "get_parallel_state", lambda: _FakePS(sp_enabled=False))

    logits = torch.zeros((1, 2, 3))
    labels = torch.tensor([[-100, 1]])

    # labels None -> assert
    with pytest.raises(AssertionError):
        m.ForSequenceClassificationLoss(logits=logits, labels=None, num_labels=3)

    # num_labels None -> assert
    with pytest.raises(AssertionError):
        m.ForSequenceClassificationLoss(logits=logits, labels=labels, num_labels=None)

    # logits and hidden_states both None -> assert
    with pytest.raises(AssertionError):
        m.ForSequenceClassificationLoss(logits=None, labels=labels, num_labels=3)
