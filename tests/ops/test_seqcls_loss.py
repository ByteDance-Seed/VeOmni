import pytest
import torch

import veomni.ops.fused_cross_entropy as m
from veomni.data.constants import IGNORE_INDEX
from veomni.utils.device import get_device_type, get_torch_device


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
    Case:
        logits provided
        hidden_states/weights = None
        sp_enabled = False
    Manually calculate the cross-entropy for a single effective token, and verify that it matches the function output.
    """
    monkeypatch.setattr(m, "get_parallel_state", lambda: _FakePS(sp_enabled=False))

    ignore = IGNORE_INDEX
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
    Case:
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
    Case:
        logits provided
        hidden_states + weights present (matrix path available)
        sp_enabled = False

    Expected (with Liger fused loss enabled):
      - loss is computed from (hidden_states, weights, labels) via fused linear cross-entropy.
        The passed-in `logits` is NOT used for loss computation in this fused path.
      - out_logits is the flattened *input* logits, because fused_liger_kernel_cross_entropy
        returns `(loss, logits)` without materializing projected logits.
    """
    dev_api = get_torch_device()
    local_rank = 0
    dev_api.set_device(f"{get_device_type()}:{local_rank}")

    device = torch.device(get_device_type(), local_rank)
    monkeypatch.setattr(m, "get_parallel_state", lambda: _FakePS(sp_enabled=False))

    ignore = IGNORE_INDEX
    B, T, H, C = 1, 2, 5, 3

    hidden_states = torch.tensor(
        [[[1.0, 0.0, -1.0, 2.0, 0.5], [0.5, -1.0, 0.0, 1.5, -0.5]]],
        device=device,
        dtype=torch.float32,
    )
    weights = torch.tensor(
        [[0.2, -0.1, 0.0, 0.3, 0.5], [-0.4, 0.6, 0.1, -0.2, 0.0], [0.1, 0.2, -0.3, 0.0, 0.4]],
        device=device,
        dtype=torch.float32,
    )

    logits = torch.randn((B, T, C), device=device, dtype=torch.float32)
    labels = torch.tensor([[ignore, 1]], device=device, dtype=torch.long)

    loss, out_logits = m.ForSequenceClassificationLoss(
        logits=logits,
        labels=labels,
        num_labels=C,
        ignore_index=ignore,
        hidden_states=hidden_states,
        weights=weights,
    )

    proj = hidden_states.reshape(-1, H) @ weights.t()  # [B*T, C]
    flat_labels = labels.reshape(-1)  # [B*T]
    valid = flat_labels != ignore

    log_probs = torch.log_softmax(proj, dim=-1)  # [B*T, C]
    nll = -log_probs[valid, flat_labels[valid]]  # [num_valid]
    expected = nll.mean() if nll.numel() > 0 else proj.sum() * 0.0

    assert torch.isfinite(loss)
    # Fused kernel may have tiny numerical differences; allow small tolerance.
    assert torch.allclose(loss.float(), expected.float(), atol=2e-3, rtol=0.0)

    assert out_logits is not None
    assert out_logits.shape == (B * T, C)
    assert out_logits.dtype == torch.float32
    assert out_logits.device == logits.device
    # Current contract: fused returns *input logits* (flattened), not projected logits
    assert torch.allclose(out_logits, logits.view(-1, C).float(), atol=1e-6)


def test_seqcls_loss_sp_enabled_calls_reduce_with_correct_num_valid_tokens(monkeypatch):
    """
    Case:
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
    ignore = IGNORE_INDEX
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
    Case:
        labels = None
        num_labels = None
        logits = None and hidden_states = None
    """
    monkeypatch.setattr(m, "get_parallel_state", lambda: _FakePS(sp_enabled=False))

    logits = torch.zeros((1, 2, 3))
    labels = torch.tensor([[-100, 1]])

    # labels None -> assert
    with pytest.raises(ValueError, match="labels must be provided"):
        m.ForSequenceClassificationLoss(logits=logits, labels=None, num_labels=3)

    # num_labels None -> assert
    with pytest.raises(ValueError, match="num_labels must be provided"):
        m.ForSequenceClassificationLoss(logits=logits, labels=labels, num_labels=None)

    # logits and hidden_states both None -> assert
    with pytest.raises(ValueError, match="Either hidden_states or logits must be provided"):
        m.ForSequenceClassificationLoss(logits=None, labels=labels, num_labels=3)
