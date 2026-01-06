import types

import torch

import veomni.ops.loss as m


def _fake_ps(sp_enabled: bool):
    return types.SimpleNamespace(sp_enabled=sp_enabled)


def test_seqcls_token_loss_labels_none(monkeypatch):
    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=False))

    H = 4
    C = 3
    B, L = 2, 5
    hidden_states = torch.randn(B, L, H)
    weight = torch.randn(C, H)

    loss, logits = m.seqcls_token_loss_function(hidden_states, weight, labels=None)

    assert loss is None
    assert logits.shape == (B, L, C)


def test_seqcls_token_loss_flattens_and_calls_fixed_cross_entropy(monkeypatch):
    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=False))
    monkeypatch.setattr(m, "fused_linear_cross_entropy", None)

    called = {}

    def fake_fixed_cross_entropy(logits, labels, num_items_in_batch, ignore_index, **kwargs):
        # check the shape after flatten
        called["logits_shape"] = tuple(logits.shape)
        called["labels_shape"] = tuple(labels.shape)
        called["ignore_index"] = ignore_index
        # Returns a returnable scalar value.
        return logits.sum() * 0.0 + labels.sum() * 0.0

    monkeypatch.setattr(m, "fixed_cross_entropy", fake_fixed_cross_entropy)

    H = 8
    C = 5
    B, L = 2, 3
    hidden_states = torch.randn(B, L, H)
    weight = torch.randn(C, H)
    labels = torch.tensor([[1, -100, 2], [3, 4, -100]], dtype=torch.long)

    loss, logits = m.seqcls_token_loss_function(hidden_states, weight, labels=labels, ignore_index=-100)

    assert loss is not None
    assert logits is not None
    assert tuple(logits.shape) == (B * L, C)

    assert called["logits_shape"] == (B * L, C)
    assert called["labels_shape"] == (B * L,)
    assert called["ignore_index"] == -100


def test_seqcls_token_loss_sp_reduce_called_with_correct_num_valid_tokens(monkeypatch):
    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=True))
    monkeypatch.setattr(m, "fused_linear_cross_entropy", None)

    # fixed_cross_entropy returns a fixed loss.
    monkeypatch.setattr(
        m, "fixed_cross_entropy", lambda logits, labels, num_items_in_batch, ignore_index, **kw: torch.tensor(3.0)
    )

    got = {}

    def fake_reduce_sequence_parallel_loss(loss, num_valid_tokens):
        got["loss"] = float(loss.item())
        got["num_valid_tokens"] = int(num_valid_tokens.item())
        return loss * 1.0

    monkeypatch.setattr(m, "reduce_sequence_parallel_loss", fake_reduce_sequence_parallel_loss)

    H = 4
    C = 2
    B, L = 1, 6
    hidden_states = torch.randn(B, L, H)
    weight = torch.randn(C, H)

    labels = torch.tensor([[0, -100, -100, 1, 0, -100]], dtype=torch.long)
    # valid tokens = 3 (positions 0,3,4)
    loss, logits = m.seqcls_token_loss_function(hidden_states, weight, labels=labels, ignore_index=-100)

    assert loss is not None
    assert got["loss"] == 3.0
    assert got["num_valid_tokens"] == 3


def test_seqcls_token_loss_sp_reduce_not_called_when_sp_disabled(monkeypatch):
    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=False))
    monkeypatch.setattr(m, "fused_linear_cross_entropy", None)
    monkeypatch.setattr(
        m, "fixed_cross_entropy", lambda logits, labels, num_items_in_batch, ignore_index, **kw: torch.tensor(2.0)
    )
    monkeypatch.setattr(
        m,
        "reduce_sequence_parallel_loss",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("reduce_sequence_parallel_loss should not be called")
        ),
    )

    H = 4
    C = 2
    B, L = 1, 4
    hidden_states = torch.randn(B, L, H)
    weight = torch.randn(C, H)
    labels = torch.tensor([[0, -100, 1, -100]], dtype=torch.long)

    loss, logits = m.seqcls_token_loss_function(hidden_states, weight, labels=labels, ignore_index=-100)
    assert float(loss.item()) == 2.0
