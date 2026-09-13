# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing limitations
# under the License.

"""Model-facing causal-LM and sequence-classification loss policy."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.loss.loss_utils import fixed_cross_entropy

import veomni.models_kernel.loss_utils.cross_entropy_loss as loss_utils
from tests.ops.tol import EAGER_ATOL, EAGER_GRAD_ATOL, EAGER_GRAD_RTOL, EAGER_RTOL
from veomni.models_kernel.loss_utils import ForCausalLMLoss, ForSequenceClassificationLoss
from veomni.ops import VeomniOp


IGNORE_INDEX = -100
VOCAB = 16
HIDDEN = 8


class _RecordingOp:
    def __init__(self, inner: VeomniOp) -> None:
        self.inner = inner
        self.labels: Tensor | None = None
        self.kwargs: dict | None = None

    def __call__(self, hidden: Tensor, labels: Tensor, weight: Tensor, **kwargs):
        self.labels = labels.detach().clone()
        self.kwargs = dict(kwargs)
        return self.inner(hidden, labels, weight, **kwargs)


def _hf_fused_ce(hidden: Tensor, weight: Tensor, target: Tensor) -> Tensor:
    hidden_flat = hidden.reshape(-1, hidden.shape[-1])
    logits = F.linear(hidden_flat, weight).float()
    return fixed_cross_entropy(logits, target.reshape(-1), ignore_index=IGNORE_INDEX)


def _packed_targets() -> tuple[Tensor, Tensor, Tensor]:
    """Two packed length-3 segments.

    Naive causal shift of the concatenated labels would train across the
    segment boundary. The pre-shifted target marks each segment tail as
    ``ignore_index``.
    """
    labels = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    naive_shift = torch.tensor([[2, 3, 4, 5, 6, IGNORE_INDEX]], dtype=torch.long)
    packed_shift = torch.tensor([[2, 3, IGNORE_INDEX, 5, 6, IGNORE_INDEX]], dtype=torch.long)
    return labels, naive_shift, packed_shift


def test_packed_shift_labels_match_hf_loss_and_grads(monkeypatch):
    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    torch.manual_seed(0)
    _labels, naive_shift, packed_shift = _packed_targets()
    hidden = torch.randn(1, 6, HIDDEN, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float32, requires_grad=True)

    hidden_h = hidden.detach().clone().requires_grad_(True)
    weight_h = weight.detach().clone().requires_grad_(True)
    expected = _hf_fused_ce(hidden_h, weight_h, packed_shift)
    expected.backward()

    hidden_o = hidden.detach().clone().requires_grad_(True)
    weight_o = weight.detach().clone().requires_grad_(True)
    op = _RecordingOp(VeomniOp("cross_entropy_loss", "standard", "eager"))
    loss, _logits, _aux = ForCausalLMLoss(
        hidden_states=hidden_o,
        weights=weight_o,
        labels=_labels,
        shift_labels=packed_shift,
        ignore_index=IGNORE_INDEX,
        op=op,
    )
    assert op.kwargs is not None
    assert "shift_labels" not in op.kwargs
    assert "loss_reduction_group" not in op.kwargs
    torch.testing.assert_close(op.labels, packed_shift.reshape(-1))
    torch.testing.assert_close(loss, expected, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    loss.backward()
    torch.testing.assert_close(hidden_o.grad, hidden_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    torch.testing.assert_close(weight_o.grad, weight_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)

    naive = _hf_fused_ce(hidden.detach(), weight.detach(), naive_shift)
    assert not torch.allclose(expected.detach(), naive, atol=EAGER_ATOL, rtol=EAGER_RTOL)


def test_shift_labels_win_when_sp_enabled(monkeypatch):
    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=True))
    recorded: dict = {}

    def fake_reduce(loss: Tensor, num_valid_tokens: Tensor, group=None) -> Tensor:
        recorded["num_valid_tokens"] = int(num_valid_tokens.detach())
        recorded["group"] = group
        return loss

    monkeypatch.setattr(loss_utils, "reduce_sequence_parallel_loss", fake_reduce)

    torch.manual_seed(1)
    labels, _naive_shift, packed_shift = _packed_targets()
    hidden = torch.randn(1, 6, HIDDEN, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float32, requires_grad=True)

    hidden_h = hidden.detach().clone().requires_grad_(True)
    weight_h = weight.detach().clone().requires_grad_(True)
    expected = _hf_fused_ce(hidden_h, weight_h, packed_shift)
    expected.backward()

    hidden_o = hidden.detach().clone().requires_grad_(True)
    weight_o = weight.detach().clone().requires_grad_(True)
    op = _RecordingOp(VeomniOp("cross_entropy_loss", "standard", "eager"))
    loss, _logits, _aux = ForCausalLMLoss(
        hidden_states=hidden_o,
        weights=weight_o,
        labels=labels,
        shift_labels=packed_shift,
        ignore_index=IGNORE_INDEX,
        op=op,
    )
    torch.testing.assert_close(op.labels, packed_shift.reshape(-1))
    assert recorded["num_valid_tokens"] == int((packed_shift != IGNORE_INDEX).sum())
    assert recorded["num_valid_tokens"] != int((labels != IGNORE_INDEX).sum())
    assert recorded["group"] is None
    torch.testing.assert_close(loss, expected, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    loss.backward()
    torch.testing.assert_close(hidden_o.grad, hidden_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    torch.testing.assert_close(weight_o.grad, weight_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_explicit_reduction_group_without_sp(monkeypatch):
    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    recorded: dict = {}
    sentinel_group = object()

    def fake_reduce(loss: Tensor, num_valid_tokens: Tensor, group=None) -> Tensor:
        recorded["num_valid_tokens"] = int(num_valid_tokens.detach())
        recorded["group"] = group
        return loss

    monkeypatch.setattr(loss_utils, "reduce_sequence_parallel_loss", fake_reduce)

    torch.manual_seed(2)
    _labels, _naive_shift, packed_shift = _packed_targets()
    hidden = torch.randn(1, 6, HIDDEN, dtype=torch.float32)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float32)
    op = _RecordingOp(VeomniOp("cross_entropy_loss", "standard", "eager"))

    ForCausalLMLoss(
        hidden_states=hidden,
        weights=weight,
        labels=None,
        shift_labels=packed_shift,
        ignore_index=IGNORE_INDEX,
        op=op,
        loss_reduction_group=sentinel_group,
    )
    assert recorded["group"] is sentinel_group
    assert recorded["num_valid_tokens"] == int((packed_shift != IGNORE_INDEX).sum())
    assert op.kwargs is not None
    assert "loss_reduction_group" not in op.kwargs


def test_no_reduce_when_sp_off_and_group_missing(monkeypatch):
    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))

    def fail_reduce(*_args, **_kwargs):
        raise AssertionError("reduction must stay off when SP is off and no group is given")

    monkeypatch.setattr(loss_utils, "reduce_sequence_parallel_loss", fail_reduce)

    labels, naive_shift, _packed_shift = _packed_targets()
    hidden = torch.randn(1, 6, HIDDEN, dtype=torch.float32)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float32)
    op = _RecordingOp(VeomniOp("cross_entropy_loss", "standard", "eager"))
    ForCausalLMLoss(
        hidden_states=hidden,
        weights=weight,
        labels=labels,
        ignore_index=IGNORE_INDEX,
        op=op,
    )
    torch.testing.assert_close(op.labels, naive_shift.reshape(-1))


def test_sp_without_shift_labels_uses_collator_labels(monkeypatch):
    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=True))
    monkeypatch.setattr(loss_utils, "reduce_sequence_parallel_loss", lambda loss, *_args, **_kwargs: loss)

    _labels, naive_shift, _packed_shift = _packed_targets()
    hidden = torch.randn(1, 6, HIDDEN, dtype=torch.float32)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float32)
    op = _RecordingOp(VeomniOp("cross_entropy_loss", "standard", "eager"))
    ForCausalLMLoss(
        hidden_states=hidden,
        weights=weight,
        labels=naive_shift,
        ignore_index=IGNORE_INDEX,
        op=op,
    )
    torch.testing.assert_close(op.labels, naive_shift.reshape(-1))


def test_causal_logits_path_matches_hf_and_returns_flattened_logits(monkeypatch):
    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    torch.manual_seed(5)
    logits = torch.randn(2, 4, VOCAB, requires_grad=True)
    labels = torch.randint(0, VOCAB, (2, 4))
    shifted = F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:].contiguous()

    loss, out_logits, aux = ForCausalLMLoss(
        logits=logits,
        labels=labels,
        vocab_size=VOCAB,
        op=VeomniOp("cross_entropy_loss", "standard", "eager"),
    )

    expected = fixed_cross_entropy(logits.float().view(-1, VOCAB), shifted.view(-1))
    torch.testing.assert_close(loss, expected, atol=EAGER_ATOL, rtol=EAGER_RTOL)
    torch.testing.assert_close(out_logits, logits.float().view(-1, VOCAB))
    assert aux is None


def test_causal_logprobs_dispatch_returns_aux_without_calling_ce(monkeypatch):
    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    hidden = torch.randn(1, 3, HIDDEN)
    weight = torch.randn(VOCAB, HIDDEN)
    labels = torch.tensor([[1, 2, 3]])
    expected_log_probs = torch.randn(1, 3)
    expected_entropy = torch.rand(1, 3)
    seen = {}

    def fake_chunk_logprobs(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return expected_log_probs, expected_entropy

    class FailOp:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("plain CE op must not run on return_log_probs=True")

    monkeypatch.setattr(loss_utils, "chunk_logprobs_function", fake_chunk_logprobs)
    loss, logits, aux = ForCausalLMLoss(
        hidden_states=hidden,
        weights=weight,
        labels=labels,
        return_log_probs=True,
        temperature=0.7,
        chunk_size=2,
        op=FailOp(),
    )

    assert loss is None and logits is None
    assert aux is not None
    assert aux.log_probs is expected_log_probs
    assert aux.entropy is expected_entropy
    assert seen["args"] == (hidden, weight, labels)
    assert seen["kwargs"]["temperature"] == 0.7
    assert seen["kwargs"]["chunk_size"] == 2


def test_causal_topk_teacher_tensors_must_be_paired(monkeypatch):
    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    with pytest.raises(ValueError, match="must be provided together"):
        ForCausalLMLoss(
            hidden_states=torch.randn(1, 3, HIDDEN),
            weights=torch.randn(VOCAB, HIDDEN),
            labels=torch.tensor([[1, 2, 3]]),
            return_log_probs=True,
            teacher_topk_ids=torch.zeros(1, 3, 2, dtype=torch.long),
        )


def test_seqcls_logits_path_matches_hf(monkeypatch):
    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    torch.manual_seed(6)
    logits = torch.randn(2, 3, 5, requires_grad=True)
    labels = torch.tensor([[IGNORE_INDEX, 1, IGNORE_INDEX], [2, IGNORE_INDEX, 4]])

    loss, out_logits, aux = ForSequenceClassificationLoss(
        logits=logits,
        labels=labels,
        num_labels=5,
        op=VeomniOp("cross_entropy_loss", "standard", "eager"),
    )

    expected = fixed_cross_entropy(logits.float().view(-1, 5), labels.view(-1))
    torch.testing.assert_close(loss, expected, atol=EAGER_ATOL, rtol=EAGER_RTOL)
    torch.testing.assert_close(out_logits, logits.float().view(-1, 5))
    assert aux is None


@pytest.mark.parametrize("impl", ["eager", "chunk_loss"])
def test_seqcls_hidden_weight_path_matches_hf_and_does_not_materialize_output_logits(monkeypatch, impl):
    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    torch.manual_seed(7)
    hidden = torch.randn(2, 3, HIDDEN, requires_grad=True)
    weight = torch.randn(5, HIDDEN, requires_grad=True)
    labels = torch.tensor([[IGNORE_INDEX, 1, IGNORE_INDEX], [2, IGNORE_INDEX, 4]])

    hidden_ref = hidden.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    expected = fixed_cross_entropy(F.linear(hidden_ref.view(-1, HIDDEN), weight_ref), labels.view(-1))
    expected.backward()

    loss, out_logits, aux = ForSequenceClassificationLoss(
        labels=labels,
        num_labels=5,
        hidden_states=hidden,
        weights=weight,
        op=VeomniOp("cross_entropy_loss", "standard", impl),
    )
    loss.backward()

    torch.testing.assert_close(loss, expected, atol=EAGER_ATOL, rtol=EAGER_RTOL)
    torch.testing.assert_close(hidden.grad, hidden_ref.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    torch.testing.assert_close(weight.grad, weight_ref.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert out_logits is None
    assert aux is None


def test_seqcls_prefers_fused_inputs_and_preserves_caller_logits(monkeypatch):
    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    logits = torch.randn(1, 3, 5)
    hidden = torch.randn(1, 3, HIDDEN)
    weight = torch.randn(5, HIDDEN)
    labels = torch.tensor([[IGNORE_INDEX, 1, 2]])
    op = _RecordingOp(VeomniOp("cross_entropy_loss", "standard", "eager"))

    loss, out_logits, _aux = ForSequenceClassificationLoss(
        logits=logits,
        labels=labels,
        num_labels=5,
        hidden_states=hidden,
        weights=weight,
        op=op,
    )

    expected = fixed_cross_entropy(F.linear(hidden.view(-1, HIDDEN), weight), labels.view(-1))
    torch.testing.assert_close(loss, expected, atol=EAGER_ATOL, rtol=EAGER_RTOL)
    torch.testing.assert_close(out_logits, logits.float().view(-1, 5))


def test_seqcls_sp_reduces_with_valid_target_count(monkeypatch):
    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=True))
    seen = {}

    def fake_reduce(loss, num_valid_tokens):
        seen["num_valid_tokens"] = int(num_valid_tokens)
        return loss

    monkeypatch.setattr(loss_utils, "reduce_sequence_parallel_loss", fake_reduce)
    ForSequenceClassificationLoss(
        logits=torch.randn(1, 4, 3),
        labels=torch.tensor([[0, IGNORE_INDEX, 1, IGNORE_INDEX]]),
        num_labels=3,
        op=VeomniOp("cross_entropy_loss", "standard", "eager"),
    )
    assert seen["num_valid_tokens"] == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"logits": torch.zeros(1, 2, 3), "labels": None, "num_labels": 3}, "labels must be provided"),
        ({"logits": torch.zeros(1, 2, 3), "labels": torch.zeros(1, 2, dtype=torch.long)}, "num_labels"),
        ({"labels": torch.zeros(1, 2, dtype=torch.long), "num_labels": 3}, "Either hidden_states or logits"),
    ],
)
def test_seqcls_validates_required_inputs(monkeypatch, kwargs, message):
    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    with pytest.raises(ValueError, match=message):
        ForSequenceClassificationLoss(**kwargs)


def test_reduce_loss_no_nan_when_sp_group_all_padding():
    """The distributed reducer must return graph-connected zero for 0/0 tokens."""
    from veomni.distributed.sequence_parallel.loss import ReduceLoss

    with (
        patch(
            "veomni.distributed.sequence_parallel.loss.get_unified_sequence_parallel_group", return_value=MagicMock()
        ),
        patch("veomni.distributed.sequence_parallel.loss.dist.get_world_size", return_value=2),
        patch("veomni.distributed.sequence_parallel.loss.dist.all_reduce", side_effect=lambda *_args, **_kwargs: None),
    ):
        value = torch.tensor(0.5, requires_grad=True)
        result = ReduceLoss.apply(value * 1.0, torch.tensor(0.0))
        assert torch.isfinite(result)
        assert result.item() == 0.0
        result.backward()
        assert value.grad is not None
        assert torch.isfinite(value.grad)
        assert value.grad.item() == 0.0
