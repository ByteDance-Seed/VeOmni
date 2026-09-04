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

"""ForCausalLMLoss target selection and caller-selected reduction."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.loss.loss_utils import fixed_cross_entropy

from tests.kernels.tol import EAGER_ATOL, EAGER_GRAD_ATOL, EAGER_GRAD_RTOL, EAGER_RTOL
from veomni.kernels import VeomniKernel
from veomni.models_kernel.utils import loss_utils
from veomni.models_kernel.utils.loss_utils import ForCausalLMLoss


IGNORE_INDEX = -100
VOCAB = 16
HIDDEN = 8


class _RecordingKernel:
    def __init__(self, inner: VeomniKernel) -> None:
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
    kernel = _RecordingKernel(VeomniKernel("cross_entropy_loss", "standard", "eager"))
    loss, _logits, _aux = ForCausalLMLoss(
        hidden_states=hidden_o,
        weights=weight_o,
        labels=_labels,
        shift_labels=packed_shift,
        ignore_index=IGNORE_INDEX,
        kernel=kernel,
    )
    assert kernel.kwargs is not None
    assert "shift_labels" not in kernel.kwargs
    assert "loss_reduction_group" not in kernel.kwargs
    torch.testing.assert_close(kernel.labels, packed_shift.reshape(-1))
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
    kernel = _RecordingKernel(VeomniKernel("cross_entropy_loss", "standard", "eager"))
    loss, _logits, _aux = ForCausalLMLoss(
        hidden_states=hidden_o,
        weights=weight_o,
        labels=labels,
        shift_labels=packed_shift,
        ignore_index=IGNORE_INDEX,
        kernel=kernel,
    )
    torch.testing.assert_close(kernel.labels, packed_shift.reshape(-1))
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
    kernel = _RecordingKernel(VeomniKernel("cross_entropy_loss", "standard", "eager"))

    ForCausalLMLoss(
        hidden_states=hidden,
        weights=weight,
        labels=None,
        shift_labels=packed_shift,
        ignore_index=IGNORE_INDEX,
        kernel=kernel,
        loss_reduction_group=sentinel_group,
    )
    assert recorded["group"] is sentinel_group
    assert recorded["num_valid_tokens"] == int((packed_shift != IGNORE_INDEX).sum())
    assert kernel.kwargs is not None
    assert "loss_reduction_group" not in kernel.kwargs


def test_no_reduce_when_sp_off_and_group_missing(monkeypatch):
    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))

    def fail_reduce(*_args, **_kwargs):
        raise AssertionError("reduction must stay off when SP is off and no group is given")

    monkeypatch.setattr(loss_utils, "reduce_sequence_parallel_loss", fail_reduce)

    labels, naive_shift, _packed_shift = _packed_targets()
    hidden = torch.randn(1, 6, HIDDEN, dtype=torch.float32)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float32)
    kernel = _RecordingKernel(VeomniKernel("cross_entropy_loss", "standard", "eager"))
    ForCausalLMLoss(
        hidden_states=hidden,
        weights=weight,
        labels=labels,
        ignore_index=IGNORE_INDEX,
        kernel=kernel,
    )
    torch.testing.assert_close(kernel.labels, naive_shift.reshape(-1))


def test_sp_without_shift_labels_uses_collator_labels(monkeypatch):
    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=True))
    monkeypatch.setattr(loss_utils, "reduce_sequence_parallel_loss", lambda loss, *_args, **_kwargs: loss)

    _labels, naive_shift, _packed_shift = _packed_targets()
    hidden = torch.randn(1, 6, HIDDEN, dtype=torch.float32)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float32)
    kernel = _RecordingKernel(VeomniKernel("cross_entropy_loss", "standard", "eager"))
    ForCausalLMLoss(
        hidden_states=hidden,
        weights=weight,
        labels=naive_shift,
        ignore_index=IGNORE_INDEX,
        kernel=kernel,
    )
    torch.testing.assert_close(kernel.labels, naive_shift.reshape(-1))
