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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for chunked fused-linear gradient gating."""

import pytest
import torch

import veomni.ops.kernels.cross_entropy.chunk_logprobs as cl
import veomni.ops.kernels.cross_entropy.chunk_topk_distill as ctkd


class _FakePS:
    sp_enabled = False


@pytest.fixture(autouse=True)
def _use_cpu_compatible_path(monkeypatch):
    monkeypatch.setattr(cl, "_FA_CE_AVAILABLE", False)
    monkeypatch.setattr(cl, "get_parallel_state", lambda: _FakePS())
    monkeypatch.setattr(ctkd, "get_parallel_state", lambda: _FakePS())


def _make_non_contiguous_hidden(batch_size: int, seq_len: int, hidden_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    leaf = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    hidden = (leaf * 1.0).permute(1, 0, 2).contiguous().permute(1, 0, 2)
    assert not hidden.is_contiguous()
    assert hidden.reshape(-1, hidden_size).data_ptr() != hidden.data_ptr()
    return leaf, hidden


@pytest.mark.parametrize("weight_requires_grad", [True, False], ids=["trainable_weight", "frozen_weight"])
def test_chunk_logprobs_non_contiguous_hidden_preserves_trunk_gradient(weight_requires_grad):
    torch.manual_seed(0)
    batch_size, seq_len, hidden_size, vocab_size = 2, 8, 4, 16
    hidden_leaf, hidden = _make_non_contiguous_hidden(batch_size, seq_len, hidden_size)
    weight = torch.randn(vocab_size, hidden_size, requires_grad=weight_requires_grad)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    log_probs, _ = cl.chunk_logprobs_function(
        hidden,
        weight,
        labels,
        shift_labels=labels,
        chunk_size=5,
    )
    assert log_probs.requires_grad
    log_probs.sum().backward()

    assert hidden_leaf.grad is not None, "trunk gradient was silently dropped"
    assert hidden_leaf.grad.abs().sum() > 0
    assert (weight.grad is not None) == weight_requires_grad


@pytest.mark.parametrize("weight_requires_grad", [True, False], ids=["trainable_weight", "frozen_weight"])
def test_chunk_topk_distill_non_contiguous_hidden_preserves_trunk_gradient(weight_requires_grad):
    torch.manual_seed(0)
    batch_size, seq_len, hidden_size, vocab_size, topk = 2, 8, 4, 16, 3
    hidden_leaf, hidden = _make_non_contiguous_hidden(batch_size, seq_len, hidden_size)
    weight = torch.randn(vocab_size, hidden_size, requires_grad=weight_requires_grad)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_log_probs = teacher_logits.log_softmax(dim=-1)
    teacher_topk_log_probs, teacher_topk_ids = teacher_log_probs.topk(topk, dim=-1)

    _, _, distill, _, _ = ctkd.chunk_topk_distill_function(
        hidden,
        weight,
        labels,
        teacher_topk_ids,
        teacher_topk_log_probs,
        shift_labels=labels,
        chunk_size=5,
    )
    assert distill.requires_grad
    distill.sum().backward()

    assert hidden_leaf.grad is not None, "trunk gradient was silently dropped"
    assert hidden_leaf.grad.abs().sum() > 0
    assert (weight.grad is not None) == weight_requires_grad
