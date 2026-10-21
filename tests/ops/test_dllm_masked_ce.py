# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Tests for the dLLM masked cross-entropy op.

Pins the op against an independent implementation of the masked-CE recipe::

    token_loss = CE(logits[mask], target[mask], reduction='none') / p_mask[mask]
    ce_loss    = sum(token_loss / answer_lengths[mask]) / B

with the op's ``loss_weight_i = 1/(p_mask_i * answer_length_i)`` and
``reduction="mean", num_samples=B``.
"""

import torch
import torch.nn.functional as F

from veomni.models.loss_utils.dllm import dllm_masked_cross_entropy


IGNORE = -100


def _dllm_recipe_loss(logits, input_ids, masked, p_mask, answer_lengths, batch_size):
    """Reference masked-CE (no shift, masked positions only)."""
    logits_f = logits.reshape(-1, logits.shape[-1]).float()
    ids = input_ids.reshape(-1)
    m = masked.reshape(-1)
    pmf = p_mask.reshape(-1)
    alf = answer_lengths.reshape(-1)

    token_loss = F.cross_entropy(logits_f[m], ids[m], reduction="none") / pmf[m]
    return (token_loss / alf[m]).sum() / batch_size


def _build_batch(seed=0):
    torch.manual_seed(seed)
    b, length, vocab = 2, 6, 16
    logits = torch.randn(b, length, vocab, dtype=torch.float32)
    input_ids = torch.randint(0, vocab, (b, length))

    # Per-sample mask ratio and answer length (the prompt is positions [0:prompt_len)).
    p_mask = torch.tensor([0.4, 0.7])  # t_s per sample
    prompt_lens = torch.tensor([2, 3])
    answer_lengths = (length - prompt_lens).float()  # N_s

    # Mask a subset of the answer region per sample.
    masked = torch.zeros(b, length, dtype=torch.bool)
    masked[0, 3] = True
    masked[0, 5] = True
    masked[1, 4] = True

    # labels: original id at masked positions, IGNORE elsewhere.
    labels = torch.full((b, length), IGNORE, dtype=torch.long)
    labels[masked] = input_ids[masked]

    # loss_weight_i = 1 / (t_s * N_s), broadcast per sample.
    p_mask_full = p_mask[:, None].expand(b, length)
    answer_len_full = answer_lengths[:, None].expand(b, length)

    return logits, input_ids, labels, masked, p_mask_full, answer_len_full, b


def test_matches_dllm_masked_ce_recipe():
    logits, input_ids, labels, masked, p_mask_full, answer_len_full, b = _build_batch()

    ref = _dllm_recipe_loss(logits, input_ids, masked, p_mask_full, answer_len_full, b)
    got = dllm_masked_cross_entropy(
        logits,
        labels,
        loss_weight=1.0 / (p_mask_full * answer_len_full),
        num_samples=b,
        reduction="mean",
    )
    torch.testing.assert_close(got, ref)


def test_sum_vs_mean_reduction():
    logits, input_ids, labels, masked, p_mask_full, answer_len_full, b = _build_batch()
    lw = 1.0 / (p_mask_full * answer_len_full)

    s = dllm_masked_cross_entropy(logits, labels, lw, reduction="sum")
    m = dllm_masked_cross_entropy(logits, labels, lw, num_samples=b, reduction="mean")
    torch.testing.assert_close(m, s / b)


def test_ignore_positions_excluded():
    """Garbage logits at IGNORE positions must not change the loss."""
    logits, input_ids, labels, masked, p_mask_full, answer_len_full, b = _build_batch()
    lw = 1.0 / (p_mask_full * answer_len_full)

    base = dllm_masked_cross_entropy(logits, labels, lw, reduction="sum")

    corrupted = logits.clone()
    corrupted[~masked] += 1000.0  # only non-loss positions
    after = dllm_masked_cross_entropy(corrupted, labels, lw, reduction="sum")
    torch.testing.assert_close(base, after)


def test_weighting_scales_contribution():
    """Doubling every weight doubles the (linear) weighted sum."""
    logits, input_ids, labels, masked, p_mask_full, answer_len_full, b = _build_batch()
    lw = 1.0 / (p_mask_full * answer_len_full)

    s1 = dllm_masked_cross_entropy(logits, labels, lw, reduction="sum")
    s2 = dllm_masked_cross_entropy(logits, labels, 2.0 * lw, reduction="sum")
    torch.testing.assert_close(s2, 2.0 * s1)


def test_no_masked_token_returns_zero_with_grad():
    torch.manual_seed(1)
    logits = torch.randn(2, 4, 8, requires_grad=True)
    labels = torch.full((2, 4), IGNORE, dtype=torch.long)
    lw = torch.ones(2, 4)

    loss = dllm_masked_cross_entropy(logits, labels, lw, reduction="sum")
    assert loss.item() == 0.0
    loss.backward()  # graph must be intact
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0


def test_reexported_from_loss_utils_package():
    from veomni.models.loss_utils import dllm_masked_cross_entropy as reexported

    assert reexported is dllm_masked_cross_entropy
