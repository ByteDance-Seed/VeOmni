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

"""Regression tests for the DPO packed-batch label alignment helper.

DPO packs each preference pair as two adjacent segments ``[chosen | rejected]``
in the packed sequence. When SP is enabled, ``SequenceParallelCollator`` applies
a single global left-shift to the packed labels — so slicing the shifted tensor
by ``seq_lens`` leaves each segment's tail position holding the *next*
segment's head token (chosen tail = rejected head). Without a boundary mask,
that cross-segment token would leak into the chosen / rejected log-prob sums.

These tests pin the segment-aware ``IGNORE_INDEX`` boundary masking so a
regression can never silently reintroduce that leak.
"""

import torch

from veomni.trainer.text_dpo_trainer import _build_dpo_labels_list
from veomni.utils.constants import IGNORE_INDEX


def _global_shift(labels: torch.Tensor) -> torch.Tensor:
    """Mirror ``SequenceParallelCollator``'s global left-shift + IGN pad."""
    shifted = labels[..., 1:].clone()
    tail = torch.full_like(labels[..., :1], IGNORE_INDEX)
    return torch.cat([shifted, tail], dim=-1)


def test_sp_on_masks_segment_boundary():
    """SP-on: chosen tail must be IGNORE_INDEX, not rejected's head token.

    Constructs a packed batch of two preference pairs (4 segments total) with
    distinct value ranges per segment so a leak from segment ``i``'s tail into
    segment ``i+1``'s head is visually detectable. Applies the global shift
    (as SequenceParallelCollator would), then asserts the helper masks every
    segment's trailing slot.
    """
    # Segment token ranges chosen to be disjoint: 10-19 chosen0, 20-29 rejected0,
    # 30-39 chosen1, 40-49 rejected1. Under a global left-shift, chosen0's tail
    # (before masking) picks up rejected0's head token = 20.
    labels = torch.tensor(
        [10, 11, 12, 13, 20, 21, 22, 30, 31, 32, 33, 34, 40, 41, 42, 43],
        dtype=torch.long,
    )
    seq_lens = [4, 3, 5, 4]
    shifted = _global_shift(labels)

    labels_list = _build_dpo_labels_list(shifted, seq_lens, sp_enabled=True)

    assert len(labels_list) == 4
    for i, (seg, sl) in enumerate(zip(labels_list, seq_lens)):
        assert seg.shape == (sl,), f"segment {i}: expected len {sl}, got {seg.shape}"
        assert int(seg[-1].item()) == IGNORE_INDEX, (
            f"segment {i}: trailing slot must be IGNORE_INDEX after boundary mask; got {seg[-1].item()}"
        )

    # Sanity: interior positions still carry the shifted (i.e., next-token)
    # labels — this is the alignment the kernel's log_probs expects.
    # chosen0 interior (positions 0..sl-2) should equal shifted[:sl-1].
    for i, (seg, sl) in enumerate(zip(labels_list, seq_lens)):
        offset = sum(seq_lens[:i])
        expected_interior = shifted[offset : offset + sl - 1]
        assert torch.equal(seg[:-1], expected_interior), f"segment {i}: interior labels corrupted"


def test_sp_off_applies_per_segment_shift():
    """SP-off: helper does per-segment causal shift + IGN pad (unchanged behavior).

    With SP disabled the collator does not shift; the trainer must apply the
    causal shift per segment (dropping index 0, appending IGNORE_INDEX). Assert
    equivalence to the pre-fix reference construction.
    """
    labels = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=torch.long)
    seq_lens = [4, 3, 5]

    labels_list = _build_dpo_labels_list(labels, seq_lens, sp_enabled=False)

    expected = [
        torch.tensor([2, 3, 4, IGNORE_INDEX], dtype=torch.long),
        torch.tensor([6, 7, IGNORE_INDEX], dtype=torch.long),
        torch.tensor([9, 10, 11, 12, IGNORE_INDEX], dtype=torch.long),
    ]
    for i, (seg, exp) in enumerate(zip(labels_list, expected)):
        assert torch.equal(seg, exp), f"segment {i}: sp-off mismatch — got {seg.tolist()}, want {exp.tolist()}"


def test_sp_on_boundary_mask_prevents_cross_segment_leak_in_logp_sum():
    """End-to-end: masking-vs-not-masking changes the chosen logsum by exactly
    the boundary log_prob.

    Simulates the trainer's ``loss_mask * per_token_logps`` reduction on a
    packed pair (chosen + rejected) with a plausible per-token log_probs
    tensor. Compares:
      * ``naive_split`` — the pre-fix behavior (``all_labels.split(seq_lens)``),
        which keeps the cross-segment token in ``loss_mask``.
      * ``_build_dpo_labels_list`` (fix) — masks the boundary slot.

    Chosen's tail under the naive path holds rejected's head token (still a
    valid label, not IGNORE_INDEX), so the fix drops exactly one term from the
    chosen sum. Rejected's tail is already IGNORE_INDEX from the packed-batch
    trailing pad, so the fix leaves rejected's sum unchanged — that invariant
    is also asserted below.
    """
    torch.manual_seed(0)
    seq_lens = [5, 4]  # chosen=5, rejected=4
    packed_L = sum(seq_lens)

    labels = torch.arange(1, packed_L + 1, dtype=torch.long)
    shifted = _global_shift(labels)  # simulates SequenceParallelCollator
    per_token_logps = torch.randn(packed_L, dtype=torch.float32)

    fixed = _build_dpo_labels_list(shifted, seq_lens, sp_enabled=True)
    fixed_chosen = (per_token_logps[: seq_lens[0]] * (fixed[0] != IGNORE_INDEX).float()).sum()
    fixed_rejected = (per_token_logps[seq_lens[0] :] * (fixed[1] != IGNORE_INDEX).float()).sum()

    naive_segs = list(shifted.split(seq_lens))
    naive_chosen = (per_token_logps[: seq_lens[0]] * (naive_segs[0] != IGNORE_INDEX).float()).sum()
    naive_rejected = (per_token_logps[seq_lens[0] :] * (naive_segs[1] != IGNORE_INDEX).float()).sum()

    # Chosen: the fix drops exactly the boundary slot (index seq_lens[0] - 1).
    assert not torch.equal(fixed_chosen, naive_chosen), "boundary mask must change chosen's log-prob sum"
    assert torch.allclose(
        naive_chosen - fixed_chosen,
        per_token_logps[seq_lens[0] - 1],
        atol=1e-6,
    ), "chosen delta must equal exactly the boundary-slot log_prob"

    # Rejected: last packed position is already IGNORE_INDEX (trailing pad from
    # the global shift), so naive and fixed sums are identical.
    assert torch.equal(fixed_rejected, naive_rejected), "rejected sum should be unchanged by the boundary mask"
