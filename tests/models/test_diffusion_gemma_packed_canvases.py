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
"""Attributing a packed row's canvases to the samples they read.

The decoder half of packing samples. `flat_seq` already carries visibility as a
per-canvas interval, and an interval over a packed cache lands inside exactly
one sample -- so the attribution is derivable rather than declared, and the case
where it is not derivable is the leak the mode exists to refuse.

What is checked here is that derivation and the positions that come off it. The
positions are the part with no shape to protect it: a canvas must continue the
visible prefix inside its own sample. Numbering it from either the packed row or
the sample's full encoder length is rotated, runs, and returns a plausible number.
"""

import pytest
import torch

from veomni.models.transformers.diffusion_gemma.dg_hole import (
    canvas_owner_segments,
    canvas_visible_intervals,
    packed_canvas_positions,
)
from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import (
    IGNORE_INDEX,
    encoder_ar_targets,
)

from .test_diffusion_gemma_canvas_batching import _toy_model


CANVAS = 16

#: Ragged on both axes, which is the case packing exists for and the one where a
#: shared `cache_length` would be wrong for every sample but the last.
SPECS = ((6, 2), (10, 1))


def cu_seqlens_of(lengths):
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)


def runs(pairs):
    """``[(lo, hi), ...]`` as the ``(1, canvases)`` pair the intervals arrive in."""
    lo = torch.tensor([[pair[0] for pair in pairs]], dtype=torch.int32)
    hi = torch.tensor([[pair[1] for pair in pairs]], dtype=torch.int32)
    return lo, hi


class TestTheOwnerComesOffTheInterval:
    def test_each_canvas_is_attributed_to_the_sample_it_reads(self):
        lengths = [40, 24, 56]
        # Sample 0's canvases read inside [0, 40), sample 1's inside [40, 64),
        # sample 2's inside [64, 120).
        lo, hi = runs([(0, 8), (0, 24), (40, 48), (64, 70), (64, 120)])
        owner = canvas_owner_segments(lo, hi, cu_seqlens_of(lengths))
        assert owner.tolist() == [[0, 0, 1, 2, 2]]

    def test_a_run_filling_its_sample_exactly_stays_inside_it(self):
        """The boundary case both comparisons have to get right at once."""
        lo, hi = runs([(0, 40), (40, 64)])
        owner = canvas_owner_segments(lo, hi, cu_seqlens_of([40, 24]))
        assert owner.tolist() == [[0, 1]]

    def test_one_sample_in_the_row_is_the_unpacked_case(self):
        lo, hi = runs([(0, 8), (0, 24), (0, 40)])
        owner = canvas_owner_segments(lo, hi, cu_seqlens_of([40]))
        assert owner.tolist() == [[0, 0, 0]]

    def test_a_run_crossing_into_the_next_sample_is_refused(self):
        """The leak: a canvas reading the prompt, and later the answer, of another sample."""
        lo, hi = runs([(0, 8), (32, 48)])
        with pytest.raises(ValueError, match="read past the end of the sample"):
            canvas_owner_segments(lo, hi, cu_seqlens_of([40, 24]))

    def test_a_run_spanning_the_whole_row_is_refused(self):
        lo, hi = runs([(0, 64)])
        with pytest.raises(ValueError, match="read past the end of the sample"):
            canvas_owner_segments(lo, hi, cu_seqlens_of([40, 24]))

    def test_an_empty_run_is_refused_rather_than_attributed_to_the_first_sample(self):
        """`canvas_visible_intervals` returns `lo == hi == 0` for an empty canvas.

        Containment would put that at sample 0 whatever sample it belongs to, so
        the emptiness has to be caught before the attribution, not by it.
        """
        lo, hi = runs([(0, 8), (0, 0)])
        with pytest.raises(ValueError, match="see nothing of the encoder cache"):
            canvas_owner_segments(lo, hi, cu_seqlens_of([40, 24]))

    def test_the_refusals_name_which_canvases_failed(self):
        lo, hi = runs([(0, 8), (32, 48), (64, 100)])
        with pytest.raises(ValueError, match=r"canvases \[1, 2\]"):
            canvas_owner_segments(lo, hi, cu_seqlens_of([40, 24, 8]))

    @pytest.mark.parametrize("shape", [(3,), (2, 3), (1, 2, 3)])
    def test_a_shape_that_is_not_one_row_is_refused(self, shape):
        lo = torch.zeros(shape, dtype=torch.int32)
        hi = torch.ones(shape, dtype=torch.int32)
        with pytest.raises(ValueError, match="one row"):
            canvas_owner_segments(lo, hi, cu_seqlens_of([40]))


class TestThePositionsFollowTheVisiblePrefix:
    def test_a_canvas_sits_after_what_it_can_see_and_not_after_the_row(self):
        """Each canvas continues its visible prefix, independently of packing."""
        lengths = [40, 24]
        lo, hi = runs([(0, 8), (40, 48)])
        positions = packed_canvas_positions(lo, hi, cu_seqlens_of(lengths), CANVAS)

        assert positions.shape == (1, 2 * CANVAS)
        assert positions[0, :CANVAS].tolist() == list(range(8, 8 + CANVAS))
        assert positions[0, CANVAS:].tolist() == list(range(8, 8 + CANVAS))

    def test_later_canvases_continue_after_their_larger_visible_prefix(self):
        lo, hi = runs([(0, 8), (0, 24), (0, 40)])
        positions = packed_canvas_positions(lo, hi, cu_seqlens_of([40]), CANVAS)
        blocks = positions.reshape(3, CANVAS)
        assert blocks[0].tolist() == list(range(8, 8 + CANVAS))
        assert blocks[1].tolist() == list(range(24, 24 + CANVAS))
        assert blocks[2].tolist() == list(range(40, 40 + CANVAS))

    def test_one_sample_uses_the_visible_prefix_not_the_full_encoder_row(self):
        lengths = [56]
        lo, hi = runs([(0, 8), (0, 24)])
        packed = packed_canvas_positions(lo, hi, cu_seqlens_of(lengths), CANVAS)

        expected = torch.cat((torch.arange(8, 8 + CANVAS), torch.arange(24, 24 + CANVAS))).unsqueeze(0)
        assert torch.equal(packed, expected)

    def test_the_sudoku_shape_starts_the_decoder_at_the_prompt_end(self):
        """A 219-token prompt plus hidden 256-token answer starts at 219, not 475."""
        canvas_length = 256
        lo, hi = runs([(0, 219)])
        positions = packed_canvas_positions(lo, hi, cu_seqlens_of([219 + canvas_length]), canvas_length)

        assert positions[0, 0].item() == 219
        assert positions[0, -1].item() == 474

    def test_the_row_length_does_not_reach_the_positions(self):
        """Adding a sample behind must not move the samples in front of it."""
        lo, hi = runs([(0, 8), (40, 48)])
        two = packed_canvas_positions(lo, hi, cu_seqlens_of([40, 24]), CANVAS)
        three = packed_canvas_positions(lo, hi, cu_seqlens_of([40, 24, 96]), CANVAS)
        assert torch.equal(two, three)

    def test_positions_come_from_the_visible_run_within_the_owner(self):
        lo, hi = runs([(0, 8), (0, 40)])
        positions = packed_canvas_positions(lo, hi, cu_seqlens_of([40, 24]), CANVAS)
        assert positions[0, :CANVAS].tolist() == list(range(8, 8 + CANVAS))
        assert positions[0, CANVAS:].tolist() == list(range(40, 40 + CANVAS))

    def test_the_refusal_reaches_through_the_positions(self):
        lo, hi = runs([(0, 8), (32, 48)])
        with pytest.raises(ValueError, match="read past the end of the sample"):
            packed_canvas_positions(lo, hi, cu_seqlens_of([40, 24]), CANVAS)


class TestTheIntervalsAndTheAttributionComposeAsTheModelUsesThem:
    """`canvas_visible_intervals` feeds the attribution, so check the pair, not each."""

    @staticmethod
    def visibility_of(lengths, canvas_counts, canvas_length=CANVAS):
        """The per-canvas encoder mask a packed collator would emit.

        Sample ``i`` occupies ``[offset, offset + lengths[i])`` of the cache, and
        its canvas ``k`` sees that sample's prompt plus the ``k`` canvases before
        it -- the transform's per-block context, re-based onto the packed row.
        """
        cache_length = sum(lengths)
        rows = []
        offset = 0
        for length, canvases in zip(lengths, canvas_counts):
            prompt = length - canvases * canvas_length
            for block in range(canvases):
                row = torch.zeros(cache_length, dtype=torch.bool)
                row[offset : offset + prompt + block * canvas_length] = True
                rows.append(row)
            offset += length
        return torch.stack(rows).unsqueeze(0)

    def test_a_packed_collators_mask_attributes_every_canvas_to_its_own_sample(self):
        lengths = [8 + 3 * CANVAS, 20 + 2 * CANVAS]
        counts = [3, 2]
        lo, hi = canvas_visible_intervals(self.visibility_of(lengths, counts))
        owner = canvas_owner_segments(lo, hi, cu_seqlens_of(lengths))
        assert owner.tolist() == [[0, 0, 0, 1, 1]]

    def test_the_positions_off_that_mask_continue_each_visible_prefix(self):
        lengths = [8 + 3 * CANVAS, 20 + 2 * CANVAS]
        lo, hi = canvas_visible_intervals(self.visibility_of(lengths, [3, 2]))
        positions = packed_canvas_positions(lo, hi, cu_seqlens_of(lengths), CANVAS)

        blocks = positions.reshape(5, CANVAS)
        starts = [8, 8 + CANVAS, 8 + 2 * CANVAS, 20, 20 + CANVAS]
        for block, start in zip(blocks, starts):
            assert block.tolist() == list(range(start, start + CANVAS))

    def test_a_mask_letting_one_sample_read_the_next_is_refused(self):
        """A collator bug that shape checks cannot see, and the model must.

        The run has to stay contiguous to reach the attribution at all, so it is
        extended from sample 0's start through one token of sample 1's prompt --
        `canvas_visible_intervals` sees one run and has no objection. A hole
        instead would be caught before this, which is the test below.
        """
        lengths = [8 + CANVAS, 20 + CANVAS]
        visible = self.visibility_of(lengths, [1, 1])
        visible[0, 0, : lengths[0] + 1] = True
        with pytest.raises(ValueError, match="read past the end of the sample"):
            canvas_owner_segments(*canvas_visible_intervals(visible), cu_seqlens_of(lengths))

    def test_a_hole_is_still_caught_before_the_attribution_sees_it(self):
        """The two refusals are independent and the interval's comes first."""
        lengths = [8 + 2 * CANVAS, 20 + CANVAS]
        visible = self.visibility_of(lengths, [2, 1])
        visible[0, 1, 3] = False
        with pytest.raises(ValueError, match="not one contiguous run"):
            canvas_visible_intervals(visible)


#: The toy tokenizer's, and what `mask_other_tail_eos` cuts the run of.
EOS = 1


def sample_features(prompt, canvases, canvas_length, vocab_size, generator, eos_run=0):
    """One sample's encoder row, canvases, labels and per-block context.

    ``eos_run`` ends the encoder row with that many EOS, which is what the
    official last-canvas fill looks like. Giving the samples runs of *different*
    lengths is what makes the AR term's trailing-EOS cut visible: a row-wide cut
    finds one run and it belongs to the last sample.
    """
    length = prompt + canvases * canvas_length
    input_ids = torch.randint(3, vocab_size, (length,), generator=generator)
    if eos_run:
        input_ids[length - eos_run :] = EOS
    return {
        "input_ids": input_ids,
        "decoder_input_ids": torch.randint(3, vocab_size, (canvases, canvas_length), generator=generator),
        "labels": torch.randint(3, vocab_size, (canvases, canvas_length), generator=generator),
        "prompt": prompt,
        "canvases": canvases,
    }


def packed_and_separate(model, specs=SPECS, seed=0):
    """The same samples as one packed row, and as one row each.

    The two differ in nothing but packing: identical tokens, identical canvases,
    identical per-block context. The separate rows need no padding because they
    are run one at a time, so there is no padded-versus-packed confound either --
    the only thing that moves is whether the samples share a row.
    """
    canvas_length = model.config.canvas_length
    vocab_size = model.config.text_config.vocab_size
    generator = torch.Generator().manual_seed(seed)
    samples = [
        sample_features(p, k, canvas_length, vocab_size, generator, eos_run=index + 2)
        for index, (p, k) in enumerate(specs)
    ]

    total = sum(len(sample["input_ids"]) for sample in samples)
    packed = {
        "input_ids": [],
        "position_ids": [],
        "decoder_input_ids": [],
        "decoder_attention_mask": [],
        "labels": [],
        "loss_weight": [],
    }
    separate = []
    offset = 0
    for sample in samples:
        prompt, canvases = sample["prompt"], sample["canvases"]
        length = len(sample["input_ids"])
        weight = torch.full((canvases, canvas_length), 1.0 / (canvases * canvas_length), dtype=model.dtype)

        packed["input_ids"].append(sample["input_ids"])
        packed["position_ids"].append(torch.arange(length))
        packed["decoder_input_ids"].append(sample["decoder_input_ids"])
        packed["labels"].append(sample["labels"])
        packed["loss_weight"].append(weight)

        own, shared = [], []
        for block in range(canvases):
            visible = prompt + block * canvas_length
            row = torch.zeros(total, dtype=torch.long)
            row[offset : offset + visible] = 1
            shared.append(torch.cat([row, torch.ones(canvas_length, dtype=torch.long)]))
            alone = torch.zeros(length, dtype=torch.long)
            alone[:visible] = 1
            own.append(torch.cat([alone, torch.ones(canvas_length, dtype=torch.long)]))
        packed["decoder_attention_mask"].extend(shared)

        separate.append(
            {
                "input_ids": sample["input_ids"].unsqueeze(0),
                "attention_mask": torch.ones(1, length, dtype=torch.long),
                "decoder_input_ids": sample["decoder_input_ids"].unsqueeze(0),
                "decoder_attention_mask": torch.stack(own).unsqueeze(0),
                "labels": sample["labels"].unsqueeze(0),
                "loss_weight": weight.unsqueeze(0),
            }
        )
        offset += length

    batch = {
        "input_ids": torch.cat(packed["input_ids"]).unsqueeze(0),
        "position_ids": torch.cat(packed["position_ids"]).unsqueeze(0),
        "attention_mask": torch.ones(1, total, dtype=torch.long),
        "decoder_input_ids": torch.cat(packed["decoder_input_ids"]).unsqueeze(0),
        "decoder_attention_mask": torch.stack(packed["decoder_attention_mask"]).unsqueeze(0),
        "labels": torch.cat(packed["labels"]).unsqueeze(0),
        "loss_weight": torch.cat(packed["loss_weight"]).unsqueeze(0),
    }
    return batch, separate


def flat_seq_model(dtype=torch.float64, encoder_ar_loss=False):
    """A packing-capable model, with the AR term off unless a test wants it.

    Off by default so the canvas term can be checked on its own: it is per canvas
    already and comes out bit-identical, while the AR term is measured per sample
    from `cu_seqlens` and reduces in float32. Keeping them apart says which one a
    failure is in.
    """
    model = _toy_model(dtype=dtype)
    model.config.canvas_batching = "flat_seq"
    model.config.encoder_ar_loss = encoder_ar_loss
    return model


def canvas_logits(outputs, canvases, canvas_length):
    """The logits as ``(canvases, canvas_length, vocab)`` whichever way they arrive."""
    logits = outputs.logits
    return logits.reshape(canvases, canvas_length, logits.shape[-1])


class TestThePackedRowMatchesTheSamplesRunSeparately:
    """The equivalence the whole stage rests on, at float64 through the model.

    Constraint 33 explains why this is asserted in float64 and not bf16: the MoE
    router's `topk` is discontinuous, so packing's different GEMM row counts can
    reassign an expert off the last bit and move a token's output by O(1). float64
    puts that reassignment far enough away to see the arithmetic underneath.
    """

    def test_every_canvas_gets_the_logits_it_would_have_alone(self):
        """Bit-identical, not merely close, and asserted that way on purpose.

        There is no tolerance to argue about at float64: a canvas's queries, the
        keys it is allowed, and the positions it is rotated to are all the same
        tensors whether or not another sample shares the row, so the arithmetic
        is the same arithmetic. Anything at all here is a real difference --
        which is what makes this the test that would catch a wrong `cache_length`
        or a boundary the mask let through.
        """
        model = flat_seq_model()
        batch, separate = packed_and_separate(model)
        canvas_length = model.config.canvas_length

        with torch.no_grad():
            packed = canvas_logits(model(**batch, pack_samples=True), sum(k for _, k in SPECS), canvas_length)
            alone = torch.cat([canvas_logits(model(**one), k, canvas_length) for one, (_, k) in zip(separate, SPECS)])

        assert packed.shape == alone.shape
        assert torch.equal(packed, alone), f"packed and separate logits differ by {(packed - alone).abs().max():.3e}"

    def test_the_canvas_term_is_the_sum_of_the_samples_canvas_terms(self):
        """`loss_weight` normalizes per sample, so a packed row's loss adds them.

        This is the property the trainer's per-example reduction rests on: a
        packed row has to contribute what its samples would have contributed
        apart, or a token's gradient depends on who it was packed with.

        Not bit-identical, unlike the logits it is built from: the loss reduces in
        float32 whatever the model's dtype, and a packed row folds every canvas
        into one sum where the separate runs make one sum each and this adds
        them. Reduction order is the whole of the 3.4e-8 below.
        """
        model = flat_seq_model()
        batch, separate = packed_and_separate(model)

        with torch.no_grad():
            packed = float(model(**batch, pack_samples=True).diffusion_loss)
            apart = sum(float(model(**one).diffusion_loss) for one in separate)

        assert packed == pytest.approx(apart, rel=1e-6)

    def test_a_sample_cannot_see_the_sample_packed_in_front_of_it(self):
        """Perturb the first sample's prompt; the second sample must not move.

        The direction that matters. Sample 1 sits *after* sample 0 in the row, so
        a missing boundary check shows up as sample 1 reading sample 0's context --
        which under a plain causal mask is exactly what would happen.
        """
        model = flat_seq_model()
        batch, _ = packed_and_separate(model)
        first_canvases = SPECS[0][1]
        canvas_length = model.config.canvas_length
        total_canvases = sum(k for _, k in SPECS)

        with torch.no_grad():
            before = canvas_logits(model(**batch, pack_samples=True), total_canvases, canvas_length)
            disturbed = {key: value.clone() for key, value in batch.items()}
            disturbed["input_ids"][0, : SPECS[0][0]] += 1
            after = canvas_logits(model(**disturbed, pack_samples=True), total_canvases, canvas_length)

        moved = (before[:first_canvases] - after[:first_canvases]).abs().max()
        held = (before[first_canvases:] - after[first_canvases:]).abs().max()
        assert moved > 1e-6, "perturbing a sample's own prompt should move its canvases"
        assert held == 0.0, f"the sample behind moved by {held:.3e} and must not have moved at all"

    def test_a_sample_cannot_see_the_sample_packed_behind_it(self):
        """And the other direction, which a causal mask would give for free.

        Worth pinning anyway: the boundaries come from `position_ids`, and a
        derivation that mislabelled them could open this side too.
        """
        model = flat_seq_model()
        batch, _ = packed_and_separate(model)
        first_canvases = SPECS[0][1]
        canvas_length = model.config.canvas_length
        total_canvases = sum(k for _, k in SPECS)

        with torch.no_grad():
            before = canvas_logits(model(**batch, pack_samples=True), total_canvases, canvas_length)
            disturbed = {key: value.clone() for key, value in batch.items()}
            disturbed["input_ids"][0, SPECS[0][0] + SPECS[0][1] * canvas_length :] += 1
            after = canvas_logits(model(**disturbed, pack_samples=True), total_canvases, canvas_length)

        held = (before[:first_canvases] - after[:first_canvases]).abs().max()
        assert held == 0.0, f"the sample in front moved by {held:.3e} and must not have moved at all"

    def test_the_whole_loss_agrees_once_the_ar_term_is_segmented(self):
        """Both terms of Eq. 13, which is what the recipe actually trains on.

        The comparison that found the AR term wrong three ways, so it is also the
        one that says the rework is right. Kept beside the canvas-term test rather
        than replacing it: if this fails and that passes, the fault is in the AR
        term, and that localisation is the reason both exist.
        """
        model = flat_seq_model(encoder_ar_loss=True)
        batch, separate = packed_and_separate(model)

        with torch.no_grad():
            packed = model(**batch, pack_samples=True)
            apart = [model(**one) for one in separate]

        assert float(packed.encoder_ar_loss) == pytest.approx(
            sum(float(one.encoder_ar_loss) for one in apart), rel=1e-6
        )
        assert float(packed.loss) == pytest.approx(sum(float(one.loss) for one in apart), rel=1e-6)

    def test_the_ar_term_is_not_diluted_by_the_number_of_samples_packed(self):
        """The failure mode that started this: a factor of S, and no shape to see it.

        Agreement with the separate runs *is* the anti-dilution assertion, and at
        several counts rather than one because the factor is the count: a row-wide
        normalization returns `apart / S`, which misses by 2x and 3x here and
        cannot be confused with arithmetic noise at any tolerance. Growing the
        count is what distinguishes it from a constant offset.
        """
        model = flat_seq_model(encoder_ar_loss=True)
        for count in (1, 2, 3):
            batch, separate = packed_and_separate(model, specs=((6, 1),) * count)
            with torch.no_grad():
                packed = float(model(**batch, pack_samples=True).encoder_ar_loss)
                apart = sum(float(model(**one).encoder_ar_loss) for one in separate)
            assert packed == pytest.approx(apart, rel=1e-6), (
                f"{count} samples packed gave {packed:.6f} against {apart:.6f} apart, a factor of {apart / packed:.3f}"
            )

    def test_the_gradients_agree_with_the_separate_runs(self):
        """Forward agreement is not enough: the backward walks the packed row once.

        The hole kernel's `dkdv` gives one program sole ownership of a key tile,
        and packing is what puts two samples' keys in one row -- so a gradient
        that agrees here is evidence about the ownership and not only the mask.
        """
        model = flat_seq_model()
        batch, separate = packed_and_separate(model)

        model.zero_grad(set_to_none=True)
        model(**batch, pack_samples=True).loss.backward()
        packed = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

        model.zero_grad(set_to_none=True)
        for one in separate:
            model(**one).loss.backward()
        apart = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)

        assert packed.keys() == apart.keys()
        worst = max((packed[name] - apart[name]).abs().max() for name in packed)
        scale = max(apart[name].abs().max() for name in apart)
        assert worst / scale < 1e-6, f"gradients differ by {worst:.3e} against a scale of {scale:.3e}"


class TestThePackedPathRefusesWhatItCannotExpress:
    def test_a_mode_other_than_flat_seq_is_refused(self):
        model = flat_seq_model(dtype=torch.float32)
        batch, _ = packed_and_separate(model, seed=1)
        for mode in ("serial", "flat_batch"):
            model.config.canvas_batching = mode
            with pytest.raises(ValueError, match="needs `canvas_batching: flat_seq`"):
                model(**batch, pack_samples=True)

    def test_a_padded_row_is_refused_rather_than_packed(self):
        """The zero is the whole point: packing a padded row reads across the padding."""
        model = flat_seq_model(dtype=torch.float32)
        batch, _ = packed_and_separate(model, seed=2)
        batch["attention_mask"][0, 0] = 0
        with pytest.raises(ValueError, match="cannot carry padding"):
            model(**batch, pack_samples=True)

    def test_an_all_ones_mask_is_accepted_and_not_merely_tolerated(self):
        """The collator emits one, so it has to pass -- but be examined, not ignored."""
        model = flat_seq_model(dtype=torch.float32)
        batch, _ = packed_and_separate(model, seed=2)
        with torch.no_grad():
            with_mask = model(**batch, pack_samples=True).logits
            without = model(**{k: v for k, v in batch.items() if k != "attention_mask"}, pack_samples=True).logits
        assert torch.equal(with_mask, without)

    def test_missing_position_ids_are_refused(self):
        model = flat_seq_model(dtype=torch.float32)
        batch, _ = packed_and_separate(model, seed=2)
        del batch["position_ids"]
        with pytest.raises(ValueError, match="restart at 0 for each sample"):
            model(**batch, pack_samples=True)

    def test_supplying_decoder_positions_alongside_is_refused(self):
        """Two answers to where a canvas sits is one more than the computation has."""
        model = flat_seq_model(dtype=torch.float32)
        batch, _ = packed_and_separate(model, seed=2)
        canvases, canvas_length = batch["decoder_input_ids"].shape[1:]
        batch["decoder_position_ids"] = torch.zeros(1, canvases * canvas_length, dtype=torch.long)
        with pytest.raises(ValueError, match="cannot also be given `decoder_position_ids`"):
            model(**batch, pack_samples=True)

    def test_more_than_one_row_is_refused(self):
        model = flat_seq_model(dtype=torch.float32)
        batch, _ = packed_and_separate(model, seed=2)
        doubled = {key: torch.cat([value, value]) for key, value in batch.items()}
        with pytest.raises(ValueError, match="every sample in one row"):
            model(**doubled, pack_samples=True)

    def test_a_decode_against_a_prebuilt_cache_is_refused(self):
        model = flat_seq_model(dtype=torch.float32)
        batch, _ = packed_and_separate(model, seed=2)
        del batch["input_ids"]
        with pytest.raises(ValueError, match="needs the encoder row it is packing"):
            model(**batch, pack_samples=True)

    def test_a_per_row_ar_weight_is_refused_on_a_packed_row(self):
        """The axis it indexes is gone: one row, several examples.

        Silently broadcasting it would scale every sample in the row alike, which
        is the opposite of what the argument exists to say.
        """
        model = flat_seq_model(dtype=torch.float32, encoder_ar_loss=True)
        batch, _ = packed_and_separate(model, seed=2)
        batch["encoder_ar_row_weight"] = torch.ones(1)
        with pytest.raises(ValueError, match="scales the AR term per example"):
            model(**batch, pack_samples=True)

    def test_a_per_sample_ar_weight_is_accepted_and_zeroes_that_sample(self):
        model = flat_seq_model(dtype=torch.float32, encoder_ar_loss=True)
        batch, separate = packed_and_separate(model, seed=2)
        batch["encoder_ar_row_weight"] = torch.tensor([1.0, 0.0])

        with torch.no_grad():
            packed = float(model(**batch, pack_samples=True).encoder_ar_loss)
            first = float(model(**separate[0]).encoder_ar_loss)

        assert packed == pytest.approx(first, rel=1e-6)


class TestTheEncoderArTermIsMeasuredPerSample:
    """Eq. 13's second term, whose three per-row steps packing makes several wide.

    Checked against the unpacked function on the same tokens rather than against
    hand-written expectations: the property wanted is that a sample's targets and
    weights do not depend on who shares its row, and the unpacked call is exactly
    "nobody shares its row".
    """

    @staticmethod
    def rows_and_pack(lengths, eos=1, seed=0):
        """One row per sample, and the same tokens packed, with EOS-tailed fills."""
        generator = torch.Generator().manual_seed(seed)
        rows = []
        for index, length in enumerate(lengths):
            row = torch.randint(3, 50, (length,), generator=generator)
            # A trailing EOS run of a different length in each sample, which is
            # what a per-row tail cut gets wrong for every sample but the last.
            row[length - (index + 2) :] = eos
            rows.append(row)
        packed = torch.cat(rows).unsqueeze(0)
        cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
        return rows, packed, cu_seqlens

    @pytest.mark.parametrize("mask_other_tail_eos", [True, False])
    @pytest.mark.parametrize("lengths", [[12, 9], [9, 12, 7], [10], [6, 6, 6, 6]])
    def test_the_labels_and_weights_match_the_samples_run_alone(self, lengths, mask_other_tail_eos):
        rows, packed, cu_seqlens = self.rows_and_pack(lengths)
        labels, weights = encoder_ar_targets(packed, cu_seqlens=cu_seqlens, mask_other_tail_eos=mask_other_tail_eos)

        offset = 0
        for row in rows:
            alone_labels, alone_weights = encoder_ar_targets(row.unsqueeze(0), mask_other_tail_eos=mask_other_tail_eos)
            span = slice(offset, offset + len(row))
            assert torch.equal(labels[0, span], alone_labels[0]), f"labels differ for the sample at {offset}"
            assert torch.equal(weights[0, span], alone_weights[0]), f"weights differ for the sample at {offset}"
            offset += len(row)

    def test_one_segment_is_identical_to_passing_none(self):
        """The unpacked path must not change meaning to make room for this one."""
        (row,), packed, cu_seqlens = self.rows_and_pack([14])
        segmented = encoder_ar_targets(packed, cu_seqlens=cu_seqlens)
        plain = encoder_ar_targets(packed)
        assert torch.equal(segmented[0], plain[0])
        assert torch.equal(segmented[1], plain[1])

    def test_every_samples_first_position_is_dropped_not_only_the_rows(self):
        """Otherwise a sample's first token is predicted from the previous sample."""
        lengths = [12, 9, 7]
        _, packed, cu_seqlens = self.rows_and_pack(lengths)
        labels, weights = encoder_ar_targets(packed, cu_seqlens=cu_seqlens)

        for start in cu_seqlens[:-1].tolist():
            assert labels[0, start] == IGNORE_INDEX, f"position {start} starts a sample and cannot be a target"
            assert weights[0, start] == 0.0

    def test_the_weights_of_each_sample_sum_to_one(self):
        """Not to `1/S` each, which is what a row-wide count would give."""
        lengths = [12, 9, 7]
        _, packed, cu_seqlens = self.rows_and_pack(lengths)
        _, weights = encoder_ar_targets(packed, cu_seqlens=cu_seqlens)

        bounds = cu_seqlens.tolist()
        for start, end in zip(bounds[:-1], bounds[1:]):
            assert float(weights[0, start:end].sum()) == pytest.approx(1.0, abs=1e-6)

    def test_each_samples_tail_eos_run_is_cut_and_not_only_the_last(self):
        """The row's `flip().cummin()` sees one run, and it is the last sample's."""
        lengths = [12, 9]
        _, packed, cu_seqlens = self.rows_and_pack(lengths)
        segmented, _ = encoder_ar_targets(packed, cu_seqlens=cu_seqlens)
        per_row, _ = encoder_ar_targets(packed)

        # The first sample's fill: cut by the segmented pass, missed by the other.
        tail = slice(cu_seqlens[1].item() - 1, cu_seqlens[1].item())
        assert (segmented[0, tail] == IGNORE_INDEX).all()
        assert (per_row[0, tail] != IGNORE_INDEX).all()

    def test_the_first_eos_of_each_samples_run_stays_the_stop_target(self):
        lengths = [12, 9]
        rows, packed, cu_seqlens = self.rows_and_pack(lengths)
        labels, _ = encoder_ar_targets(packed, cu_seqlens=cu_seqlens)

        offset = 0
        for index, row in enumerate(rows):
            run_start = len(row) - (index + 2)
            assert labels[0, offset + run_start] == 1, "the first EOS of a sample's fill is its stop signal"
            offset += len(row)

    def test_more_than_one_row_is_refused(self):
        rows, _, cu_seqlens = self.rows_and_pack([8, 8])
        stacked = torch.stack(rows)
        with pytest.raises(ValueError, match="a packed row is one row"):
            encoder_ar_targets(stacked, cu_seqlens=cu_seqlens)

    @staticmethod
    def agrees_sample_by_sample(rows, prompts=None, types=None, **kwargs):
        """Whether the packed targets restrict to what each sample gets alone."""
        packed = torch.cat(rows).unsqueeze(0)
        lengths = [len(row) for row in rows]
        cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
        extra = {}
        if prompts is not None:
            extra["encoder_prompt_mask"] = torch.cat(prompts).unsqueeze(0)
        if types is not None:
            extra["mm_token_type_ids"] = torch.cat(types).unsqueeze(0)
        labels, weights = encoder_ar_targets(packed, cu_seqlens=cu_seqlens, **kwargs, **extra)

        offset = 0
        for index, row in enumerate(rows):
            alone_extra = {}
            if prompts is not None:
                alone_extra["encoder_prompt_mask"] = prompts[index].unsqueeze(0)
            if types is not None:
                alone_extra["mm_token_type_ids"] = types[index].unsqueeze(0)
            alone = encoder_ar_targets(row.unsqueeze(0), **kwargs, **alone_extra)
            span = slice(offset, offset + len(row))
            if not torch.equal(labels[0, span], alone[0][0]) or not torch.equal(weights[0, span], alone[1][0]):
                return False
            offset += len(row)
        return True

    @pytest.mark.parametrize(
        "name,rows",
        [
            ("a sample that is entirely EOS", [[EOS, EOS, EOS, EOS], [5, 6, 7, EOS, EOS]]),
            ("a sample of length one", [[9], [5, 6, 7, EOS, EOS]]),
            ("a sample of length one that is EOS", [[EOS], [5, 6, EOS]]),
            ("every sample entirely EOS", [[EOS, EOS], [EOS, EOS, EOS]]),
            ("no EOS anywhere", [[4, 5, 6], [7, 8]]),
            ("EOS in the middle and not the tail", [[4, EOS, 6], [7, EOS, 8, 9]]),
        ],
    )
    def test_the_degenerate_samples_agree_too(self, name, rows):
        """The cases the mutation check cannot reach, since they have no interior.

        A sample with no non-EOS position, or none at all after its first, drives
        the suffix and prefix sums to their boundaries -- where an off-by-one lives
        if there is one, and where reverting a fix looks the same as keeping it.
        """
        assert self.agrees_sample_by_sample([torch.tensor(row) for row in rows]), name

    def test_the_other_cuts_are_applied_per_sample_as_well(self):
        """`mask_prompt_ar_loss` and the image cut are elementwise, but sit upstream
        of the count -- so a sample whose prompt swallows it changes its neighbours'
        weights if the normalization is not segmented."""
        rows = [torch.tensor([4, 5, 6, EOS]), torch.tensor([7, 8, 9, EOS, EOS])]
        assert self.agrees_sample_by_sample(
            rows,
            prompts=[torch.tensor([1, 1, 0, 0]).bool(), torch.tensor([1, 0, 0, 0, 0]).bool()],
            mask_prompt_ar_loss=True,
        )
        assert self.agrees_sample_by_sample(
            [torch.tensor([4, 5, 6]), torch.tensor([7, 8, EOS])],
            prompts=[torch.tensor([1, 1, 1]).bool(), torch.tensor([1, 0, 0]).bool()],
            mask_prompt_ar_loss=True,
        ), "a sample whose prompt is the whole sample"
        assert self.agrees_sample_by_sample(rows, types=[torch.tensor([0, 1, 1, 0]), torch.tensor([0, 0, 1, 0, 0])])
