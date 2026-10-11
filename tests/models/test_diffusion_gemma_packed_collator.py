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
"""Where the packed row actually comes from: the collator and the mode that picks it.

The model-side equivalence is pinned in `test_diffusion_gemma_packed_canvases.py`
against a packed batch built by hand. That leaves the gap this file closes -- the
hand-built batch agreeing with the model says nothing about the collator agreeing
with the hand-built batch, and the collator is what training runs on.

So the equivalence here is run through the real collator on both sides: the same
features packed into one row, and the same features collated one at a time. And
because `canvas_batching: flat_seq_pack` is what turns packing on, the mode is
checked to reach *both* halves that branch on it -- the attention and the encoder
AR term -- since a mode that reached only the attention would train on a quietly
diluted objective.
"""

from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import veomni.trainer.diffusion_gemma_trainer as dg_trainer
from veomni.data.data_collator import DiffusionGemmaCollator
from veomni.models.transformers.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaConfig
from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import expand_self_conditioning_mask
from veomni.trainer.diffusion_gemma_trainer import DiffusionGemmaTrainer, diffusion_gemma_sample_cost

from .test_diffusion_gemma_canvas_batching import _toy_model


#: Ragged on both axes -- prompt length and canvas count -- which is the case
#: packing exists for and the one where padding to the batch's worst case costs
#: the most.
SPECS = ((6, 2), (11, 1), (4, 2))

EOS = 1


def collator_features(specs, canvas_length, vocab_size, seed=0, dtype=torch.float64):
    """The per-sample dicts the data transform hands the collator.

    Mirrors the transform's contract rather than inventing one: `input_ids` is the
    prompt followed by every canvas, `decoder_context_mask` has one row per canvas
    holding the prompt plus the canvases before it, and `loss_weight` normalizes
    within the sample so a sample contributes the same whatever it is batched with.
    """
    generator = torch.Generator().manual_seed(seed)
    features = []
    for index, (prompt, canvases) in enumerate(specs):
        length = prompt + canvases * canvas_length
        input_ids = torch.randint(3, vocab_size, (length,), generator=generator)
        # Runs of different lengths, so a row-wide trailing-EOS cut would find one
        # run and give it to the last sample -- see the AR term's segmented cut.
        input_ids[length - (index + 2) :] = EOS
        context = torch.zeros(canvases, length, dtype=torch.long)
        for block in range(canvases):
            context[block, : prompt + block * canvas_length] = 1
        prompt_mask = torch.zeros(length, dtype=torch.bool)
        prompt_mask[:prompt] = True
        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones(length, dtype=torch.long),
                "encoder_prompt_mask": prompt_mask,
                "decoder_context_mask": context,
                "decoder_input_ids": torch.randint(3, vocab_size, (canvases, canvas_length), generator=generator),
                "labels": torch.randint(3, vocab_size, (canvases, canvas_length), generator=generator),
                "loss_weight": torch.full((canvases, canvas_length), 1.0 / (canvases * canvas_length), dtype=dtype),
            }
        )
    return features


def features_for(model, specs=SPECS, seed=0):
    return collator_features(
        specs, model.config.canvas_length, model.config.text_config.vocab_size, seed=seed, dtype=model.dtype
    )


def packed_and_one_at_a_time(model, specs=SPECS, seed=0):
    """The same features through the collator packed, and through it singly.

    The single batches need no padding -- one feature is already its own longest --
    so the only difference between the two arms is whether the samples share a row.
    """
    features = features_for(model, specs, seed)
    packed = DiffusionGemmaCollator(pad_token_id=0, pack_samples=True)(features)
    alone = [DiffusionGemmaCollator(pad_token_id=0)([feature]) for feature in features]
    return packed, alone


@contextmanager
def unpacked(model):
    """Run the single-sample arm in the mode that matches its batch.

    A batch of one sample is not a packed batch: the padded collator emits no
    `position_ids`, and the packed mode refuses a row whose boundaries it cannot
    read rather than assuming the row is one sample. So the comparison arm has to
    say which mode it is, and the model has to be asked twice.
    """
    previous = model.config.canvas_batching
    model.config.canvas_batching = "flat_seq"
    try:
        yield model
    finally:
        model.config.canvas_batching = previous


def packing_model(encoder_ar_loss):
    model = _toy_model(dtype=torch.float64)
    model.config.canvas_batching = "flat_seq_pack"
    model.config.encoder_ar_loss = encoder_ar_loss
    return model


def canvas_logits(outputs, canvases, canvas_length):
    logits = outputs.logits
    return logits.reshape(canvases, canvas_length, logits.shape[-1])


def model_input(batch):
    """`num_supervised_examples` is the trainer's, not the model's."""
    return {key: value for key, value in batch.items() if key != "num_supervised_examples"}


class TestThePackedBatchPadsNothing:
    def test_the_encoder_rows_are_concatenated_into_one(self):
        features = collator_features(SPECS, canvas_length=8, vocab_size=32)
        batch = DiffusionGemmaCollator(pack_samples=True)(features)
        lengths = [len(feature["input_ids"]) for feature in features]

        assert batch["input_ids"].shape == (1, sum(lengths))
        assert torch.equal(batch["input_ids"][0], torch.cat([feature["input_ids"] for feature in features]))

    def test_the_boundaries_arrive_as_position_restarts(self):
        """The convention the rest of the stack reads, and the only record of them.

        Nothing else in the packed batch says where one sample ends: the encoder
        row is a plain concatenation and the mask is all ones. Lose the restarts
        and the model sees one long sample, with no shape to object to.
        """
        features = collator_features(SPECS, canvas_length=8, vocab_size=32)
        batch = DiffusionGemmaCollator(pack_samples=True)(features)
        lengths = [len(feature["input_ids"]) for feature in features]

        expected = torch.cat([torch.arange(length) for length in lengths])
        assert torch.equal(batch["position_ids"][0], expected)
        assert (batch["position_ids"][0] == 0).sum() == len(features)

    def test_no_row_is_padded_so_the_mask_is_all_ones(self):
        features = collator_features(SPECS, canvas_length=8, vocab_size=32)
        batch = DiffusionGemmaCollator(pack_samples=True)(features)

        assert batch["attention_mask"].shape == batch["input_ids"].shape
        assert bool(batch["attention_mask"].all())

    def test_the_canvas_counts_are_not_padded_to_the_largest(self):
        """Where the padded path spends the most on ragged data.

        Padding a mixed-K batch to max K adds dummy canvases with zero weight:
        real decoder passes over tokens that contribute nothing. Packing puts
        every sample's canvases on the axis and stops.
        """
        features = collator_features(SPECS, canvas_length=8, vocab_size=32)
        batch = DiffusionGemmaCollator(pack_samples=True)(features)
        total = sum(k for _, k in SPECS)

        assert batch["decoder_input_ids"].shape == (1, total, 8)
        assert batch["labels"].shape == (1, total, 8)
        assert batch["loss_weight"].shape == (1, total, 8)
        assert float(batch["loss_weight"].sum()) == pytest.approx(len(SPECS))

    def test_a_single_sample_still_comes_out_as_a_packed_row(self):
        """The K==1 batch the padded path squeezes to two dimensions.

        Packing must not squeeze: the decoder's packed path reads the canvas axis
        to attribute canvases to samples, and a two-dimensional batch has none.
        """
        features = collator_features(((6, 1),), canvas_length=8, vocab_size=32)
        batch = DiffusionGemmaCollator(pack_samples=True)(features)

        assert batch["decoder_input_ids"].ndim == 3
        assert batch["decoder_input_ids"].shape[:2] == (1, 1)

    def test_the_supervised_count_still_counts_examples(self):
        """What the trainer divides by, and packing must not change it.

        Count rows instead and a packed row is one example however many samples
        it holds, which scales the effective learning rate by the packing factor.
        """
        features = collator_features(SPECS, canvas_length=8, vocab_size=32)
        features[1]["loss_weight"] = torch.zeros_like(features[1]["loss_weight"])
        batch = DiffusionGemmaCollator(pack_samples=True)(features)

        assert int(batch["num_supervised_examples"]) == len(SPECS) - 1


class TestThePackedMaskKeepsTheSamplesApart:
    """The encoder half of `decoder_attention_mask` is what attributes a canvas.

    It is also the only thing stopping a canvas reading a neighbour's prompt, so
    these two jobs are checked as one: the visible run has to be inside the
    canvas's own sample and be exactly the context that sample gave it.
    """

    @staticmethod
    def offsets_of(features):
        bounds, start = [], 0
        for feature in features:
            length = len(feature["input_ids"])
            bounds.append((start, start + length))
            start += length
        return bounds

    def test_each_canvas_sees_only_inside_its_own_sample(self):
        features = collator_features(SPECS, canvas_length=8, vocab_size=32)
        batch = DiffusionGemmaCollator(pack_samples=True)(features)
        total = int(batch["input_ids"].shape[1])
        encoder_half = batch["decoder_attention_mask"][0, :, :total]

        row = 0
        for (start, end), (_prompt, canvases) in zip(self.offsets_of(features), SPECS):
            for _ in range(canvases):
                outside = torch.cat([encoder_half[row, :start], encoder_half[row, end:]])
                assert not bool(outside.any()), f"canvas {row} can see outside its sample"
                row += 1

    def test_the_visible_run_is_the_samples_own_per_block_context(self):
        canvas_length = 8
        features = collator_features(SPECS, canvas_length=canvas_length, vocab_size=32)
        batch = DiffusionGemmaCollator(pack_samples=True)(features)
        total = int(batch["input_ids"].shape[1])
        encoder_half = batch["decoder_attention_mask"][0, :, :total]

        row = 0
        for (start, _end), (prompt, canvases) in zip(self.offsets_of(features), SPECS):
            for block in range(canvases):
                visible = encoder_half[row].nonzero().flatten().tolist()
                assert visible == list(range(start, start + prompt + block * canvas_length))
                row += 1

    def test_the_canvas_half_stays_visible(self):
        """A canvas reads itself; only the cache side is per sample."""
        features = collator_features(SPECS, canvas_length=8, vocab_size=32)
        batch = DiffusionGemmaCollator(pack_samples=True)(features)
        total = int(batch["input_ids"].shape[1])

        assert bool(batch["decoder_attention_mask"][0, :, total:].all())


class TestThePackedCollatorAgreesWithCollatingOneAtATime:
    """End to end through the real collator, at float64.

    Constraint 33 covers why float64: the MoE router's `topk` is discontinuous,
    so a different GEMM row count can reassign an expert off the last bit and
    move a token's output by O(1). float64 leaves the arithmetic visible.
    """

    def test_every_canvas_gets_the_logits_it_would_have_alone(self):
        model = packing_model(encoder_ar_loss=False)
        packed, alone = packed_and_one_at_a_time(model)
        canvas_length = model.config.canvas_length

        with torch.no_grad():
            got = canvas_logits(model(**model_input(packed)), sum(k for _, k in SPECS), canvas_length)
            with unpacked(model):
                want = torch.cat(
                    [canvas_logits(model(**model_input(one)), k, canvas_length) for one, (_, k) in zip(alone, SPECS)]
                )

        assert got.shape == want.shape
        assert torch.equal(got, want), f"collated packed and singly differ by {(got - want).abs().max():.3e}"

    def test_the_whole_loss_is_the_sum_of_the_samples_losses(self):
        """Both terms of Eq. 13, from the collator, with the AR term on.

        Approximate rather than exact because the losses reduce in float32
        whatever the model's dtype, and a packed row folds every canvas into one
        sum where the separate runs make one sum each and this adds them.
        """
        model = packing_model(encoder_ar_loss=True)
        packed, alone = packed_and_one_at_a_time(model)

        with torch.no_grad():
            got = model(**model_input(packed))
            with unpacked(model):
                want = [model(**model_input(one)) for one in alone]

        assert float(got.diffusion_loss) == pytest.approx(sum(float(one.diffusion_loss) for one in want), rel=1e-6)
        assert float(got.encoder_ar_loss) == pytest.approx(sum(float(one.encoder_ar_loss) for one in want), rel=1e-6)
        assert float(got.loss) == pytest.approx(sum(float(one.loss) for one in want), rel=1e-6)

    def test_the_gradients_agree_with_the_samples_run_singly(self):
        model = packing_model(encoder_ar_loss=True)
        packed, alone = packed_and_one_at_a_time(model)

        model.zero_grad()
        model(**model_input(packed)).loss.backward()
        got = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

        model.zero_grad()
        with unpacked(model):
            for one in alone:
                model(**model_input(one)).loss.backward()
        want = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

        assert got.keys() == want.keys()
        assert got, "no parameter received a gradient, so this compares nothing"
        # Each tensor against its own norm rather than element by element. An
        # embedding gradient is a sparse sum, so an entry near zero carries a
        # large *relative* error from float32 reduction order alone and says
        # nothing about the packing; the losses reduce in float32 whatever the
        # model's dtype. Measured spread is ~3e-7, which is float32 eps. What
        # this guards against is a scale error -- an AR term normalized over the
        # row divides each sample's share by the number packed with it, a third
        # of the gradient here -- so the margin is five orders of magnitude.
        worst, culprit = max(
            ((got[name] - want[name]).norm() / want[name].norm().clamp(min=1e-30), name) for name in got
        )
        assert float(worst) < 1e-6, f"{culprit}: gradients differ by {float(worst):.3e} of the gradient's norm"


class TestTheModeIsTheOnlySwitch:
    """`flat_seq_pack` has to reach every place that branches on packing.

    Two do, in two classes: the attention segments the encoder row, and the
    encoder AR term measures itself per sample. A mode that reached only the
    first would train on an objective divided by the number of samples packed --
    a changed loss with no changed shape and nothing in the logs.
    """

    def test_the_mode_packs_without_being_asked_a_second_time(self):
        model = packing_model(encoder_ar_loss=False)
        packed, _ = packed_and_one_at_a_time(model)
        canvas_length = model.config.canvas_length
        total = sum(k for _, k in SPECS)

        with torch.no_grad():
            # No `pack_samples=True` anywhere: the config is the switch.
            asked = canvas_logits(model(**model_input(packed), pack_samples=True), total, canvas_length)
            unasked = canvas_logits(model(**model_input(packed)), total, canvas_length)

        assert torch.equal(asked, unasked)

    def test_the_mode_reaches_the_ar_term_and_not_only_the_attention(self):
        """The dilution this mode would otherwise ship.

        Left unsegmented, the AR weights normalize over the row, so each sample's
        share is divided by the number of samples -- roughly a third here. The
        equivalence below is what catches that, and it catches it at the size of
        the packing factor rather than at a tolerance.
        """
        model = packing_model(encoder_ar_loss=True)
        packed, alone = packed_and_one_at_a_time(model)

        with torch.no_grad():
            got = float(model(**model_input(packed)).encoder_ar_loss)
            with unpacked(model):
                want = sum(float(model(**model_input(one)).encoder_ar_loss) for one in alone)

        assert got > 0.0
        assert got == pytest.approx(want, rel=1e-6)

    def test_the_mode_decodes_through_the_flat_seq_path(self):
        """`flat_seq_pack` differs from `flat_seq` in the batch, not the decoder."""
        from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import _canvas_mode

        assert _canvas_mode(SimpleNamespace(canvas_batching="flat_seq_pack")) == "flat_seq"
        assert _canvas_mode(SimpleNamespace(canvas_batching="flat_seq")) == "flat_seq"
        assert _canvas_mode(SimpleNamespace(canvas_batching="serial")) == "serial"

    @pytest.mark.parametrize("mode", ["serial", "flat_batch"])
    def test_packing_a_mode_that_cannot_express_it_is_refused(self, mode):
        model = _toy_model(dtype=torch.float64)
        model.config.canvas_batching = mode
        packed, _ = packed_and_one_at_a_time(model)

        with pytest.raises(ValueError, match="flat_seq"):
            model(**model_input(packed), pack_samples=True)


class TestThePackedCollatorRefusesWhatItCannotPack:
    @pytest.mark.parametrize("key", ["mm_token_type_ids", "pixel_values", "image_position_ids"])
    def test_image_features_are_refused_rather_than_dropped(self, key):
        """Silence here would attend an image span as text.

        The packed encoder runs varlen kernels that take `cu_seqlens` and nothing
        else, so there is nowhere to put an image block boundary. Dropping the
        key would leave a batch that runs and trains on the wrong attention.
        """
        features = collator_features(SPECS, canvas_length=8, vocab_size=32)
        features[0][key] = torch.zeros(len(features[0]["input_ids"]), dtype=torch.long)

        with pytest.raises(ValueError, match=f"cannot pack `{key}`"):
            DiffusionGemmaCollator(pack_samples=True)(features)

    def test_an_empty_batch_is_still_refused(self):
        with pytest.raises(ValueError, match="empty feature list"):
            DiffusionGemmaCollator(pack_samples=True)([])

    @pytest.mark.parametrize(
        ("key", "broken", "match"),
        [
            ("encoder_prompt_mask", lambda f: torch.ones(3, dtype=torch.bool), "same length as `input_ids`"),
            ("decoder_context_mask", lambda f: torch.ones(2, 3, dtype=torch.long), "same length as `input_ids`"),
            ("labels", lambda f: torch.zeros(1, 8, dtype=torch.long), "must match `decoder_input_ids`"),
        ],
    )
    def test_the_shape_checks_apply_to_the_packed_path_too(self, key, broken, match):
        """Both paths share one set of checks, so neither can be the loose one."""
        features = collator_features(SPECS, canvas_length=8, vocab_size=32)
        features[0][key] = broken(features[0])

        with pytest.raises(ValueError, match=match):
            DiffusionGemmaCollator(pack_samples=True)(features)


class TestTheStepDoesNotPadTheCanvasAxisUnderPacking:
    """The step's K alignment and packing cannot both hold, and packing wins.

    `preforward` pads every rank's canvas axis up to the step's global max K so
    that `serial`'s per-canvas collectives stay in lockstep. The dummy slots it
    adds are given an all-visible mask, which under packing reads the whole row
    -- and a canvas reading past its own sample is the leak this mode exists to
    refuse. The alignment is what gives way: with the decoder running one pass,
    no collective count depends on K.
    """

    @staticmethod
    def trainer(model):
        """The real config off the model, so `canvas_batching` is where production has it."""
        trainer = SimpleNamespace(base=SimpleNamespace(model=SimpleNamespace(model_config=model.config)))
        for name in (
            "_align_all_canvas_batch",
            "_self_conditioning_sample_count",
            "_noisy_token_ids",
            "_decoder_sample_count",
        ):
            setattr(trainer, name, MethodType(getattr(DiffusionGemmaTrainer, name), trainer))
        return trainer

    @staticmethod
    @contextmanager
    def another_rank_drew_more_canvases(extra=2):
        """Stand in for the rank that packed more, which is the only reason to align."""
        with (
            patch.object(dg_trainer.dist, "is_initialized", return_value=True),
            patch.object(dg_trainer, "all_reduce", side_effect=lambda value, op: value + extra),
        ):
            yield

    def align(self, model, batch, extra=2):
        with self.another_rank_drew_more_canvases(extra):
            return self.trainer(model)._align_all_canvas_batch(batch)

    def test_the_unpacked_modes_still_pad(self):
        """Control: the stand-in above really does reach the padding.

        Without it the packed assertion below would pass on a step that never
        tried to pad, and would keep passing if the alignment were deleted.
        """
        model = _toy_model(dtype=torch.float64)
        with unpacked(model):
            batch = DiffusionGemmaCollator(pad_token_id=0)(features_for(model))
            before = batch["decoder_input_ids"].shape[1]
            after = self.align(model, batch)
        assert after["decoder_input_ids"].shape[1] == before + 2

    def test_the_packed_row_keeps_the_canvases_it_packed(self):
        model = packing_model(encoder_ar_loss=False)
        packed = DiffusionGemmaCollator(pad_token_id=0, pack_samples=True)(features_for(model))
        before = packed["decoder_input_ids"].shape[1]
        assert self.align(model, packed)["decoder_input_ids"].shape[1] == before

    @pytest.mark.parametrize("encoder_ar_loss", [False, True])
    def test_the_step_runs_in_the_order_the_step_has(self, encoder_ar_loss):
        """Collate, align, forward -- the order `forward_backward_step` uses.

        Asserting on the helper alone would not have caught this: the helper
        returned a batch that looked fine and the refusal came two calls later,
        inside the decoder.
        """
        model = packing_model(encoder_ar_loss)
        packed = DiffusionGemmaCollator(pad_token_id=0, pack_samples=True)(features_for(model))
        model(**model_input(self.align(model, packed)))

    def test_the_self_conditioning_draw_follows_the_packed_canvas_count(self):
        """Skipping the alignment is what leaves this count ragged, so it is pinned here.

        The bits are per canvas, and the decoder splits them by cutting its
        sequence into as many equal blocks as there are bits. A count taken from
        anything but the row's own K -- the sample count, say -- would cut the
        row into blocks that are not canvases and condition the wrong tokens,
        without dividing unevenly often enough to raise.
        """
        model = packing_model(encoder_ar_loss=False)
        canvas_length = model.config.canvas_length
        packed = self.align(model, DiffusionGemmaCollator(pad_token_id=0, pack_samples=True)(features_for(model)))
        canvases = packed["decoder_input_ids"].shape[1]
        # Ragged on purpose: were the draw per sample, this would be 3, not 5.
        assert canvases == sum(count for _, count in SPECS)

        assert self.trainer(model)._self_conditioning_sample_count(packed) == canvases

        flags = torch.zeros(canvases, dtype=torch.bool)
        flags[1] = True
        spread = expand_self_conditioning_mask(flags, 1, canvases * canvas_length)[0, :, 0]
        assert bool(spread[canvas_length : 2 * canvas_length].all())
        assert not bool(spread[:canvas_length].any())
        assert not bool(spread[2 * canvas_length :].any())

    def test_a_dummy_canvas_would_have_been_refused(self):
        """Why the skip is required rather than tidy.

        The dummy slot's mask is all-visible, which over a packed row is every
        sample in it. One is enough to stop the step.
        """
        model = packing_model(encoder_ar_loss=False)
        packed = DiffusionGemmaCollator(pad_token_id=0, pack_samples=True)(features_for(model))
        padded = dg_trainer.pad_decoder_canvases(dict(packed), packed["decoder_input_ids"].shape[1] + 1)

        with pytest.raises(ValueError, match="read past the end of the sample"):
            model(**model_input(padded))


class TestTheTrainerBuildsTheCollatorTheModeAsksFor:
    @staticmethod
    def stub(canvas_batching, multimodal=False, dyn_bsz=False, micro_batch_size=1, runtime="main"):
        # A real config, not a namespace of the fields this happens to read. `canvas_length`
        # and `max_canvases` live on the outer config while `pad_token_id` lives on the text
        # one, and a stub that invents its own placement passes while production raises
        # `AttributeError` -- which is exactly how a crash on the only usable `dyn_bsz`
        # combination shipped past a green suite once already.
        trainer = SimpleNamespace(
            base=SimpleNamespace(
                args=SimpleNamespace(
                    data=SimpleNamespace(max_seq_len=4096),
                    train=SimpleNamespace(dyn_bsz=dyn_bsz, dyn_bsz_runtime=runtime, micro_batch_size=micro_batch_size),
                ),
                model=SimpleNamespace(
                    model_config=DiffusionGemmaConfig(
                        canvas_batching=canvas_batching, canvas_length=256, max_canvases=8
                    )
                ),
            ),
            _is_multimodal=multimodal,
        )
        # The real methods, not no-ops: what building the collator has to *say* about a
        # token budget is part of what it is responsible for.
        for name in (
            "_warn_if_the_budget_unit_does_not_match_train_size",
            "_check_the_budget_can_hold_a_sample",
        ):
            setattr(trainer, name, MethodType(getattr(DiffusionGemmaTrainer, name), trainer))
        return trainer

    @classmethod
    def build(cls, canvas_batching, multimodal=False, dyn_bsz=False, micro_batch_size=1, runtime="main"):
        trainer = cls.stub(canvas_batching, multimodal, dyn_bsz, micro_batch_size, runtime)
        DiffusionGemmaTrainer._build_collate_fn(trainer)
        return trainer.base

    @pytest.mark.parametrize(("mode", "packs"), [("serial", False), ("flat_seq", False), ("flat_seq_pack", True)])
    def test_only_the_packed_mode_gets_the_packing_collator(self, mode, packs):
        base = self.build(mode)
        assert base.collate_fn.pack_samples is packs
        assert base.collate_fn.pad_token_id == DiffusionGemmaConfig().get_text_config().pad_token_id

    def test_packing_the_multimodal_data_is_refused_at_startup(self):
        """Before the first step rather than during it.

        The collator refuses image features per feature, which is a step-one
        crash after the model is built and sharded. The mode and the data type
        are both known at build time, so the contradiction is knowable then.
        """
        with pytest.raises(ValueError, match="cannot pack the multimodal data types"):
            self.build("flat_seq_pack", multimodal=True)

    def test_the_cost_measure_is_supplied_beside_the_collator(self):
        """The two have to agree, so they are set together.

        A budget costed one way and a batch built another describes a step that
        is not the step being run.
        """
        base = self.build("flat_seq_pack")
        assert base.dyn_bsz_length_fn is diffusion_gemma_sample_cost

    @pytest.mark.parametrize("mode", ["serial", "flat_batch", "flat_seq"])
    def test_a_token_budget_without_packing_is_refused(self, mode):
        """The budget would describe a batch the collator does not build.

        Without packing every prompt is padded to the longest in the micro-batch,
        so a selection of N real tokens is handed to the model as up to
        `micro_batch_size * max(L)`. That overshoot is in the direction that
        OOMs, and nothing between the buffer and the model would notice.
        """
        with pytest.raises(ValueError, match="needs `canvas_batching: flat_seq_pack`"):
            self.build(mode, dyn_bsz=True)

    def test_the_packed_mode_accepts_a_token_budget(self):
        base = self.build("flat_seq_pack", dyn_bsz=True)
        assert base.collate_fn.pack_samples is True

    @staticmethod
    @contextmanager
    def warnings():
        said = []
        with patch.object(dg_trainer.logger, "warning_rank0", side_effect=said.append):
            yield said

    def test_a_token_budget_says_what_train_size_now_has_to_be_counted_in(self):
        """The one consequence of the charge that no code path can catch.

        `compute_train_steps` divides `data.train_size` by the budget per step, so
        a dataset declared in plain tokens is spent up to twice as fast as that
        estimate and the run stops with the learning-rate schedule unfinished.
        The ratio depends on the data, so this is said rather than corrected --
        but it has to be said, because a run that ends early looks like a run
        that ended.
        """
        with self.warnings() as said:
            self.build("flat_seq_pack", dyn_bsz=True)
        assert [message for message in said if "train_size" in message]

    def test_nothing_is_said_about_train_size_when_there_is_no_budget(self):
        """`train_steps` comes from the sample count then, and is unaffected."""
        with self.warnings() as said:
            self.build("flat_seq_pack")
        assert not [message for message in said if "train_size" in message]

    def test_a_budget_too_small_for_one_sample_is_called_out(self):
        """The recipe's own `micro_batch_size: 1` is this case, which is the point.

        The budget is `micro_batch_size * max_seq_len` = 4096 while a sample costs up
        to 4096 + 8*256 = 6144, so under `dyn_bsz_runtime: main` the buffer's
        `force=True` takes an oversized first sample and packs nothing beside it. The
        dial reads as set and does nothing except stop bounding the row.
        """
        with self.warnings() as said:
            self.build("flat_seq_pack", dyn_bsz=True, micro_batch_size=1)
        assert [message for message in said if "6144" in message and "micro_batch_size" in message]

    def test_a_budget_with_room_for_two_samples_is_left_alone(self):
        """`micro_batch_size: 3` gives 12288, which is twice the worst case."""
        with self.warnings() as said:
            self.build("flat_seq_pack", dyn_bsz=True, micro_batch_size=3)
        assert not [message for message in said if "costs up to" in message]

    def test_the_worker_runtime_refuses_the_budget_that_main_only_warns_about(self):
        """The two runtimes handle an oversized sample in opposite directions.

        `DynBszBuffer` passes it through; `DynamicBatchingSizeDataset` *discards* it,
        and `force_generate_long_sequence` cannot be turned on. So the same budget
        that costs `main` its packing costs `worker` the training data -- at
        `micro_batch_size: 1` every sample with canvases exceeds 4096 and an epoch can
        produce nothing. A warning shared between the two would describe one of them
        wrongly, which is what this pins.
        """
        with pytest.raises(ValueError, match="discards a sample that does not fit"):
            self.build("flat_seq_pack", dyn_bsz=True, micro_batch_size=1, runtime="worker")

    def test_the_worker_runtime_is_accepted_once_the_budget_holds_a_sample(self):
        """`micro_batch_size: 2` gives 8192, past the 6144 worst case, so nothing drops."""
        with self.warnings() as said:
            self.build("flat_seq_pack", dyn_bsz=True, micro_batch_size=2, runtime="worker")
        # Still short of twice the cost, so the packing warning stands -- that one is
        # about wasted configuration and applies to both runtimes.
        assert [message for message in said if "one sample per row" in message]
