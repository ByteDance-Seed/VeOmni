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
from types import SimpleNamespace

import pytest
import torch

from tests.models.test_diffusion_gemma_canvas_batching import _toy_model
from veomni.data.data_collator import DiffusionGemmaCollator


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
