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
"""``canvas_batching="flat_batch"`` has to be the serial loop, not a second model.

Flat folds a sample's K canvases into the decoder's batch instead of looping over them.
The claim that buys is scheduling only -- one set of FSDP/EP collectives and one MoE
call instead of K -- so what it must earn is that no canvas's inputs changed: same keys,
same sliding window, same isolation from the answer it is denoising.

Two kinds of check, because either alone is weak. The serial path is a numerical oracle,
which catches a fold that mixes rows up but says nothing about whether *both* paths leak.
So the visibility boundary is also probed directly on the flat path: perturb a clean
canvas, a future canvas, another noisy canvas, and assert which outputs are allowed to
move. A leak that predates this change would show up there and not in the comparison.
"""

from typing import get_args

import pytest
import torch
from huggingface_hub.errors import StrictDataclassFieldValidationError

from tests.models.test_diffusion_gemma import build_toy_model
from veomni.data.data_collator import DiffusionGemmaCollator
from veomni.models.transformers.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaConfig
from veomni.models.transformers.diffusion_gemma.dg_hole import canvas_visible_intervals, sliding_visible_intervals
from veomni.models.transformers.diffusion_gemma.dg_prefix import pack_prefix_kv
from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import (
    _decoder_cache_span,
    _decoder_positions_from_context_mask,
    _fold_canvas_axis,
    _repeat_cache_onto_canvas_rows,
)


#: Enough canvases that a middle one has both a clean past and a clean future.
CANVASES = 3
PROMPT_LENGTH = 8
#: Keys the model itself takes; the collator also emits bookkeeping the trainer consumes.
MODEL_INPUTS = (
    "input_ids",
    "attention_mask",
    "decoder_input_ids",
    "decoder_attention_mask",
    "labels",
    "loss_weight",
)


def build_all_canvas_batch(
    config,
    batch_size=2,
    num_canvases=CANVASES,
    prompt_length=PROMPT_LENGTH,
    seed=0,
    prompt_lengths=None,
    canvas_counts=None,
):
    """A stacked ``(B, K, C)`` batch carrying the transform's per-block context masks.

    Built through the real collator so ``decoder_attention_mask`` gets the layout the
    model is entitled to assume: the encoder half is 1 on the prompt and the canvases
    *before* the draw, 0 from the draw on, and the canvas half is all ones.

    ``prompt_lengths`` / ``canvas_counts`` override the uniform defaults per sample, and
    the collator then does what it does in training -- left-pad the shorter prompt, pad
    the shorter sample up to max K with dummy canvases.
    """
    torch.manual_seed(seed)
    canvas_length = config.canvas_length
    vocab_size = config.text_config.vocab_size
    prompt_lengths = list(prompt_lengths) if prompt_lengths is not None else [prompt_length] * batch_size
    canvas_counts = list(canvas_counts) if canvas_counts is not None else [num_canvases] * batch_size
    features = []
    for prompt, canvases in zip(prompt_lengths, canvas_counts):
        encoder_length = prompt + canvases * canvas_length
        context = torch.zeros(canvases, encoder_length, dtype=torch.long)
        for block in range(canvases):
            context[block, : prompt + block * canvas_length] = 1
        features.append(
            {
                "input_ids": torch.randint(3, vocab_size, (encoder_length,)),
                "decoder_context_mask": context,
                "decoder_input_ids": torch.randint(3, vocab_size, (canvases, canvas_length)),
                "labels": torch.randint(3, vocab_size, (canvases, canvas_length)),
                "loss_weight": torch.full((canvases, canvas_length), 1.0 / (canvases * canvas_length)),
            }
        )
    batch = DiffusionGemmaCollator(pad_token_id=0)(features)
    return {key: batch[key] for key in MODEL_INPUTS}


def _double(batch):
    return {key: value.double() if value.is_floating_point() else value for key, value in batch.items()}


def _conditioning_logits(model, batch_size=2, canvases=CANVASES, seed=5):
    torch.manual_seed(seed)
    return torch.randn(
        batch_size,
        canvases,
        model.config.canvas_length,
        model.config.text_config.vocab_size,
        dtype=torch.float64,
    )


def _toy_model(dtype=torch.float32):
    """A toy model in train mode, which a stacked batch requires.

    ``SealableCache(truncate_sliding=not self.training)`` keeps the whole encoder history
    only while training. Eval truncates the sliding layers to the window, and a per-block
    context mask hides a suffix of the encoder -- precisely what the model refuses to
    serve out of a truncated cache. Nothing here is stochastic in train mode (attention
    dropout is 0 and the canvases are not resampled), so the forward stays deterministic.
    """
    return build_toy_model(dtype=dtype).train()


def _forward(model, batch, mode):
    model.config.canvas_batching = mode
    with torch.no_grad():
        return model(**{key: value.clone() for key, value in batch.items()})


def _forward_backward(model, batch, mode):
    model.config.canvas_batching = mode
    model.zero_grad(set_to_none=True)
    outputs = model(**{key: value.clone() for key, value in batch.items()})
    outputs.loss.backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return outputs, gradients


#: Both recipes ship `micro_batch_size: 1`, so that is the shape that will actually
#: train; 2 is the smallest one where a canvas-major fold is distinguishable from a
#: sample-major one, since at B=1 the two orderings coincide.
BATCH_SIZES = (1, 2)
#: The two modes that pack K canvases into one decoder pass. They pack different
#: axes -- `flat_batch` the batch, `flat_seq` the sequence -- and the serial loop is
#: the oracle for both.
PACKED_MODES = ("flat_batch", "flat_seq")


class TestFlatMatchesSerial:
    @pytest.mark.parametrize("mode", PACKED_MODES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_the_hidden_states_logits_and_losses_agree(self, batch_size, mode):
        model = _toy_model()
        batch = build_all_canvas_batch(model.config, batch_size=batch_size)

        serial = _forward(model, batch, "serial")
        flat = _forward(model, batch, mode)

        assert serial.logits.shape == (
            batch_size,
            CANVASES,
            model.config.canvas_length,
            model.config.text_config.vocab_size,
        )
        assert flat.logits.shape == serial.logits.shape
        torch.testing.assert_close(flat.logits, serial.logits)
        torch.testing.assert_close(flat.loss, serial.loss)
        torch.testing.assert_close(flat.diffusion_loss, serial.diffusion_loss)
        torch.testing.assert_close(flat.encoder_ar_loss, serial.encoder_ar_loss)

    @pytest.mark.parametrize("mode", PACKED_MODES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_every_parameter_gradient_agrees(self, batch_size, mode):
        """A fold that lined the rows up wrong could still average to the same loss.

        The two paths do add a gradient up in different orders -- the tied embedding
        alone collects one by three routes (encoder embedding, decoder embedding, tied
        head) -- so they are not expected to agree bitwise. Run in float64, because in
        float32 the disagreement reaches the size of the toy model's smaller gradients
        and no tolerance separates reordering from a real fault.

        float64 does not take it to a float64 floor: RMSNorm, the attention softmax, the
        rotary tables and the logits are all hardcoded to float32 in this model, so the
        floor stays at roughly one float32 ulp -- measured at 2.2e-7 of the gradient's
        own scale at B=1 and 9.3e-8 at B=2. The tolerance is therefore relative to each
        gradient *tensor's* scale rather than to each element: the error rides in on
        shared intermediates, so it is the same size across a tensor whether the element
        it lands on is large or small, and an elementwise `rtol` would demand of the
        near-zero entries a precision the arithmetic never had. A fold that crossed rows
        misses by O(1) of that same scale, four decades above this.
        """
        model = _toy_model(dtype=torch.float64)
        batch = _double(build_all_canvas_batch(model.config, batch_size=batch_size))

        _, serial = _forward_backward(model, batch, "serial")
        _, flat = _forward_backward(model, batch, mode)

        assert serial.keys() == flat.keys()
        assert "model.decoder.embed_tokens.weight" in serial
        assert len(serial) > 100, f"only {len(serial)} parameters received a gradient"
        for name in serial:
            scale = serial[name].abs().max()
            worst = (flat[name] - serial[name]).abs().max()
            assert worst <= 1e-6 * scale, (
                f"{name}: gradients differ by {worst:.3e}, which is {worst / scale:.3e} of the "
                f"tensor's own scale {scale:.3e} -- above the float32 floor the two orders leave"
            )

    @pytest.mark.parametrize("mode", PACKED_MODES)
    def test_the_self_conditioning_signal_stays_on_its_own_canvas(self, mode):
        """`self_conditioning_prob: 0.5` drives this path in the recipe, and it is the
        one place the packing has to line a per-canvas Bernoulli up with the tokens it
        was drawn for -- a decoder row under `flat_batch`, a span of the sequence under
        `flat_seq`."""
        model = _toy_model(dtype=torch.float64)
        batch = _double(build_all_canvas_batch(model.config))
        batch["self_conditioning_logits"] = _conditioning_logits(model)
        # Rows 1 and 4 are sample 0 canvas 1 and sample 1 canvas 1. Read canvas-major
        # instead they are sample 1 canvas 0 and sample 0 canvas 2, so a transposed fold
        # conditions the wrong canvases. Rows 0 and 5 would not catch that: they are the
        # two canvases the transpose leaves where they were.
        batch["self_conditioning_mask"] = torch.tensor([False, True, False, False, True, False])

        serial = _forward(model, batch, "serial")
        flat = _forward(model, batch, mode)

        torch.testing.assert_close(flat.logits, serial.logits)

    @pytest.mark.parametrize("mode", PACKED_MODES)
    def test_one_conditioning_flag_moves_exactly_one_canvas(self, mode):
        """Checked against the packed path with the signal off rather than against
        serial, because a transposed fold that both paths shared would agree with
        itself."""
        model = _toy_model(dtype=torch.float64)
        batch = _double(build_all_canvas_batch(model.config))
        batch["self_conditioning_logits"] = _conditioning_logits(model)
        batch["self_conditioning_mask"] = torch.zeros(2 * CANVASES, dtype=torch.bool)
        baseline = _forward(model, batch, mode).logits

        lit = 1  # sample 0, canvas 1
        batch["self_conditioning_mask"][lit] = True
        moved = _forward(model, batch, mode).logits

        for sample in range(2):
            for canvas in range(CANVASES):
                unchanged = torch.allclose(moved[sample, canvas], baseline[sample, canvas])
                assert unchanged == (sample * CANVASES + canvas != lit), (
                    f"sample {sample} canvas {canvas} was {'unmoved' if unchanged else 'moved'} "
                    f"by a flag meant for row {lit}"
                )

    @pytest.mark.parametrize("mode", PACKED_MODES)
    def test_ragged_prompts_and_canvas_counts_agree(self, mode):
        """The rows that actually differ are where a fold that crossed them would show:
        the collator left-pads the shorter prompt and pads the shorter sample up to
        max K with dummy canvases, so this batch has both. For `flat_seq` it is also
        where a canvas's interval starts somewhere other than the cache's first row."""
        model = _toy_model(dtype=torch.float64)
        batch = _double(build_all_canvas_batch(model.config, prompt_lengths=(8, 20), canvas_counts=(3, 1)))
        assert batch["decoder_input_ids"].shape[1] == 3

        serial = _forward(model, batch, "serial")
        flat = _forward(model, batch, mode)

        torch.testing.assert_close(flat.logits, serial.logits)

    @pytest.mark.parametrize("mode", PACKED_MODES)
    def test_the_fold_survives_recompute(self, mode):
        """The recipe trains with `gradient_checkpointing.enable: true`, and the packing
        rides into the attention layer as a keyword. A checkpointed layer replays its
        forward from keywords its first call already consumed once -- `canvas_groups`
        and `canvas_interval` are both popped there -- so the backward is the pass that
        would read the wrong keys if the replay lost them.
        """
        model = _toy_model(dtype=torch.float64)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        batch = _double(build_all_canvas_batch(model.config))

        serial_out, serial = _forward_backward(model, batch, "serial")
        flat_out, flat = _forward_backward(model, batch, mode)

        torch.testing.assert_close(flat_out.logits, serial_out.logits)
        assert serial.keys() == flat.keys()
        for name in serial:
            torch.testing.assert_close(
                flat[name], serial[name], rtol=1e-5, atol=1e-8, msg=lambda message, n=name: f"{n}: {message}"
            )

    @pytest.mark.parametrize("mode", PACKED_MODES)
    def test_the_encoder_still_runs_on_one_row(self, mode):
        """Packing K canvases must not clone the encoder; that OOMs 26B."""
        model = _toy_model()
        batch = build_all_canvas_batch(model.config)
        widths = []

        encoder = model.model.encoder.language_model
        handle = encoder.register_forward_hook(
            lambda _module, _args, output: widths.append(output.last_hidden_state.shape[0])
        )
        try:
            _forward(model, batch, mode)
        finally:
            handle.remove()

        assert widths == [batch["input_ids"].shape[0]]

    @pytest.mark.parametrize("mode", PACKED_MODES)
    def test_a_single_canvas_batch_is_untouched(self, mode):
        """``(B, C)`` never reaches the packing, so the flag cannot change Eq. 13 runs."""
        model = _toy_model()
        batch = build_all_canvas_batch(model.config, num_canvases=1)
        squeezed = dict(batch)
        for key in ("decoder_input_ids", "labels", "loss_weight", "decoder_attention_mask"):
            squeezed[key] = batch[key].squeeze(1)

        serial = _forward(model, squeezed, "serial")
        flat = _forward(model, squeezed, mode)

        assert serial.logits.ndim == 3
        torch.testing.assert_close(flat.logits, serial.logits)


@pytest.mark.parametrize("mode", PACKED_MODES)
class TestTheVisibilityBoundaryHolds:
    """Probe the packed paths directly: the oracle above cannot see a shared leak.

    `flat_seq` needs this more than `flat_batch` does. Under `flat_batch` a canvas is
    alone in its row, so the keys it cannot reach are absent; under `flat_seq` every
    canvas's keys are present in the same row and only the interval keeps them apart.
    """

    @staticmethod
    def _hidden(model, batch, mode):
        return _forward(model, batch, mode).logits

    def _perturbed(self, model, batch, span, mode):
        """Rerun with a slice of the encoder's clean answer replaced."""
        disturbed = {key: value.clone() for key, value in batch.items()}
        start, stop = span
        torch.manual_seed(7)
        disturbed["input_ids"][:, start:stop] = torch.randint(
            3, model.config.text_config.vocab_size, (batch["input_ids"].shape[0], stop - start)
        )
        return self._hidden(model, disturbed, mode)

    def test_a_canvas_cannot_read_the_clean_answer_it_is_denoising(self, mode):
        """The whole point of the per-block mask: canvas k must not see clean canvas k."""
        model = _toy_model()
        batch = build_all_canvas_batch(model.config)
        canvas_length = model.config.canvas_length
        target = 1
        start = PROMPT_LENGTH + target * canvas_length

        baseline = self._hidden(model, batch, mode)
        disturbed = self._perturbed(model, batch, (start, start + canvas_length), mode)

        torch.testing.assert_close(disturbed[:, target], baseline[:, target])

    def test_a_canvas_cannot_read_the_clean_answer_after_it(self, mode):
        model = _toy_model()
        batch = build_all_canvas_batch(model.config)
        canvas_length = model.config.canvas_length
        target = 1
        start = PROMPT_LENGTH + (target + 1) * canvas_length

        baseline = self._hidden(model, batch, mode)
        disturbed = self._perturbed(model, batch, (start, start + canvas_length), mode)

        torch.testing.assert_close(disturbed[:, target], baseline[:, target])

    def test_a_canvas_does_read_the_clean_answer_before_it(self, mode):
        """Without this the two tests above would pass on a decoder that reads nothing."""
        model = _toy_model()
        batch = build_all_canvas_batch(model.config)
        canvas_length = model.config.canvas_length
        target = 2
        start = PROMPT_LENGTH + (target - 1) * canvas_length

        baseline = self._hidden(model, batch, mode)
        disturbed = self._perturbed(model, batch, (start, start + canvas_length), mode)

        assert not torch.allclose(disturbed[:, target], baseline[:, target])

    def test_a_canvas_cannot_read_another_noisy_canvas(self, mode):
        """Folding K into the batch is what keeps the canvases apart; check it does."""
        model = _toy_model()
        batch = build_all_canvas_batch(model.config)

        baseline = self._hidden(model, batch, mode)
        disturbed_batch = {key: value.clone() for key, value in batch.items()}
        torch.manual_seed(11)
        disturbed_batch["decoder_input_ids"][:, 0] = torch.randint(
            3, model.config.text_config.vocab_size, batch["decoder_input_ids"][:, 0].shape
        )
        disturbed = self._hidden(model, disturbed_batch, mode)

        assert not torch.allclose(disturbed[:, 0], baseline[:, 0]), "the perturbed canvas ignored its own tokens"
        torch.testing.assert_close(disturbed[:, 1], baseline[:, 1])
        torch.testing.assert_close(disturbed[:, 2], baseline[:, 2])

    def test_the_sliding_layers_keep_their_window_on_each_canvas_prefix(self, mode):
        """Sliding layers see the last `W-1` *visible* keys, and visibility is per canvas.

        A fold that let one row's window be measured against another's prefix would still
        match the serial path only if the serial path had the same bug, so this pins the
        window where the config puts it: with `sliding_window == canvas_length` here, a
        token one full canvas back is outside every sliding layer's reach and only the
        single full-attention layer can carry it.
        """
        model = _toy_model()
        assert model.config.text_config.sliding_window == model.config.canvas_length
        layer_types = model.config.text_config.layer_types
        assert layer_types.count("sliding_attention") == 5 and layer_types.count("full_attention") == 1

        batch = build_all_canvas_batch(model.config)
        baseline = self._hidden(model, batch, mode)
        # The first prompt token is inside canvas 2's visible prefix but far outside the
        # sliding window, so it reaches canvas 2 through the full layer alone.
        disturbed = self._perturbed(model, batch, (0, 1), mode)

        assert not torch.allclose(disturbed[:, 2], baseline[:, 2])


class TestTheDgPrefixGatherFolds:
    """DG_PREFIX is the shipped decoder path, and the one that folds rows itself.

    Concatenation has to repeat the cache to reach ``B * K`` rows. DG_PREFIX does not:
    it gathers each canvas's visible prefix straight out of the one cached row, so the
    fold lives in an index and the only copy is the packed prefix it would build anyway.
    That index is what these pin.
    """

    def test_more_segments_than_cache_rows_gather_from_their_own_sample(self):
        # Two cached samples, two canvases each, in row-major order: segment i reads
        # cache row i // 2, with a longer visible prefix on the later canvas.
        visible = torch.tensor(
            [
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        encoder_k = torch.arange(2 * 1 * 8 * 2, dtype=torch.float32).view(2, 1, 8, 2)
        encoder_v = encoder_k + 1000

        packed_k, packed_v, cu = pack_prefix_kv(encoder_k, encoder_v, visible, canvas_groups=2)

        assert cu.tolist() == [0, 3, 8, 10, 13]
        expected = torch.stack(
            [
                encoder_k[0, 0, 2],
                encoder_k[0, 0, 3],
                encoder_k[0, 0, 4],
                encoder_k[0, 0, 2],
                encoder_k[0, 0, 3],
                encoder_k[0, 0, 4],
                encoder_k[0, 0, 5],
                encoder_k[0, 0, 6],
                encoder_k[1, 0, 1],
                encoder_k[1, 0, 2],
                encoder_k[1, 0, 1],
                encoder_k[1, 0, 2],
                encoder_k[1, 0, 3],
            ]
        )
        torch.testing.assert_close(packed_k[0, 0], expected)
        torch.testing.assert_close(packed_v[0, 0], expected + 1000)

    def test_extra_rows_are_refused_unless_the_fold_was_declared(self):
        """The fold is declared, never inferred from the two shapes.

        Inferring it would turn any other shape mismatch into a silent gather from the
        wrong sample's cache, and at the shipped `micro_batch_size: 1` a one-row cache
        divides into everything, so the check would never fire. Serial therefore keeps
        the exact equality it has always had.
        """
        encoder_k = torch.randn(2, 1, 4, 3)
        with pytest.raises(ValueError, match=r"batch \* canvas_groups"):
            pack_prefix_kv(encoder_k, encoder_k.clone(), torch.ones(4, 4, dtype=torch.bool))
        with pytest.raises(ValueError, match=r"batch \* canvas_groups"):
            pack_prefix_kv(encoder_k, encoder_k.clone(), torch.ones(3, 4, dtype=torch.bool), canvas_groups=2)

    def test_the_concatenating_path_also_refuses_an_undeclared_fold(self):
        """The sdpa branch used to be guarded by `torch.cat` raising on the row count."""
        keys = torch.randn(2, 1, 4, 3)
        with pytest.raises(ValueError, match="against an encoder cache"):
            _repeat_cache_onto_canvas_rows(keys, keys.clone(), query_rows=6, canvas_groups=1)
        folded, _ = _repeat_cache_onto_canvas_rows(keys, keys.clone(), query_rows=6, canvas_groups=3)
        assert folded.shape[0] == 6
        torch.testing.assert_close(folded[3], keys[1])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="DG_PREFIX kernel needs a GPU")
    def test_every_canvas_reaches_the_kernel_with_the_visibility_serial_gave_it(self, monkeypatch):
        """Compare what goes *into* the kernel, not what comes out.

        The kernel's own arithmetic departs from sdpa by ~0.8 on these logits, far more
        than folding does, so an output comparison here would be measuring the kernel.
        Its inputs are exact: a canvas's visible prefix is a boolean row, and the fold is
        correct exactly when canvas ``(b, k)`` arrives with the row the serial loop gave
        it. Combined with the gather test above -- segment ``i`` reads cache row
        ``i // K`` -- that fixes both halves of the index. The numerics are pinned in
        float32/float64 on the CPU path, which shares this mask construction.
        """
        import veomni.models.transformers.diffusion_gemma.dg_prefix as dg_prefix

        model = build_toy_model(dtype=torch.bfloat16).cuda().train()
        model.config.text_config.attn_dg_prefix = True
        batch = {key: value.cuda() for key, value in build_all_canvas_batch(model.config).items()}
        batch_size = batch["input_ids"].shape[0]
        layers = model.config.text_config.num_hidden_layers

        seen = []
        pack = dg_prefix.pack_prefix_kv
        monkeypatch.setattr(
            dg_prefix,
            "pack_prefix_kv",
            lambda keys, values, visible, groups=1: (
                seen.append(visible.clone()),
                pack(keys, values, visible, groups),
            )[1],
        )

        _forward(model, batch, "serial")
        serial_masks, seen[:] = list(seen), []
        _forward(model, batch, "flat_batch")
        flat_masks = list(seen)

        assert len(serial_masks) == layers * CANVASES
        assert all(mask.shape[0] == batch_size for mask in serial_masks)
        assert len(flat_masks) == layers
        for layer in range(layers):
            folded = flat_masks[layer]
            assert folded.shape[0] == batch_size * CANVASES
            for canvas in range(CANVASES):
                # The serial loop is canvas-outer, layer-inner: one whole decoder pass
                # per canvas.
                looped = serial_masks[canvas * layers + layer]
                for sample in range(batch_size):
                    assert torch.equal(folded[sample * CANVASES + canvas], looped[sample]), (
                        f"layer {layer}, sample {sample}, canvas {canvas} was folded onto the wrong prefix"
                    )


class TestTheFlagIsRefusedWhenItCannotBeHonoured:
    def test_an_unknown_mode_is_refused_on_assignment(self):
        """The strict dataclass enforces the `Literal` both ways, so a typo in a config
        or a checkpoint cannot quietly leave training on the serial path."""
        model = _toy_model()
        with pytest.raises(StrictDataclassFieldValidationError, match="canvas_batching"):
            model.config.canvas_batching = "fullstream"

    def test_every_mode_is_either_compared_against_serial_or_named_here(self):
        """A mode added to the annotation would be accepted by the config and then
        decoded by whichever branch it fell into, untested. Tying the annotation to
        `PACKED_MODES` means adding one lands it in every equivalence test above
        rather than only here.

        `flat_seq_pack` is the one exception and is listed rather than tolerated. It
        is not the same computation as `serial` on the same collated batch -- it puts
        several *examples* in the row, so it is the same computation as running those
        examples one at a time, which is what
        `tests/models/test_diffusion_gemma_packed_collator.py` pins end to end. Adding
        it to `PACKED_MODES` would make every comparison above fail for the right
        reason and the wrong test.
        """
        annotation = DiffusionGemmaConfig.__annotations__["canvas_batching"]
        assert get_args(annotation) == ("serial", *PACKED_MODES, "flat_seq_pack")

    @pytest.mark.parametrize("mode", PACKED_MODES)
    def test_a_prebuilt_mask_mapping_is_refused(self, mode):
        """A mask mapping is already sized to one canvas per row; reusing it would
        silently give every canvas the first row's visibility."""
        model = _toy_model()
        batch = build_all_canvas_batch(model.config)
        model.config.canvas_batching = mode
        with pytest.raises(ValueError, match="mask mapping"):
            model.model._decode_canvases(
                decoder_input_ids=batch["decoder_input_ids"],
                past_key_values=None,
                self_conditioning_logits=None,
                self_conditioning_embeddings=None,
                self_conditioning_mask=None,
                decoder_attention_mask={"full_attention": None, "sliding_attention": None},
                decoder_position_ids=None,
            )

    def test_a_single_mask_row_is_refused_rather_than_broadcast(self):
        """The tensor form of the same hazard. Broadcasting one row onto `B * K` would
        give every sample the first one's visibility, which is exactly what the serial
        path refuses when it checks a decoder mask against `(batch_size, kv_length)`."""
        one_row_mask = torch.ones(1, PROMPT_LENGTH + CANVASES, dtype=torch.bool)
        with pytest.raises(ValueError, match="Expected 2 decoder rows"):
            _fold_canvas_axis(one_row_mask, batch_size=2, num_canvases=CANVASES, stacked_ndim=3)


class TestFlatSeqRefusesAVisibilityItCannotExpress:
    """`flat_seq` carries visibility as one interval per canvas, which is strictly
    less expressive than the mask the other two modes take. Everything the pipeline
    builds fits -- left padding in front, future canvases behind -- so the mode's
    safety rests on refusing what does not, rather than on that never happening."""

    def test_a_hole_in_the_middle_of_the_visible_run_is_refused(self):
        """An interval cannot say "these keys but not that one in the middle", and a
        kernel handed one would read straight across the gap."""
        visible = torch.tensor([[[True, True, False, True, False]]])
        with pytest.raises(ValueError, match="not one contiguous run"):
            canvas_visible_intervals(visible)

    def test_left_padding_and_a_hidden_suffix_are_both_accepted(self):
        """The two shapes the pipeline actually produces: zeros before the run from
        left padding, zeros after it from the canvases not yet visible."""
        visible = torch.tensor([[[False, True, True, False, False], [False, False, False, False, False]]])
        lo, hi = canvas_visible_intervals(visible)

        assert lo.tolist() == [[1, 0]]
        assert hi.tolist() == [[3, 0]]
        assert lo.dtype == torch.int32

    def test_a_sliding_layer_measures_its_window_against_each_canvas_prefix(self):
        """Not against the cache: canvas 0 and canvas 2 end in different places, so
        the same window puts their lower bounds in different places too."""
        lo = torch.tensor([[0, 0]], dtype=torch.int32)
        hi = torch.tensor([[4, 16]], dtype=torch.int32)

        narrowed, unchanged = sliding_visible_intervals(lo, hi, sliding_window=8)

        assert narrowed.tolist() == [[0, 9]]
        assert torch.equal(unchanged, hi)

    def test_a_mask_that_hides_part_of_a_canvas_is_refused(self):
        """The canvas half is always all ones -- a canvas attends to itself in full.
        `flat_seq` has nowhere to put an exception to that."""
        model = _toy_model()
        batch = build_all_canvas_batch(model.config)
        batch["decoder_attention_mask"][:, :, -1] = 0

        with pytest.raises(ValueError, match="canvas half"):
            _forward(model, batch, "flat_seq")

    def test_a_two_dimensional_mask_is_refused(self):
        """Without a canvas axis there is no per-canvas prefix to read, and taking the
        one row for all of them would show every canvas the whole answer."""
        model = _toy_model()
        batch = build_all_canvas_batch(model.config)
        batch["decoder_attention_mask"] = batch["decoder_attention_mask"][:, 0]

        with pytest.raises(ValueError, match="per-canvas decoder mask"):
            _forward(model, batch, "flat_seq")


class TestDecoderPositionsFollowTheVisiblePrefix:
    def test_the_dispatch_uses_a_static_caches_mask_width(self, monkeypatch):
        class StaticCacheShape:
            is_compileable = True
            max_cache_len = 475

            @staticmethod
            def get_seq_length():
                return 219

        assert _decoder_cache_span(StaticCacheShape()) == 475
        model = _toy_model()
        canvas_length = model.config.canvas_length
        mask = torch.zeros(1, StaticCacheShape.max_cache_len + canvas_length, dtype=torch.bool)
        mask[:, :219] = True
        mask[:, StaticCacheShape.max_cache_len :] = True
        captured = {}

        def record_positions(**kwargs):
            captured["positions"] = kwargs["decoder_position_ids"]
            return object()

        monkeypatch.setattr(model.model.decoder, "forward", record_positions)
        model.model._decode_canvases(
            decoder_input_ids=torch.zeros(1, canvas_length, dtype=torch.long),
            past_key_values=StaticCacheShape(),
            self_conditioning_logits=None,
            self_conditioning_embeddings=None,
            self_conditioning_mask=None,
            decoder_attention_mask=mask,
            decoder_position_ids=None,
        )

        assert captured["positions"][0, 0].item() == 219
        assert captured["positions"][0, -1].item() == 219 + canvas_length - 1

    @pytest.mark.parametrize("mode", ("serial", *PACKED_MODES))
    def test_every_decoder_dispatch_receives_prefix_relative_positions(self, mode, monkeypatch):
        model = _toy_model()
        batch = build_all_canvas_batch(model.config, batch_size=1)
        seen = []
        original = model.model.decoder.forward

        def record_positions(*args, decoder_position_ids=None, **kwargs):
            seen.append(decoder_position_ids.detach().clone())
            return original(*args, decoder_position_ids=decoder_position_ids, **kwargs)

        monkeypatch.setattr(model.model.decoder, "forward", record_positions)
        _forward(model, batch, mode)

        if mode == "serial":
            positions = torch.stack(seen, dim=1)
        else:
            positions = seen[0].reshape(1, CANVASES, model.config.canvas_length)
        starts = PROMPT_LENGTH + torch.arange(CANVASES) * model.config.canvas_length
        expected = starts[:, None] + torch.arange(model.config.canvas_length)
        assert torch.equal(positions[0], expected)

    def test_a_hidden_answer_does_not_shift_the_canvas(self):
        prompt_length = 219
        canvas_length = 256
        cache_length = prompt_length + canvas_length
        mask = torch.zeros(1, cache_length + canvas_length, dtype=torch.bool)
        mask[:, :prompt_length] = True
        mask[:, cache_length:] = True

        positions = _decoder_positions_from_context_mask(mask, cache_length, canvas_length)

        assert positions.shape == (1, canvas_length)
        assert positions[0, 0].item() == 219
        assert positions[0, -1].item() == 474

    def test_each_stacked_canvas_uses_its_own_visible_prefix(self):
        cache_length = 20
        canvas_length = 4
        mask = torch.zeros(1, 3, cache_length + canvas_length, dtype=torch.bool)
        for canvas, prefix_length in enumerate((8, 12, 16)):
            mask[:, canvas, :prefix_length] = True
        mask[..., cache_length:] = True

        positions = _decoder_positions_from_context_mask(mask, cache_length, canvas_length)

        assert positions.tolist() == [[list(range(8, 12)), list(range(12, 16)), list(range(16, 20))]]

    def test_left_padding_keeps_positions_adjacent_to_the_visible_encoder_tokens(self):
        cache_length = 12
        canvas_length = 4
        mask = torch.zeros(2, cache_length + canvas_length, dtype=torch.bool)
        mask[0, 2:8] = True
        mask[1, :10] = True
        mask[:, cache_length:] = True

        positions = _decoder_positions_from_context_mask(mask, cache_length, canvas_length)

        assert positions.tolist() == [list(range(8, 12)), list(range(10, 14))]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the hole kernel is Triton/CUDA only")
class TestTheHoleKernelIsTheReasonForFlatSeq:
    """`flat_seq` without the kernel is a dense `K*C` by `P + K*C` mask, which pays
    for exactly the canvases it masks out. The kernel is the point of the mode, so it
    is checked against the dense mask it replaces and against the serial loop.

    Run in fp32, where both `tl.dot` (`input_precision="ieee"`) and sdpa are exact
    enough that the comparison measures the kernel's visibility rather than bf16's
    reduction order.
    """

    @staticmethod
    def _model(kernel):
        model = build_toy_model(dtype=torch.float32).cuda().train()
        model.config.text_config.attn_dg_prefix = kernel
        return model

    @staticmethod
    def _batch(config):
        return {key: value.cuda() for key, value in build_all_canvas_batch(config).items()}

    def test_the_kernel_actually_runs_rather_than_falling_back(self, monkeypatch):
        """Everything below would still pass if the kernel silently never ran."""
        import veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma as modeling

        model = self._model(kernel=True)
        calls = []
        original = modeling.hole_decoder_attention
        monkeypatch.setattr(
            modeling,
            "hole_decoder_attention",
            lambda *args, **kwargs: (calls.append(args[5].shape), original(*args, **kwargs))[1],
        )

        _forward(model, self._batch(model.config), "flat_seq")

        assert len(calls) == model.config.text_config.num_hidden_layers
        assert all(shape == (2, CANVASES) for shape in calls)

    def test_the_kernel_agrees_with_the_dense_mask_it_replaces(self):
        """Same mode, same intervals, two ways of applying them."""
        batch = self._batch(self._model(kernel=False).config)
        dense = _forward(self._model(kernel=False), batch, "flat_seq").logits
        kernel = _forward(self._model(kernel=True), batch, "flat_seq").logits

        assert (kernel - dense).abs().max() < 1e-4 * dense.abs().max()

    def test_the_kernel_agrees_with_the_serial_loop(self):
        """End to end: the packing and the kernel together against K plain passes."""
        model = self._model(kernel=False)
        batch = self._batch(model.config)
        serial = _forward(model, batch, "serial").logits
        kernel = _forward(self._model(kernel=True), batch, "flat_seq").logits

        assert (kernel - serial).abs().max() < 1e-4 * serial.abs().max()

    def test_every_parameter_gradient_agrees_with_the_serial_loop(self):
        model = self._model(kernel=False)
        batch = self._batch(model.config)
        _, serial = _forward_backward(model, batch, "serial")
        _, kernel = _forward_backward(self._model(kernel=True), batch, "flat_seq")

        assert serial.keys() == kernel.keys()
        for name in serial:
            scale = serial[name].abs().max()
            if scale == 0:
                continue
            worst = (kernel[name] - serial[name]).abs().max()
            assert worst <= 1e-3 * scale, f"{name}: {worst:.3e} against a scale of {scale:.3e}"

    def test_a_canvas_still_cannot_read_the_answer_it_is_denoising(self):
        """Under the kernel every canvas's keys sit in the same row, and only the
        interval keeps them apart -- so this is the leak the mode could introduce."""
        model = self._model(kernel=True)
        batch = self._batch(model.config)
        canvas_length = model.config.canvas_length
        target = 1
        start = PROMPT_LENGTH + target * canvas_length

        baseline = _forward(model, batch, "flat_seq").logits
        disturbed = {key: value.clone() for key, value in batch.items()}
        torch.manual_seed(7)
        disturbed["input_ids"][:, start : start + canvas_length] = torch.randint(
            3,
            model.config.text_config.vocab_size,
            (batch["input_ids"].shape[0], canvas_length),
            device=batch["input_ids"].device,
        )
        moved = _forward(model, disturbed, "flat_seq").logits

        torch.testing.assert_close(moved[:, target], baseline[:, target])
        assert not torch.allclose(moved[:, target + 1], baseline[:, target + 1])
