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
"""What a dynamic-batch token budget charges a DiffusionGemma sample.

The default charge is the sample's `attention_mask` sum, which is right when a
sample is one sequence through one stack and wrong for DiffusionGemma: its
canvases sit in the encoder row *and* get a second pass through the decoder, so
two samples with the same row length and different canvas counts cost
differently while the default calls them equal.

`scripts/bench_dg_step_cost.py` is where that was measured -- 1.3-1.6x of step
time across a canvas sweep at fixed row length. The generic plumbing that hands a
measure to the dataloader is covered by `test_dyn_bsz_length_fn.py`; this file
checks the arithmetic of the DG charge and that a buffer spends it.
"""

import pytest
import torch

from veomni.data.dynamic_batching import DynBszBuffer
from veomni.trainer.diffusion_gemma_trainer import diffusion_gemma_sample_cost


CANVAS = 256


def dg_sample(prompt, canvases, canvas_length=CANVAS, stacked=True):
    """A DiffusionGemma sample as the data transform emits it, cost-relevant keys only."""
    length = prompt + canvases * canvas_length
    canvas_shape = (canvases, canvas_length) if stacked else (canvas_length,)
    return {
        "input_ids": torch.zeros(length, dtype=torch.long),
        "attention_mask": torch.ones(length, dtype=torch.long),
        "decoder_input_ids": torch.zeros(canvas_shape, dtype=torch.long),
    }


class TestTheDiffusionGemmaChargeCountsCanvasesTwice:
    def test_the_charge_is_the_encoder_row_plus_the_canvas_tokens(self):
        sample = dg_sample(prompt=1024, canvases=4)
        row = 1024 + 4 * CANVAS
        assert diffusion_gemma_sample_cost(sample) == row + 4 * CANVAS

    def test_a_single_unstacked_canvas_is_charged_the_same_way(self):
        """`compute_all_canvas_df_loss: false` gives `(C,)` rather than `(1, C)`."""
        assert diffusion_gemma_sample_cost(dg_sample(1024, 1, stacked=False)) == (1024 + CANVAS) + CANVAS

    @pytest.mark.parametrize("canvases", [1, 2, 4, 7])
    def test_samples_the_default_charge_calls_equal_are_charged_differently(self, canvases):
        """The whole point, stated as the property the default measure lacks.

        Every sample here has the same encoder row, so `attention_mask.sum()` is
        identical for all of them -- and the measured step time is not, by up to
        1.6x. The charge has to separate them or the budget cannot convert to a
        time at all.
        """
        row = 2048
        sample = dg_sample(prompt=row - canvases * CANVAS, canvases=canvases)

        assert int(sample["attention_mask"].sum()) == row
        assert diffusion_gemma_sample_cost(sample) == row + canvases * CANVAS

    def test_the_charge_rises_with_the_canvas_count_at_a_fixed_row_length(self):
        """And by roughly as much as the step time does, which is why 2x and not more.

        Across this sweep the charge spreads 1.67x where the measured step time
        spreads about 1.43x. Erring high is the safe direction for a budget, but
        not by much: charging a canvas token the fitted 2.69x instead of 2x would
        spread it 2.05x, well past the time it is standing in for.
        """
        row = 2048
        costs = [diffusion_gemma_sample_cost(dg_sample(row - k * CANVAS, k)) for k in (1, 2, 4, 7)]

        assert costs == sorted(costs)
        assert costs[-1] / costs[0] == pytest.approx(1.67, abs=0.01)

    def test_it_never_charges_less_than_the_default_would(self):
        """A charge below the encoder row would under-fill nothing and over-fill the step."""
        for prompt, canvases in ((0, 8), (1024, 4), (4096, 1)):
            sample = dg_sample(prompt, canvases)
            assert diffusion_gemma_sample_cost(sample) >= int(sample["attention_mask"].sum())

    @pytest.mark.parametrize("missing", ["input_ids", "decoder_input_ids"])
    def test_a_sample_missing_either_half_is_refused(self, missing):
        """Defaulting the canvas half to zero is the silent version of the old bug."""
        sample = dg_sample(1024, 4)
        del sample[missing]
        with pytest.raises(KeyError, match=missing):
            diffusion_gemma_sample_cost(sample)

    def test_the_encoder_half_is_the_row_the_collator_concatenates(self):
        """`input_ids`, not the mask, so it does not rest on the mask being all ones.

        The packed collator concatenates the whole of `input_ids`, so that is the
        row the model is handed whatever the mask says.
        """
        sample = dg_sample(1024, 4)
        sample["attention_mask"] = torch.zeros_like(sample["attention_mask"])
        assert diffusion_gemma_sample_cost(sample) == (1024 + 4 * CANVAS) + 4 * CANVAS

    def test_it_is_the_charge_a_buffer_can_be_built_with(self):
        """End of the plumbing: the DG measure selecting a DG micro-batch.

        Four samples of the same 2048-token row, canvas counts 7, 7, 1, 1. The
        default charge calls all four identical and hands three of them to a
        6144-unit budget -- 9984 units of real work, 62% over. The DG charge takes
        two, for exactly the budget.
        """
        samples = [dg_sample(2048 - k * CANVAS, k) for k in (7, 7, 1, 1)]

        default = DynBszBuffer()
        for sample in samples:
            default.append(sample)
        charged = DynBszBuffer(get_length_fn=diffusion_gemma_sample_cost)
        for sample in samples:
            charged.append(sample)

        picked_by_default = default.get_samples(6144)
        assert len(picked_by_default) == 3
        assert sum(diffusion_gemma_sample_cost(sample) for sample in picked_by_default) == 9984

        picked = charged.get_samples(6144)
        assert len(picked) == 2
        assert sum(diffusion_gemma_sample_cost(sample) for sample in picked) == 6144
