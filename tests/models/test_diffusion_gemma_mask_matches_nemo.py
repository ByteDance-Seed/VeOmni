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
"""The decoder's visibility is the same set NeMo's reference recipe attends through.

This is the cheapest correctness check there is on the SFT path and the one worth
having first: it needs no weights, no GPU and no checkpoint, and if it fails then
every numerical comparison downstream is comparing two different objectives rather
than two implementations of one.

The two stacks arrive at that set by different machinery. NeMo builds a dense
``[B, 1, response, enc + response]`` boolean directly from block arithmetic -- a
strict ``block_q > block_kv`` over encoder columns, block-diagonal over canvas
columns. VeOmni carries the same thing as a per-canvas half-open *interval*, because
``flat_seq`` packs the canvases onto one row and the Triton kernel reads a run rather
than a mask (see ``dg_hole``). Two representations of one set is exactly the shape of
bug that survives unit tests on either side alone.

Two layers, because each is weak alone:

* **The closed form** (always runs). Canvas ``k`` of an example with prompt length
  ``p`` sees encoder keys ``[0, p + k*C)`` on a full-attention layer, narrowed to
  ``[max(0, p + k*C - (W-1)), p + k*C)`` on a sliding one, plus its own canvas block.
  These assertions are the contract and gate CI on their own.
* **The cross-stack comparison** (skipped when the NeMo checkout is absent). It feeds
  *NeMo's own* visibility into VeOmni's interval helpers and compares the dense masks,
  so nothing in the comparison is transcribed by hand. This is the evidence that the
  closed form above is NeMo's, not merely self-consistent.

Point the comparison at a checkout with ``NEMO_AUTOMODEL_REPO=/path/to/repo``.
NeMo's mask module imports nothing but ``torch``, so it is loaded by path and the
NeMo package itself is never imported -- the two stacks pin incompatible
``transformers`` versions and cannot share a process.
"""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from veomni.models.transformers.diffusion_gemma.dg_hole import (
    canvas_visible_intervals,
    dense_hole_mask,
    sliding_visible_intervals,
)


NEMO_MASK_RELATIVE_PATH = "nemo_automodel/components/models/diffusion_gemma/attention_mask.py"


def _load_nemo_mask_builder():
    """NeMo's ``build_block_diffusion_training_mask``, loaded by path, or ``None``.

    An explicitly configured path that holds no such module raises rather than
    skipping: a typo there would otherwise read as "CI has no checkout" and the
    comparison would be reported green without having run.
    """
    configured = os.environ.get("NEMO_AUTOMODEL_REPO")
    if configured:
        candidates = [Path(configured)]
    else:
        # The two checkouts sit side by side on a dev box; CI has neither.
        siblings = Path(__file__).resolve().parents[2].parent
        candidates = [siblings / "diffusion-gemma-automodel", siblings / "Automodel"]

    for repo in candidates:
        module_path = repo / NEMO_MASK_RELATIVE_PATH
        if not module_path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("nemo_dg_attention_mask", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.build_block_diffusion_training_mask

    if configured:
        raise FileNotFoundError(
            f"NEMO_AUTOMODEL_REPO={configured} has no {NEMO_MASK_RELATIVE_PATH}. Unset it to skip the "
            "cross-stack comparison, or point it at a NeMo AutoModel checkout."
        )
    return None


NEMO_MASK_BUILDER = _load_nemo_mask_builder()
requires_nemo = pytest.mark.skipif(
    NEMO_MASK_BUILDER is None,
    reason=f"no NeMo checkout with {NEMO_MASK_RELATIVE_PATH}; set NEMO_AUTOMODEL_REPO to enable",
)


# (prompt lengths per example, canvases, canvas length, encoder tail padding)
GEOMETRIES = [
    ([5], 1, 4, 0),
    ([5], 3, 4, 0),
    ([0], 3, 4, 0),  # no prompt: canvas 0 sees nothing of the encoder
    ([7], 2, 8, 0),
    ([13], 5, 4, 0),  # prompt not a multiple of the canvas length
    ([6], 2, 4, 5),  # encoder columns past the response are never attendable
    ([3, 9], 3, 4, 0),  # per-example prompts against one shared key axis
    ([0, 1, 12], 2, 8, 3),
]
# W = 1 narrows to nothing; W - 1 past the whole cache narrows to nothing at all.
SLIDING_WINDOWS = [None, 1, 2, 5, 9, 64]


def _ids(geometries):
    return [f"p{'-'.join(map(str, p))}_k{k}_c{c}_pad{pad}" for p, k, c, pad in geometries]


def geometry(prompts, canvases, canvas_length, tail_pad):
    response_length = canvases * canvas_length
    return {
        "prefix_lengths": torch.tensor(prompts, dtype=torch.long),
        "response_length": response_length,
        "enc_len": max(prompts) + response_length + tail_pad,
        "block_size": canvas_length,
    }


def veomni_dense_mask(encoder_visible, canvas_length, sliding_window=None):
    """VeOmni's chain end to end: visibility -> interval -> (narrow) -> dense mask."""
    lo, hi = canvas_visible_intervals(encoder_visible)
    if sliding_window is not None:
        lo, hi = sliding_visible_intervals(lo, hi, sliding_window)
    return dense_hole_mask(lo, hi, encoder_visible.shape[-1], canvas_length)


def expected_dense_mask(prompts, canvases, canvas_length, enc_len, sliding_window=None):
    """The contract, written as the set rather than as either stack's machinery."""
    batch = len(prompts)
    tokens = canvases * canvas_length
    canvas_of = torch.arange(tokens) // canvas_length
    keys = torch.arange(enc_len)

    clean = torch.zeros(batch, tokens, enc_len, dtype=torch.bool)
    for example, prompt in enumerate(prompts):
        # A canvas reads the clean cache up to its own block's boundary and no further:
        # the first token of its own block is the answer it is being trained to denoise.
        boundary = prompt + canvas_of * canvas_length
        low = torch.zeros_like(boundary)
        if sliding_window is not None:
            low = (boundary - (sliding_window - 1)).clamp(min=0)
        clean[example] = (keys >= low.unsqueeze(-1)) & (keys < boundary.unsqueeze(-1))

    same_canvas = canvas_of.unsqueeze(-1) == canvas_of.unsqueeze(0)
    noise = same_canvas.unsqueeze(0).expand(batch, tokens, tokens)
    return torch.cat([clean, noise], dim=-1).unsqueeze(1)


def pipeline_visibility(prompts, canvases, canvas_length, enc_len):
    """What the collator hands the decoder: canvas ``k`` sees ``prompt + k*C``."""
    boundary = torch.tensor(prompts).unsqueeze(-1) + torch.arange(canvases) * canvas_length
    return torch.arange(enc_len) < boundary.unsqueeze(-1)


class TestTheClosedForm:
    """VeOmni's intervals expand to the set the contract names. Gates CI alone."""

    @pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
    @pytest.mark.parametrize("prompts, canvases, canvas_length, tail_pad", GEOMETRIES, ids=_ids(GEOMETRIES))
    def test_the_dense_mask_is_the_interval_the_contract_names(
        self, prompts, canvases, canvas_length, tail_pad, sliding_window
    ):
        spec = geometry(prompts, canvases, canvas_length, tail_pad)
        enc_len = spec["enc_len"]
        visible = pipeline_visibility(prompts, canvases, canvas_length, enc_len)

        actual = veomni_dense_mask(visible, canvas_length, sliding_window)
        expected = expected_dense_mask(prompts, canvases, canvas_length, enc_len, sliding_window)
        assert torch.equal(actual, expected), (
            f"visibility differs at {(actual != expected).sum().item()} of {actual.numel()} positions"
        )

    @pytest.mark.parametrize("prompts, canvases, canvas_length, tail_pad", GEOMETRIES, ids=_ids(GEOMETRIES))
    def test_no_canvas_reads_its_own_block_or_later(self, prompts, canvases, canvas_length, tail_pad):
        """The leakage invariant, stated on the boundary column rather than the whole set.

        Off-by-one here is total leakage and not a tolerance question: the canvas
        would read the clean token it is denoising, and the loss would fall without
        the model having learned anything.
        """
        enc_len = geometry(prompts, canvases, canvas_length, tail_pad)["enc_len"]
        visible = pipeline_visibility(prompts, canvases, canvas_length, enc_len)
        mask = veomni_dense_mask(visible, canvas_length)[:, 0, :, :enc_len]

        for example, prompt in enumerate(prompts):
            for canvas in range(canvases):
                rows = slice(canvas * canvas_length, (canvas + 1) * canvas_length)
                boundary = prompt + canvas * canvas_length
                if boundary < enc_len:
                    assert not mask[example, rows, boundary].any(), (
                        f"canvas {canvas} of example {example} reads the clean encoder column at its own "
                        f"block start ({boundary})"
                    )
                assert mask[example, rows, :boundary].all(), (
                    f"canvas {canvas} of example {example} is missing part of the prefix it is entitled to"
                )

    @pytest.mark.parametrize("sliding_window", [2, 5, 9])
    def test_the_sliding_window_is_anchored_to_the_block_not_to_the_query(self, sliding_window):
        """Every query in a block gets the same window, and it ends at the block boundary.

        A per-query band is the plausible wrong answer here -- and the one NeMo's own
        comment records having shipped and reverted -- because it starves the later
        queries in a block of the previous block's context that generation feeds them.
        It would also make the run depend on the query, which is precisely what an
        interval cannot express, so the kernel could not carry it either.
        """
        prompts, canvases, canvas_length = [6], 3, 4
        enc_len = prompts[0] + canvases * canvas_length
        visible = pipeline_visibility(prompts, canvases, canvas_length, enc_len)
        mask = veomni_dense_mask(visible, canvas_length, sliding_window)[0, 0, :, :enc_len]

        for canvas in range(canvases):
            rows = mask[canvas * canvas_length : (canvas + 1) * canvas_length]
            assert torch.equal(rows, rows[:1].expand_as(rows)), (
                f"queries inside canvas {canvas} disagree, so the window follows the query"
            )
            boundary = prompts[0] + canvas * canvas_length
            attended = rows[0].nonzero(as_tuple=True)[0]
            if attended.numel():
                assert int(attended[-1]) == boundary - 1, "the window does not end at the block boundary"
                assert int(attended[0]) == max(0, boundary - (sliding_window - 1))
                assert attended.numel() <= sliding_window - 1


@requires_nemo
class TestAgainstNemo:
    """VeOmni's interval helpers, fed NeMo's own visibility, reproduce NeMo's dense mask."""

    @staticmethod
    def nemo_masks(spec, sliding_window):
        return NEMO_MASK_BUILDER(
            spec["prefix_lengths"],
            spec["response_length"],
            spec["enc_len"],
            spec["block_size"],
            sliding_window=sliding_window,
            device="cpu",
        )

    @pytest.mark.parametrize("prompts, canvases, canvas_length, tail_pad", GEOMETRIES, ids=_ids(GEOMETRIES))
    def test_the_full_attention_mask_is_the_same_set(self, prompts, canvases, canvas_length, tail_pad):
        spec = geometry(prompts, canvases, canvas_length, tail_pad)
        nemo_full, _ = self.nemo_masks(spec, None)

        # M_OBC is constant inside a block, so every C-th row of NeMo's mask *is* the
        # per-canvas visibility VeOmni's intervals are built from. Taking it from NeMo
        # keeps the comparison free of any rule restated here.
        visible = nemo_full[:, 0, :: spec["block_size"], : spec["enc_len"]]
        assert visible.shape == (len(prompts), canvases, spec["enc_len"])

        actual = veomni_dense_mask(visible, canvas_length)
        assert torch.equal(actual, nemo_full), (
            f"{(actual != nemo_full).sum().item()} of {nemo_full.numel()} positions differ from NeMo"
        )

    @pytest.mark.parametrize("sliding_window", [w for w in SLIDING_WINDOWS if w is not None])
    @pytest.mark.parametrize("prompts, canvases, canvas_length, tail_pad", GEOMETRIES, ids=_ids(GEOMETRIES))
    def test_the_sliding_mask_is_the_same_set(self, prompts, canvases, canvas_length, tail_pad, sliding_window):
        """The one that had to be checked: the narrowing rules are written differently.

        NeMo lower-bounds the encoder columns at ``valid_cache - W + 1`` and leaves the
        canvas columns unbanded; VeOmni narrows the interval to ``hi - (W - 1)``. Those
        agree, but nothing about either expression says so.
        """
        spec = geometry(prompts, canvases, canvas_length, tail_pad)
        nemo_full, nemo_sliding = self.nemo_masks(spec, sliding_window)

        visible = nemo_full[:, 0, :: spec["block_size"], : spec["enc_len"]]
        actual = veomni_dense_mask(visible, canvas_length, sliding_window)
        assert torch.equal(actual, nemo_sliding), (
            f"{(actual != nemo_sliding).sum().item()} of {nemo_sliding.numel()} positions differ from NeMo"
        )

    def test_the_production_geometry(self):
        """Once at the shipped shape, since the small cases cannot see a 256/512 mixup."""
        spec = geometry([712], 3, 256, 0)
        nemo_full, nemo_sliding = self.nemo_masks(spec, 512)
        visible = nemo_full[:, 0, ::256, : spec["enc_len"]]

        assert torch.equal(veomni_dense_mask(visible, 256), nemo_full)
        assert torch.equal(veomni_dense_mask(visible, 256, 512), nemo_sliding)

    @pytest.mark.parametrize("prompts, canvases, canvas_length, tail_pad", GEOMETRIES, ids=_ids(GEOMETRIES))
    def test_nemos_visibility_is_one_contiguous_run(self, prompts, canvases, canvas_length, tail_pad):
        """``flat_seq`` can only carry NeMo's set at all if it has no hole in it.

        ``canvas_visible_intervals`` refuses a gap rather than reading across it, so
        this is really a check that the refusal never fires on the reference geometry
        -- if NeMo ever made the set non-contiguous, packing would have to go.
        """
        spec = geometry(prompts, canvases, canvas_length, tail_pad)
        nemo_full, _ = self.nemo_masks(spec, None)
        visible = nemo_full[:, 0, :: spec["block_size"], : spec["enc_len"]]

        lo, hi = canvas_visible_intervals(visible)
        assert torch.equal(visible.sum(dim=-1).to(torch.int32), (hi - lo).to(torch.int32))
        assert torch.equal(lo, torch.zeros_like(lo)), "a run that does not start at 0 means left padding"
