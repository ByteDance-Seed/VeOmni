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
"""The hole kernel against a dense oracle that shares none of its arithmetic.

The oracle slices ``[lo, hi)`` out of the cache, concatenates the canvas, and
calls a plain softmax -- no mask, no tiling, no online rescaling. So a tile
boundary the kernel got wrong, an interval it read from the wrong row, or a
canvas it let see its neighbour all show up as a numerical disagreement rather
than as a second implementation of the same mistake.

Tolerances are relative to each tensor's own scale. The error rides in on
shared intermediates -- the online softmax rescales a whole row at a time -- so
it is the same size across a tensor whether the element it lands on is large or
small, and an elementwise `rtol` would demand of the near-zero entries a
precision the arithmetic never had.
"""

import pytest
import torch

from veomni.ops.kernels.dllm_hole_attention import dllm_hole_attention, hole_attention_reference


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the hole kernel is Triton/CUDA only")

#: fp32 runs `tl.dot` with `input_precision="ieee"`, so the only gap left is the
#: online softmax's rescaling order. bf16 keeps ~8 mantissa bits.
FLOAT32_TOLERANCE = 1e-5
BFLOAT16_TOLERANCE = 5e-2

SCALE = 0.125


def build_case(
    batch=2,
    heads_q=4,
    heads_kv=2,
    canvases=3,
    canvas_length=32,
    clean_length=40,
    head_dim=64,
    dtype=torch.float32,
    left_pad=True,
    prompt=8,
    seed=0,
):
    """A batch whose intervals look like the decoder's: a growing visible prefix.

    Canvas ``k`` sees the prompt plus the ``k`` canvases before it, and sample 1
    carries a left pad so ``lo`` is not uniformly zero -- a kernel that ignored
    ``lo`` would still pass if every interval started at the cache's first row.

    ``prompt`` is the visible run canvas 0 gets, and it is a parameter because
    the default of 8 keys is shorter than any tile: every shape in ``SHAPES``
    inherits it, so none of them crosses a tile boundary inside the visible run.
    The recipe's is 221 (see ``TestTheGeometryTheRecipeRuns``).
    """
    torch.manual_seed(seed)
    tokens = canvases * canvas_length
    device = "cuda"

    def tensor(*shape):
        return torch.randn(*shape, dtype=dtype, device=device, requires_grad=True)

    query = tensor(batch, heads_q, tokens, head_dim)
    clean_key = tensor(batch, heads_kv, clean_length, head_dim)
    clean_value = tensor(batch, heads_kv, clean_length, head_dim)
    noise_key = tensor(batch, heads_kv, tokens, head_dim)
    noise_value = tensor(batch, heads_kv, tokens, head_dim)

    lo = torch.zeros(batch, canvases, dtype=torch.int32, device=device)
    hi = torch.zeros(batch, canvases, dtype=torch.int32, device=device)
    prompt = min(prompt, clean_length)
    for sample in range(batch):
        pad = sample if left_pad else 0
        for canvas in range(canvases):
            lo[sample, canvas] = min(pad, clean_length)
            hi[sample, canvas] = min(prompt + canvas * canvas_length, clean_length)
    hi = torch.maximum(hi, lo)
    return query, clean_key, clean_value, noise_key, noise_value, lo, hi


def worst_relative(got, want):
    """Largest absolute disagreement as a fraction of the tensor's own scale."""
    if want.numel() == 0:
        return 0.0
    scale = want.abs().max()
    if scale == 0:
        return got.abs().max().item()
    return ((got - want).abs().max() / scale).item()


SHAPES = [
    pytest.param({}, id="baseline"),
    pytest.param({"heads_q": 2, "heads_kv": 2}, id="no-gqa"),
    pytest.param({"heads_q": 8, "heads_kv": 1}, id="wide-gqa"),
    pytest.param({"canvases": 1}, id="single-canvas"),
    pytest.param({"canvas_length": 16, "canvases": 4}, id="minimum-canvas"),
    # A legal canvas that is not a power of two: the tile has to fall back to a
    # power-of-two divisor of it rather than to the canvas itself.
    pytest.param({"canvas_length": 48, "canvases": 2}, id="ragged-canvas"),
    pytest.param({"head_dim": 128}, id="two-d-groups"),
    pytest.param({"head_dim": 48}, id="ragged-head-dim"),
    pytest.param({"head_dim": 256, "canvas_length": 64, "canvases": 2, "clean_length": 100}, id="shipped-head-dim"),
    pytest.param({"head_dim": 512, "canvas_length": 64, "canvases": 2, "clean_length": 100}, id="widest-head-dim"),
    # The tile sizes are picked for bf16; in fp32 the same tile is twice the
    # bytes, and this is the widest canvas/head pair that has to keep fitting.
    pytest.param({"head_dim": 128, "canvas_length": 256, "canvases": 2, "clean_length": 300}, id="shipped-canvas"),
    pytest.param({"clean_length": 0}, id="empty-cache"),
    pytest.param({"clean_length": 4}, id="cache-shorter-than-a-tile"),
    pytest.param({"batch": 1}, id="single-sample"),
]


@requires_cuda
@pytest.mark.parametrize("overrides", SHAPES)
def test_the_kernel_matches_the_dense_oracle(overrides):
    inputs = build_case(**overrides)
    got = dllm_hole_attention(*inputs, SCALE)
    want = hole_attention_reference(*inputs, SCALE)

    assert got.shape == inputs[0].shape
    assert worst_relative(got, want) < FLOAT32_TOLERANCE


@requires_cuda
@pytest.mark.parametrize("overrides", SHAPES)
def test_every_gradient_matches_the_dense_oracle(overrides):
    inputs = build_case(**overrides)
    tensors = list(inputs[:5])
    upstream = torch.randn_like(inputs[0])

    got = torch.autograd.grad(dllm_hole_attention(*inputs, SCALE), tensors, upstream)
    want = torch.autograd.grad(hole_attention_reference(*inputs, SCALE), tensors, upstream)

    names = ("query", "clean_key", "clean_value", "noise_key", "noise_value")
    for name, mine, theirs in zip(names, got, want):
        assert worst_relative(mine, theirs) < FLOAT32_TOLERANCE, f"{name} disagrees with the oracle"


@requires_cuda
def test_bfloat16_matches_the_oracle_at_its_own_precision():
    """The shipped dtype. Run the oracle in fp32 so the comparison measures the
    kernel rather than two bf16 reduction orders."""
    inputs = build_case(dtype=torch.bfloat16)
    got = dllm_hole_attention(*inputs, SCALE)
    want = hole_attention_reference(*[t.float() if t.is_floating_point() else t for t in inputs], SCALE)

    assert got.dtype == torch.bfloat16
    assert worst_relative(got.float(), want) < BFLOAT16_TOLERANCE


@requires_cuda
def test_bfloat16_gradients_match_the_oracle_at_their_own_precision():
    """The backward sums in fp32 registers but stores in the input dtype, so
    bf16 is where that store is the only rounding between the two. fp32 cannot
    see it: there the store is exact."""
    inputs = build_case(dtype=torch.bfloat16)
    tensors = list(inputs[:5])
    upstream = torch.randn_like(inputs[0])
    wide = [t.float() if t.is_floating_point() else t for t in inputs]

    got = torch.autograd.grad(dllm_hole_attention(*inputs, SCALE), tensors, upstream)
    want = torch.autograd.grad(hole_attention_reference(*wide, SCALE), wide[:5], upstream.float())

    names = ("query", "clean_key", "clean_value", "noise_key", "noise_value")
    for name, mine, theirs in zip(names, got, want):
        assert mine.dtype == torch.bfloat16, f"{name} came back in the wrong dtype"
        assert worst_relative(mine.float(), theirs) < BFLOAT16_TOLERANCE, f"{name} disagrees with the oracle"


@requires_cuda
def test_a_canvas_cannot_read_the_cache_past_its_interval():
    """`hi` is the leak boundary, so move the key just outside it and require
    the output not to notice. The oracle would agree with a kernel that read
    too far only if it read too far as well; this asks the kernel alone."""
    inputs = build_case()
    query, clean_key, clean_value, noise_key, noise_value, lo, hi = inputs
    baseline = dllm_hole_attention(*inputs, SCALE)

    canvas = 0
    boundary = int(hi[0, canvas])
    disturbed = clean_key.detach().clone()
    disturbed[0, :, boundary] += 10.0
    moved = dllm_hole_attention(query, disturbed, clean_value, noise_key, noise_value, lo, hi, SCALE)

    span = slice(0, query.shape[2] // lo.shape[1])
    torch.testing.assert_close(moved[0, :, span], baseline[0, :, span])
    # The same key one row earlier is inside the interval, so it must move.
    inside = clean_key.detach().clone()
    inside[0, :, boundary - 1] += 10.0
    shifted = dllm_hole_attention(query, inside, clean_value, noise_key, noise_value, lo, hi, SCALE)
    assert not torch.allclose(shifted[0, :, span], baseline[0, :, span])


@requires_cuda
def test_a_canvas_cannot_read_the_cache_before_its_interval():
    """`lo` is the other boundary: left padding below it, and on a sliding layer
    everything the window has dropped."""
    inputs = build_case()
    query, clean_key, clean_value, noise_key, noise_value, lo, hi = inputs
    lo = lo + 4
    hi = torch.maximum(hi, lo)
    baseline = dllm_hole_attention(query, clean_key, clean_value, noise_key, noise_value, lo, hi, SCALE)

    disturbed = clean_key.detach().clone()
    disturbed[:, :, 0] += 10.0
    moved = dllm_hole_attention(query, disturbed, clean_value, noise_key, noise_value, lo, hi, SCALE)

    torch.testing.assert_close(moved, baseline)


@requires_cuda
def test_a_canvas_cannot_read_another_canvas_noise():
    """The holes. Only the perturbed canvas's own outputs may move."""
    inputs = build_case()
    query, clean_key, clean_value, noise_key, noise_value, lo, hi = inputs
    canvases = lo.shape[1]
    canvas_length = query.shape[2] // canvases
    baseline = dllm_hole_attention(*inputs, SCALE)

    disturbed = noise_key.detach().clone()
    disturbed[:, :, canvas_length : 2 * canvas_length] += 10.0
    moved = dllm_hole_attention(query, clean_key, clean_value, disturbed, noise_value, lo, hi, SCALE)

    for canvas in range(canvases):
        span = slice(canvas * canvas_length, (canvas + 1) * canvas_length)
        if canvas == 1:
            assert not torch.allclose(moved[:, :, span], baseline[:, :, span])
        else:
            torch.testing.assert_close(moved[:, :, span], baseline[:, :, span])


#: Intervals that select no clean key at all. `build_case` runs a 40-long cache
#: with a 32-wide tile, so these cover an empty run on a tile boundary and off
#: it, an inverted one, one that clamping inverts, and negative bounds.
DEGENERATE_INTERVALS = [
    pytest.param(0, 0, id="empty-on-a-tile"),
    pytest.param(5, 5, id="empty-mid-tile"),
    pytest.param(37, 37, id="empty-in-the-last-tile"),
    pytest.param(20, 10, id="inverted"),
    pytest.param(50, 60, id="inverted-by-the-clamp"),
    pytest.param(-8, -2, id="wholly-negative"),
    pytest.param(0, -2, id="negative-hi"),
]


@requires_cuda
@pytest.mark.parametrize("lo_value, hi_value", DEGENERATE_INTERVALS)
def test_an_interval_that_selects_nothing_still_lets_a_canvas_see_itself(lo_value, hi_value):
    """A padded canvas can have nothing visible in the cache. Its row must still
    reach softmax with a key, or the output is a NaN that only surfaces in the
    loss -- and it must see no clean key at all, not almost all of them.

    Where the interval sits is the point of the parametrisation. An empty run
    off a tile boundary still has a tile below it, and walking into that tile
    finds nothing visible and rescales a running max of -inf against itself. A
    negative bound is the mirror image: read as an offset from the end of the
    cache rather than as nothing, it turns the emptiest interval there is into
    nearly the whole cache.
    """
    inputs = build_case()
    query, clean_key, clean_value, noise_key, noise_value, lo, hi = inputs
    lo = torch.full_like(lo, lo_value)
    hi = torch.full_like(hi, hi_value)

    out = dllm_hole_attention(query, clean_key, clean_value, noise_key, noise_value, lo, hi, SCALE)
    want = hole_attention_reference(query, clean_key, clean_value, noise_key, noise_value, lo, hi, SCALE)
    empty = torch.zeros_like(lo)
    noise_only = hole_attention_reference(query, clean_key, clean_value, noise_key, noise_value, empty, empty, SCALE)

    assert torch.isfinite(out).all()
    assert worst_relative(out, want) < FLOAT32_TOLERANCE
    assert worst_relative(out, noise_only) < FLOAT32_TOLERANCE


@requires_cuda
@pytest.mark.parametrize(
    "reaching, clamped",
    [
        pytest.param((0, 200), (0, 40), id="past-the-end"),
        pytest.param((-64, 24), (0, 24), id="before-the-start"),
    ],
)
def test_an_interval_reaching_outside_the_cache_reads_only_the_cache(reaching, clamped):
    """Clamped rather than trusted: the alternative to clamping a bound the
    caller put outside the cache is reading outside the cache, which is a fault
    or a silently wrong answer depending on what is allocated next."""
    query, clean_key, clean_value, noise_key, noise_value, lo, hi = build_case()
    assert clean_key.shape[2] == 40, "the clamped bounds above are written against this cache length"

    def intervals(bounds):
        return torch.full_like(lo, bounds[0]), torch.full_like(hi, bounds[1])

    out = dllm_hole_attention(query, clean_key, clean_value, noise_key, noise_value, *intervals(reaching), SCALE)
    want = hole_attention_reference(query, clean_key, clean_value, noise_key, noise_value, *intervals(clamped), SCALE)

    assert worst_relative(out, want) < FLOAT32_TOLERANCE


#: The 26B recipe's decoder geometry, off `diffusiongemma-26B-A4B-it/config.json`
#: and the Sudoku recipe: 16 query heads, `head_dim` 256 with 8 KV heads on the 25
#: sliding layers and `global_head_dim` 512 with 2 on the 5 full-attention ones,
#: `canvas_length` 256, and a 477-key encoder row of which 221 are the prompt.
SHIPPED = dict(batch=1, heads_q=16, canvases=1, canvas_length=256, clean_length=477, prompt=221)

SHIPPED_LAYERS = [
    pytest.param(dict(SHIPPED, heads_kv=8, head_dim=256), id="sliding-layer"),
    pytest.param(dict(SHIPPED, heads_kv=2, head_dim=512), id="global-layer"),
]

#: What the recipe passes for `sm_scale`. `SCALE` is 0.125, where softmax is soft
#: and a last-bit difference in a logit is averaged away; at 1.0 the same inputs
#: go nearly one-hot -- mean top weight 0.20 against 0.88 on the sliding layer
#: and 0.35 against 0.92 on the global one -- so that difference decides which
#: key a row reads instead. Two numerical regimes, and only one of them ships.
SHIPPED_SCALE = 1.0


class TestTheGeometryTheRecipeRuns:
    """`SHAPES` reaches `head_dim` 256 only at `canvas_length` 64 and
    `canvas_length` 256 only at `head_dim` 128, so the pair the sliding layers
    run is covered by neither -- and `head_dim` 512 meets `canvas_length` 256
    nowhere, which is all five full-attention layers. Every shape there also
    inherits `build_case`'s 8-key visible run, shorter than a tile, where the
    recipe's is 221 keys inside a 477-key cache.

    These pass. They are here because the suite passing says nothing about the
    shape that ships unless the shape that ships is in it, and a tiling change
    is free to stop being true at one pair and not the others.
    """

    @requires_cuda
    @pytest.mark.parametrize("overrides", SHIPPED_LAYERS)
    def test_the_forward_matches_the_oracle(self, overrides):
        inputs = build_case(**overrides)
        assert worst_relative(dllm_hole_attention(*inputs, SCALE), hole_attention_reference(*inputs, SCALE)) < (
            FLOAT32_TOLERANCE
        )

    @requires_cuda
    @pytest.mark.parametrize("overrides", SHIPPED_LAYERS)
    def test_every_gradient_matches_the_oracle(self, overrides):
        inputs = build_case(**overrides)
        tensors = list(inputs[:5])
        upstream = torch.randn_like(inputs[0])

        got = torch.autograd.grad(dllm_hole_attention(*inputs, SCALE), tensors, upstream)
        want = torch.autograd.grad(hole_attention_reference(*inputs, SCALE), tensors, upstream)

        names = ("query", "clean_key", "clean_value", "noise_key", "noise_value")
        for name, mine, theirs in zip(names, got, want):
            assert worst_relative(mine, theirs) < FLOAT32_TOLERANCE, f"{name} disagrees with the oracle"

    @requires_cuda
    @pytest.mark.parametrize("visible", [8, 64, 65, 221, 477])
    def test_a_visible_run_of_any_length_matches_the_oracle(self, visible):
        """Swept on the global layer, which is the wider tile and the one no
        shape covers at this canvas length. 64 and 65 straddle a tile edge; 477
        is the whole cache, where the run ends mid-tile."""
        query, clean_key, clean_value, noise_key, noise_value, lo, hi = build_case(
            **dict(SHIPPED, heads_kv=2, head_dim=512)
        )
        inputs = (
            query,
            clean_key,
            clean_value,
            noise_key,
            noise_value,
            torch.zeros_like(lo),
            torch.full_like(hi, visible),
        )
        tensors = list(inputs[:5])
        upstream = torch.randn_like(query)

        forward = worst_relative(dllm_hole_attention(*inputs, SCALE), hole_attention_reference(*inputs, SCALE))
        got = torch.autograd.grad(dllm_hole_attention(*inputs, SCALE), tensors, upstream)
        want = torch.autograd.grad(hole_attention_reference(*inputs, SCALE), tensors, upstream)

        assert forward < FLOAT32_TOLERANCE
        for mine, theirs in zip(got, want):
            assert worst_relative(mine, theirs) < FLOAT32_TOLERANCE


@requires_cuda
@pytest.mark.parametrize("overrides", SHIPPED_LAYERS)
def test_the_shipped_softmax_scale_matches_the_oracle(overrides):
    """At `sm_scale` 1.0 the row is nearly one-hot, so the online softmax's
    running maximum is doing almost all of the work and almost none of the
    accumulation. Every other test here runs 0.125, which exercises the
    opposite. Asserted on the backward as well: near-one-hot weights make
    `dp - delta` a difference of close quantities, which is where a rescaling
    order shows up if it is going to."""
    inputs = build_case(**overrides)
    tensors = list(inputs[:5])
    upstream = torch.randn_like(inputs[0])

    got = dllm_hole_attention(*inputs, SHIPPED_SCALE)
    want = hole_attention_reference(*inputs, SHIPPED_SCALE)
    assert worst_relative(got, want) < FLOAT32_TOLERANCE

    # Measure the regime rather than asserting the scale was passed. If these
    # inputs happened to give a soft row at 1.0 the test would be a duplicate of
    # the fp32 one above, and the docstring's claim would be decoration.
    query, clean_key = inputs[0], inputs[1]
    visible = int(inputs[6][0, 0])
    heads_per_group = query.shape[1] // clean_key.shape[1]
    logits = torch.einsum("hqd,hkd->hqk", query[0].float(), clean_key[0].float().repeat_interleave(heads_per_group, 0))
    top = torch.softmax(logits[:, :, :visible] * SHIPPED_SCALE, dim=-1).max(dim=-1).values
    assert top.mean() > 0.5, (
        f"softmax is not near one-hot at sm_scale {SHIPPED_SCALE} (mean top weight {top.mean():.3f})"
    )

    mine = torch.autograd.grad(dllm_hole_attention(*inputs, SHIPPED_SCALE), tensors, upstream)
    theirs = torch.autograd.grad(hole_attention_reference(*inputs, SHIPPED_SCALE), tensors, upstream)
    names = ("query", "clean_key", "clean_value", "noise_key", "noise_value")
    for name, a, b in zip(names, mine, theirs):
        assert worst_relative(a, b) < FLOAT32_TOLERANCE, f"{name} disagrees with the oracle at sm_scale 1.0"


@requires_cuda
def test_each_canvas_carries_its_own_interval():
    """`build_case` staggers `lo` by sample and holds it constant across the
    canvases of a row, so a kernel that read canvas 0's bounds for the whole row
    would pass every shape in `SHAPES`. A packed row is where that matters: it
    carries canvases from several samples, whose visible prefixes start at
    different places in the cache.
    """
    query, clean_key, clean_value, noise_key, noise_value, _, _ = build_case(
        batch=1, canvases=4, canvas_length=32, clean_length=64
    )
    device = query.device
    lo = torch.tensor([[0, 17, 33, 48]], dtype=torch.int32, device=device)
    hi = torch.tensor([[16, 33, 49, 64]], dtype=torch.int32, device=device)
    staggered = (query, clean_key, clean_value, noise_key, noise_value, lo, hi)

    got = dllm_hole_attention(*staggered, SCALE)
    assert worst_relative(got, hole_attention_reference(*staggered, SCALE)) < FLOAT32_TOLERANCE

    # The bounds must not be interchangeable, or the assertion above would hold
    # for a kernel that ignored three of the four intervals.
    uniform = (query, clean_key, clean_value, noise_key, noise_value, torch.zeros_like(lo), torch.full_like(hi, 16))
    assert worst_relative(got, hole_attention_reference(*uniform, SCALE)) > 1e-2

    tensors = list(staggered[:5])
    upstream = torch.randn_like(query)
    mine = torch.autograd.grad(dllm_hole_attention(*staggered, SCALE), tensors, upstream)
    theirs = torch.autograd.grad(hole_attention_reference(*staggered, SCALE), tensors, upstream)
    names = ("query", "clean_key", "clean_value", "noise_key", "noise_value")
    for name, a, b in zip(names, mine, theirs):
        assert worst_relative(a, b) < FLOAT32_TOLERANCE, f"{name} disagrees with the oracle"


class TestTheKernelRefusesWhatItCannotHonour:
    def test_cpu_tensors_are_refused_rather_than_run(self):
        inputs = [t.cpu() if torch.is_tensor(t) else t for t in build_case()] if torch.cuda.is_available() else None
        if inputs is None:
            tokens = 3 * 32
            inputs = [
                torch.randn(2, 4, tokens, 64),
                torch.randn(2, 2, 40, 64),
                torch.randn(2, 2, 40, 64),
                torch.randn(2, 2, tokens, 64),
                torch.randn(2, 2, tokens, 64),
                torch.zeros(2, 3, dtype=torch.int32),
                torch.zeros(2, 3, dtype=torch.int32),
            ]
        with pytest.raises(ValueError, match="CUDA"):
            dllm_hole_attention(*inputs, SCALE)

    @requires_cuda
    def test_a_missing_scale_is_refused(self):
        with pytest.raises(ValueError, match="sm_scale"):
            dllm_hole_attention(*build_case())

    @requires_cuda
    def test_intervals_must_cover_every_sample(self):
        query, clean_key, clean_value, noise_key, noise_value, lo, hi = build_case()
        with pytest.raises(ValueError, match="clean_lo/clean_hi must be"):
            dllm_hole_attention(query, clean_key, clean_value, noise_key, noise_value, lo[:1], hi[:1], SCALE)

    @requires_cuda
    def test_a_canvas_split_that_does_not_divide_is_refused(self):
        query, clean_key, clean_value, noise_key, noise_value, lo, hi = build_case(canvases=3, canvas_length=32)
        wider = torch.zeros(lo.shape[0], 5, dtype=torch.int32, device=lo.device)
        with pytest.raises(ValueError, match="equal canvases"):
            dllm_hole_attention(query, clean_key, clean_value, noise_key, noise_value, wider, wider, SCALE)

    @requires_cuda
    def test_a_canvas_below_the_dot_minimum_is_refused(self):
        inputs = build_case(canvas_length=32, canvases=4)
        query, clean_key, clean_value, noise_key, noise_value, lo, hi = inputs
        # 128 query tokens read as 16 canvases would put 8 tokens in each.
        narrow = torch.zeros(lo.shape[0], 16, dtype=torch.int32, device=lo.device)
        with pytest.raises(ValueError, match="must be a multiple of 16"):
            dllm_hole_attention(query, clean_key, clean_value, noise_key, noise_value, narrow, narrow, SCALE)

    @requires_cuda
    def test_a_noise_cache_that_does_not_match_the_query_is_refused(self):
        query, clean_key, clean_value, noise_key, noise_value, lo, hi = build_case()
        with pytest.raises(ValueError, match="one key per query token"):
            dllm_hole_attention(
                query, clean_key, clean_value, noise_key[:, :, :-16], noise_value[:, :, :-16], lo, hi, SCALE
            )

    @requires_cuda
    def test_a_mixed_dtype_batch_is_refused(self):
        query, clean_key, clean_value, noise_key, noise_value, lo, hi = build_case()
        with pytest.raises(ValueError, match="does not match the query"):
            dllm_hole_attention(query, clean_key.half(), clean_value, noise_key, noise_value, lo, hi, SCALE)

    @requires_cuda
    def test_float64_is_refused_rather_than_silently_narrowed(self):
        inputs = build_case(dtype=torch.float64)
        with pytest.raises(ValueError, match="fp16/bf16/fp32"):
            dllm_hole_attention(*inputs, SCALE)
