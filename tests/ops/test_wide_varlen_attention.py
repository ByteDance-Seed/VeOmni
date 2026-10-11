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
"""The varlen kernel against two oracles that share none of its arithmetic.

The first slices the keys each query is entitled to and calls a plain softmax,
one position at a time -- no mask, no tiling, no online rescaling. The second is
sdpa on the unpacked segments, which is the thing packing has to be equal to and
is the reason to trust the first: two independent implementations agreeing on a
packed row is a much narrower target than either agreeing with the kernel alone.

That second comparison is the point of the whole exercise. Packing is only worth
doing if a packed row computes what S separate rows computed, so it is asserted
elementwise rather than inferred from a run finishing -- and both directions of
the leak are probed directly, since an oracle cannot catch a kernel that reads
across a segment boundary if it reads across the same boundary itself.

Tolerances are relative to each tensor's own scale. The error rides in on shared
intermediates -- the online softmax rescales a whole row at a time -- so it is
the same size across a tensor whether the element it lands on is large or small,
and an elementwise `rtol` would demand of the near-zero entries a precision the
arithmetic never had. `worst_relative_in_group` is the one exception, and only
for a reference that has collapsed to noise; see its docstring for why the
fallback is floored rather than applied throughout.
"""

import pytest
import torch
import torch.nn.functional as F

from veomni.ops.kernels.wide_varlen_attention import wide_varlen_attention, wide_varlen_attention_reference


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the varlen kernel is Triton/CUDA only")

#: fp32 runs `tl.dot` with `input_precision="ieee"`, so the only gap left is the
#: online softmax's rescaling order. bf16 keeps ~8 mantissa bits.
FLOAT32_TOLERANCE = 1e-5
BFLOAT16_TOLERANCE = 5e-2

SCALE = 0.125


def build_case(lengths=(12, 20, 5), heads_q=4, heads_kv=2, head_dim=64, dtype=torch.float32, seed=0):
    """One packed row holding `lengths` segments end to end."""
    torch.manual_seed(seed)
    total = sum(lengths)
    device = "cuda"

    def tensor(heads):
        return torch.randn(1, heads, total, head_dim, dtype=dtype, device=device, requires_grad=True)

    query = tensor(heads_q)
    key = tensor(heads_kv)
    value = tensor(heads_kv)

    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0)
    return query, key, value, cu


def worst_relative(got, want):
    """Largest absolute disagreement as a fraction of the tensor's own scale."""
    if want.numel() == 0:
        return 0.0
    scale = want.abs().max()
    if scale == 0:
        return got.abs().max().item()
    return ((got - want).abs().max() / scale).item()


def worst_relative_in_group(got, want, group):
    """`worst_relative`, but with a floor for a reference that has collapsed.

    A gradient can be analytically zero while the arithmetic that produces it is
    not: in a one-token segment the softmax is 1.0 whatever the score is, so `dq`
    and `dk` are exactly zero, and both sides instead return the cancellation
    residue of `dp - delta`, whose operands are the size of `dv`. Measured
    against its own maximum that residue reads as a 117% error -- two noise
    tensors divided by one of themselves. Measured against the scale of the
    gradients it shares its intermediates with, it reads as the 7e-7 it is.

    The floor is deliberately narrow. Swapping the group's scale in
    unconditionally would divide every assertion by the largest gradient in the
    group, and under grouped-query attention `dk` and `dv` sum over a whole group
    of query heads and so run several times `dq` -- which would quietly relax the
    `dq` assertions by that ratio on every shape, not just on the degenerate one
    this exists for. So the tensor's own scale is used whenever it is meaningful,
    and only a reference that has collapsed to a thousandth of the computation
    around it falls back.
    """
    scale = want.abs().max().item()
    group_scale = max(tensor.abs().max().item() for tensor in group)
    if scale < group_scale * 1e-3:
        scale = group_scale
    if scale == 0:
        return got.abs().max().item()
    return ((got - want).abs().max() / scale).item()


SHAPES = [
    pytest.param({}, id="baseline"),
    pytest.param({"heads_q": 2, "heads_kv": 2}, id="no-gqa"),
    pytest.param({"heads_q": 8, "heads_kv": 1}, id="wide-gqa"),
    pytest.param({"lengths": (37,)}, id="single-segment"),
    pytest.param({"lengths": (1, 1, 1, 1)}, id="single-token-segments"),
    pytest.param({"lengths": (64, 64)}, id="tile-aligned"),
    # One past and one short of the 16-wide tile, so the masked tail is neither
    # empty nor the whole tile.
    pytest.param({"lengths": (65, 63)}, id="off-by-one-tiles"),
    pytest.param({"lengths": (96, 3, 17, 1, 40)}, id="ragged"),
    # `pack_prefix_kv` keeps empty prefixes as zero-length segments on purpose,
    # so they arrive here and must read nothing rather than fault.
    pytest.param({"lengths": (0, 20, 0, 13, 0)}, id="empty-segments"),
    pytest.param({"head_dim": 128}, id="two-d-groups"),
    pytest.param({"head_dim": 48}, id="ragged-head-dim"),
    pytest.param({"head_dim": 256, "lengths": (24, 40)}, id="shipped-sliding-head-dim"),
    # The five layers this kernel exists for.
    pytest.param({"head_dim": 512, "lengths": (24, 40)}, id="shipped-global-head-dim"),
    pytest.param({"head_dim": 512, "heads_q": 16, "heads_kv": 2, "lengths": (20, 12)}, id="shipped-global-geometry"),
]


@requires_cuda
@pytest.mark.parametrize("overrides", SHAPES)
def test_the_kernel_matches_the_sliced_oracle(overrides):
    query, key, value, cu = build_case(**overrides)
    got = wide_varlen_attention(query, key, value, cu, sm_scale=SCALE)
    want = wide_varlen_attention_reference(query, key, value, cu, SCALE)

    assert got.shape == query.shape
    assert torch.isfinite(got).all()
    assert worst_relative(got, want) < FLOAT32_TOLERANCE


@requires_cuda
@pytest.mark.parametrize("overrides", SHAPES)
def test_every_gradient_matches_the_sliced_oracle(overrides):
    query, key, value, cu = build_case(**overrides)
    tensors = [query, key, value]
    upstream = torch.randn_like(query)

    got = torch.autograd.grad(wide_varlen_attention(query, key, value, cu, sm_scale=SCALE), tensors, upstream)
    want = torch.autograd.grad(wide_varlen_attention_reference(query, key, value, cu, SCALE), tensors, upstream)

    for name, mine, theirs in zip(("query", "key", "value"), got, want):
        assert torch.isfinite(mine).all(), f"{name} came back with a non-finite entry"
        assert worst_relative_in_group(mine, theirs, want) < FLOAT32_TOLERANCE, f"{name} disagrees with the oracle"


def unpacked_sdpa(query, key, value, cu, scale):
    """sdpa over each segment on its own row -- what packing must be equal to.

    This is the padding-free half of the comparison: every segment is its own
    call at its own length, so nothing here has an opinion about packing. A
    kernel that let one segment see another would have to make sdpa do the same
    to agree, and sdpa is never shown two segments at once.
    """
    repeat = query.shape[1] // key.shape[1]
    outputs = torch.zeros_like(query)
    bounds = [int(x) for x in cu.tolist()]
    for start, end in zip(bounds[:-1], bounds[1:]):
        if end <= start:
            continue
        span = slice(start, end)
        outputs[:, :, span] = F.scaled_dot_product_attention(
            query[:, :, span],
            key[:, :, span].repeat_interleave(repeat, dim=1),
            value[:, :, span].repeat_interleave(repeat, dim=1),
            is_causal=True,
            scale=scale,
        )
    return outputs


@requires_cuda
@pytest.mark.parametrize("overrides", SHAPES)
def test_a_packed_row_computes_what_the_separate_rows_computed(overrides):
    """The claim the mode rests on, asserted rather than assumed."""
    query, key, value, cu = build_case(**overrides)
    got = wide_varlen_attention(query, key, value, cu, sm_scale=SCALE)
    want = unpacked_sdpa(query, key, value, cu, SCALE)

    assert worst_relative(got, want) < FLOAT32_TOLERANCE


@requires_cuda
@pytest.mark.parametrize("overrides", SHAPES)
def test_packed_gradients_match_the_separate_rows(overrides):
    query, key, value, cu = build_case(**overrides)
    tensors = [query, key, value]
    upstream = torch.randn_like(query)

    got = torch.autograd.grad(wide_varlen_attention(query, key, value, cu, sm_scale=SCALE), tensors, upstream)
    want = torch.autograd.grad(unpacked_sdpa(query, key, value, cu, SCALE), tensors, upstream)

    for name, mine, theirs in zip(("query", "key", "value"), got, want):
        assert worst_relative_in_group(mine, theirs, want) < FLOAT32_TOLERANCE, (
            f"{name} disagrees with the unpacked rows"
        )


@requires_cuda
def test_bfloat16_matches_the_oracle_at_its_own_precision():
    """The shipped dtype. Run the oracle in fp32 so the comparison measures the
    kernel rather than two bf16 reduction orders."""
    query, key, value, cu = build_case(dtype=torch.bfloat16, head_dim=512)
    got = wide_varlen_attention(query, key, value, cu, sm_scale=SCALE)
    want = wide_varlen_attention_reference(query.float(), key.float(), value.float(), cu, SCALE)

    assert got.dtype == torch.bfloat16
    assert worst_relative(got.float(), want) < BFLOAT16_TOLERANCE


@requires_cuda
def test_bfloat16_gradients_match_the_oracle_at_their_own_precision():
    """The backward sums in fp32 registers but stores in the input dtype, so
    bf16 is where that store is the only rounding between the two."""
    query, key, value, cu = build_case(dtype=torch.bfloat16, head_dim=512)
    wide = [
        query.float().detach().requires_grad_(),
        key.float().detach().requires_grad_(),
        value.float().detach().requires_grad_(),
    ]
    upstream = torch.randn_like(query)

    got = torch.autograd.grad(
        wide_varlen_attention(query, key, value, cu, sm_scale=SCALE), [query, key, value], upstream
    )
    want = torch.autograd.grad(wide_varlen_attention_reference(*wide, cu, SCALE), wide, upstream.float())

    for name, mine, theirs in zip(("query", "key", "value"), got, want):
        assert mine.dtype == torch.bfloat16, f"{name} came back in the wrong dtype"
        assert worst_relative_in_group(mine.float(), theirs, want) < BFLOAT16_TOLERANCE, (
            f"{name} disagrees with the oracle"
        )


@requires_cuda
def test_float16_stays_finite_through_the_backward():
    """fp16 is accepted, so it is exercised. Its exponent range is far narrower
    than bf16's, and the backward's `probabilities * (dp - delta) * scale` is
    cast to the input dtype before it is stored -- so overflow to inf is a real
    failure mode here that bf16 would never show."""
    query, key, value, cu = build_case(dtype=torch.float16, head_dim=128)
    output = wide_varlen_attention(query, key, value, cu, sm_scale=SCALE)
    want = wide_varlen_attention_reference(query.float(), key.float(), value.float(), cu, SCALE)

    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    assert worst_relative(output.float(), want) < BFLOAT16_TOLERANCE

    gradients = torch.autograd.grad(output.sum(), [query, key, value])
    for name, gradient in zip(("query", "key", "value"), gradients):
        assert gradient.dtype == torch.float16, f"{name} came back in the wrong dtype"
        assert torch.isfinite(gradient).all(), f"{name} gradient overflowed fp16"


@requires_cuda
def test_a_token_cannot_see_the_one_after_it():
    """Causality inside a segment. The oracle would agree with a kernel that
    read one key too far only if it read one too far as well; this asks the
    kernel alone."""
    query, key, value, cu = build_case()
    baseline = wide_varlen_attention(query, key, value, cu, sm_scale=SCALE)

    position = 5
    disturbed = key.detach().clone()
    disturbed[:, :, position + 1] += 10.0
    moved = wide_varlen_attention(query, disturbed, value, cu, sm_scale=SCALE)

    torch.testing.assert_close(moved[:, :, : position + 1], baseline[:, :, : position + 1])
    # Its own key is inside the causal set, so perturbing that must move it.
    inside = key.detach().clone()
    inside[:, :, position] += 10.0
    shifted = wide_varlen_attention(query, inside, value, cu, sm_scale=SCALE)
    assert not torch.allclose(shifted[:, :, position], baseline[:, :, position])


@requires_cuda
def test_a_segment_cannot_see_another_segment():
    """The boundary packing exists to hold. Only the perturbed segment's own
    outputs may move -- including the segments *before* it, which a kernel that
    forgot to offset its keys would disturb."""
    lengths = (12, 20, 5)
    query, key, value, cu = build_case(lengths=lengths)
    baseline = wide_varlen_attention(query, key, value, cu, sm_scale=SCALE)

    bounds = [int(x) for x in cu.tolist()]
    target = 1
    disturbed = key.detach().clone()
    disturbed[:, :, bounds[target] : bounds[target + 1]] += 10.0
    moved = wide_varlen_attention(query, disturbed, value, cu, sm_scale=SCALE)

    for segment in range(len(lengths)):
        span = slice(bounds[segment], bounds[segment + 1])
        if segment == target:
            assert not torch.allclose(moved[:, :, span], baseline[:, :, span])
        else:
            torch.testing.assert_close(moved[:, :, span], baseline[:, :, span])


@requires_cuda
def test_a_trusted_max_seqlen_gives_the_same_answer_as_a_read_one():
    """The production path passes `max_seqlen` to skip a host synchronisation.
    It must not be a different computation from the path the tests above take."""
    query, key, value, cu = build_case(lengths=(96, 3, 17, 1, 40))
    trusted = wide_varlen_attention(query, key, value, cu, max_seqlen=96, sm_scale=SCALE)
    read = wide_varlen_attention(query, key, value, cu, sm_scale=SCALE)

    torch.testing.assert_close(trusted, read)


@requires_cuda
def test_an_overstated_max_seqlen_is_harmless():
    """Too large only costs empty programs that return before doing anything."""
    query, key, value, cu = build_case()
    generous = wide_varlen_attention(query, key, value, cu, max_seqlen=256, sm_scale=SCALE)
    exact = wide_varlen_attention(query, key, value, cu, sm_scale=SCALE)

    torch.testing.assert_close(generous, exact)


@requires_cuda
def test_an_understated_max_seqlen_truncates_instead_of_summing_rows_nobody_computed():
    """The direction that is not safe, held to the bound the kernel promises.

    `max_seqlen` sizes the forward and `dq` grids, so understating it leaves the
    rows past it uncomputed. `dkdv` walks its queries by *segment length*
    instead, so unless all three kernels clamp to `max_seqlen` alike, it folds
    those uncomputed rows -- their `lse` and `delta` never written -- into the
    dK/dV of key tiles that are perfectly valid.

    The assertion has to be on the gradients, and this is the subtle part: under
    a causal mask row `i` reads only keys up to `i`, so every row the forward
    *did* compute has the same value truncated or not. The forward output cannot
    distinguish the two, and neither can finiteness, which only fires when the
    allocator happens to hand back a dirty page. The backward is where the
    spurious rows actually land.
    """
    lengths = (96, 40)
    cut = 32
    query, key, value, cu = build_case(lengths=lengths)
    upstream_scale = 1.0

    truncated = wide_varlen_attention(query, key, value, cu, max_seqlen=cut, sm_scale=SCALE)
    got = torch.autograd.grad(truncated.sum() * upstream_scale, [query, key, value])

    # The same computation, expressed as a row that really is cut to `cut`.
    short = [min(length, cut) for length in lengths]
    kept = torch.cat([torch.arange(start, start + keep) for start, keep in zip((0, lengths[0]), short)]).to(cu.device)
    short_cu = torch.zeros(len(short) + 1, dtype=torch.int32, device=cu.device)
    short_cu[1:] = torch.tensor(short, dtype=torch.int32, device=cu.device).cumsum(0)
    leaves = [tensor[:, :, kept].detach().clone().requires_grad_() for tensor in (query, key, value)]
    expected = wide_varlen_attention(*leaves, short_cu, max_seqlen=cut, sm_scale=SCALE)
    want = torch.autograd.grad(expected.sum() * upstream_scale, leaves)

    assert torch.isfinite(truncated).all()
    torch.testing.assert_close(truncated[:, :, kept], expected)

    dropped = torch.ones(query.shape[2], dtype=torch.bool, device=cu.device)
    dropped[kept] = False
    for name, mine, theirs in zip(("query", "key", "value"), got, want):
        assert torch.isfinite(mine).all(), f"{name} gradient went non-finite"
        assert worst_relative(mine[:, :, kept], theirs) < FLOAT32_TOLERANCE, (
            f"{name} gradient picked up rows past max_seqlen"
        )
        assert not mine[:, :, dropped].abs().any(), f"{name} gradient reached past the truncation"


@requires_cuda
def test_a_row_of_only_empty_segments_still_returns_a_differentiable_zero():
    """`cdiv(0, tile)` is 0, so no program runs. The layer must still be part of
    the graph -- a detached zero here would leave this layer's q/k/v with no
    gradient while every other layer kept theirs."""
    query, key, value, cu = build_case(lengths=(0, 0, 0))
    output = wide_varlen_attention(query, key, value, cu, max_seqlen=0, sm_scale=SCALE)

    assert output.grad_fn is not None
    assert not output.abs().any()
    for gradient in torch.autograd.grad(output.sum(), [query, key, value]):
        assert not gradient.abs().any()


#: Prefix sums the trusted path cannot reject without the synchronisation it
#: exists to avoid, so the kernel clamps them instead. Each stays inside the row
#: once clamped, and each keeps the segments disjoint -- an overlapping one is
#: excluded on purpose, since two programs would then own one key tile.
MALFORMED_PREFIX_SUMS = [
    pytest.param(lambda cu, total: torch.tensor([0, 12, 32, total + 64], dtype=torch.int32), id="past-the-row-end"),
    pytest.param(lambda cu, total: torch.tensor([0, 12, 12, total], dtype=torch.int32), id="repeated-bound"),
    pytest.param(lambda cu, total: torch.tensor([8, 12, 32, total], dtype=torch.int32), id="does-not-start-at-zero"),
    pytest.param(lambda cu, total: torch.tensor([0, 12, 32, total - 7], dtype=torch.int32), id="stops-short"),
    pytest.param(lambda cu, total: torch.tensor([-5, 12, 32, total], dtype=torch.int32), id="negative-bound"),
]


@requires_cuda
@pytest.mark.parametrize("build_prefix_sum", MALFORMED_PREFIX_SUMS)
def test_a_malformed_prefix_sum_on_the_trusted_path_is_clamped_not_read_out_of_bounds(build_prefix_sum):
    """The clamp is what makes trusting `cu_seqlens` tolerable, so it is tested.

    On the trusted path a malformed prefix sum cannot be rejected -- reading it
    is the synchronisation the path exists to avoid -- so `_segment_bounds`
    clamps every bound into the row and the result is a defined wrong answer.
    `wide_varlen_attention_reference` clamps identically, which makes it the
    oracle for exactly this: agreement means the kernel read the same clamped
    span, and finiteness means it read nothing outside the row at all.
    """
    query, key, value, cu = build_case(lengths=(12, 20, 5))
    total = query.shape[2]
    malformed = build_prefix_sum(cu, total).to(cu.device)

    got = wide_varlen_attention(query, key, value, malformed, max_seqlen=total, sm_scale=SCALE)
    want = wide_varlen_attention_reference(query, key, value, malformed, SCALE)

    assert torch.isfinite(got).all()
    assert worst_relative(got, want) < FLOAT32_TOLERANCE


class TestTheKernelRefusesWhatItCannotHonour:
    def test_cpu_tensors_are_refused_rather_than_run(self):
        query = torch.randn(1, 4, 32, 64)
        cu = torch.tensor([0, 16, 32], dtype=torch.int32)
        with pytest.raises(ValueError, match="CUDA"):
            wide_varlen_attention(query, query[:, :2], query[:, :2], cu, sm_scale=SCALE)

    @requires_cuda
    def test_a_missing_scale_is_refused(self):
        query, key, value, cu = build_case()
        with pytest.raises(ValueError, match="sm_scale"):
            wide_varlen_attention(query, key, value, cu)

    @requires_cuda
    def test_a_second_packed_row_is_refused(self):
        query, key, value, cu = build_case()
        with pytest.raises(ValueError, match="one packed row"):
            wide_varlen_attention(
                query.repeat(2, 1, 1, 1), key.repeat(2, 1, 1, 1), value.repeat(2, 1, 1, 1), cu, sm_scale=SCALE
            )

    @requires_cuda
    def test_a_cache_shorter_than_the_query_is_refused(self):
        query, key, value, cu = build_case()
        with pytest.raises(ValueError, match="one entry per query token"):
            wide_varlen_attention(query, key[:, :, :-4], value[:, :, :-4], cu, sm_scale=SCALE)

    @requires_cuda
    def test_a_head_count_that_does_not_group_is_refused(self):
        query, key, value, cu = build_case(heads_q=4, heads_kv=2)
        with pytest.raises(ValueError, match="multiple of kv heads"):
            wide_varlen_attention(query[:, :3], key, value, cu, sm_scale=SCALE)

    @requires_cuda
    def test_a_head_wider_than_the_kernel_accumulates_is_refused(self):
        query, key, value, cu = build_case(head_dim=1024, lengths=(8, 8))
        with pytest.raises(ValueError, match="head_dim up to 512"):
            wide_varlen_attention(query, key, value, cu, sm_scale=SCALE)

    @requires_cuda
    def test_a_float64_input_is_refused_rather_than_silently_narrowed(self):
        query, key, value, cu = build_case(dtype=torch.float64)
        with pytest.raises(ValueError, match="fp16/bf16/fp32"):
            wide_varlen_attention(query, key, value, cu, sm_scale=SCALE)

    @requires_cuda
    def test_a_non_int32_prefix_sum_is_refused(self):
        query, key, value, cu = build_case()
        with pytest.raises(ValueError, match="int32"):
            wide_varlen_attention(query, key, value, cu.long(), sm_scale=SCALE)

    @requires_cuda
    def test_a_prefix_sum_that_does_not_cover_the_row_is_refused_when_it_is_read(self):
        """Only on the path that already synchronises -- the trusted path cannot
        see this without putting that synchronisation back."""
        query, key, value, cu = build_case()
        with pytest.raises(ValueError, match="must end at the packed length"):
            wide_varlen_attention(query, key, value, cu[:-1], sm_scale=SCALE)

    @requires_cuda
    def test_a_decreasing_prefix_sum_is_refused_when_it_is_read(self):
        query, key, value, cu = build_case()
        broken = cu.clone()
        broken[1] = cu[2] + 1
        with pytest.raises(ValueError, match="non-decreasing"):
            wide_varlen_attention(query, key, value, broken, sm_scale=SCALE)
