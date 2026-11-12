"""The expert-parallel token permutation has to give the same answer twice, both ways.

A token is sent to each of its ``top_k`` experts, so the permutation index carries
``top_k`` duplicates, and each direction has a sum over them: the unpermute adds a
token's expert outputs, and the permute's backward adds the gradient of its copies.
Spelled the obvious way -- ``scatter_add_`` on the token index, or letting autograd
invert a duplicated ``index_select`` -- CUDA serves both with atomics and the order
changes from call to call.

On the 26B-A4B recipe that was not a last-bit curiosity. Two forwards of one batch first
diverged at an expert output by a single bfloat16 step, and the residual stack turned
that into 0.25 one layer later; the backward is coarser by six orders of magnitude,
since the gradient is bfloat16 and every atomic add rounds to it, so eight values summed
in a different order landed up to 1.25e-1 apart -- a few percent of the gradient, on 614
of 631 parameters. Two runs on identical data with frozen weights reported ``grad_norm``
anywhere between 380 and 3300, and ``enable_full_determinism`` did not change that: only
the 1-D CUDA form of ``scatter_add_`` raises, and the ``index_add`` behind a gather's
backward is never checked at all.

Both directions are therefore tested for reproducibility, and both against a float64
oracle, because the grouping that buys the determinism is only correct while every token
has exactly ``top_k`` entries.

Which parametrisation actually gates which direction is not symmetric, and the tests say
so where it matters: a bfloat16 forward cannot catch a revert of the unpermute, because
the cast absorbs the difference, while the permute's backward is caught in every dtype
because its atomics accumulated in bfloat16 to begin with.
"""

import pytest
import torch

from veomni.distributed.moe.moe_utils import permute, unpermute
from veomni.utils.device import IS_CUDA_AVAILABLE


# The 26B-A4B recipe's routing width, for fidelity rather than for the tests to bite.
NUM_EXPERTS, TOP_K, HIDDEN = 128, 8, 128

DEVICES = ["cpu"] + (["cuda"] if IS_CUDA_AVAILABLE else [])

# The recipe runs bfloat16 and the fused entry point accepts only bfloat16 or float16,
# so those two are the production widths. float32 is here for a reason of its own, and
# the bit-for-bit tests below depend on it: see the note there.
UNIT_ROUNDOFF = {torch.float32: 2**-24, torch.bfloat16: 2**-8, torch.float16: 2**-11}

# The sum accumulates in FP32 whatever it was handed, so that is the width that bounds
# the accumulation error, and the output dtype only pays for the cast back.
ACCUMULATOR_UNIT_ROUNDOFF = 2**-24


def build_routing(num_tokens: int, device: str, dtype: torch.dtype = torch.bfloat16):
    """The tensors the expert-parallel path builds, in its entry order."""
    generator = torch.Generator(device=device).manual_seed(0)
    logits = torch.randn(num_tokens, NUM_EXPERTS, device=device, generator=generator)
    weights, experts = torch.topk(torch.softmax(logits, dim=-1), TOP_K, dim=-1)

    routing_map = torch.zeros(num_tokens, NUM_EXPERTS, device=device)
    routing_map.scatter_(1, experts, 1.0)
    routing_map = routing_map.T.contiguous()

    weights_idx = torch.zeros(num_tokens, NUM_EXPERTS, device=device, dtype=dtype)
    weights_idx.scatter_(1, experts, weights.to(dtype))

    hidden = torch.randn(num_tokens, HIDDEN, device=device, dtype=torch.float32, generator=generator).to(dtype)
    entries, mapping = permute(hidden, routing_map, TOP_K)
    return hidden, entries, weights_idx, mapping, routing_map


def oracle_sum_per_token(values, mapping, num_tokens):
    """Each token's sum, taken entry by entry in float64 so the order cannot matter."""
    out = torch.zeros(num_tokens, values.shape[-1], dtype=torch.float64, device=values.device)
    for position, token in enumerate(mapping.tolist()):
        out[token] += values[position].double()
    return out


def assert_within_rounding_budget(actual, expected, terms, mapping, num_tokens, dtype):
    """Hold the sum to the error a correct one can commit, element by element.

    A single tolerance cannot do this. Pinning ``atol`` to the largest value in the
    tensor is generous enough to hide an accumulator that dropped to bfloat16 -- it did,
    when this was tried -- while a bare ``rtol`` is unmeetable for a token whose entries
    cancelled, since the answer is then small but the roundings that produced it are not.

    The standard bound has both parts: ``top_k - 1`` additions each round to the FP32
    accumulator, in proportion to the terms actually summed rather than to their total,
    and the cast back to ``dtype`` rounds once more against the result. Which part binds
    depends on the width -- the cast dominates in bfloat16, the accumulation in float32 --
    so neither is redundant across the dtypes tested, though only one bites at a time.
    """
    scale = oracle_sum_per_token(terms.abs(), mapping, num_tokens)
    budget = UNIT_ROUNDOFF[dtype] * expected.abs() + (TOP_K - 1) * ACCUMULATOR_UNIT_ROUNDOFF * scale

    excess = (actual.double() - expected).abs() - budget
    worst = excess.max().item()
    assert worst <= 0, (
        f"the sum is off by more than {TOP_K - 1} FP32 roundings plus a cast to {dtype}: worst element exceeds "
        f"its budget by {worst:.3e}, so it is accumulating at the wrong width or over the wrong entries"
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", list(UNIT_ROUNDOFF))
@pytest.mark.parametrize("num_tokens", [64, 257])
def test_the_unpermute_sums_each_token_s_own_entries(device, dtype, num_tokens):
    """The grouping is only valid if the sorted entries line up with their tokens."""
    hidden, _, weights_idx, mapping, routing_map = build_routing(num_tokens, device, dtype)

    # Not the entries ``permute`` produced. Those are ``top_k`` copies of one row scaled
    # by positive weights, so a token's terms all share a sign and nothing cancels --
    # the easiest case there is for any accumulator. In production the entries are
    # separate experts' outputs, which is what this draws.
    entries = torch.randn(mapping.numel(), HIDDEN, device=device, dtype=torch.float32).to(dtype)
    out = unpermute(entries, weights_idx, hidden.shape, mapping, routing_map, TOP_K)

    weights = weights_idx.T.contiguous().masked_select(routing_map.bool())
    # The weighting happens in ``dtype`` in the implementation too, so the oracle differs
    # from it only in the order and the width of the accumulation.
    terms = entries * weights.unsqueeze(-1)
    expected = oracle_sum_per_token(terms, mapping, num_tokens)

    assert out.shape == (num_tokens, HIDDEN)
    assert_within_rounding_budget(out, expected, terms, mapping, num_tokens, dtype)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", list(UNIT_ROUNDOFF))
@pytest.mark.parametrize("num_tokens", [64, 257])
def test_the_permute_backward_sums_each_token_s_own_copies(device, dtype, num_tokens):
    hidden, _, _, mapping, routing_map = build_routing(num_tokens, device, dtype)
    upstream = torch.randn(mapping.numel(), HIDDEN, device=device, dtype=torch.float32).to(dtype)

    leaf = hidden.detach().clone().requires_grad_(True)
    permute(leaf, routing_map, TOP_K)[0].backward(upstream)

    expected = oracle_sum_per_token(upstream, mapping, num_tokens)
    assert_within_rounding_budget(leaf.grad, expected, upstream, mapping, num_tokens, dtype)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="the accumulation order only varies on device")
@pytest.mark.parametrize("dtype", list(UNIT_ROUNDOFF))
@pytest.mark.parametrize("num_tokens", [512, 1024])
def test_the_unpermute_agrees_bit_for_bit(dtype, num_tokens):
    """float32 is the parametrisation that can fail; bfloat16 is here to record that.

    Measured on one A800 over 16 repeats: the ``scatter_add_`` this replaced disagreed
    with itself in every float32 configuration tried, by 1e-6 or so, and in no bfloat16
    one -- the accumulator is FP32 either way, and one cast to bfloat16 discards a
    difference that small. So a bfloat16 forward cannot catch a revert here, and saying
    otherwise would misrepresent the gate. The real model still diverged in bfloat16, and
    reverting this reduction is what stopped it, but that has 30 layers on 8 ranks to
    accumulate chances with and this test has one call.
    """
    hidden, entries, weights_idx, mapping, routing_map = build_routing(num_tokens, "cuda", dtype)
    args = (entries, weights_idx, hidden.shape, mapping, routing_map, TOP_K)

    first = unpermute(*args)
    for _ in range(16):
        assert torch.equal(unpermute(*args), first), (
            "unpermute returned a different sum for the same input, so a step's loss no longer follows "
            "from its data and weights"
        )


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="the accumulation order only varies on device")
@pytest.mark.parametrize("dtype", list(UNIT_ROUNDOFF))
@pytest.mark.parametrize("num_tokens", [512, 1024])
def test_the_permute_backward_agrees_bit_for_bit(dtype, num_tokens):
    hidden, _, _, mapping, routing_map = build_routing(num_tokens, "cuda", dtype)
    upstream = torch.randn(mapping.numel(), HIDDEN, device="cuda", dtype=dtype)

    def gradient():
        leaf = hidden.detach().clone().requires_grad_(True)
        permute(leaf, routing_map, TOP_K)[0].backward(upstream)
        return leaf.grad

    first = gradient()
    for _ in range(16):
        assert torch.equal(gradient(), first), (
            "permute handed back a different gradient for the same upstream, so the optimizer does not "
            "follow from the step's data and weights"
        )


@pytest.mark.parametrize("device", DEVICES)
def test_a_permutation_it_cannot_group_is_refused(device):
    """Fewer entries than ``num_tokens * top_k`` would silently shift the grouping."""
    hidden, entries, weights_idx, mapping, routing_map = build_routing(64, device)

    with pytest.raises(ValueError, match="exactly top_k"):
        unpermute(entries[:-TOP_K], weights_idx, hidden.shape, mapping[:-TOP_K], routing_map, TOP_K)

    with pytest.raises(ValueError, match="exactly top_k"):
        unpermute(entries, weights_idx, hidden.shape, mapping, routing_map, TOP_K + 1)


@pytest.mark.parametrize("device", DEVICES)
def test_a_top_k_the_routing_contradicts_is_refused_before_the_experts_run(device):
    """The permute has to reject it going in, not on the way back.

    Its backward checks the same thing, but by then every rank has entered the expert
    all-to-alls, and one rank leaving a collective to raise hangs the rest until a
    watchdog fires -- which reports a timeout somewhere else entirely.
    """
    hidden, _, _, _, routing_map = build_routing(64, device)
    leaf = hidden.detach().clone().requires_grad_(True)

    with pytest.raises(ValueError, match="exactly top_k"):
        permute(leaf, routing_map, TOP_K + 1)
