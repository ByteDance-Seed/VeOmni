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
"""A packed encoder row must compute what the separate rows computed.

That is the whole claim of packing, so it is asserted elementwise rather than
inferred from a forward finishing. The oracle is `F.scaled_dot_product_attention`
run on one sample at a time: it never sees two samples at once, so it cannot
reproduce a cross-sample read even in principle -- which a block-diagonal mask
through the same backend could.

Both leak directions are probed on top of that, because an oracle agreeing to
2e-3 in bf16 leaves room for a small leak to hide, whereas perturbing one sample
and requiring the others not to move does not.

The window convention is pinned against the mask transformers actually builds
rather than against `sliding_window` read as a number. `create_sliding_window_
causal_mask` lets a query see `sliding_window` keys *including itself*, while
flash's `window_size` counts the ones besides it, and an off-by-one between the
two is a silent accuracy change that no shape check would catch.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from veomni.models.transformers.diffusion_gemma.dg_varlen import (
    block_diagonal_causal_mask,
    encoder_segments,
    should_use_varlen_encoder,
    varlen_encoder_attention,
)


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="both varlen backends are CUDA only")

#: bf16 keeps ~8 mantissa bits, and the packed and unpacked runs reduce in
#: different orders, so this measures the kernels rather than the arithmetic.
BFLOAT16_TOLERANCE = 5e-2
#: A leak is not a rounding difference. One sample moving another must show up
#: far above the noise, or the probe is not a probe.
LEAK_TOLERANCE = 1e-3

SCALE = 1.0

#: The two geometries the encoder actually has: 25 layers sliding at 256 with a
#: window, 5 layers full at 512 with none.
SLIDING = {"heads": 8, "kv_heads": 4, "head_dim": 256, "sliding_window": 512}
GLOBAL = {"heads": 16, "kv_heads": 2, "head_dim": 512, "sliding_window": None}

LENGTH_SETS = [
    pytest.param([600, 600, 600], id="equal"),
    pytest.param([1024, 613, 97, 5], id="ragged"),
    pytest.param([777], id="one-sample-is-not-a-special-case"),
    pytest.param([1, 1, 1, 1], id="single-token-samples"),
    pytest.param([1200, 1], id="a-long-sample-beside-a-token"),
    # Longer than the sliding window, so the window has to bite inside a segment
    # rather than only at its start.
    pytest.param([900, 1500], id="both-past-the-window"),
]


def build(lengths, geometry, dtype=torch.bfloat16, seed=0):
    """One packed row, plus the positions that describe its segment boundaries."""
    torch.manual_seed(seed)
    total = sum(lengths)

    def tensor(heads):
        return torch.randn(1, heads, total, geometry["head_dim"], dtype=dtype, device="cuda", requires_grad=True)

    positions = torch.cat([torch.arange(length, device="cuda") for length in lengths]).unsqueeze(0)
    return tensor(geometry["heads"]), tensor(geometry["kv_heads"]), tensor(geometry["kv_heads"]), positions


def unpacked_oracle(query, key, value, lengths, geometry):
    """sdpa on one sample at a time, in fp32, with the window written out."""
    window = geometry["sliding_window"]
    outputs = []
    start = 0
    for length in lengths:
        span = slice(start, start + length)
        positions = torch.arange(length, device=query.device)
        keep = positions[:, None] >= positions[None, :]
        if window:
            keep &= positions[:, None] - positions[None, :] < window
        outputs.append(
            F.scaled_dot_product_attention(
                query[:, :, span].float(),
                key[:, :, span].float(),
                value[:, :, span].float(),
                attn_mask=keep[None, None],
                scale=SCALE,
                enable_gqa=key.shape[1] != query.shape[1],
            )
        )
        start += length
    return torch.cat(outputs, dim=2).transpose(1, 2)


def run_packed(query, key, value, positions, geometry):
    cu_seqlens, max_seqlen = encoder_segments(positions)
    return varlen_encoder_attention(
        query,
        key,
        value,
        cu_seqlens,
        max_seqlen,
        head_dim=geometry["head_dim"],
        sliding_window=geometry["sliding_window"],
        sm_scale=SCALE,
    )


def worst_relative(got, want):
    scale = want.abs().max()
    if scale == 0:
        return got.abs().max().item()
    return ((got - want).abs().max() / scale).item()


class TestSegmentsComeOffThePositions:
    def test_a_packed_row_yields_one_entry_per_sample(self):
        positions = torch.cat([torch.arange(n) for n in (4, 7, 1)]).unsqueeze(0)
        cu_seqlens, max_seqlen = encoder_segments(positions)

        assert cu_seqlens.tolist() == [0, 4, 11, 12]
        assert cu_seqlens.dtype == torch.int32
        assert max_seqlen == 7

    def test_an_unpacked_row_is_one_segment_rather_than_an_error(self):
        """The ordinary encoder row is the S=1 case of a packed one, so it must
        pass through unchanged -- otherwise every caller needs a branch."""
        cu_seqlens, max_seqlen = encoder_segments(torch.arange(12).unsqueeze(0))

        assert cu_seqlens.tolist() == [0, 12]
        assert max_seqlen == 12

    def test_positions_that_do_not_start_a_sample_are_refused(self):
        """Left padding is what this looks like, and it has no place in a packed
        row: the pad tokens would become a sample of their own."""
        with pytest.raises(ValueError, match="must start a sample at index 0"):
            encoder_segments(torch.arange(1, 9).unsqueeze(0))

    def test_positions_that_restart_but_skip_are_refused(self):
        """The dangerous shape: boundaries look right, RoPE was applied from
        positions describing a different sequence."""
        positions = torch.tensor([[0, 1, 2, 0, 1, 3]])
        with pytest.raises(ValueError, match="0..L-1 within each sample"):
            encoder_segments(positions)

    def test_positions_that_repeat_within_a_sample_are_refused(self):
        positions = torch.tensor([[0, 1, 1, 2]])
        with pytest.raises(ValueError, match="0..L-1 within each sample"):
            encoder_segments(positions)

    def test_a_second_row_is_refused(self):
        with pytest.raises(ValueError, match="one row"):
            encoder_segments(torch.arange(8).view(2, 4))

    def test_an_empty_row_is_refused(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            encoder_segments(torch.zeros(1, 0, dtype=torch.long))


class TestTheMaskAgreesWithTransformers:
    """The fallback mask and the kernels must implement one convention, and it
    has to be the one the unpacked encoder already uses."""

    @pytest.mark.parametrize("window", [None, 1, 2, 8, 64])
    def test_the_window_counts_the_query_among_the_keys_it_may_see(self, window):
        cu_seqlens = torch.tensor([0, 40], dtype=torch.int32)
        keep = block_diagonal_causal_mask(cu_seqlens, 40, sliding_window=window)
        counts = keep[0, 0].sum(-1)

        # Query i sees keys [max(0, i - window + 1), i], so the count climbs to
        # the window and then holds.
        expected = min(window, 40) if window else 40
        assert counts[-1].item() == expected
        assert counts[0].item() == 1

    def test_the_mask_matches_what_create_sliding_window_causal_mask_builds(self):
        """Pinned against transformers rather than against the docstring, since
        this convention is the one flash's `window_size` is derived from."""
        from transformers.masking_utils import create_sliding_window_causal_mask

        from veomni.models.transformers.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaTextConfig

        length, window = 40, 8
        config = DiffusionGemmaTextConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            global_head_dim=32,
            sliding_window=window,
            intermediate_size=128,
        )
        config._attn_implementation = "sdpa"
        theirs = create_sliding_window_causal_mask(
            config=config,
            inputs_embeds=torch.zeros(1, length, config.hidden_size),
            attention_mask=None,
            past_key_values=None,
            position_ids=torch.arange(length).unsqueeze(0),
        )
        mine = block_diagonal_causal_mask(torch.tensor([0, length], dtype=torch.int32), length, window)

        assert torch.equal(mine, theirs.bool())

    def test_segments_cannot_see_each_other(self):
        keep = block_diagonal_causal_mask(torch.tensor([0, 3, 7], dtype=torch.int32), 7)

        assert not keep[0, 0, :3, 3:].any(), "the first sample reached the second"
        assert not keep[0, 0, 3:, :3].any(), "the second sample reached the first"

    def test_a_prefix_sum_that_does_not_cover_the_row_is_refused(self):
        with pytest.raises(ValueError, match="must end at the packed length"):
            block_diagonal_causal_mask(torch.tensor([0, 3], dtype=torch.int32), 7)

    def test_a_window_inside_a_pack_is_measured_from_the_sample_not_the_row(self):
        """The combination the two single-purpose tests above leave out: with
        several segments *and* a window, an implementation measuring the window
        from the row's start rather than the sample's would still box correctly
        and still be wrong inside the box."""
        lengths = [10, 6]
        window = 3
        cu_seqlens = torch.tensor([0, 10, 16], dtype=torch.int32)
        keep = block_diagonal_causal_mask(cu_seqlens, sum(lengths), sliding_window=window)[0, 0]

        offset = 0
        for length in lengths:
            for query in range(length):
                visible = keep[offset + query].nonzero().flatten() - offset
                expected = list(range(max(0, query - window + 1), query + 1))
                assert visible.tolist() == expected, f"query {query} of a {length}-token sample"
            offset += length


@requires_cuda
class TestThePackedRowMatchesTheSeparateRows:
    @pytest.mark.parametrize("lengths", LENGTH_SETS)
    @pytest.mark.parametrize("geometry,name", [(SLIDING, "sliding"), (GLOBAL, "global")])
    def test_the_forward_matches(self, lengths, geometry, name):
        query, key, value, positions = build(lengths, geometry)
        got = run_packed(query, key, value, positions, geometry)
        want = unpacked_oracle(query, key, value, lengths, geometry)

        assert got.shape == want.shape
        assert worst_relative(got.float(), want) < BFLOAT16_TOLERANCE, f"{name} layer disagrees"

    @pytest.mark.parametrize("lengths", LENGTH_SETS)
    @pytest.mark.parametrize("geometry,name", [(SLIDING, "sliding"), (GLOBAL, "global")])
    def test_every_gradient_matches(self, lengths, geometry, name):
        query, key, value, positions = build(lengths, geometry)
        upstream = torch.randn_like(run_packed(query, key, value, positions, geometry))

        got = torch.autograd.grad(run_packed(query, key, value, positions, geometry), [query, key, value], upstream)
        wide = [tensor.float().detach().requires_grad_() for tensor in (query, key, value)]
        want = torch.autograd.grad(unpacked_oracle(*wide, lengths, geometry), wide, upstream.float())

        for tensor_name, mine, theirs in zip(("query", "key", "value"), got, want):
            assert torch.isfinite(mine).all(), f"{name}: {tensor_name} gradient went non-finite"
            assert worst_relative(mine.float(), theirs) < BFLOAT16_TOLERANCE, (
                f"{name}: {tensor_name} gradient disagrees"
            )


@requires_cuda
class TestNoSampleReadsAnother:
    """Perturbation probes, which a tolerance cannot absorb."""

    @pytest.mark.parametrize("geometry,name", [(SLIDING, "sliding"), (GLOBAL, "global")])
    @pytest.mark.parametrize("target", [0, 1, 2])
    def test_moving_one_sample_leaves_the_others_where_they_were(self, geometry, name, target):
        lengths = [300, 400, 200]
        query, key, value, positions = build(lengths, geometry)
        baseline = run_packed(query, key, value, positions, geometry)

        bounds = [0]
        for length in lengths:
            bounds.append(bounds[-1] + length)
        span = slice(bounds[target], bounds[target + 1])

        moved_key = key.detach().clone()
        moved_key[:, :, span] += 3.0
        moved = run_packed(query, moved_key, value, positions, geometry)

        for index in range(len(lengths)):
            other = slice(bounds[index], bounds[index + 1])
            if index == target:
                assert not torch.allclose(moved[:, other], baseline[:, other]), (
                    f"{name}: perturbing sample {target} did not change it, so the probe proves nothing"
                )
            else:
                assert worst_relative(moved[:, other].float(), baseline[:, other].float()) < LEAK_TOLERANCE, (
                    f"{name}: sample {index} moved when sample {target} did"
                )

    @pytest.mark.parametrize("geometry,name", [(SLIDING, "sliding"), (GLOBAL, "global")])
    def test_a_token_cannot_see_the_token_after_it(self, geometry, name):
        lengths = [64, 64]
        query, key, value, positions = build(lengths, geometry)
        baseline = run_packed(query, key, value, positions, geometry)

        moved_key = key.detach().clone()
        moved_key[:, :, 40:] += 3.0
        moved = run_packed(query, moved_key, value, positions, geometry)

        assert worst_relative(moved[:, :40].float(), baseline[:, :40].float()) < LEAK_TOLERANCE, (
            f"{name}: causality leaked backwards"
        )
        assert not torch.allclose(moved[:, 40:], baseline[:, 40:])


@requires_cuda
class TestTheWindowActuallyBites:
    """The sliding backend must lose the far past, or it is a full-attention
    layer wearing a window's name and 25 of 30 layers are wrong."""

    def test_a_key_outside_the_window_does_not_reach_the_query(self):
        geometry = {**SLIDING, "sliding_window": 64}
        lengths = [512]
        query, key, value, positions = build(lengths, geometry)
        baseline = run_packed(query, key, value, positions, geometry)

        moved_key = key.detach().clone()
        moved_key[:, :, :100] += 3.0
        moved = run_packed(query, moved_key, value, positions, geometry)

        # Query 400 is 300 keys past the perturbation and its window is 64.
        assert worst_relative(moved[:, 400:].float(), baseline[:, 400:].float()) < LEAK_TOLERANCE, (
            "a key outside the window reached the query, so the window was ignored"
        )
        assert not torch.allclose(moved[:, :100], baseline[:, :100])

    def test_a_narrow_window_and_no_window_disagree(self):
        lengths = [400, 400]
        query, key, value, positions = build(lengths, {**SLIDING, "sliding_window": 32})
        narrow = run_packed(query, key, value, positions, {**SLIDING, "sliding_window": 32})
        unbounded = run_packed(query, key, value, positions, {**SLIDING, "sliding_window": None})

        assert not torch.allclose(narrow, unbounded)


class TestItRefusesWhatItCannotHonour:
    def test_a_sliding_layer_too_wide_for_flash_is_refused(self):
        """The dangerous combination: `wide_varlen_attention` has no window, so
        dispatching a wide sliding layer to it would silently attend over the
        whole prefix."""
        with pytest.raises(ValueError, match="no varlen backend"):
            varlen_encoder_attention(
                torch.zeros(1, 2, 4, 512),
                torch.zeros(1, 2, 4, 512),
                torch.zeros(1, 2, 4, 512),
                torch.tensor([0, 4], dtype=torch.int32),
                4,
                head_dim=512,
                sliding_window=512,
                sm_scale=SCALE,
            )

    def test_dropout_is_refused_rather_than_dropped(self):
        class Config:
            use_bidirectional_attention = None

        with pytest.raises(ValueError, match="dropout"):
            should_use_varlen_encoder(Config(), torch.device("cpu"), dropout=0.1)

    def test_a_bidirectional_encoder_is_refused(self):
        class Config:
            use_bidirectional_attention = "all"

        with pytest.raises(ValueError, match="bidirectional"):
            should_use_varlen_encoder(Config(), torch.device("cpu"), dropout=0.0)

    def test_a_cpu_falls_to_the_mask_rather_than_raising(self):
        """There is a correct slow path, so the CPU is allowed to take it."""

        class Config:
            use_bidirectional_attention = "vision"
            _attn_implementation = "sdpa"

        assert should_use_varlen_encoder(Config(), torch.device("cpu"), dropout=0.0) is False

    @pytest.mark.parametrize("implementation", ["eager", "flex_attention"])
    def test_the_mask_fallback_is_refused_to_backends_that_would_misread_it(self, implementation):
        """`block_diagonal_causal_mask` is boolean, and eager *adds* its mask to
        the scores -- so it would add 1.0 where the mask means keep and 0.0 where
        it means drop, and every sample would stay visible to every other. The
        decoder mask already carries this scar in `to_4d`."""

        class Config:
            use_bidirectional_attention = None
            _attn_implementation = implementation

        with pytest.raises(ValueError, match="only sdpa reads"):
            should_use_varlen_encoder(Config(), torch.device("cpu"), dropout=0.0)

    def test_sequence_parallelism_is_refused(self):
        """It shards the very axis the samples are packed onto."""

        class Config:
            use_bidirectional_attention = None
            _attn_implementation = "sdpa"

        class State:
            sp_enabled = True

        with mock.patch(
            "veomni.models.transformers.diffusion_gemma.dg_varlen.get_parallel_state", return_value=State()
        ):
            with pytest.raises(ValueError, match="sequence parallelism"):
                should_use_varlen_encoder(Config(), torch.device("cpu"), dropout=0.0)

    def test_a_missing_backend_names_itself(self):
        """A bare ImportError from a deferred load would name a module and leave
        the reader to work out which layer wanted it."""
        with mock.patch.dict("sys.modules", {"flash_attn": None}):
            import veomni.models.transformers.diffusion_gemma.dg_varlen as module

            saved = module._FLASH_VARLEN
            module._FLASH_VARLEN = None
            try:
                with pytest.raises((ImportError, TypeError, AttributeError)):
                    module._load_flash_varlen()
            finally:
                module._FLASH_VARLEN = saved

    @requires_cuda
    def test_float32_is_refused_where_only_flash_could_serve(self):
        """Otherwise the error comes from inside flash and names neither the
        layer width nor the dtype the model should have been run in."""
        wide_enough_for_neither = torch.zeros(1, 2, 4, 128, dtype=torch.float32, device="cuda")
        with pytest.raises(ValueError, match="no torch.float32 path"):
            varlen_encoder_attention(
                wide_enough_for_neither,
                wide_enough_for_neither,
                wide_enough_for_neither,
                torch.tensor([0, 4], dtype=torch.int32, device="cuda"),
                4,
                head_dim=128,
                sliding_window=None,
                sm_scale=SCALE,
            )


TOY_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "toy_config", "diffusion_gemma_toy")

#: Where the encoder reads the predicate, and so where a test forces the mask.
MASK_PATH = "veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma.should_use_varlen_encoder"


def toy_encoder(dtype=torch.bfloat16, all_wide=False):
    """The repository's toy encoder, optionally with every layer past flash.

    ``all_wide`` exists to reach `wide_varlen_attention` in fp32: the toy config's
    global layers are 64 wide, which flash serves and which therefore has no fp32
    path, so without widening them the precision control cannot run at all.
    """
    from veomni.models.transformers.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaConfig
    from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaEncoderTextModel

    config = DiffusionGemmaConfig.from_pretrained(TOY_CONFIG_PATH).get_text_config()
    config.attention_dropout = 0.0
    config._attn_implementation = "sdpa"
    if all_wide:
        config.layer_types = ["full_attention"] * config.num_hidden_layers
        config.global_head_dim = 512
    torch.manual_seed(0)
    model = DiffusionGemmaEncoderTextModel(config).to(device="cuda", dtype=dtype)
    return model.train(), config


def packed_row(config, lengths, seed=1):
    torch.manual_seed(seed)
    tokens = torch.randint(0, config.vocab_size, (1, sum(lengths)), device="cuda")
    positions = torch.cat([torch.arange(n, device="cuda") for n in lengths]).unsqueeze(0)
    return tokens, positions


def encode_separately(model, tokens, positions, lengths):
    """Each sample alone, through the same path, as a pack of one segment."""
    pieces, start = [], 0
    for size in lengths:
        pieces.append(
            model(
                input_ids=tokens[:, start : start + size],
                position_ids=positions[:, start : start + size],
                pack_samples=True,
                use_cache=True,
            ).last_hidden_state
        )
        start += size
    return torch.cat(pieces, dim=1)


@requires_cuda
class TestTheEncoderConsumesAPackedRow:
    """Whole-stack behaviour, where bf16 equivalence is *not* assertable.

    Packing changes how many rows each GEMM is handed, which moves the result by
    ~1e-4 in bf16 with no attention involved at all. `DiffusionGemmaTextRouter`
    then takes `torch.topk` over near-tied probabilities, so that last bit
    reassigns an expert and the token's output moves by O(1): measured over six
    toy layers, 0 tokens rerouted in layers 0-2 and 7 of 71 by layer 5, carrying
    the output difference to 3.3e-1.

    That is a property of packing a MoE model in bf16, not of these kernels --
    the block-diagonal mask path through sdpa moves by the same 3.3e-1. So the
    assertions here are the ones precision cannot wash out, and the numerical
    equivalence is pinned per kernel above, against an fp32 oracle with no MoE
    in the way.
    """

    @pytest.mark.parametrize("all_wide", [False, True], ids=["mixed-widths", "all-past-flash"])
    def test_a_sample_packed_beside_its_own_copy_gives_two_identical_halves(self, all_wide):
        """No oracle and no tolerance: whatever the arithmetic does to one half
        it must do to the other, so any difference is the row leaking across the
        boundary."""
        model, config = toy_encoder(all_wide=all_wide)
        length = 40
        tokens, positions = packed_row(config, [length])
        twinned = model(
            input_ids=torch.cat([tokens, tokens], dim=1),
            position_ids=torch.cat([positions, positions], dim=1),
            pack_samples=True,
            use_cache=True,
        ).last_hidden_state

        assert torch.equal(twinned[:, length:], twinned[:, :length])

    def test_the_halves_stay_identical_in_float32(self):
        """The same claim where rounding is 24 bits finer, so a leak too small to
        see in bf16 has nowhere left to hide."""
        model, config = toy_encoder(dtype=torch.float32, all_wide=True)
        length = 40
        tokens, positions = packed_row(config, [length])
        twinned = model(
            input_ids=torch.cat([tokens, tokens], dim=1),
            position_ids=torch.cat([positions, positions], dim=1),
            pack_samples=True,
            use_cache=True,
        ).last_hidden_state

        assert torch.equal(twinned[:, length:], twinned[:, :length])

    def test_the_kernels_diverge_no_further_than_the_mask_does(self):
        """The comparison that survives the amplification: run the same packed
        row through the kernels and through the block-diagonal mask, and ask
        whether the kernels drifted from the separate samples by more than sdpa
        did. Both are answering the same question, so a real kernel bug shows up
        as a ratio rather than as an absolute error."""
        lengths = [40, 23, 7, 1]
        model, config = toy_encoder(dtype=torch.float32, all_wide=True)
        tokens, positions = packed_row(config, lengths)

        with_kernels = model(
            input_ids=tokens, position_ids=positions, pack_samples=True, use_cache=True
        ).last_hidden_state
        with mock.patch(MASK_PATH, return_value=False):
            with_mask = model(
                input_ids=tokens, position_ids=positions, pack_samples=True, use_cache=True
            ).last_hidden_state
            separate = encode_separately(model, tokens, positions, lengths)

        kernel_drift = worst_relative(with_kernels.float(), separate.float())
        mask_drift = worst_relative(with_mask.float(), separate.float())

        assert kernel_drift < max(4 * mask_drift, 1e-5), (
            f"the kernels drifted {kernel_drift:.2e} where the mask drifted {mask_drift:.2e}"
        )

    def test_the_cache_holds_the_whole_packed_row(self):
        model, config = toy_encoder()
        lengths = [40, 23, 7, 1]
        tokens, positions = packed_row(config, lengths)
        out = model(input_ids=tokens, position_ids=positions, pack_samples=True, use_cache=True)

        for layer in range(config.num_hidden_layers):
            assert out.past_key_values.layers[layer].keys.shape[2] == sum(lengths)

    def test_gradients_reach_the_embedding_through_a_packed_row(self):
        """A packed forward that silently dropped the graph would still return
        the right numbers."""
        model, config = toy_encoder()
        tokens, positions = packed_row(config, [40, 23])
        out = model(input_ids=tokens, position_ids=positions, pack_samples=True, use_cache=True)
        (grad,) = torch.autograd.grad(out.last_hidden_state.float().pow(2).mean(), [model.embed_tokens.weight])

        assert grad is not None and grad.abs().sum() > 0

    def test_a_mask_the_caller_built_is_refused_rather_than_dropped(self):
        """`DiffusionGemmaEncoderModel` always passes a mapping, so a packed call
        arriving through the outer model would otherwise lose it silently -- and
        with it the bidirectional vision blocks the production checkpoint's
        `use_bidirectional_attention: "vision"` puts in one."""
        model, config = toy_encoder()
        tokens, positions = packed_row(config, [40, 23])

        with pytest.raises(ValueError, match="cannot honour the"):
            model(
                input_ids=tokens,
                position_ids=positions,
                attention_mask=torch.ones_like(tokens),
                pack_samples=True,
                use_cache=True,
            )

    def test_a_prebuilt_mask_mapping_is_refused_too(self):
        model, config = toy_encoder()
        tokens, positions = packed_row(config, [40, 23])

        with pytest.raises(ValueError, match="cannot honour the"):
            model(
                input_ids=tokens,
                position_ids=positions,
                attention_mask=dict.fromkeys(config.layer_types),
                pack_samples=True,
                use_cache=True,
            )

    def test_packing_onto_a_prefilled_cache_is_refused(self):
        """`cu_seqlens` covers the row being packed and says nothing about what
        is already cached, so on the flash layers those keys would go unread."""
        model, config = toy_encoder()
        tokens, positions = packed_row(config, [40, 23])
        prefill = model.eval()(input_ids=tokens[:, :16], use_cache=True)

        with pytest.raises(ValueError, match="empty cache"):
            model(
                input_ids=tokens,
                position_ids=positions,
                past_key_values=prefill.past_key_values,
                pack_samples=True,
                use_cache=True,
            )

    def test_packing_without_positions_is_refused(self):
        """The positions this stack builds for itself run straight through the
        row, so packing would silently attend across every boundary."""
        model, config = toy_encoder()
        tokens, _ = packed_row(config, [40, 23])

        with pytest.raises(ValueError, match="restart at 0 for each sample"):
            model(input_ids=tokens, pack_samples=True, use_cache=True)

    def test_gradient_checkpointing_replays_a_packed_row_the_same_way(self):
        """The production configuration, and the one place a recomputed forward
        could see different segment boundaries than the original: recompute
        replays the cache write, and `packed_segments` has to survive the
        `partial(super().__call__, **kwargs)` the checkpoint wraps."""
        model, config = toy_encoder()
        length = 40
        tokens, positions = packed_row(config, [length])
        twinned = torch.cat([tokens, tokens], dim=1)
        twin_positions = torch.cat([positions, positions], dim=1)

        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        out = model(
            input_ids=twinned, position_ids=twin_positions, pack_samples=True, use_cache=True
        ).last_hidden_state
        (grad,) = torch.autograd.grad(out.float().pow(2).mean(), [model.embed_tokens.weight])

        assert torch.equal(out[:, length:], out[:, :length])
        assert grad.abs().sum() > 0

    def test_the_ordinary_path_is_untouched_by_the_new_argument(self):
        """`pack_samples` defaults off and the encoder it shares must not have
        moved: same row, same masks, same answer."""
        model, config = toy_encoder()
        tokens = torch.randint(0, config.vocab_size, (1, 40), device="cuda")

        first = model(input_ids=tokens, use_cache=True).last_hidden_state
        second = model(input_ids=tokens, pack_samples=False, use_cache=True).last_hidden_state

        assert torch.equal(first, second)
