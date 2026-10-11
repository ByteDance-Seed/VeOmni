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
"""DiffusionGemma under Ulysses sequence parallelism, single process.

What can be asserted without a world: which configurations are refused, what
shape the collator hands over, and which slice of a row a given rank claims.
The collectives and the sp=2-against-sp=1 equivalence need two ranks and live in
``tests/distributed/test_diffusion_gemma_ulysses.py`` -- beside
``test_diffusion_gemma_fsdp2.py`` rather than with the framework's other Ulysses
tests in ``tests/parallel/ulysses/``, because what those cover is the shared
primitives and what this covers is one model's use of them.

`sp_enabled` and `sp_size` are properties over `ulysses_size`, so a state
carrying the size drives every branch that reads them.
"""

from contextlib import contextmanager
from dataclasses import MISSING
from unittest.mock import patch

import pytest
import torch

import veomni.data.data_collator as collator_module
import veomni.models.transformers.diffusion_gemma.dg_sequence_parallel as dg_sp
from veomni.data.data_collator import DiffusionGemmaCollator
from veomni.distributed.parallel_state import ParallelState
from veomni.utils.constants import IGNORE_INDEX

from .test_diffusion_gemma import build_toy_model


CANVAS = 8


def parallel_state(ulysses_size: int, cp_size: int = 1, ep_size: int = 1) -> ParallelState:
    """A real `ParallelState` minus the process groups its `__post_init__` would build.

    Not a namespace declaring `sp_enabled=True`: that is a derived property over
    `ulysses_size` and `cp_size`, and a stub asserting the answer directly would keep
    passing if the formula behind it moved. `__post_init__` is bypassed because it
    calls `init_sequence_parallel`, which needs a live process group; `sp_rank` and
    `ulysses_group` are overridden below for the same reason.
    """
    state = ParallelState.__new__(ParallelState)
    for name, spec in ParallelState.__dataclass_fields__.items():
        default = spec.default if spec.default_factory is MISSING else spec.default_factory()
        object.__setattr__(state, name, default)
    object.__setattr__(state, "ulysses_size", ulysses_size)
    object.__setattr__(state, "cp_size", cp_size)
    object.__setattr__(state, "extra_parallel_sizes", {"ep": ep_size})
    return state


class _RankedState:
    """`ParallelState` with the two properties that need a process group answered here.

    `ulysses_group` is pinned to `None` rather than left to the real property, which
    would fall through to the module-global `get_ulysses_sequence_parallel_group()`.
    That global is set by any earlier test in the process that initialised sequence
    parallelism, and if it leaked in here `_ulysses` would read a world size off a
    group that has nothing to do with this fabricated topology -- making these tests
    depend on execution order. `None` is what forces the documented fallback.
    """

    def __init__(self, state: ParallelState, rank: int):
        self._state = state
        self._rank = rank

    def __getattr__(self, name):
        if name == "sp_rank":
            return self._rank
        if name == "ulysses_group":
            return None
        return getattr(self._state, name)


@contextmanager
def sequence_parallel(ulysses_size: int, rank: int = 0, cp_size: int = 1, ep_size: int = 1, module=dg_sp):
    state = _RankedState(parallel_state(ulysses_size, cp_size=cp_size, ep_size=ep_size), rank)
    with patch.object(module, "get_parallel_state", return_value=state):
        yield


def feature(prompt_tokens: int, canvases: int, first_id: int = 3):
    ids = torch.arange(first_id, first_id + prompt_tokens, dtype=torch.long)
    labels = torch.full((canvases, CANVAS), IGNORE_INDEX, dtype=torch.long)
    labels[:, ::2] = 5
    weight = torch.zeros(canvases, CANVAS)
    weight[:, ::2] = 2.0 / CANVAS
    return {
        "input_ids": ids,
        "encoder_prompt_mask": torch.ones(prompt_tokens, dtype=torch.bool),
        "decoder_input_ids": torch.full((canvases, CANVAS), 7, dtype=torch.long),
        "labels": labels,
        "loss_weight": weight,
        "decoder_context_mask": torch.ones(canvases, prompt_tokens, dtype=torch.bool),
    }


class TestTheConfigurationsThatAreRefused:
    """The refusals matter more than usual here, because the failure is not a crash.

    Nothing in the padded path slices anything and no all-to-all is issued, so an
    SP run would replicate the micro-batch across the group, agree on it, and
    report a healthy loss at `sp_size` times the cost.
    """

    @pytest.mark.parametrize("ulysses_size", [2, 4, 8])
    def test_a_model_that_does_not_pack_refuses_to_build(self, ulysses_size):
        with sequence_parallel(ulysses_size):
            with pytest.raises(ValueError, match="needs `canvas_batching: flat_seq_pack`"):
                build_toy_model()

    def test_the_message_says_what_would_have_happened(self):
        """ "Unsupported" reads as "might work"; the cost is the reason not to try."""
        with sequence_parallel(2):
            with pytest.raises(ValueError) as raised:
                build_toy_model()
        assert "sp_size times the cost" in str(raised.value)
        assert "ulysses_size" in str(raised.value)

    def test_a_packing_model_builds(self):
        model = build_toy_model()
        model.config.canvas_batching = "flat_seq_pack"
        model.config.text_config.attn_dg_prefix = True
        with sequence_parallel(2):
            dg_sp.sequence_parallel_preflight(model.config)

    def test_attn_dg_prefix_is_refused_at_build_time(self):
        """Without it the decoder would reach its refusal only after allocating L².

        `flat_seq_pack` alone is not enough: that flag is what routes the decoder
        to the hole kernel. Without it `_flat_seq_sdpa_mask` builds a dense mask
        over the full encoder row first, so a long-sequence misconfiguration OOMs
        before the clearer refusal fires. Checked here so the docs' "refused at
        model build" claim is true of both requirements.
        """
        model = build_toy_model()
        model.config.canvas_batching = "flat_seq_pack"
        model.config.text_config.attn_dg_prefix = False
        with sequence_parallel(2):
            with pytest.raises(ValueError, match="needs `attn_dg_prefix: true`"):
                dg_sp.sequence_parallel_preflight(model.config)

    def test_the_attn_dg_prefix_refusal_says_what_would_have_happened(self):
        model = build_toy_model()
        model.config.canvas_batching = "flat_seq_pack"
        model.config.text_config.attn_dg_prefix = False
        with sequence_parallel(2):
            with pytest.raises(ValueError) as raised:
                dg_sp.sequence_parallel_preflight(model.config)
        assert "dense mask over the full encoder row" in str(raised.value)
        assert "L²" in str(raised.value) or "L^2" in str(raised.value)

    def test_context_parallelism_is_refused_separately(self):
        """`sp_enabled` is `cp_size > 1 or ulysses_size > 1`, and only one of those is Ulysses.

        Ring attention splits the keys across ranks, which every kernel here would
        have to implement rather than merely be handed.

        Defence in depth rather than the only gate: `ParallelState.__post_init__`
        raises `NotImplementedError` on `cp_size > 1` before a real run could reach
        this, and it is only the fabricated state above -- which bypasses
        `__post_init__` -- that gets here at all. Kept because the framework's
        refusal is not this model's to rely on.
        """
        model = build_toy_model()
        model.config.canvas_batching = "flat_seq_pack"
        with sequence_parallel(1, cp_size=2):
            with pytest.raises(ValueError, match="Ulysses sequence parallelism only"):
                dg_sp.sequence_parallel_preflight(model.config)

    def test_expert_parallelism_coexists_with_it(self):
        """Same path as Qwen3-MoE: EP mesh excludes SP peers, world_size divide covers it.

        `extra_parallel_gradient_divide_factor` returns `world_size`, so the SP dimension
        is folded into the expert grad average without re-apportioning the topology.
        The shipped recipe is exactly `ep_size: 8` + (now) `ulysses_size: 2`.
        """
        model = build_toy_model()
        model.config.canvas_batching = "flat_seq_pack"
        model.config.text_config.attn_dg_prefix = True
        with sequence_parallel(2, ep_size=8):
            dg_sp.sequence_parallel_preflight(model.config)

    def test_expert_parallelism_alone_is_fine(self):
        model = build_toy_model()
        model.config.canvas_batching = "flat_seq_pack"
        model.config.text_config.attn_dg_prefix = True
        with sequence_parallel(1, ep_size=8):
            dg_sp.sequence_parallel_preflight(model.config)

    def test_a_decoder_path_that_was_not_built_for_sp_is_refused(self):
        with sequence_parallel(2):
            with pytest.raises(ValueError, match="does not run under sequence parallelism"):
                dg_sp.refuse_decoder_path_under_sequence_parallelism("`canvas_batching: serial`", "because.")

    def test_the_refusal_names_the_path_that_does_work(self):
        """The alternative to naming it is a user turning SP off, which is the wrong fix."""
        with sequence_parallel(2):
            with pytest.raises(ValueError) as raised:
                dg_sp.refuse_decoder_path_under_sequence_parallelism("a path", "for a reason.")
        assert "attn_dg_prefix: true" in str(raised.value)
        assert "flat_seq_pack" in str(raised.value)

    def test_a_single_rank_leaves_everything_alone(self):
        """Control: the gates are on `sp_enabled`, not on the code path being present."""
        with sequence_parallel(1):
            build_toy_model()
            dg_sp.refuse_decoder_path_under_sequence_parallelism("anything", "for any reason.")


class TestWhatTheDecoderThinksTheCacheIsWorth:
    """The intervals are the row's, the cache is a shard, and one number bridges them."""

    def test_the_length_is_the_row_and_not_the_shard(self):
        with sequence_parallel(4):
            assert dg_sp.full_cache_length(256) == 1024

    def test_one_rank_reads_what_it_holds(self):
        with sequence_parallel(1):
            assert dg_sp.full_cache_length(256) == 256

    def test_the_canvas_intervals_are_measured_against_that_length(self):
        """Left as the shard's length, every canvas's visible run would be clamped.

        `dllm_hole_attention` clamps `[lo, hi)` to the cache it is given, so a
        canvas whose prefix starts past the shard would read *nothing* rather than
        fail -- a batch that trains and never sees its context.
        """
        with sequence_parallel(2):
            assert dg_sp.full_cache_length(1000) > 1000


class TestTheCollatorAlignsTheRow:
    """`sp_size` is a field, not a lookup: the collator is pickled into the workers."""

    def test_a_row_is_padded_up_to_a_multiple_of_sp_size(self):
        batch = DiffusionGemmaCollator(pack_samples=True, sp_size=4)([feature(5, 2), feature(6, 2)])
        assert batch["input_ids"].shape == (1, 12)
        assert batch["attention_mask"].tolist() == [[1] * 11 + [0]]

    def test_a_row_that_already_divides_is_untouched(self):
        batch = DiffusionGemmaCollator(pack_samples=True, sp_size=4)([feature(6, 2), feature(6, 2)])
        assert batch["input_ids"].shape == (1, 12)
        assert bool(batch["attention_mask"].all())

    def test_the_tail_is_its_own_segment(self):
        """Folded into the last sample it would hand that sample extra keys to attend to.

        The restart at 0 is what `encoder_segments` reads as a boundary, so this is
        the same statement as "the tail gets its own `cu_seqlens` entry".
        """
        batch = DiffusionGemmaCollator(pack_samples=True, sp_size=4)([feature(5, 2), feature(6, 2)])
        positions = batch["position_ids"][0].tolist()
        assert positions == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0]

    def test_no_canvas_can_read_the_tail(self):
        batch = DiffusionGemmaCollator(pack_samples=True, sp_size=4)([feature(5, 2), feature(6, 2)])
        encoder_half = batch["decoder_attention_mask"][0, :, :12]
        assert not bool(encoder_half[:, 11].any())

    def test_the_tail_is_not_the_prompt(self):
        batch = DiffusionGemmaCollator(pack_samples=True, sp_size=4)([feature(5, 2), feature(6, 2)])
        assert not bool(batch["encoder_prompt_mask"][0, 11])

    def test_sp_without_packing_is_refused_at_build_time(self):
        with pytest.raises(ValueError, match="`pack_samples` is off"):
            DiffusionGemmaCollator(pack_samples=False, sp_size=2)

    def test_one_rank_pads_nothing(self):
        """Control: alignment is the SP tax, and it is not paid when SP is off."""
        batch = DiffusionGemmaCollator(pack_samples=True)([feature(5, 2), feature(6, 2)])
        assert batch["input_ids"].shape == (1, 11)
        assert bool(batch["attention_mask"].all())

    def test_the_trainer_hands_the_topology_to_the_collator(self):
        """The wiring, not the constructor.

        An earlier version of this asserted that constructing the collator does not
        call `get_parallel_state` -- true by construction, and green on the day the
        trainer stopped passing `sp_size` at all, which is the thing the name
        promised. So this drives `_build_collate_fn` and reads back what it built.
        """
        from types import MethodType, SimpleNamespace

        from veomni.models.transformers.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaConfig
        from veomni.trainer import diffusion_gemma_trainer as trainer_module
        from veomni.trainer.diffusion_gemma_trainer import DiffusionGemmaTrainer

        trainer = SimpleNamespace(
            base=SimpleNamespace(
                args=SimpleNamespace(
                    data=SimpleNamespace(max_seq_len=4096),
                    train=SimpleNamespace(dyn_bsz=False, dyn_bsz_runtime="main", micro_batch_size=1),
                ),
                model=SimpleNamespace(
                    model_config=DiffusionGemmaConfig(
                        canvas_batching="flat_seq_pack", canvas_length=256, max_canvases=8
                    )
                ),
            ),
            _is_multimodal=False,
        )
        for name in ("_warn_if_the_budget_unit_does_not_match_train_size", "_check_the_budget_can_hold_a_sample"):
            setattr(trainer, name, MethodType(getattr(DiffusionGemmaTrainer, name), trainer))

        with patch.object(trainer_module, "get_parallel_state", return_value=parallel_state(4)):
            DiffusionGemmaTrainer._build_collate_fn(trainer)
        assert trainer.base.collate_fn.sp_size == 4

    def test_the_collator_does_not_read_the_topology_itself(self):
        """It is pickled into dataloader workers, whose global state is not this one."""
        with patch.object(collator_module, "get_parallel_state") as spy:
            DiffusionGemmaCollator(pack_samples=True, sp_size=1)
        spy.assert_not_called()


class TestWhichSliceARankClaims:
    def test_the_slices_are_contiguous_and_cover_the_row(self):
        embeds = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1)
        positions = torch.arange(12).reshape(1, 12)
        seen = []
        for rank in range(4):
            with sequence_parallel(4, rank=rank):
                shard, shard_positions = dg_sp.shard_packed_row(embeds, positions)
            assert shard.shape == (1, 3, 1)
            assert torch.equal(shard_positions, positions[:, rank * 3 : rank * 3 + 3])
            seen.append(shard.reshape(-1))
        assert torch.equal(torch.cat(seen), embeds.reshape(-1))

    def test_a_row_that_does_not_divide_is_refused(self):
        embeds = torch.zeros(1, 11, 1)
        with sequence_parallel(4):
            with pytest.raises(ValueError, match="does not divide"):
                dg_sp.shard_packed_row(embeds, torch.arange(11).reshape(1, 11))

    def test_positions_that_were_already_sliced_are_refused(self):
        """The boundaries are read off these positions before the shard is taken.

        A pre-sliced `position_ids` would describe segments that do not exist, and
        the shapes downstream would still line up.
        """
        embeds = torch.zeros(1, 12, 1)
        with sequence_parallel(4):
            with pytest.raises(ValueError, match="expected at full length"):
                dg_sp.shard_packed_row(embeds, torch.arange(3).reshape(1, 3))


class TestWhatCountsAsAlignmentPadding:
    """A packed row may not carry padding. SP alignment is the one exception, and
    it has to be recognisable rather than trusted -- a genuinely padded row that
    happens to arrive under SP must still be refused."""

    def test_a_trailing_run_shorter_than_sp_size_is_alignment(self):
        mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
        with sequence_parallel(4):
            assert dg_sp.sp_pad_tokens(mask) == 1

    def test_an_all_ones_row_has_no_padding(self):
        with sequence_parallel(4):
            assert dg_sp.sp_pad_tokens(torch.ones(1, 8, dtype=torch.long)) == 0

    def test_padding_in_the_middle_is_refused(self):
        mask = torch.tensor([[1, 1, 0, 1, 1, 1]])
        with sequence_parallel(4):
            with pytest.raises(ValueError, match="not all trailing"):
                dg_sp.sp_pad_tokens(mask)

    def test_a_run_as_long_as_sp_size_is_refused(self):
        """Alignment can never need `sp_size` tokens: that row already divided."""
        mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
        with sequence_parallel(4):
            with pytest.raises(ValueError, match="too many to be alignment"):
                dg_sp.sp_pad_tokens(mask)

    def test_nothing_is_allowed_when_sp_is_off(self):
        mask = torch.tensor([[1, 1, 1, 0]])
        with sequence_parallel(1):
            assert dg_sp.sp_pad_tokens(mask) == 0


class TestTheHeadSplitConstraints:
    """Checked before any collective, so the message names the config field.

    A mismatch would otherwise surface as an all-to-all size error on one rank
    and a hang on the others.
    """

    def test_query_heads_must_divide(self):
        query = torch.zeros(1, 6, 8, 4)
        with sequence_parallel(4):
            with pytest.raises(ValueError, match="num_attention_heads"):
                dg_sp.ulysses_head_split(query, query, query, seq_dim=2, head_dim=1)

    def test_kv_heads_below_sp_size_must_divide_it(self):
        query = torch.zeros(1, 8, 8, 4)
        key = torch.zeros(1, 3, 8, 4)
        with sequence_parallel(4):
            with pytest.raises(ValueError, match="num_key_value_heads"):
                dg_sp.ulysses_head_split(query, key, key, seq_dim=2, head_dim=1)

    def test_plenty_of_kv_heads_still_have_to_divide(self):
        """8 query heads and 6 KV heads over 4 ranks: no rank gets a whole KV group.

        Nothing is repeated in this case, so the check has to be its own -- the
        repetition branch never runs and would not have caught it.
        """
        with sequence_parallel(4):
            with pytest.raises(ValueError, match="must be divisible"):
                dg_sp._kv_repeats(8, 6, 4)

    def test_the_repeat_count_is_one_when_there_are_enough(self):
        assert dg_sp._kv_repeats(16, 8, 2) == 1

    def test_the_kv_heads_are_repeated_up_to_one_per_rank(self):
        assert dg_sp._kv_repeats(16, 2, 4) == 2


class TestWhichHeadsARankClaims:
    """The replicated half is sliced, the sharded half arrives by all-to-all, and
    the two have to name the same heads. `all_to_all_tensor` cuts its scatter
    dimension with `tensor_split`, so this does too."""

    def test_the_slices_are_contiguous_and_cover_the_heads(self):
        tensor = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1)
        seen = [dg_sp._local_heads(tensor, 1, 4, rank).reshape(-1) for rank in range(4)]
        assert [block.tolist() for block in seen] == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]

    def test_heads_that_do_not_divide_are_refused(self):
        with pytest.raises(ValueError, match="do not divide"):
            dg_sp._local_heads(torch.zeros(1, 6, 1, 1), 1, 4, 0)

    def test_a_group_stays_with_its_kv_head(self):
        """The alignment that makes GQA survive the split, stated as an assertion.

        16 query heads over 2 KV heads: queries 0-7 belong to KV 0. At sp=2 rank 0
        must get exactly those queries and exactly KV 0 -- which holds because
        `repeat_interleave` and `tensor_split` cut in the same order.
        """
        queries = torch.arange(16).reshape(1, 16, 1, 1)
        kv = torch.arange(2).reshape(1, 2, 1, 1)
        repeats = dg_sp._kv_repeats(16, 2, 2)
        assert repeats == 1
        for rank, expected_kv in enumerate([0, 1]):
            mine = dg_sp._local_heads(queries, 1, 2, rank).reshape(-1).tolist()
            my_kv = dg_sp._local_heads(kv, 1, 2, rank).reshape(-1).tolist()
            assert mine == list(range(rank * 8, rank * 8 + 8))
            assert my_kv == [expected_kv]

    @pytest.mark.parametrize(
        ("q_heads", "kv_heads", "sp_size"),
        [(4, 2, 4), (8, 2, 4), (16, 4, 8), (8, 1, 4)],
    )
    def test_a_group_stays_with_its_kv_head_when_the_kv_is_repeated(self, q_heads, kv_heads, sp_size):
        """The same alignment on the branch that exists for it.

        The case above pins `repeats == 1`, which is every configuration reachable
        on two ranks -- so the repetition, which is the whole reason `_kv_repeats`
        is not just a divisibility check, went unasserted. It is not remote: the
        bench config has two global KV heads, so `sp_size = 4` already repeats.

        What could go wrong is an ordering mismatch. `repeat_interleave` gives
        `[h0, h0, h1, h1]` and a tile would give `[h0, h1, h0, h1]`; both hand
        every rank one KV head of the right shape, and only the first hands each
        rank the head its query group actually belongs to. `repeat_kv_heads` is
        called rather than restated so that the ordering this asserts is the
        ordering the model uses; the expected pairing is computed from the GQA
        rule for the same reason.
        """
        queries = torch.arange(q_heads).reshape(1, q_heads, 1, 1)
        kv = torch.arange(kv_heads).reshape(1, kv_heads, 1, 1)
        repeats = dg_sp._kv_repeats(q_heads, kv_heads, sp_size)
        assert repeats > 1, "this case is meant to exercise the repetition"

        repeated = dg_sp.repeat_kv_heads(kv, 1, repeats)
        group_size = q_heads // kv_heads
        for rank in range(sp_size):
            mine = dg_sp._local_heads(queries, 1, sp_size, rank).reshape(-1).tolist()
            my_kv = dg_sp._local_heads(repeated, 1, sp_size, rank).reshape(-1).tolist()
            wanted = {head // group_size for head in mine}
            assert len(wanted) == 1, (
                f"rank {rank} was given query heads {mine}, which span KV groups {sorted(wanted)}; "
                "Ulysses cannot serve that with one KV head"
            )
            assert my_kv == [wanted.pop()], (
                f"rank {rank} holds query heads {mine}, whose KV group is "
                f"{mine[0] // group_size}, but was handed KV head {my_kv}"
            )
