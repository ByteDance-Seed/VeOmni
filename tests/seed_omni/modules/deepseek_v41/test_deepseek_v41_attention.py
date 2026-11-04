# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""DeepSeek-V4.1 attention: the properties a forward-output test cannot see.

Numeric agreement with the released ``inference/model.py`` is checked by a
development harness, because it needs the checkpoint repository on disk and the
TileLang kernels on an SM90 device -- neither available here. What is left is
everything that a wrong answer would still look plausible for:

* the packed-sample boundary, which the reference has no concept of and which
  fails silently by pooling two samples into one latent or letting a query read
  the previous sample's history;
* the shared publication order between source and consumer layers, where reading
  a stale or absent publication produces a well-shaped tensor;
* the two-level candidate prefilter actually restricting the second level.

All of it runs on CPU in fp32 and never calls the sparse attention kernel.
"""

import pytest
import torch

from veomni.models.seed_omni.modules.deepseek_v41.llm.attention import (
    DeepseekV41Attention,
    DeepseekV41Compressor,
    DeepseekV41Indexer,
    DeepseekV41RotaryEmbedding,
    SharedAttentionState,
    apply_interleaved_rope,
    build_compression_plans,
    indexer_kl_terms,
)
from veomni.models.seed_omni.modules.deepseek_v41.llm.configuration import DeepseekV41LlmConfig


HIDDEN = 64
N_LAYERS = 8
RATIO = 2


def make_config(**overrides) -> DeepseekV41LlmConfig:
    kwargs = dict(
        hidden_size=HIDDEN,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=48,
        o_lora_rank=16,
        o_groups=2,
        sliding_window=6,
        index_n_heads=8,
        index_head_dim=32,
        index_topk=4,
        rms_norm_eps=1e-20,
        compress_ratios=[RATIO] * N_LAYERS + [0, 0, 0],
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0, 4],
        candidate_source_layer_id=0,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        engram_layer_ids=[1],
        dspark_target_layer_ids=[5, 6, 7],
        max_position_embeddings=256,
    )
    kwargs.update(overrides)
    return DeepseekV41LlmConfig(**kwargs)


def packed_positions(lengths):
    return torch.cat([torch.arange(length) for length in lengths]).unsqueeze(0)


def packed_slices(lengths):
    bounds = torch.tensor(lengths).cumsum(0).tolist()
    return tuple(zip([0] + bounds[:-1], bounds))


class TestRotary:
    def test_inverse_undoes_forward(self):
        config = make_config()
        rotary = DeepseekV41RotaryEmbedding(config)
        cos, sin = rotary(torch.arange(16), compressed=True)
        x = torch.randn(1, 16, 3, config.head_dim)
        rotated = apply_interleaved_rope(x, cos.unsqueeze(-2), sin.unsqueeze(-2))
        restored = apply_interleaved_rope(rotated, cos.unsqueeze(-2), -sin.unsqueeze(-2))
        torch.testing.assert_close(restored, x, atol=1e-5, rtol=1e-5)

    def test_only_the_rope_tail_moves(self):
        config = make_config()
        rotary = DeepseekV41RotaryEmbedding(config)
        cos, sin = rotary(torch.arange(16), compressed=False)
        x = torch.randn(1, 16, config.head_dim)
        rotated = apply_interleaved_rope(x, cos, sin)
        nope = config.head_dim - config.qk_rope_head_dim
        torch.testing.assert_close(rotated[..., :nope], x[..., :nope])
        assert not torch.allclose(rotated[..., nope:], x[..., nope:])

    def test_compress_schedule_is_yarn_scaled(self):
        """The compressed schedule stretches the low frequencies, the main one does not.

        The two differ by more than their base: ``main`` disables YaRN outright,
        which is what lets a pure sliding-window layer keep full resolution over
        the only distances it can reach.
        """
        config = make_config()
        rotary = DeepseekV41RotaryEmbedding(config)
        assert not torch.allclose(rotary.main_inv_freq, rotary.compress_inv_freq)
        plain = 1.0 / (
            config.compress_rope_theta
            ** (torch.arange(0, config.qk_rope_head_dim, 2).float() / config.qk_rope_head_dim)
        )
        # YaRN divides the slowest dimensions by `factor` and leaves the fastest
        # alone, so the ramp shows up as a ratio that is not constant.
        ratio = plain / rotary.compress_inv_freq
        assert ratio.max() > ratio.min() * 2


class TestCompressionPlan:
    def test_groups_never_straddle_a_packed_boundary(self):
        config = make_config()
        lengths = (11, 13)
        plan = build_compression_plans(config, packed_positions(lengths), packed_slices(lengths))[RATIO]
        boundary = lengths[0]
        starts = plan.window_indices[:, 0]
        ends = plan.window_indices[:, -1]
        crossing = ((starts < boundary) & (ends >= boundary)).sum().item()
        assert crossing == 0

    def test_incomplete_tail_contributes_no_group(self):
        config = make_config()
        lengths = (11, 13)
        plan = build_compression_plans(config, packed_positions(lengths), packed_slices(lengths))[RATIO]
        # 11 // 2 + 13 // 2 == 11, with one token dropped from each sample's tail.
        assert plan.compressed_len == sum(length // RATIO for length in lengths)

    def test_queries_cannot_see_another_samples_groups(self):
        config = make_config()
        lengths = (11, 13)
        plan = build_compression_plans(config, packed_positions(lengths), packed_slices(lengths))[RATIO]
        visible = plan.visibility_mask(torch.device("cpu"))
        first_groups = lengths[0] // RATIO
        assert not visible[lengths[0] :, :first_groups].any()
        assert not visible[: lengths[0], first_groups:].any()

    def test_group_positions_restart_each_sample(self):
        config = make_config()
        lengths = (11, 13)
        plan = build_compression_plans(config, packed_positions(lengths), packed_slices(lengths))[RATIO]
        first_groups = lengths[0] // RATIO
        # RoPE positions are per-sample, so the second sample starts over at 0.
        assert plan.window_positions[first_groups].item() == 0

    def test_rows_with_different_positions_are_refused(self):
        config = make_config()
        positions = torch.stack([torch.arange(8), torch.arange(8).flip(0)])
        with pytest.raises(ValueError, match="share one position layout"):
            build_compression_plans(config, positions)


class TestCompressor:
    def test_ratio_one_has_no_gate(self):
        config = make_config(compress_ratios=[1] * N_LAYERS + [0, 0, 0])
        assert DeepseekV41Compressor(config, layer_idx=0).wgate is None

    def test_ratio_two_pools_with_a_gate(self):
        config = make_config()
        assert DeepseekV41Compressor(config, layer_idx=0).wgate is not None

    def test_packed_output_matches_samples_compressed_alone(self):
        torch.manual_seed(0)
        config = make_config()
        compressor = DeepseekV41Compressor(config, layer_idx=0).float()
        for param in compressor.parameters():
            param.data.normal_(0, 0.05)

        lengths = (11, 13)
        parts = [torch.randn(1, length, HIDDEN) for length in lengths]
        plans = build_compression_plans(config, packed_positions(lengths), packed_slices(lengths))
        packed = compressor(torch.cat(parts, dim=1), plans[RATIO])
        separate = torch.cat(
            [
                compressor(part, build_compression_plans(config, torch.arange(length).unsqueeze(0))[RATIO])
                for part, length in zip(parts, lengths)
            ],
            dim=1,
        )
        torch.testing.assert_close(packed, separate)

    def test_plan_ratio_must_match_the_layer(self):
        config = make_config()
        compressor = DeepseekV41Compressor(config, layer_idx=0)
        plan = build_compression_plans(
            make_config(compress_ratios=[1] * N_LAYERS + [0, 0, 0]), torch.arange(8).unsqueeze(0)
        )[1]
        with pytest.raises(ValueError, match="compresses by"):
            compressor(torch.randn(1, 8, HIDDEN), plan)


class TestWindow:
    def test_window_stays_inside_the_packed_sample(self):
        config = make_config()
        lengths = (11, 13)
        positions = packed_positions(lengths)
        layer = DeepseekV41Attention(config, layer_idx=0)
        window = layer._window_indices(positions, positions.shape[1])
        second = window[:, lengths[0] :]
        assert not ((second >= 0) & (second < lengths[0])).any()

    def test_every_query_sees_itself(self):
        config = make_config()
        positions = torch.arange(12).unsqueeze(0)
        layer = DeepseekV41Attention(config, layer_idx=0)
        window = layer._window_indices(positions, 12)
        queries = torch.arange(12, dtype=torch.int32)
        assert bool((window == queries[None, :, None]).any(dim=-1).all())


class TestIndexerContract:
    """The shared publication chain, which fails by producing a plausible tensor."""

    def _indexer_inputs(self, config, seq_len=16):
        torch.manual_seed(0)
        hidden = torch.randn(1, seq_len, HIDDEN)
        q_residual = torch.randn(1, seq_len, config.q_lora_rank)
        positions = torch.arange(seq_len).unsqueeze(0)
        plan = build_compression_plans(config, positions)[RATIO]
        rotary = DeepseekV41RotaryEmbedding(config)
        cos, sin = rotary(positions, compressed=True)
        index_k = torch.randn(1, plan.compressed_len, config.index_head_dim)
        return hidden, q_residual, plan, index_k, cos, sin

    def test_selection_never_leaves_the_visible_range(self):
        config = make_config()
        indexer = DeepseekV41Indexer(config, layer_idx=0).float()
        for param in indexer.parameters():
            param.data.normal_(0, 0.05)
        hidden, q_residual, plan, index_k, cos, sin = self._indexer_inputs(config)
        selected, _ = indexer(hidden, q_residual, index_k, plan, SharedAttentionState(), cos, sin)

        visible = plan.visibility_mask(hidden.device)
        for query in range(selected.shape[1]):
            for slot in selected[0, query].tolist():
                assert slot == -1 or visible[query, slot]

    def test_candidate_mask_restricts_the_second_level(self):
        """With candidates to spare, level two picks only from inside them."""
        config = make_config()
        consumer = DeepseekV41Indexer(config, layer_idx=4).float()
        for param in consumer.parameters():
            param.data.normal_(0, 0.05)
        hidden, q_residual, plan, index_k, cos, sin = self._indexer_inputs(config)

        allowed = {0, 1, 2, 3, 5, 6}
        shared = SharedAttentionState()
        shared.candidates = torch.zeros(1, hidden.shape[1], plan.compressed_len, dtype=torch.bool)
        shared.candidates[..., sorted(allowed)] = True
        selected, _ = consumer(hidden, q_residual, index_k, plan, shared, cos, sin)

        visible = plan.visibility_mask(hidden.device)
        for query in range(selected.shape[1]):
            reachable = {group for group in allowed if visible[query, group]}
            if len(reachable) < config.index_topk:
                continue  # too few to fill the selection; see the test below
            assert set(selected[0, query].tolist()) <= allowed

    def test_scarce_candidates_fall_back_to_merely_visible_entries(self):
        """A query with fewer candidates than ``index_topk`` pads from outside them.

        The padding slots are entries the candidate mask scored at ``-inf``, and
        ``topk`` hands them back anyway because the row has nothing better. They
        survive as real indices because the only filter applied afterwards is
        visibility, so the attention then reads a compressed entry level one had
        rejected.

        This reproduces the released ``Indexer.forward``, whose final step is the
        same ``torch.where(idxs < compress_lens, ...)`` over visibility alone, and
        it is pinned here because it looks like a bug and "fixing" it would be a
        silent divergence. At the shipped settings it is unreachable:
        ``candidate_topk_blocks * candidate_block_size`` is 16384 against an
        ``index_topk`` of 512, so the selection is never short of candidates.
        """
        config = make_config()
        consumer = DeepseekV41Indexer(config, layer_idx=4).float()
        for param in consumer.parameters():
            param.data.normal_(0, 0.05)
        hidden, q_residual, plan, index_k, cos, sin = self._indexer_inputs(config)

        shared = SharedAttentionState()
        shared.candidates = torch.zeros(1, hidden.shape[1], plan.compressed_len, dtype=torch.bool)
        shared.candidates[..., :1] = True  # a single candidate, against index_topk=4
        selected, _ = consumer(hidden, q_residual, index_k, plan, shared, cos, sin)

        visible = plan.visibility_mask(hidden.device)
        last = selected.shape[1] - 1
        picked = {group for group in selected[0, last].tolist() if group != -1}
        assert picked - {0}, "expected padding beyond the single candidate"
        assert all(visible[last, group] for group in picked)

    def test_consumer_without_candidates_is_refused(self):
        config = make_config()
        consumer = DeepseekV41Indexer(config, layer_idx=4).float()
        hidden, q_residual, plan, index_k, cos, sin = self._indexer_inputs(config)
        with pytest.raises(RuntimeError, match="candidate blocks but none were published"):
            consumer(hidden, q_residual, index_k, plan, SharedAttentionState(), cos, sin)


class TestIndexerKL:
    """The auxiliary objective, checked against distributions with known answers.

    The indexer emits indices, so nothing reaches its weights through the
    attention output; this KL is the only gradient they ever get, and a wrong one
    would still produce a decreasing curve.
    """

    def test_perfect_student_pays_nothing(self):
        torch.manual_seed(0)
        target = torch.rand(2, 5, 8)
        target = target / target.sum(-1, keepdim=True)
        kl, _ = indexer_kl_terms(target.log(), target)
        torch.testing.assert_close(kl, torch.zeros_like(kl), atol=1e-5, rtol=0)

    def test_uniform_student_pays_exactly_the_reference(self):
        """A student that knows only the candidate set is what the reference prices."""
        torch.manual_seed(0)
        target = torch.rand(2, 5, 8)
        target = target / target.sum(-1, keepdim=True)
        kl, uniform = indexer_kl_terms(torch.zeros_like(target), target)
        torch.testing.assert_close(kl, uniform, atol=1e-5, rtol=0)

    def test_divergences_are_non_negative(self):
        torch.manual_seed(0)
        target = torch.rand(2, 5, 8)
        target = target / target.sum(-1, keepdim=True)
        kl, uniform = indexer_kl_terms(torch.randn(2, 5, 8), target)
        assert bool((kl >= 0).all()) and bool((uniform >= 0).all())

    def test_a_row_with_no_teacher_mass_is_excluded_from_both_terms(self):
        target = torch.zeros(1, 2, 4)
        # Deliberately not uniform: the reference prices how much there was to
        # learn, so a uniform target would legitimately price at zero and the row
        # would be indistinguishable from the excluded one above it.
        target[0, 1] = torch.tensor([0.7, 0.1, 0.1, 0.1])
        kl, uniform = indexer_kl_terms(torch.randn(1, 2, 4), target)
        assert kl[0, 0] == 0 and uniform[0, 0] == 0
        assert uniform[0, 1] > 0

    def test_a_uniform_teacher_had_nothing_to_teach(self):
        """The reference is what there *was* to learn, so it is zero for a flat target."""
        target = torch.full((1, 1, 4), 0.25)
        _, uniform = indexer_kl_terms(torch.randn(1, 1, 4), target)
        torch.testing.assert_close(uniform, torch.zeros_like(uniform), atol=1e-6, rtol=0)

    def test_an_all_miss_row_does_not_produce_nan_gradients(self):
        """`log_softmax` of an all -inf row is NaN, and its backward stays NaN.

        The first ``ratio - 1`` positions of every packed sample have no complete
        compression group behind them, so this is the common case rather than a
        corner one -- a NaN here poisons the indexer's whole backward.
        """
        student = torch.randn(1, 2, 4, requires_grad=True)
        masked = student.masked_fill(
            torch.tensor([[[True] * 4, [False] * 4]]),
            -torch.inf,
        )
        target = torch.zeros(1, 2, 4)
        target[0, 1] = 0.25
        kl, _ = indexer_kl_terms(masked, target)
        kl.sum().backward()
        assert bool(student.grad.isfinite().all())

    def test_the_objective_is_off_by_default(self):
        config = make_config()
        assert not config.dsa_indexer_loss
        assert not DeepseekV41Attention(config, layer_idx=0).builds_indexer_loss

    def test_a_non_positive_coefficient_counts_as_off(self):
        """`loss + 0.0 * kl` still builds the graph and writes zero grads, which an
        optimizer that decays whatever it steps would then decay."""
        config = make_config(dsa_indexer_loss=True, dsa_indexer_loss_coef=0.0)
        assert not DeepseekV41Attention(config, layer_idx=0).builds_indexer_loss

    def test_only_indexer_owning_layers_build_it(self):
        config = make_config(dsa_indexer_loss=True)
        building = {idx for idx in range(N_LAYERS) if DeepseekV41Attention(config, idx).builds_indexer_loss}
        assert building == set(config.index_source_layer_ids)


class TestSharedStateContract:
    def _layer_inputs(self, config, seq_len=16):
        hidden = torch.randn(1, seq_len, HIDDEN)
        positions = torch.arange(seq_len).unsqueeze(0)
        return hidden, positions, DeepseekV41RotaryEmbedding(config), build_compression_plans(config, positions)

    def test_compressed_layer_without_a_source_is_refused(self):
        config = make_config()
        layer = DeepseekV41Attention(config, layer_idx=1)  # neither kv nor index source
        hidden, positions, rotary, plans = self._layer_inputs(config)
        with pytest.raises(RuntimeError, match="no source layer has published"):
            layer(hidden, positions, rotary, SharedAttentionState(), plans)

    def test_reusing_layer_without_a_selection_is_refused(self):
        config = make_config()
        layer = DeepseekV41Attention(config, layer_idx=1)
        hidden, positions, rotary, plans = self._layer_inputs(config)
        shared = SharedAttentionState(compress_kv=torch.zeros(1, plans[RATIO].compressed_len, config.head_dim))
        with pytest.raises(RuntimeError, match="reuses a top-k selection"):
            layer(hidden, positions, rotary, shared, plans)

    def test_only_declared_layers_own_a_compressor_or_indexer(self):
        config = make_config()
        owns = {
            layer_idx: (
                DeepseekV41Attention(config, layer_idx).compressor is not None,
                DeepseekV41Attention(config, layer_idx).indexer is not None,
            )
            for layer_idx in range(N_LAYERS)
        }
        assert {idx for idx, (kv, _) in owns.items() if kv} == set(config.kv_source_layer_ids)
        assert {idx for idx, (_, index) in owns.items() if index} == set(config.index_source_layer_ids)
