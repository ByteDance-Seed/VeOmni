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

"""DeepSeek-V4.1 backbone graph hooks: the mask, the plan and the meter.

Three things here would be wrong without being visibly wrong:

* an image mask built from a second pass over the conversation can disagree with
  the packing it is supposed to describe, and routes the wrong tokens without
  changing any shape;
* the FLOPs estimate inherited from a dense backbone is quadratic in sequence
  length, and V4.1's attention is not -- the number it reports stays plausible
  while the MFU it implies is off by an order of magnitude at long context;
* Ulysses SP would run, because nothing in the shared packing path knows that
  this model builds its compression plan in Python before any kernel sees the
  sequence.
"""

import pytest
import torch

from veomni.models.seed_omni.modules.deepseek_v41.llm.accelerated.accelerated import (
    DeepseekV41LlmAccelerated,
    build_packed_image_mask,
    fold_indexer_kl_loss,
)
from veomni.models.seed_omni.modules.deepseek_v41.llm.configuration import DeepseekV41LlmConfig
from veomni.models.seed_omni.utils.conversation import ConversationItem


HIDDEN = 64
N_LAYERS = 8


RELEASED_QUANTIZATION = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4",
}


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
        hc_mult=4,
        compress_ratios=[2] * N_LAYERS + [0, 0, 0],
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0, 4],
        candidate_source_layer_id=0,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        engram_layer_ids=[1, 5],
        dspark_target_layer_ids=[5, 6, 7],
        dspark_n_routed_experts=8,
        max_position_embeddings=256,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        n_shared_experts=1,
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=8,
    )
    kwargs.update(overrides)
    return DeepseekV41LlmConfig(**kwargs)


def text(length, role="user"):
    return ConversationItem(type="text", value=torch.randn(length, HIDDEN), role=role)


def image(length):
    return ConversationItem(type="image", value=torch.randn(length, HIDDEN), role="user")


def dummy(length):
    return ConversationItem(type="text", value=torch.randn(length, HIDDEN), role="dummy")


class TestImageMask:
    def test_marks_exactly_the_image_segments(self):
        conversation = [[text(4), image(3), text(2)]]
        mask = build_packed_image_mask(conversation, torch.device("cpu"))
        expected = torch.tensor([[False] * 4 + [True] * 3 + [False] * 2])
        assert torch.equal(mask, expected)

    def test_text_only_batch_gets_no_mask(self):
        """`None` rather than all-false: both consumers skip their select on it."""
        assert build_packed_image_mask([[text(4), text(2)]], torch.device("cpu")) is None

    def test_dummy_segments_are_skipped_like_the_packer_skips_them(self):
        """The mask must walk the conversation the way the packer does, or it
        describes a row one segment longer than the one that exists."""
        conversation = [[text(4), dummy(5), image(3)]]
        mask = build_packed_image_mask(conversation, torch.device("cpu"))
        assert mask.shape == (1, 7)
        assert torch.equal(mask[0], torch.tensor([False] * 4 + [True] * 3))

    def test_mask_length_matches_the_packing(self):
        from veomni.models.seed_omni.modules.base.llm_packing import pack_llm_conversations_for_forward

        conversation = [[text(4), image(3), dummy(2), text(5)]]
        embeds, _, _, _ = pack_llm_conversations_for_forward(conversation, torch.device("cpu"))
        mask = build_packed_image_mask(conversation, torch.device("cpu"))
        assert mask.shape[1] == embeds.shape[1]

    def test_multiple_samples_concatenate_in_order(self):
        conversation = [[text(2), image(2)], [image(1), text(3)]]
        mask = build_packed_image_mask(conversation, torch.device("cpu"))
        assert torch.equal(mask[0], torch.tensor([False, False, True, True, True, False, False, False]))


class TestNgramAlignment:
    def _model(self, **overrides):
        return DeepseekV41LlmAccelerated(make_config(**overrides))

    def test_rows_matching_the_packed_length_pass_through(self):
        model = self._model()
        rows = [torch.randn(1, 12, 6, 8) for _ in range(2)]
        aligned = model._align_ngram_embeds(rows, 12)
        assert len(aligned) == 2 and all(row.shape[1] == 12 for row in aligned)

    def test_a_length_mismatch_is_refused(self):
        """Positional rows: past the divergence every token gets another token's
        memory, at the right shape and on every layer that reads it."""
        model = self._model()
        with pytest.raises(ValueError, match="covers 10 positions but the packed row has 12"):
            model._align_ngram_embeds([torch.randn(1, 10, 6, 8)], 12)

    def test_absent_rows_stay_absent(self):
        assert self._model()._align_ngram_embeds(None, 12) is None


class TestParallelPlan:
    def test_expert_parallel_shards_both_expert_tensors(self):
        plan = DeepseekV41LlmAccelerated(make_config()).get_parallel_plan()
        ep = plan.extra_parallel_plan["ep"]
        assert set(ep) == {
            "model.layers.*.mlp.experts.gate_up_proj",
            "model.layers.*.mlp.experts.down_proj",
            "model.mtp.*.mlp.experts.gate_up_proj",
            "model.mtp.*.mlp.experts.down_proj",
        }

    def test_plan_fqns_resolve_against_the_real_module_tree(self):
        """A stale prefix produces an empty match and silently unsharded experts."""
        model = DeepseekV41LlmAccelerated(make_config())
        names = {name for name, _ in model.named_parameters()}
        for pattern in model.get_parallel_plan().extra_parallel_plan["ep"]:
            concrete = pattern.replace("*", "0")
            assert concrete in names, f"{pattern} matches nothing in the module tree"

    def test_quantized_experts_shard_their_scales_too(self):
        """A quantized expert weight is two tensors and both are indexed by expert.

        Sharding the slab alone leaves every rank holding the full scales --
        ~15.6 GiB at the released dimensions, which is what takes eight H20s
        past their 97 GiB.
        """
        # o_lora_rank is raised off the toy default so that a group's rows are
        # whole 32-row scale tiles, as they are at the released 1024.
        config = make_config(weight_quantization=RELEASED_QUANTIZATION, o_lora_rank=64)
        # On meta because the real path is `from_pretrained`, which never
        # initializes: there is no normal_ for fp8 on CPU to fall back on.
        with torch.device("meta"):
            model = DeepseekV41LlmAccelerated(config)
        ep = model.get_parallel_plan().extra_parallel_plan["ep"]

        assert set(ep) == {
            "model.layers.*.mlp.experts.gate_up_proj",
            "model.layers.*.mlp.experts.down_proj",
            "model.layers.*.mlp.experts.gate_up_proj_scale",
            "model.layers.*.mlp.experts.down_proj_scale",
            "model.mtp.*.mlp.experts.gate_up_proj",
            "model.mtp.*.mlp.experts.down_proj",
            "model.mtp.*.mlp.experts.gate_up_proj_scale",
            "model.mtp.*.mlp.experts.down_proj_scale",
        }
        names = {name for name, _ in model.named_parameters()}
        for pattern in ep:
            assert pattern.replace("*", "0") in names


class TestMeter:
    """V4.1's attention is linear in sequence length; the dense meters are not.

    Measured at the released dimensions rather than the toy ones the rest of this
    file uses. The shapes are the whole subject here -- at ``hidden_size=64`` the
    indexer's heads outweigh everything else and a meter could be quadratic
    without the ratio showing it. The config is passed to the unbound method on a
    stub rather than by building the model, because a 5120-wide 8-layer stack is
    several GB of weights that ``estimate_flops`` never reads.
    """

    class _Stub:
        def __init__(self, config):
            self.config = config

    def _flops(self, seqlens, **overrides):
        config = make_config(
            hidden_size=5120,
            num_attention_heads=64,
            head_dim=512,
            qk_rope_head_dim=64,
            q_lora_rank=2048,
            o_lora_rank=512,
            o_groups=4,
            sliding_window=2048,
            index_n_heads=64,
            index_head_dim=128,
            index_topk=2048,
            moe_intermediate_size=2048,
            n_routed_experts=256,
            num_experts_per_tok=8,
            max_position_embeddings=32768,
            **overrides,
        )
        return DeepseekV41LlmAccelerated.estimate_flops(self._Stub(config), seqlens)

    def test_doubling_the_tokens_roughly_doubles_the_cost(self):
        """Attention reads `sliding_window + index_topk` entries per query however
        long the sequence is, so the dense `12 * S^2` term does not apply.

        Not exactly 2x: the eager indexer really is quadratic, and it is the one
        term that should grow faster. At the released dimensions it is a small
        share, so the total stays close to linear -- which is the claim.
        """
        assert 2.0 <= self._flops([8192]) / self._flops([4096]) <= 2.2

    def test_it_stays_linear_where_a_quadratic_meter_would_have_run_away(self):
        """Four doublings. A `12 * S^2` attention term would grow 256x here; the
        point of the check is that the reported cost does not."""
        base = self._flops([2048])
        assert self._flops([32768]) / base <= 20.0  # 16x tokens, plus the indexer

    def test_splitting_a_batch_into_samples_does_not_change_the_linear_part(self):
        """Packing two 2048-token samples costs what two separate ones do: every
        term but the indexer's is per-token, and the indexer is per-sample."""
        assert self._flops([2048, 2048]) == pytest.approx(2 * self._flops([2048]), rel=1e-9)

    def test_a_longer_sample_costs_more_than_the_same_tokens_split_up(self):
        """The indexer is the one term that is not packing-invariant: one 4096
        sample scores more pairs than two 2048 ones."""
        assert self._flops([4096]) > self._flops([2048, 2048])

    def test_more_layers_cost_more(self):
        assert self._flops([2048], num_hidden_layers=N_LAYERS) < self._flops(
            [2048],
            num_hidden_layers=N_LAYERS + 4,
            compress_ratios=[2] * (N_LAYERS + 4) + [0, 0, 0],
        )

    def test_the_estimate_is_positive_and_finite(self):
        import math

        value = self._flops([32768])
        assert value > 0 and math.isfinite(value)


class TestIndexerKlFoldIn:
    def test_the_mean_kl_becomes_the_module_loss(self):
        loss = fold_indexer_kl_loss({"indexer_kl_total": torch.tensor(6.0), "indexer_query_tokens": 3}, 0.5)
        assert loss is not None
        assert loss.item() == pytest.approx(1.0)

    def test_a_missing_or_zero_coef_drops_the_term(self):
        assert fold_indexer_kl_loss({}, 1.0) is None
        assert fold_indexer_kl_loss({"indexer_kl_total": torch.tensor(1.0)}, 0.0) is None


class TestSequenceParallelRefusal:
    def test_ulysses_sp_is_refused_rather_than_silently_wrong(self, monkeypatch):
        """The compression plan is built in Python over the whole sequence, before
        any kernel sees it, so a sequence-sharded rank would pool across shards."""
        import veomni.models.seed_omni.modules.deepseek_v41.llm.accelerated.accelerated as module

        class FakeState:
            sp_size = 4

        monkeypatch.setattr(module, "get_parallel_state", lambda: FakeState())
        model = DeepseekV41LlmAccelerated(make_config())
        with pytest.raises(NotImplementedError, match="does not support Ulysses sequence parallelism"):
            model.forward_pre(conversation_list=[[text(4)]])


class TestRegistration:
    def test_accelerated_class_is_registered(self):
        from veomni.models.seed_omni.modules import OMNI_ACCELERATED_MODEL_REGISTRY

        assert OMNI_ACCELERATED_MODEL_REGISTRY["deepseek_v41_llm"]() is DeepseekV41LlmAccelerated

    def test_accelerated_class_is_the_native_one_plus_mixins(self):
        from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import DeepseekV41Llm

        assert issubclass(DeepseekV41LlmAccelerated, DeepseekV41Llm)
