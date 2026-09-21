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

"""DeepSeek-V4.1 backbone: the layer loop's orderings, and what it refuses.

The sublayers need the TileLang sparse attention kernel, so the loop is
exercised with the decoder layers stubbed out. That is also the right scope:
each layer's own arithmetic is checked against the released implementation
elsewhere, and what is left here is the three orderings that a wrong answer
would still look plausible for --

* the Engram write lands before its layer, not after (after would hand the
  n-gram to the next layer and leave this one reading a stream the trained
  weights never produced);
* the DSpark hook reads a target layer's *input*, matching where
  ``main_hiddens.append`` sits in the released loop;
* the final collapse uses the last layer's outgoing FFN ``pre``, which is the
  reason ``DeepseekV4HyperHead`` cannot be reused.

-- plus the refusals, each of which guards a config that would otherwise train
nothing while producing a falling loss.
"""

import pytest
import torch

from veomni.models.seed_omni.modules.deepseek_v41.llm.attention import sequence_slices_from_positions
from veomni.models.seed_omni.modules.deepseek_v41.llm.configuration import DeepseekV41LlmConfig
from veomni.models.seed_omni.modules.deepseek_v41.llm.mhc import mhc_collapse
from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import DeepseekV41Backbone, DeepseekV41Llm


HIDDEN = 64
N_LAYERS = 8
HC = 4
ENGRAM_LAYERS = [1, 5]
DRAFT_TARGETS = [5, 6, 7]


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
        hc_mult=HC,
        hc_sinkhorn_iters=6,
        compress_ratios=[2] * N_LAYERS + [0, 0, 0],
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0, 4],
        candidate_source_layer_id=0,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        engram_layer_ids=list(ENGRAM_LAYERS),
        dspark_target_layer_ids=list(DRAFT_TARGETS),
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


def ngram_rows(config, batch, seq_len):
    cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads
    return [torch.randn(batch, seq_len, cols, config.engram_head_dim) for _ in range(len(config.engram_layer_ids))]


def stub_backbone(config, indexer_layers=()):
    """A backbone whose layers record their input and pass the stream through.

    Each stub returns a fresh, layer-specific ``pre`` so that the final collapse
    can be checked against a known value rather than against whatever the last
    layer happened to be handed.
    """
    from veomni.models.seed_omni.modules.deepseek_v41.llm.attention import IndexerLossTerms

    backbone = DeepseekV41Backbone(config).float()
    with torch.no_grad():
        for param in backbone.parameters():
            param.normal_(0, 0.05)
    backbone.seen = {}

    def make_stub(layer_idx):
        def stub(
            hidden_streams,
            pre_mix,
            position_ids,
            rotary_emb,
            shared,
            plans,
            image_mask=None,
            capture=None,
            capture_seq_len=None,
        ):
            backbone.seen[layer_idx] = {
                "streams": hidden_streams.detach().clone(),
                "pre_mix": pre_mix.detach().clone(),
                "image_mask": image_mask,
            }
            outgoing = torch.full_like(pre_mix, 0.1 * (layer_idx + 1))
            logits = torch.zeros(hidden_streams.shape[0] * hidden_streams.shape[1], config.n_routed_experts)
            terms = None
            if layer_idx in indexer_layers:
                terms = IndexerLossTerms(
                    kl=torch.tensor(float(layer_idx + 1)), uniform=torch.tensor(10.0 * (layer_idx + 1))
                )
            return hidden_streams + 1.0, outgoing, logits, terms

        return stub

    for layer_idx, layer in enumerate(backbone.layers):
        layer.forward = make_stub(layer_idx)
    return backbone


class TestSequenceSlices:
    def test_packed_row_splits_at_every_zero(self):
        positions = torch.cat([torch.arange(5), torch.arange(3), torch.arange(4)]).unsqueeze(0)
        assert sequence_slices_from_positions(positions) == ((0, 5), (5, 8), (8, 12))

    def test_unpacked_row_is_one_span(self):
        assert sequence_slices_from_positions(torch.arange(7).unsqueeze(0)) == ((0, 7),)

    def test_a_row_not_starting_at_zero_is_refused(self):
        """Decode offsets are not a training input, and silently treating a
        mid-sequence chunk as a fresh sample would compress across its history."""
        with pytest.raises(ValueError, match="must start a sample at index 0"):
            sequence_slices_from_positions(torch.arange(3, 10).unsqueeze(0))

    def test_slices_agree_with_the_positions_they_came_from(self):
        lengths = (4, 6, 2)
        positions = torch.cat([torch.arange(n) for n in lengths]).unsqueeze(0)
        for (begin, end), length in zip(sequence_slices_from_positions(positions), lengths):
            assert end - begin == length
            assert positions[0, begin] == 0


class TestLoopOrdering:
    def _inputs(self, config, batch=1, seq_len=12):
        torch.manual_seed(0)
        return (
            torch.randn(batch, seq_len, HIDDEN),
            torch.arange(seq_len).unsqueeze(0).expand(batch, -1),
            ngram_rows(config, batch, seq_len),
        )

    def test_engram_lands_before_its_layer(self):
        """The layer at an Engram index must see a stream the connection changed;
        the layer before it must not."""
        config = make_config()
        backbone = stub_backbone(config)
        embeds, positions, rows = self._inputs(config)
        backbone(embeds, positions, rows)

        for engram_layer in ENGRAM_LAYERS:
            before = backbone.seen[engram_layer - 1]["streams"]
            at = backbone.seen[engram_layer]["streams"]
            # The stub adds exactly 1.0, so anything beyond that is the connection.
            assert not torch.allclose(at, before + 1.0), f"layer {engram_layer} saw no engram contribution"

    def test_layers_without_an_engram_see_only_the_previous_layer(self):
        config = make_config()
        backbone = stub_backbone(config)
        embeds, positions, rows = self._inputs(config)
        backbone(embeds, positions, rows)

        for layer_idx in range(1, N_LAYERS):
            if layer_idx in ENGRAM_LAYERS:
                continue
            torch.testing.assert_close(
                backbone.seen[layer_idx]["streams"], backbone.seen[layer_idx - 1]["streams"] + 1.0
            )

    def test_draft_hook_reads_the_layer_input(self):
        """`main_hiddens.append(h.mean(dim=2))` sits *above* `layer(...)` in the
        released loop: the MTP head reads the attention input of its target."""
        config = make_config()
        backbone = stub_backbone(config)
        embeds, positions, rows = self._inputs(config)
        outputs = backbone(embeds, positions, rows, return_draft_hiddens=True)

        drafts = outputs["draft_hiddens"]
        assert len(drafts) == len(DRAFT_TARGETS)
        for draft, layer_idx in zip(drafts, DRAFT_TARGETS):
            torch.testing.assert_close(draft, backbone.seen[layer_idx]["streams"].mean(dim=2))

    def test_draft_hiddens_are_off_by_default(self):
        config = make_config()
        backbone = stub_backbone(config)
        embeds, positions, rows = self._inputs(config)
        assert "draft_hiddens" not in backbone(embeds, positions, rows)

    def test_first_layer_collapses_with_the_identity_mix(self):
        config = make_config()
        backbone = stub_backbone(config)
        embeds, positions, rows = self._inputs(config)
        backbone(embeds, positions, rows)
        first = backbone.seen[0]["pre_mix"]
        assert bool((first[..., 0] == 1).all()) and bool((first[..., 1:] == 0).all())

    def test_each_layer_receives_the_previous_layers_outgoing_pre(self):
        config = make_config()
        backbone = stub_backbone(config)
        embeds, positions, rows = self._inputs(config)
        backbone(embeds, positions, rows)
        for layer_idx in range(1, N_LAYERS):
            expected = 0.1 * layer_idx  # the stub at layer_idx - 1 returned this
            torch.testing.assert_close(
                backbone.seen[layer_idx]["pre_mix"],
                torch.full_like(backbone.seen[layer_idx]["pre_mix"], expected),
            )

    def test_final_collapse_uses_the_last_outgoing_pre(self):
        """Not a `pre` derived from the final stream -- that is what
        `DeepseekV4HyperHead` does, and why it is not reused here."""
        config = make_config()
        backbone = stub_backbone(config)
        embeds, positions, rows = self._inputs(config)
        outputs = backbone(embeds, positions, rows)

        last_input = backbone.seen[N_LAYERS - 1]["streams"]
        final_streams = last_input + 1.0
        last_pre = torch.full(final_streams.shape[:-1], 0.1 * N_LAYERS)
        expected = backbone.norm(mhc_collapse(final_streams, last_pre))
        torch.testing.assert_close(outputs["hidden_states"], expected, atol=1e-5, rtol=1e-5)


class TestMasks:
    def _inputs(self, config, seq_len=12):
        torch.manual_seed(0)
        return torch.randn(1, seq_len, HIDDEN), torch.arange(seq_len).unsqueeze(0), ngram_rows(config, 1, seq_len)

    def test_image_mask_reaches_the_layers_unnegated(self):
        """The layers route on image tokens; only the Engram gate takes `~mask`."""
        config = make_config()
        backbone = stub_backbone(config)
        embeds, positions, rows = self._inputs(config)
        mask = torch.zeros(1, 12, dtype=torch.bool)
        mask[0, 3:6] = True
        backbone(embeds, positions, rows, image_mask=mask)
        for layer_idx in range(N_LAYERS):
            assert torch.equal(backbone.seen[layer_idx]["image_mask"], mask)

    def test_image_tokens_get_no_engram_contribution(self):
        config = make_config()
        backbone = stub_backbone(config)
        embeds, positions, rows = self._inputs(config)
        mask = torch.zeros(1, 12, dtype=torch.bool)
        mask[0, 3:6] = True

        backbone(embeds, positions, rows, image_mask=mask)
        masked = backbone.seen[ENGRAM_LAYERS[0]]["streams"]
        backbone(embeds, positions, rows, image_mask=None)
        plain = backbone.seen[ENGRAM_LAYERS[0]]["streams"]

        # Inside the span the gate is zeroed, so the stream is the untouched one.
        assert not torch.allclose(masked[:, 3:6], plain[:, 3:6])
        torch.testing.assert_close(masked[:, :3], plain[:, :3])


class TestIndexerKLAggregation:
    def _run(self, config, indexer_layers):
        backbone = stub_backbone(config, indexer_layers=indexer_layers)
        torch.manual_seed(0)
        embeds = torch.randn(1, 12, HIDDEN)
        positions = torch.arange(12).unsqueeze(0)
        return backbone(embeds, positions, ngram_rows(config, 1, 12))

    def test_terms_are_summed_over_the_layers_that_built_them(self):
        config = make_config(dsa_indexer_loss=True)
        outputs = self._run(config, indexer_layers=(0, 4))
        assert outputs["indexer_kl_total"].item() == pytest.approx(1.0 + 5.0)
        assert outputs["indexer_uniform_total"].item() == pytest.approx(10.0 + 50.0)
        assert outputs["indexer_kl_layers"] == 2

    def test_the_layer_count_travels_rather_than_dividing_the_sum(self):
        """Dividing here would make each indexer's gradient depend on how many
        other layers happened to carry one."""
        config = make_config(dsa_indexer_loss=True)
        two = self._run(config, indexer_layers=(0, 4))
        one = self._run(config, indexer_layers=(0,))
        assert two["indexer_kl_total"].item() > one["indexer_kl_total"].item()
        assert one["indexer_kl_layers"] == 1

    def test_no_kl_keys_when_the_objective_is_off(self):
        outputs = self._run(make_config(), indexer_layers=())
        assert "indexer_kl_total" not in outputs
        assert "indexer_kl_layers" not in outputs

    def test_enabled_but_unbuilt_is_refused(self):
        """A flag that is accepted and trains nothing is the failure this guards."""
        config = make_config(dsa_indexer_loss=True)
        with pytest.raises(RuntimeError, match="no layer built an indexer KL"):
            self._run(config, indexer_layers=())


class TestRefusals:
    def test_missing_ngram_embeds_is_refused(self):
        config = make_config()
        backbone = stub_backbone(config)
        with pytest.raises(ValueError, match="no ngram_embeds were passed"):
            backbone(torch.randn(1, 12, HIDDEN), torch.arange(12).unsqueeze(0))

    def test_wrong_ngram_embeds_length_is_refused(self):
        config = make_config()
        backbone = stub_backbone(config)
        rows = ngram_rows(config, 1, 12)[:1]
        with pytest.raises(ValueError, match="engram_layer_ids names"):
            backbone(torch.randn(1, 12, HIDDEN), torch.arange(12).unsqueeze(0), rows)

    def test_wrapper_requires_inputs_embeds(self):
        model = DeepseekV41Llm(make_config())
        with pytest.raises(ValueError, match="requires inputs_embeds"):
            model(position_ids=torch.arange(4).unsqueeze(0))

    def test_wrapper_requires_position_ids(self):
        """Assuming a single arange would compress across packed boundaries."""
        model = DeepseekV41Llm(make_config())
        with pytest.raises(ValueError, match="requires position_ids"):
            model(inputs_embeds=torch.randn(1, 4, HIDDEN))

    def test_embeddings_belong_to_the_text_encoder(self):
        model = DeepseekV41Llm(make_config())
        with pytest.raises(NotImplementedError, match="deepseek_v41_text_encoder"):
            model.get_input_embeddings()
        with pytest.raises(NotImplementedError, match="deepseek_v41_text_encoder"):
            model.set_input_embeddings(torch.nn.Embedding(4, HIDDEN))


class TestModuleShape:
    def test_engram_connections_only_at_declared_layers(self):
        backbone = DeepseekV41Backbone(make_config())
        assert sorted(int(key) for key in backbone.engram.keys()) == ENGRAM_LAYERS

    def test_registered_under_its_model_type(self):
        from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY

        assert OMNI_CONFIG_REGISTRY["deepseek_v41_llm"]() is DeepseekV41LlmConfig
        assert OMNI_MODEL_REGISTRY["deepseek_v41_llm"]() is DeepseekV41Llm

    def test_no_split_modules_covers_the_layer_and_the_connection(self):
        """The connections hang off the model, so the layer entry alone would
        leave them unwrapped."""
        assert set(DeepseekV41Llm._no_split_modules) == {
            "DeepseekV41DecoderLayer",
            "DeepseekV41EngramConnection",
        }
