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

"""DeepSeek-V4.1 mHC wiring and MoE routing: the invariants, not the arithmetic.

Numeric agreement with the released ``inference/model.py`` is checked by a
development harness, which needs that repository on disk. What is pinned here is
everything that would still produce a well-shaped, plausible tensor if it were
wrong:

* the one-site-delayed ``pre`` -- collapsing with the wrong site's coefficients
  is a working model that is not this model;
* the correction biases being buffers, so an optimizer never steps them;
* the routing weights coming from unbiased scores, so the balancing controller
  cannot change how much an expert contributes.

All CPU, fp32, and none of it calls the sparse attention kernel.
"""

import pytest
import torch

from veomni.models.seed_omni.modules.deepseek_v41.llm.configuration import DeepseekV41LlmConfig
from veomni.models.seed_omni.modules.deepseek_v41.llm.decoder_layer import DeepseekV41DecoderLayer
from veomni.models.seed_omni.modules.deepseek_v41.llm.mhc import (
    DeepseekV41HyperConnection,
    expand_to_streams,
    identity_pre_mix,
    mhc_collapse,
    mhc_expand,
)
from veomni.models.seed_omni.modules.deepseek_v41.llm.moe import (
    DeepseekV41SparseMoeBlock,
    DeepseekV41TopKRouter,
)


HIDDEN = 64
N_LAYERS = 8
HC = 4


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
        engram_layer_ids=[1],
        dspark_target_layer_ids=[5, 6, 7],
        max_position_embeddings=256,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        n_shared_experts=1,
    )
    kwargs.update(overrides)
    return DeepseekV41LlmConfig(**kwargs)


def randomized(module: torch.nn.Module, std: float = 0.05) -> torch.nn.Module:
    """``nn.Parameter(torch.empty(...))`` leaves uninitialised memory behind."""
    module = module.float()
    with torch.no_grad():
        for param in module.parameters():
            param.normal_(0, std)
    return module


class TestHyperConnection:
    def test_comb_is_doubly_stochastic(self):
        """Sinkhorn's entire purpose: the stream mixer preserves total mass."""
        torch.manual_seed(0)
        hc = randomized(DeepseekV41HyperConnection(make_config()))
        _, _, comb = hc(torch.randn(2, 7, HC, HIDDEN))
        torch.testing.assert_close(comb.sum(-1), torch.ones_like(comb.sum(-1)), atol=1e-3, rtol=0)
        torch.testing.assert_close(comb.sum(-2), torch.ones_like(comb.sum(-2)), atol=1e-3, rtol=0)

    def test_post_stays_in_its_range(self):
        """``post`` is a doubled sigmoid, so a value outside [0, 2] means the
        scale or base got applied to the wrong slice of the projection."""
        torch.manual_seed(0)
        hc = randomized(DeepseekV41HyperConnection(make_config()), std=1.0)
        _, post, _ = hc(torch.randn(2, 7, HC, HIDDEN))
        assert bool(((post >= 0) & (post <= 2)).all())

    def test_pre_is_returned_rather_than_applied(self):
        """The V4/V4.1 divergence in one assertion.

        ``DeepseekV4HyperConnection`` returns ``(post, comb, collapsed)`` and
        keeps ``pre`` to itself. V4.1 needs it for the *next* site, so it comes
        back as the first element and nothing here is collapsed.
        """
        torch.manual_seed(0)
        hc = randomized(DeepseekV41HyperConnection(make_config()))
        pre, post, comb = hc(torch.randn(2, 7, HC, HIDDEN))
        assert pre.shape == (2, 7, HC)
        assert post.shape == (2, 7, HC)
        assert comb.shape == (2, 7, HC, HC)


class TestCollapseAndExpand:
    def test_expand_matches_the_released_spelling(self):
        """The matmul rewrite is the same contraction, without the [b,s,hc,hc,d].

        The released ``hc_post`` materialises an intermediate ``hc_mult`` times
        the size of the stream it is mixing. Transposing ``comb`` instead is the
        same sum over the source stream -- and getting the transpose backwards
        would mix the streams in the wrong direction, which is a doubly
        stochastic matrix either way and so passes every shape and mass check.
        """
        torch.manual_seed(0)
        output = torch.randn(2, 5, HIDDEN)
        residual = torch.randn(2, 5, HC, HIDDEN)
        post = torch.rand(2, 5, HC) * 2
        comb = torch.softmax(torch.randn(2, 5, HC, HC), dim=-1)
        released = post.unsqueeze(-1) * output.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
        )
        torch.testing.assert_close(mhc_expand(output, residual, post, comb), released, atol=1e-5, rtol=1e-5)

    def test_identity_pre_mix_recovers_the_embedding(self):
        """Every copy holds the same vector at the start, and the first collapse
        has to return that vector rather than ``hc_mult`` times it."""
        embeddings = torch.randn(2, 5, HIDDEN)
        streams = expand_to_streams(embeddings, HC)
        collapsed = mhc_collapse(streams, identity_pre_mix(streams, HC))
        torch.testing.assert_close(collapsed, embeddings)

    def test_collapse_is_a_weighted_sum_over_copies(self):
        streams = torch.randn(1, 3, HC, HIDDEN)
        pre = torch.rand(1, 3, HC)
        expected = (pre.unsqueeze(-1) * streams).sum(dim=2)
        torch.testing.assert_close(mhc_collapse(streams, pre), expected)


class TestRouter:
    def test_correction_biases_are_buffers(self):
        """They are load-balancing controllers updated by a sign rule on expert
        load, not by the objective. As parameters the optimizer would apply
        momentum and weight decay to a quantity whose job is to track load."""
        router = DeepseekV41TopKRouter(make_config(enable_vision_routing_bias=True))
        names = {name for name, _ in router.named_parameters()}
        assert "e_score_correction_bias" not in names
        assert "e_score_correction_bias_vl" not in names
        buffers = {name for name, _ in router.named_buffers()}
        assert {"e_score_correction_bias", "e_score_correction_bias_vl"} <= buffers

    def test_no_vl_bias_without_vision(self):
        router = DeepseekV41TopKRouter(make_config(enable_vision_routing_bias=False))
        assert router.e_score_correction_bias_vl is None

    def test_weights_are_unbiased_scores(self):
        """The bias picks experts; it must not scale them.

        Checked with the renormalisation off, because renormalising divides by
        the sum over the selected set -- so a shared expert legitimately gets a
        different weight once the selection moves.
        """
        torch.manual_seed(0)
        config = make_config(norm_topk_prob=False, routed_scaling_factor=1.0)
        router = randomized(DeepseekV41TopKRouter(config))
        with torch.no_grad():
            router.e_score_correction_bias.normal_(0, 1.0)
        hidden = torch.randn(1, 6, HIDDEN)
        _, weights, indices = router(hidden)
        unbiased = torch.nn.functional.softplus(
            torch.nn.functional.linear(hidden.reshape(-1, HIDDEN), router.weight)
        ).sqrt()
        torch.testing.assert_close(weights, unbiased.gather(1, indices), atol=1e-6, rtol=1e-5)

    def test_vl_bias_moves_the_selection(self):
        torch.manual_seed(0)
        config = make_config(enable_vision_routing_bias=True)
        router = randomized(DeepseekV41TopKRouter(config))
        with torch.no_grad():
            router.e_score_correction_bias.normal_(0, 1.0)
            router.e_score_correction_bias_vl.normal_(0, 1.0)
        hidden = torch.randn(1, 8, HIDDEN)
        _, _, text = router(hidden, torch.zeros(1, 8, dtype=torch.bool))
        _, _, image = router(hidden, torch.ones(1, 8, dtype=torch.bool))
        assert bool((text.sort(-1).values != image.sort(-1).values).any())

    def test_image_mask_is_ignored_without_a_vl_bias(self):
        """A text-only checkpoint must route image tokens like any other."""
        torch.manual_seed(0)
        router = randomized(DeepseekV41TopKRouter(make_config(enable_vision_routing_bias=False)))
        hidden = torch.randn(1, 6, HIDDEN)
        _, _, masked = router(hidden, torch.ones(1, 6, dtype=torch.bool))
        _, _, plain = router(hidden, None)
        assert bool((masked == plain).all())

    def test_gate_temp_divides_the_logits(self):
        torch.manual_seed(0)
        hidden = torch.randn(1, 6, HIDDEN)
        plain = randomized(DeepseekV41TopKRouter(make_config(gate_temp=1.0)))
        warmed = DeepseekV41TopKRouter(make_config(gate_temp=2.0)).float()
        with torch.no_grad():
            warmed.weight.copy_(plain.weight)
        torch.testing.assert_close(warmed(hidden)[0] * 2.0, plain(hidden)[0], atol=1e-5, rtol=1e-5)

    def test_single_expert_is_never_renormalised(self):
        """``weights / weights.sum()`` at top-1 is identically 1, which throws the
        score away. V4 renormalises unconditionally; V4.1 guards on ``top_k > 1``."""
        torch.manual_seed(0)
        config = make_config(num_experts_per_tok=1, norm_topk_prob=True, routed_scaling_factor=1.0)
        router = randomized(DeepseekV41TopKRouter(config))
        _, weights, _ = router(torch.randn(1, 6, HIDDEN))
        assert not bool(torch.allclose(weights, torch.ones_like(weights)))

    def test_renormalised_weights_sum_to_the_scaling_factor(self):
        torch.manual_seed(0)
        config = make_config(num_experts_per_tok=2, norm_topk_prob=True, routed_scaling_factor=1.5)
        router = randomized(DeepseekV41TopKRouter(config))
        _, weights, _ = router(torch.randn(1, 6, HIDDEN))
        torch.testing.assert_close(weights.sum(-1), torch.full((6,), 1.5), atol=1e-5, rtol=1e-5)


class TestSparseMoeBlock:
    def test_reused_v4_experts_read_the_config_aliases(self):
        """``DeepseekV4Experts`` and ``DeepseekV4MLP`` read ``num_local_experts``
        / ``intermediate_size`` / ``mlp_bias``, which V4.1 exposes as properties
        over its own field names. A rename on either side lands here."""
        config = make_config()
        block = DeepseekV41SparseMoeBlock(config)
        assert block.experts.gate_up_proj.shape == (
            config.n_routed_experts,
            2 * config.moe_intermediate_size,
            config.hidden_size,
        )
        assert block.experts.down_proj.shape == (
            config.n_routed_experts,
            config.hidden_size,
            config.moe_intermediate_size,
        )
        assert block.shared_experts.gate_proj.bias is None

    def test_forward_shape_and_router_logits(self):
        torch.manual_seed(0)
        config = make_config()
        block = randomized(DeepseekV41SparseMoeBlock(config))
        hidden = torch.randn(2, 5, HIDDEN)
        output, logits = block(hidden)
        assert output.shape == hidden.shape
        assert logits.shape == (2 * 5, config.n_routed_experts)


class TestDecoderLayerWiring:
    """The one-site-delayed ``pre``, with the sublayers stubbed out.

    The sublayers need the TileLang sparse attention kernel and an SM90 device,
    and they are not what is under test: the question is which coefficients
    reach which collapse.
    """

    @staticmethod
    def _stubbed(config, layer_idx=0):
        layer = randomized(DeepseekV41DecoderLayer(config, layer_idx))

        def attention(hidden_states, *args, **kwargs):
            return hidden_states * 1.5 - 0.25, None

        def mlp(hidden_states, image_mask=None):
            return hidden_states.tanh(), None

        layer.self_attn.forward = attention
        layer.mlp.forward = mlp
        return layer

    def test_attention_collapses_with_the_incoming_pre(self):
        torch.manual_seed(0)
        config = make_config()
        layer = self._stubbed(config)
        streams = torch.randn(2, 5, HC, HIDDEN)
        incoming = torch.rand(2, 5, HC) + 0.1

        output, outgoing, _, _ = layer(streams, incoming, torch.arange(5).unsqueeze(0), None, None, {})

        attn_pre, attn_post, attn_comb = layer.attn_hc(streams)
        expected = mhc_expand(
            layer.self_attn.forward(layer.input_layernorm(mhc_collapse(streams, incoming)))[0],
            streams,
            attn_post,
            attn_comb,
        )
        ffn_pre, ffn_post, ffn_comb = layer.ffn_hc(expected)
        expected = mhc_expand(
            layer.mlp.forward(layer.post_attention_layernorm(mhc_collapse(expected, attn_pre)))[0],
            expected,
            ffn_post,
            ffn_comb,
        )
        torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(outgoing, ffn_pre)

    def test_using_the_ffns_own_pre_would_differ(self):
        """Without this the test above passes for a model wired like V4's."""
        torch.manual_seed(0)
        config = make_config()
        layer = self._stubbed(config)
        streams = torch.randn(2, 5, HC, HIDDEN)
        incoming = torch.rand(2, 5, HC) + 0.1

        output, _, _, _ = layer(streams, incoming, torch.arange(5).unsqueeze(0), None, None, {})

        attn_pre, attn_post, attn_comb = layer.attn_hc(streams)
        after_attn = mhc_expand(
            layer.self_attn.forward(layer.input_layernorm(mhc_collapse(streams, incoming)))[0],
            streams,
            attn_post,
            attn_comb,
        )
        ffn_pre, ffn_post, ffn_comb = layer.ffn_hc(after_attn)
        v4_style = mhc_expand(
            layer.mlp.forward(layer.post_attention_layernorm(mhc_collapse(after_attn, ffn_pre)))[0],
            after_attn,
            ffn_post,
            ffn_comb,
        )
        assert (output - v4_style).abs().max() > 1e-4

    def test_outgoing_pre_feeds_the_next_layer(self):
        """A two-layer chain has to thread the first block's FFN ``pre`` into the
        second block's attention collapse, and nothing else."""
        torch.manual_seed(0)
        config = make_config()
        first = self._stubbed(config, layer_idx=0)
        second = self._stubbed(config, layer_idx=1)
        streams = expand_to_streams(torch.randn(2, 5, HIDDEN), HC)

        pre = identity_pre_mix(streams, HC)
        streams, pre, _, _ = first(streams, pre, torch.arange(5).unsqueeze(0), None, None, {})
        expected_collapse = mhc_collapse(streams, pre)
        streams_after, _, _, _ = second(streams, pre, torch.arange(5).unsqueeze(0), None, None, {})

        # The second block's attention input is the collapse of the first
        # block's output with the first block's FFN pre.
        attn_post, attn_comb = second.attn_hc(streams)[1:]
        expected = mhc_expand(
            second.self_attn.forward(second.input_layernorm(expected_collapse))[0], streams, attn_post, attn_comb
        )
        attn_pre = second.attn_hc(streams)[0]
        ffn_pre, ffn_post, ffn_comb = second.ffn_hc(expected)
        expected = mhc_expand(
            second.mlp.forward(second.post_attention_layernorm(mhc_collapse(expected, attn_pre)))[0],
            expected,
            ffn_post,
            ffn_comb,
        )
        torch.testing.assert_close(streams_after, expected, atol=1e-5, rtol=1e-5)

    def test_layer_owns_two_independent_hyper_connections(self):
        """One shared module would tie the attention and FFN sites' coefficients,
        which is a smaller model that trains and never reaches this accuracy."""
        layer = DeepseekV41DecoderLayer(make_config(), 0)
        assert layer.attn_hc is not layer.ffn_hc
        assert layer.attn_hc.fn is not layer.ffn_hc.fn


@pytest.mark.parametrize("layer_idx", [0, 3, 7])
def test_every_backbone_layer_builds(layer_idx):
    layer = DeepseekV41DecoderLayer(make_config(), layer_idx)
    assert layer.layer_idx == layer_idx
