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

"""Decoding a sequence one token at a time must equal prefilling it whole.

This is the only check that can fail for the right reason. An incremental
decoder's mistakes are all of one kind -- a group formed from the wrong tokens, a
window trimmed one row early, a compressed index not shifted by the window's
length -- and every one of them yields finite, confident, wrong logits. Shapes
stay right, nothing raises, and generated text stays fluent.

So the sequence is run both ways and compared position by position. The sparse
kernel is swapped for ``dense_sparse_attn`` on *both* sides, which is the point:
the two are already pinned to each other in ``test_deepseek_v41_decode.py``, and
holding the attention arithmetic fixed leaves the caching and the index
arithmetic as the only things this file can blame. It also lets the check run on
CPU, where it is cheap enough to cover several layer roles.

The config deliberately makes every mechanism bite within a short sequence:
``sliding_window=6`` against 17 tokens forces the window to wrap several times,
``ratio=2`` leaves a half-filled group pending on every odd position, and layer 0
is simultaneously the KV source, an index source and the candidate source.
"""

import pytest
import torch

from veomni.models.seed_omni.modules.deepseek_v41.llm import attention as attention_module
from veomni.models.seed_omni.modules.deepseek_v41.llm.attention import (
    DeepseekV41Attention,
    DeepseekV41RotaryEmbedding,
    SharedAttentionState,
    build_compression_plans,
    trailing_partial_group,
)
from veomni.models.seed_omni.modules.deepseek_v41.llm.configuration import DeepseekV41LlmConfig
from veomni.models.seed_omni.modules.deepseek_v41.llm.decode import (
    DeepseekV41AttentionCache,
    DeepseekV41DecodeCache,
    decode_attention_step,
    decode_backbone_step,
    dense_sparse_attn,
)
from veomni.models.seed_omni.modules.deepseek_v41.llm.inference import build_image_mask
from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import DeepseekV41Backbone, DeepseekV41Llm
from veomni.models.seed_omni.utils.conversation import ConversationItem


HIDDEN = 64
N_LAYERS = 8
RATIO = 2
SEQ_LEN = 17


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
        # 8 candidate slots against index_topk=4, mirroring the shipped ratio
        # (16384 against 512). Starving the candidate budget below index_topk puts
        # the two paths in the regime TestStarvedCandidates describes.
        candidate_topk_blocks=4,
        candidate_block_size=2,
        engram_layer_ids=[1],
        dspark_target_layer_ids=[5, 6, 7],
        dspark_n_routed_experts=8,
        max_position_embeddings=256,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
    )
    kwargs.update(overrides)
    return DeepseekV41LlmConfig(**kwargs)


def _build_layers(config, layer_ids, seed=0):
    """Layers with every parameter initialized -- ``torch.empty`` leaves some out."""
    torch.manual_seed(seed)
    layers = []
    for layer_idx in layer_ids:
        layer = DeepseekV41Attention(config, layer_idx=layer_idx).double()
        for param in layer.parameters():
            param.data.normal_(0, 0.05)
        # Zero-initialized by construction, so a sink bug would be invisible.
        layer.attn_sink.data.normal_(0, 0.5)
        layers.append(layer)
    return layers


@pytest.fixture(autouse=True)
def _dense_everywhere(monkeypatch):
    """Hold the attention arithmetic fixed on both sides of the comparison."""
    monkeypatch.setattr(attention_module, "sparse_attn_tilelang", dense_sparse_attn)


# The compressor pools in fp32 whatever dtype the model carries -- the released
# module does, and ``DeepseekV41Compressor`` reproduces the arithmetic rather than
# the storage. Prefill runs that fp32 matmul over the whole sequence and gathers;
# a decode step runs it over one group. Different shapes, different reduction
# order, ~1 ULP of fp32 apart. Paths that stay in the model's own dtype -- ratio 1
# and window-only -- are held to EXACT below, and the selection itself is compared
# as indices, where floating point has no say.
EXACT = dict(atol=1e-9, rtol=1e-9)
FP32_POOLED = dict(atol=1e-6, rtol=1e-4)


def _prefill(layers, hidden, config, shared=None):
    position_ids = torch.arange(hidden.shape[1]).unsqueeze(0)
    plans = build_compression_plans(config, position_ids)
    rotary = DeepseekV41RotaryEmbedding(config).double()
    shared = shared if shared is not None else SharedAttentionState()
    outputs = []
    for layer in layers:
        out, _ = layer(hidden, position_ids, rotary, shared, plans)
        outputs.append(out)
    return outputs


def _decode(layers, hidden, config):
    rotary = DeepseekV41RotaryEmbedding(config).double()
    caches = [
        DeepseekV41AttentionCache(
            sliding_window=config.sliding_window,
            compress_ratio=config.compress_ratios[layer.layer_idx],
        )
        for layer in layers
    ]
    outputs = [[] for _ in layers]
    selections = []
    for position in range(hidden.shape[1]):
        token = hidden[:, position : position + 1]
        # Fresh per step, exactly as prefill builds one per forward: the shared
        # state carries across layers, never across positions.
        shared = SharedAttentionState()
        for index, layer in enumerate(layers):
            outputs[index].append(decode_attention_step(layer, token, position, rotary, shared, caches[index]))
        selections.append(shared.topk_idxs)
    return [torch.cat(chunks, dim=1) for chunks in outputs], caches, selections


class TestDecodeMatchesPrefill:
    """The load-bearing equivalence, over each role a layer can play."""

    @pytest.mark.parametrize(
        ("layer_ids", "role"),
        [
            ([0], "kv source, index source and candidate source at once"),
            ([0, 1], "a consumer reading the source's compressed KV and selection"),
            ([0, 4], "a second index source re-ranking the shared keys"),
            ([0, 1, 4, 5], "the full publish chain"),
        ],
    )
    def test_every_position_agrees(self, layer_ids, role):
        config = make_config()
        layers = _build_layers(config, layer_ids)
        torch.manual_seed(1)
        hidden = torch.randn(1, SEQ_LEN, HIDDEN, dtype=torch.float64)

        expected = _prefill(layers, hidden, config)
        actual, _, _ = _decode(layers, hidden, config)

        for layer_idx, want, got in zip(layer_ids, expected, actual):
            torch.testing.assert_close(got, want, msg=f"layer {layer_idx} ({role})", **FP32_POOLED)

    def test_the_selection_is_identical(self):
        """Which groups were chosen -- discrete, so no tolerance can hide a shift.

        The tolerance above is a concession to fp32 pooling, and a concession is
        exactly where an off-by-one in the cache could shelter: attending to group
        j-1 instead of j moves the output by far more than 1e-6, but only on the
        inputs that happen to separate them. Indices cannot be nearly right.
        """
        config = make_config()
        layers = _build_layers(config, [0])
        torch.manual_seed(1)
        hidden = torch.randn(1, SEQ_LEN, HIDDEN, dtype=torch.float64)

        prefill_shared = SharedAttentionState()
        _prefill(layers, hidden, config, shared=prefill_shared)
        _, _, selections = _decode(layers, hidden, config)

        for position in range(SEQ_LEN):
            want_row = prefill_shared.topk_idxs[0, position]
            want = sorted(int(i) for i in want_row[want_row >= 0])
            got_row = selections[position]
            got = [] if got_row is None else sorted(int(i) for i in got_row[0, 0])
            assert got == want, f"position {position}: decode chose {got}, prefill chose {want}"

    def test_a_ratio_one_layer_agrees(self):
        """Ratio 1 takes the compressor's ungated branch, so this stays exact."""
        config = make_config(compress_ratios=[1] * N_LAYERS + [0, 0, 0])
        layers = _build_layers(config, [0, 1])
        torch.manual_seed(2)
        hidden = torch.randn(1, SEQ_LEN, HIDDEN, dtype=torch.float64)

        expected = _prefill(layers, hidden, config)
        actual, _, _ = _decode(layers, hidden, config)

        for want, got in zip(expected, actual):
            torch.testing.assert_close(got, want, **EXACT)

    def test_a_window_only_layer_agrees(self):
        """``compress_ratio == 0`` never touches the compressed cache at all."""
        config = make_config()
        layers = _build_layers(config, [N_LAYERS])  # the first of the trailing zeros
        torch.manual_seed(3)
        hidden = torch.randn(1, SEQ_LEN, HIDDEN, dtype=torch.float64)

        expected = _prefill(layers, hidden, config)
        actual, caches, _ = _decode(layers, hidden, config)

        torch.testing.assert_close(actual[0], expected[0], **EXACT)
        assert caches[0].compress_kv is None


def make_backbone_config(**overrides) -> DeepseekV41LlmConfig:
    """The attention config, shrunk where the defaults are sized for the real model.

    384 routed experts and an engram head dim of 256 are the shipped numbers and
    are not what a per-position comparison needs; everything that decides
    *behaviour* -- the source-layer roles, the window, the ratio -- is inherited
    from :func:`make_config` unchanged.
    """
    fields = dict(
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        engram_n_heads=2,
        engram_head_dim=8,
        hc_mult=2,
    )
    fields.update(overrides)
    return make_config(**fields)


class TestBackboneDecodeMatchesPrefill:
    """The whole stack, not just its attention -- mHC, Engram and MoE included.

    ``decode_layer_step`` is a hand copy of ``DeepseekV41DecoderLayer.forward``
    with one call swapped, and a copy is exactly the thing that rots quietly. This
    is what notices: anything ``forward`` grows that the copy misses moves the
    hidden states, and every position is compared.
    """

    def _backbone(self, config, seed=0):
        torch.manual_seed(seed)
        backbone = DeepseekV41Backbone(config).double()
        for param in backbone.parameters():
            param.data.normal_(0, 0.05)
        for layer in backbone.layers:
            layer.self_attn.attn_sink.data.normal_(0, 0.5)
        return backbone

    def _ngram_embeds(self, config, seq_len):
        cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        return [
            torch.randn(1, seq_len, cols, config.engram_head_dim, dtype=torch.float64) for _ in config.engram_layer_ids
        ]

    def test_every_position_agrees(self):
        config = make_backbone_config()
        backbone = self._backbone(config)
        torch.manual_seed(1)
        embeds = torch.randn(1, SEQ_LEN, HIDDEN, dtype=torch.float64)
        ngram_embeds = self._ngram_embeds(config, SEQ_LEN)
        position_ids = torch.arange(SEQ_LEN).unsqueeze(0)

        expected = backbone(embeds, position_ids, ngram_embeds=ngram_embeds)["hidden_states"]

        cache = DeepseekV41DecodeCache.empty(config)
        rows = [
            decode_backbone_step(
                backbone,
                embeds[:, position : position + 1],
                position,
                cache,
                ngram_embeds=[e[:, position : position + 1] for e in ngram_embeds],
            )
            for position in range(SEQ_LEN)
        ]
        torch.testing.assert_close(torch.cat(rows, dim=1), expected, **FP32_POOLED)

    def test_a_missing_ngram_lookup_is_refused(self):
        """Silently skipping Engram would leave the stack running, and wrong."""
        config = make_backbone_config()
        backbone = self._backbone(config)
        assert backbone.engram, "this config is supposed to have an Engram layer"
        with pytest.raises(RuntimeError, match="ngram_embeds"):
            decode_backbone_step(
                backbone,
                torch.randn(1, 1, HIDDEN, dtype=torch.float64),
                0,
                DeepseekV41DecodeCache.empty(config),
            )


class TestGenerateMatchesTeacherForcing:
    """``generate`` end to end, against the forward it has to agree with.

    The steps above test the pieces; this drives the module the FSM drives, so
    it also covers what only the wiring can get wrong -- the carrier walk, the
    cache being created and kept on the module, the position counter advancing
    once per step, and the ``output`` item the text encoder samples from.

    The text encoder is not in the loop, so its half of the contract is played
    by hand: it overwrites the tail item's value with the next token's
    embedding, which is exactly what ``DeepseekV41TextEncoder.generate`` does
    after sampling.
    """

    def _llm(self, config, seed=0):
        torch.manual_seed(seed)
        model = DeepseekV41Llm(config).double()
        for param in model.parameters():
            param.data.normal_(0, 0.05)
        for layer in model.model.layers:
            layer.self_attn.attn_sink.data.normal_(0, 0.5)
        return model.eval()

    def _ngram_embeds(self, config, seq_len):
        cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        return [
            torch.randn(1, seq_len, cols, config.engram_head_dim, dtype=torch.float64) for _ in config.engram_layer_ids
        ]

    @torch.no_grad()
    def test_a_prompt_then_steps_matches_one_forward(self):
        config = make_backbone_config()
        model = self._llm(config)
        torch.manual_seed(1)
        embeds = torch.randn(1, SEQ_LEN, HIDDEN, dtype=torch.float64)
        ngram_embeds = self._ngram_embeds(config, SEQ_LEN)

        expected = model(
            inputs_embeds=embeds,
            position_ids=torch.arange(SEQ_LEN).unsqueeze(0),
            ngram_embeds=ngram_embeds,
        )["hidden_states"]

        prompt_len = 5
        carrier = [ConversationItem(type="text", value=embeds[0, :prompt_len], role="user")]
        outputs = model.generate(
            conversation_list=carrier,
            ngram_embeds=[e[:, :prompt_len] for e in ngram_embeds],
        )
        carrier = outputs["conversation_list"]
        assert carrier[-1].type == "output"
        torch.testing.assert_close(carrier[-1].value[0], expected[0, prompt_len - 1], **FP32_POOLED)

        for position in range(prompt_len, SEQ_LEN):
            # Standing in for the text encoder: it samples a token and writes
            # that token's embedding onto this same item.
            carrier[-1].value = embeds[0, position : position + 1]
            carrier = model.generate(
                conversation_list=carrier,
                ngram_embeds=[e[:, position : position + 1] for e in ngram_embeds],
            )["conversation_list"]
            torch.testing.assert_close(carrier[-1].value[0], expected[0, position], **FP32_POOLED)

    @torch.no_grad()
    def test_reset_lets_a_second_conversation_start_clean(self):
        """Without it the next request decodes against this one's window.

        Nothing would raise: the cache is the right shape and the positions keep
        counting, so the second conversation would simply attend to the first
        one's tokens and read fluently.
        """
        config = make_backbone_config()
        model = self._llm(config)
        torch.manual_seed(2)
        embeds = torch.randn(1, 4, HIDDEN, dtype=torch.float64)
        ngram_embeds = self._ngram_embeds(config, 4)

        def run():
            carrier = [ConversationItem(type="text", value=embeds[0], role="user")]
            return model.generate(conversation_list=carrier, ngram_embeds=ngram_embeds)["conversation_list"][-1].value

        first = run()
        assert model._next_position == 4
        model.reset_global_inference_state()
        assert model._decode_cache is None and model._next_position == 0
        torch.testing.assert_close(run(), first, **EXACT)

    @torch.no_grad()
    def test_a_carrier_that_does_not_end_on_output_is_refused(self):
        config = make_backbone_config()
        model = self._llm(config)
        embeds = torch.randn(1, 4, HIDDEN, dtype=torch.float64)
        ngram_embeds = self._ngram_embeds(config, 4)
        carrier = [ConversationItem(type="text", value=embeds[0], role="user")]
        model.generate(conversation_list=carrier, ngram_embeds=ngram_embeds)

        carrier.append(ConversationItem(type="text", value=embeds[0, :1], role="user"))
        with pytest.raises(ValueError, match="'output' item"):
            model.generate(conversation_list=carrier, ngram_embeds=ngram_embeds)

    def test_the_image_mask_is_none_without_an_image(self):
        """None, not all-false: the MoE bias and the Engram gate both skip on None."""
        carrier = [ConversationItem(type="text", value=torch.zeros(3, HIDDEN), role="user")]
        assert build_image_mask(carrier, torch.device("cpu")) is None

    def test_the_image_mask_spans_the_image_item(self):
        carrier = [
            ConversationItem(type="text", value=torch.zeros(2, HIDDEN), role="user"),
            ConversationItem(type="image", value=torch.zeros(3, HIDDEN), role="user"),
            ConversationItem(type="text", value=torch.zeros(1, HIDDEN), role="user"),
        ]
        mask = build_image_mask(carrier, torch.device("cpu"))
        assert mask.shape == (1, 6)
        assert mask[0].tolist() == [False, False, True, True, True, False]


class TestStarvedCandidates:
    """The one regime where decode and prefill legitimately disagree.

    When the candidate mask leaves a query fewer than ``index_topk`` slots, both
    paths pick ``index_topk`` anyway and pad the remainder with entries scored
    ``-inf``. Which ones is a tie-break among equals, and ``topk`` settles it by
    array width: prefill ranks over every group in the sequence, a decode step
    over the groups that exist so far. Same finite picks, different padding.

    The padding is inert for the window-plus-compressed attention that consumes
    it -- a padded slot is one the indexer scored at the floor -- but it is a real
    difference in which rows get read, so it is pinned here rather than left to be
    rediscovered as a mystery. ``DeepseekV41Indexer``'s own docstring notes this
    cannot arise at the shipped settings, where 16384 candidate positions back an
    ``index_topk`` of 512; the config above keeps the ratio on that side.
    """

    def test_a_starved_budget_diverges_only_in_the_padding(self):
        config = make_config(candidate_topk_blocks=2)  # 4 slots, equal to index_topk
        assert config.candidate_topk_blocks * config.candidate_block_size <= config.index_topk
        layers = _build_layers(config, [0, 4])
        torch.manual_seed(1)
        hidden = torch.randn(1, SEQ_LEN, HIDDEN, dtype=torch.float64)

        prefill_shared = SharedAttentionState()
        _prefill(layers, hidden, config, shared=prefill_shared)
        _, _, selections = _decode(layers, hidden, config)

        divergent = 0
        for position in range(SEQ_LEN):
            want_row = prefill_shared.topk_idxs[0, position]
            want = {int(i) for i in want_row[want_row >= 0]}
            got_row = selections[position]
            got = set() if got_row is None else {int(i) for i in got_row[0, 0]}
            if got != want:
                divergent += 1
                # Prefill dropped padding that fell outside the sequence's visible
                # groups; decode had no such slots to pick and padded inside.
                assert want.issubset(got), f"position {position}: decode lost {want - got}"
        assert divergent, "the starved regime stopped reproducing -- retune the config or drop this test"

    def test_the_shipped_ratio_leaves_room(self):
        """The guard that keeps the default config out of the regime above."""
        config = make_config()
        assert config.candidate_topk_blocks * config.candidate_block_size > config.index_topk


class TestCacheState:
    """What the cache holds, since a wrong-but-consistent cache still agrees."""

    def test_the_window_is_trimmed_not_grown(self):
        config = make_config()
        layers = _build_layers(config, [0])
        hidden = torch.randn(1, SEQ_LEN, HIDDEN, dtype=torch.float64)
        _, caches, _ = _decode(layers, hidden, config)
        assert caches[0].window_len == config.sliding_window

    def test_one_group_per_ratio_tokens(self):
        config = make_config()
        layers = _build_layers(config, [0])
        hidden = torch.randn(1, SEQ_LEN, HIDDEN, dtype=torch.float64)
        _, caches, _ = _decode(layers, hidden, config)
        assert caches[0].compressed_len == SEQ_LEN // RATIO
        # 17 tokens at ratio 2 leaves the 17th waiting for a partner.
        assert caches[0].pending is not None
        assert caches[0].pending.shape[1] == SEQ_LEN % RATIO

    def test_the_compressed_cache_matches_prefills_latents(self):
        """Append-only is only sound if group j never changes after it is written."""
        config = make_config()
        layers = _build_layers(config, [0])
        hidden = torch.randn(1, SEQ_LEN, HIDDEN, dtype=torch.float64)

        position_ids = torch.arange(SEQ_LEN).unsqueeze(0)
        plans = build_compression_plans(config, position_ids)
        rotary = DeepseekV41RotaryEmbedding(config).double()
        shared = SharedAttentionState()
        layers[0](hidden, position_ids, rotary, shared, plans)

        _, caches, _ = _decode(layers, hidden, config)
        torch.testing.assert_close(caches[0].compress_kv, shared.compress_kv, **FP32_POOLED)
        torch.testing.assert_close(caches[0].index_k, shared.index_k, **FP32_POOLED)

    def test_empty_cache_reports_zero(self):
        cache = DeepseekV41AttentionCache(sliding_window=6, compress_ratio=2)
        assert cache.window_len == 0
        assert cache.compressed_len == 0

    def test_decode_cache_spans_the_backbone(self):
        config = make_config()
        cache = DeepseekV41DecodeCache.empty(config)
        assert len(cache.layers) == config.num_hidden_layers
        assert cache.layers[0].compress_ratio == config.compress_ratios[0]
        assert len(cache.dspark) == config.num_nextn_predict_layers
        assert all(slot.compress_ratio == 0 for slot in cache.dspark)
        assert cache.position == 0

    def test_a_multi_token_step_is_refused(self):
        config = make_config()
        layers = _build_layers(config, [0])
        rotary = DeepseekV41RotaryEmbedding(config).double()
        cache = DeepseekV41AttentionCache(sliding_window=config.sliding_window, compress_ratio=RATIO)
        with pytest.raises(ValueError, match="one token"):
            decode_attention_step(
                layers[0],
                torch.randn(1, 2, HIDDEN, dtype=torch.float64),
                0,
                rotary,
                SharedAttentionState(),
                cache,
            )


def _clone_tensor(tensor):
    return None if tensor is None else tensor.detach().clone()


def _snapshot_layer_cache(cache):
    return {
        "window": _clone_tensor(cache.window),
        "compress_kv": _clone_tensor(cache.compress_kv),
        "index_k": _clone_tensor(cache.index_k),
        "pending": _clone_tensor(cache.pending),
    }


def _assert_same_tensor(got, want, name, **tol):
    if want is None or got is None:
        assert got is None and want is None, f"{name}: one side is None"
        return
    assert got.shape == want.shape, f"{name} shape {tuple(got.shape)} != {tuple(want.shape)}"
    if tol:
        torch.testing.assert_close(got, want, msg=name, **tol)
        return
    assert torch.equal(got, want), f"{name} is not bitwise equal; max abs {(got - want).abs().max().item():.3g}"


class TestBatchedPrefillMatchesPerPosition:
    """The capture the batched prefill writes must be the per-position cache.

    ``_prefill`` fills the decode cache from one training forward; ``_prefill_per_position``
    is the loop it replaced. If they disagree, decode after a fast prefill attends
    to the wrong history and the text stays fluent.

    Bitwise equality holds for anything that never enters the compressor's fp32
    pool: the source layer's window and pending (functions of the prompt
    embeddings) and the None-ness of every non-source field. The compressed
    history, later layers, and the tail hidden go through that pool -- prefill
    matmuls over the whole sequence, a decode step over one group -- and are
    held to the same ``FP32_POOLED`` tolerance as :class:`TestDecodeMatchesPrefill`.
    Discrete token ids after further decode steps must match exactly.
    """

    def _llm(self, config, seed=0):
        torch.manual_seed(seed)
        model = DeepseekV41Llm(config).double()
        for param in model.parameters():
            param.data.normal_(0, 0.05)
        for layer in model.model.layers:
            layer.self_attn.attn_sink.data.normal_(0, 0.5)
        return model.eval()

    def _ngram_embeds(self, config, seq_len):
        cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        return [
            torch.randn(1, seq_len, cols, config.engram_head_dim, dtype=torch.float64) for _ in config.engram_layer_ids
        ]

    def _run_prefill(self, model, method, embeds, ngram_embeds):
        carrier = [ConversationItem(type="text", value=embeds[0].clone(), role="user")]
        outputs = method(
            carrier,
            ngram_embeds=[e.clone() for e in ngram_embeds],
        )
        return (
            outputs["conversation_list"][-1].value.detach().clone(),
            [_snapshot_layer_cache(layer) for layer in model._decode_cache.layers],
            int(model._next_position),
        )

    def _decode_tokens(self, model, embeds, ngram_embeds, start, n_steps, lm_head):
        tokens = []
        carrier = None
        for position in range(start, start + n_steps):
            if carrier is None:
                # Standing in for the text encoder after prefill: it samples from
                # the tail hidden and writes the next token's embedding on it.
                carrier = [
                    ConversationItem(type="text", value=embeds[0, :start], role="user"),
                    ConversationItem(type="output", value=embeds[0, position : position + 1], role="assistant"),
                ]
            else:
                carrier[-1].value = embeds[0, position : position + 1]
            carrier = model.generate(
                conversation_list=carrier,
                ngram_embeds=[e[:, position : position + 1] for e in ngram_embeds],
            )["conversation_list"]
            tokens.append(int((carrier[-1].value.squeeze(0).double() @ lm_head).argmax()))
        return tokens

    @torch.no_grad()
    def test_tail_and_every_cache_field_agree_bitwise(self):
        config = make_backbone_config()
        # 5 tokens at ratio 2: two complete groups and a trailing partial, so
        # window, compress_kv, index_k and pending all have something to say.
        prompt_len = 5
        assert prompt_len > RATIO and prompt_len % RATIO == 1

        torch.manual_seed(1)
        embeds = torch.randn(1, prompt_len, HIDDEN, dtype=torch.float64)
        ngram_embeds = self._ngram_embeds(config, prompt_len)

        stepped = self._llm(config)
        want_tail, want_cache, want_pos = self._run_prefill(
            stepped, stepped._prefill_per_position, embeds, ngram_embeds
        )

        batched = self._llm(config)
        got_tail, got_cache, got_pos = self._run_prefill(batched, batched._prefill, embeds, ngram_embeds)

        assert got_pos == want_pos == prompt_len
        _assert_same_tensor(got_tail, want_tail, "tail hidden", **FP32_POOLED)
        assert len(got_cache) == len(want_cache) == config.num_hidden_layers
        source_ids = set(config.kv_source_layer_ids)
        for layer_idx, (got, want) in enumerate(zip(got_cache, want_cache)):
            # The source layer's window and pending are functions of the prompt
            # embeddings, which both paths share, so they must agree bit for bit.
            # Everything downstream of the compressor's fp32 pool is close, not equal.
            window_tol = {} if layer_idx == 0 else FP32_POOLED
            _assert_same_tensor(got["window"], want["window"], f"layer {layer_idx} window", **window_tol)
            if layer_idx in source_ids:
                _assert_same_tensor(
                    got["compress_kv"], want["compress_kv"], f"layer {layer_idx} compress_kv", **FP32_POOLED
                )
                _assert_same_tensor(got["index_k"], want["index_k"], f"layer {layer_idx} index_k", **FP32_POOLED)
                pending_tol = {} if layer_idx == 0 else FP32_POOLED
                _assert_same_tensor(got["pending"], want["pending"], f"layer {layer_idx} pending", **pending_tol)
            else:
                for field in ("compress_kv", "index_k", "pending"):
                    _assert_same_tensor(got[field], want[field], f"layer {layer_idx} {field}")

        source = config.kv_source_layer_ids[0]
        assert got_cache[source]["pending"] is not None
        assert got_cache[source]["pending"].shape[1] == prompt_len % RATIO
        assert got_cache[source]["compress_kv"] is not None
        assert got_cache[source]["compress_kv"].shape[1] == prompt_len // RATIO
        for layer_idx, snapshot in enumerate(got_cache):
            if layer_idx in config.kv_source_layer_ids:
                continue
            assert snapshot["compress_kv"] is None
            assert snapshot["index_k"] is None
            assert snapshot["pending"] is None

    @torch.no_grad()
    def test_decode_after_either_prefill_emits_the_same_tokens(self):
        config = make_backbone_config()
        prompt_len = 5
        n_steps = 4
        total = prompt_len + n_steps

        torch.manual_seed(1)
        embeds = torch.randn(1, total, HIDDEN, dtype=torch.float64)
        ngram_embeds = self._ngram_embeds(config, total)
        torch.manual_seed(0)
        lm_head = torch.randn(HIDDEN, 32, dtype=torch.float64)

        tokens = []
        for method_name in ("_prefill_per_position", "_prefill"):
            model = self._llm(config)
            getattr(model, method_name)(
                [ConversationItem(type="text", value=embeds[0, :prompt_len].clone(), role="user")],
                [e[:, :prompt_len].clone() for e in ngram_embeds],
            )
            tokens.append(self._decode_tokens(model, embeds, ngram_embeds, prompt_len, n_steps, lm_head))

        assert tokens[0] == tokens[1]
        assert len(tokens[0]) == n_steps

    def test_pending_is_the_last_packed_samples_partial_group(self):
        """Packed remainders must not be concatenated -- that would look like a full group."""
        hidden = torch.arange(10, dtype=torch.float64).view(1, 10, 1).expand(1, 10, HIDDEN).contiguous()
        position_ids = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]])
        pending = trailing_partial_group(hidden, position_ids, RATIO)
        assert pending is not None
        assert pending.shape[1] == 1
        torch.testing.assert_close(pending[0, 0], hidden[0, 9])
