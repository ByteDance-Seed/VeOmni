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

"""The n-gram node's packing, against the backbone row it has to line up with.

The rows this node emits are indexed positionally by the backbone. That makes
one property load-bearing above all others: the sequence packed here and the
sequence packed by the backbone must have the same length and the same
positions, *including* image spans -- which the shared ``pack_input_ids`` drops,
because the text encoder has no use for them.

The other half is the id copy in ``meta``. The text encoder overwrites each text
item's ``value`` with its embedding, so this node cannot read ids from ``value``
unless the graph happens to schedule it first -- and nothing in the DAG says it
will, since both nodes depend only on vision.
"""

import pytest
import torch

from veomni.models.seed_omni.modules.base.llm_packing import pack_llm_conversations_for_forward
from veomni.models.seed_omni.modules.deepseek_v41.ngram.accelerated.accelerated import pack_ngram_inputs
from veomni.models.seed_omni.modules.deepseek_v41.ngram.hashing import DEAD
from veomni.models.seed_omni.utils.conversation import ConversationItem


PAD = 2
HIDDEN = 16
CPU = torch.device("cpu")


def text_item(ids, role="user"):
    """A text item as it looks *after* the text encoder has run: value is an
    embedding, and the ids survive only in meta."""
    tokens = torch.tensor(ids, dtype=torch.long)
    return ConversationItem(
        type="text",
        value=torch.randn(len(ids), HIDDEN),
        role=role,
        meta={"input_ids": tokens},
    )


def text_item_before_encode(ids, role="user"):
    tokens = torch.tensor(ids, dtype=torch.long)
    return ConversationItem(type="text", value=tokens, role=role, meta={"input_ids": tokens.clone()})


def image_item(n_tokens):
    return ConversationItem(type="image", value=torch.randn(n_tokens, HIDDEN), role="user")


def dummy_item(n):
    return ConversationItem(type="text", value=torch.randn(n, HIDDEN), role="user", is_dummy=True, meta={})


class TestAlignmentWithTheBackboneRow:
    """The property the backbone's length check exists to catch."""

    def test_length_matches_the_packed_backbone_row(self):
        conversation = [[text_item([5, 6, 7]), image_item(4), text_item([8, 9])]]
        ids, mask = pack_ngram_inputs(conversation, CPU, PAD)
        embeds, _, _, _ = pack_llm_conversations_for_forward(conversation, CPU)
        assert ids.shape[1] == embeds.shape[1] == 9
        assert mask.shape == ids.shape

    def test_image_spans_are_covered_rather_than_skipped(self):
        """`pack_input_ids` drops them; this packer must not, or every position
        after the first image is off by the image's length."""
        conversation = [[text_item([5, 6]), image_item(3), text_item([7])]]
        ids, mask = pack_ngram_inputs(conversation, CPU, PAD)
        assert torch.equal(ids[0], torch.tensor([5, 6, PAD, PAD, PAD, 7]))
        assert torch.equal(mask[0], torch.tensor([True, True, False, False, False, True]))

    def test_alignment_holds_across_several_samples(self):
        conversation = [
            [text_item([1, 2]), image_item(2)],
            [image_item(1), text_item([3, 4, 5])],
        ]
        ids, _ = pack_ngram_inputs(conversation, CPU, PAD)
        embeds, _, _, _ = pack_llm_conversations_for_forward(conversation, CPU)
        assert ids.shape[1] == embeds.shape[1]

    def test_dummy_items_are_skipped_the_way_the_packer_skips_them(self):
        conversation = [[text_item([1, 2]), dummy_item(5), image_item(2)]]
        ids, _ = pack_ngram_inputs(conversation, CPU, PAD)
        embeds, _, _, _ = pack_llm_conversations_for_forward(conversation, CPU)
        assert ids.shape[1] == embeds.shape[1] == 4


class TestIdsSurviveTheTextEncoder:
    def test_ids_are_read_from_meta_not_value(self):
        """After encode, `value` is an embedding. Reading it would either raise
        on the dtype or hash float bits."""
        conversation = [[text_item([11, 12, 13])]]
        ids, _ = pack_ngram_inputs(conversation, CPU, PAD)
        assert torch.equal(ids[0], torch.tensor([11, 12, 13]))

    def test_same_result_before_and_after_the_encoder_has_run(self):
        """Order-independence is the reason the copy exists: both nodes depend
        only on vision, so the DAG does not fix which runs first."""
        before = pack_ngram_inputs([[text_item_before_encode([4, 5, 6])]], CPU, PAD)[0]
        after = pack_ngram_inputs([[text_item([4, 5, 6])]], CPU, PAD)[0]
        assert torch.equal(before, after)

    def test_a_text_item_without_the_copy_is_refused(self):
        item = ConversationItem(type="text", value=torch.randn(3, HIDDEN), role="user", meta={})
        with pytest.raises(ValueError, match=r"meta\['input_ids'\]"):
            pack_ngram_inputs([[item]], CPU, PAD)


class TestTemplateStashesTheIds:
    def _template(self):
        from veomni.models.seed_omni.modules.deepseek_v41.text_encoder.chat_template import DeepseekV41ChatTemplate

        class FakeTokenizer:
            unk_token_id = 0

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [ord(char) for char in text]}

            def convert_tokens_to_ids(self, token):
                return 7

        return DeepseekV41ChatTemplate(FakeTokenizer())

    def test_tokenize_leaves_a_copy_in_meta(self):
        template = self._template()
        parts = [ConversationItem(type="text", value="abc", role="user", meta={"loss_mask": 1})]
        template.tokenize(parts)
        assert torch.equal(parts[0].meta["input_ids"], parts[0].value)

    def test_the_copy_is_not_an_alias_of_value(self):
        """`encode` rebinds `value`, but a later in-place edit of either must not
        reach through to the other."""
        template = self._template()
        parts = [ConversationItem(type="text", value="abc", role="user", meta={"loss_mask": 1})]
        template.tokenize(parts)
        parts[0].value[0] = 999
        assert parts[0].meta["input_ids"][0] != 999

    def test_merging_keeps_the_copy_in_step_with_value(self):
        """The base merge concatenates `value`; a stale copy would leave every
        n-gram past the join hashed from the wrong tokens."""
        template = self._template()
        parts = [
            ConversationItem(type="text", value="ab", role="user", meta={"loss_mask": 1}),
            ConversationItem(type="text", value="cd", role="user", meta={"loss_mask": 1}),
        ]
        template.tokenize(parts)
        merged = template.merge_text_embeds(parts)
        assert len(merged) == 1
        assert torch.equal(merged[0].meta["input_ids"], merged[0].value)
        assert merged[0].meta["input_ids"].tolist() == [ord(c) for c in "abcd"]


class TestContractWithTheBackbone:
    """The two packers meet at a length check; this exercises both sides of it."""

    def _conversation(self):
        return [
            [text_item([5, 6, 7]), image_item(4), text_item([8, 9])],
            [text_item([1]), image_item(2)],
        ]

    def test_the_ngram_rows_are_accepted_by_the_backbone(self):
        from tests.seed_omni.modules.deepseek_v41.test_deepseek_v41_accelerated import make_config
        from veomni.models.seed_omni.modules.deepseek_v41.llm.accelerated.accelerated import (
            DeepseekV41LlmAccelerated,
        )

        conversation = self._conversation()
        ids, _ = pack_ngram_inputs(conversation, CPU, PAD)
        embeds, _, _, _ = pack_llm_conversations_for_forward(conversation, CPU)

        # One row per position, as the lookup would emit.
        rows = [torch.randn(1, ids.shape[1], 6, 8) for _ in range(2)]
        model = DeepseekV41LlmAccelerated(make_config())
        assert model._align_ngram_embeds(rows, embeds.shape[1]) is not None

    def test_the_text_only_packing_would_be_rejected(self):
        """Guards the check itself: `pack_input_ids` drops image spans, and a row
        built that way is exactly what the backbone must refuse."""
        from tests.seed_omni.modules.deepseek_v41.test_deepseek_v41_accelerated import make_config
        from veomni.models.seed_omni.modules.deepseek_v41.llm.accelerated.accelerated import (
            DeepseekV41LlmAccelerated,
        )

        conversation = self._conversation()
        text_only = sum(
            item.meta["input_ids"].numel() for sample in conversation for item in sample if item.type == "text"
        )
        embeds, _, _, _ = pack_llm_conversations_for_forward(conversation, CPU)
        assert text_only < embeds.shape[1]  # the images are the difference

        model = DeepseekV41LlmAccelerated(make_config())
        with pytest.raises(ValueError, match="positions but the packed row has"):
            model._align_ngram_embeds([torch.randn(1, text_only, 6, 8)], embeds.shape[1])


class TestMaskReachesTheHashing:
    def test_masked_positions_compress_to_dead(self):
        """The mask is what stops a lookback from reading across an image; the
        id put there is incidental."""
        from veomni.models.seed_omni.modules.deepseek_v41.ngram.configuration import DeepseekV41NgramConfig
        from veomni.models.seed_omni.modules.deepseek_v41.ngram.modeling import DeepseekV41Ngram

        config = DeepseekV41NgramConfig(
            layer_ids=[0],
            num_embeddings=[1024],
            max_ngram_size=4,
            n_heads=2,
            head_dim=8,
            vocab_size=64,
            hash_vocab_size=64,
            compressed_vocab_size=32,
            pad_token_id=PAD,
        )
        model = DeepseekV41Ngram(config)
        with torch.no_grad():
            model.token_map.copy_(torch.arange(config.vocab_size) % config.compressed_vocab_size)

        conversation = [[text_item([5, 6]), image_item(3), text_item([7])]]
        ids, mask = pack_ngram_inputs(conversation, CPU, PAD)
        compressed = model.compress(ids, mask)
        assert (compressed[0, 2:5] == DEAD).all()
        assert (compressed[0, :2] != DEAD).all() and compressed[0, 5] != DEAD


class TestTheTwoVocabSizes:
    """``vocab_size`` (129280, the tokenizer's) and ``hash_vocab_size``
    (16000000, the base the bucket primes are drawn above) are unrelated
    quantities three orders of magnitude apart. Folding them into one field
    sizes ``token_map`` at 16M int64 rows and makes the convert step ask for a
    tokenizer with 16M tokens -- which is how this was found."""

    @staticmethod
    def _config(**overrides):
        from veomni.models.seed_omni.modules.deepseek_v41.ngram.configuration import DeepseekV41NgramConfig

        fields = dict(
            layer_ids=[0],
            num_embeddings=[1024],
            max_ngram_size=4,
            n_heads=2,
            head_dim=8,
            vocab_size=64,
            hash_vocab_size=64,
            compressed_vocab_size=32,
            pad_token_id=PAD,
        )
        fields.update(overrides)
        return DeepseekV41NgramConfig(**fields)

    def test_released_defaults_keep_them_apart(self):
        config = self._config(vocab_size=129280, hash_vocab_size=16000000, num_embeddings=[384006168])
        assert (config.vocab_size, config.hash_vocab_size) == (129280, 16000000)

    def test_token_map_is_sized_by_the_tokenizer(self):
        from veomni.models.seed_omni.modules.deepseek_v41.ngram.modeling import DeepseekV41Ngram

        model = DeepseekV41Ngram(self._config(vocab_size=48, hash_vocab_size=64))
        assert model.token_map.shape == (48,)

    def test_the_bucket_floor_follows_the_hash_base(self):
        """``num_embeddings`` sized against the tokenizer instead would pass a
        floor 250x too low and leave the tables reading wrong rows."""
        with pytest.raises(ValueError, match="hash_vocab_size"):
            self._config(hash_vocab_size=1024, num_embeddings=[1024]).validate_bucket_layout()
        self._config(hash_vocab_size=64, num_embeddings=[1024]).validate_bucket_layout()


class TestFreeze:
    """``freeze: true`` in a module config is only honoured when the module
    implements ``freeze_model`` -- the runtime looks the attribute up rather
    than requiring it, so a missing method is accepted and silently ignored.
    G1 settled on training these tables frozen, so that silence would cost
    optimizer state for 196.6B parameters before anyone noticed."""

    @staticmethod
    def _build(freeze: bool):
        from veomni.models.seed_omni.modules.deepseek_v41.ngram.configuration import DeepseekV41NgramConfig
        from veomni.models.seed_omni.modules.deepseek_v41.ngram.modeling import DeepseekV41Ngram

        config = DeepseekV41NgramConfig(
            layer_ids=[0],
            num_embeddings=[1024],
            max_ngram_size=4,
            n_heads=2,
            head_dim=8,
            vocab_size=64,
            hash_vocab_size=64,
            compressed_vocab_size=32,
            pad_token_id=PAD,
            freeze=freeze,
        )
        return DeepseekV41Ngram(config)

    def test_the_runtime_can_find_the_hook(self):
        assert callable(getattr(self._build(freeze=True), "freeze_model", None))

    def test_freeze_reaches_every_parameter(self):
        model = self._build(freeze=True)
        model.freeze_model()
        assert [p.requires_grad for p in model.parameters()] == [False]

    def test_the_tables_are_the_whole_module(self):
        """Freezing ``tables`` is freezing the module only while that holds; a
        parameter added outside them would quietly stay trainable."""
        names = [name for name, _ in self._build(freeze=False).named_parameters()]
        assert names and all(name.startswith("tables.") for name in names)

    def test_default_leaves_the_tables_trainable(self):
        model = self._build(freeze=False)
        model.freeze_model()
        assert all(p.requires_grad for p in model.parameters())
