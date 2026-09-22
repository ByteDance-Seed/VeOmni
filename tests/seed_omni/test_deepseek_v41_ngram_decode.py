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

"""Hashing a sequence one token at a time must equal hashing it whole.

Hash ids are integers, so this file gets what the attention cache's tests could
not: exact equality, with no tolerance for a mistake to hide under. Either the
lookback read the same tokens or it did not.

What is actually under test is the claim that lets the cache stay small. An
n-gram reaches back ``max_ngram_size - 1`` tokens, so the cache keeps a tail of
that width and nothing else -- a thousand-token conversation carries the same
three ids a four-token one does. The tests below drive sequences several times
longer than the lookback so a tail that was trimmed too eagerly, or read from
the wrong end, cannot come out even by accident.
"""

import pytest
import torch

from veomni.models.seed_omni.modules.deepseek_v41.ngram.configuration import DeepseekV41NgramConfig
from veomni.models.seed_omni.modules.deepseek_v41.ngram.decode import (
    DeepseekV41NgramCache,
    ngram_lookup_step,
)
from veomni.models.seed_omni.modules.deepseek_v41.ngram.hashing import DEAD
from veomni.models.seed_omni.modules.deepseek_v41.ngram.modeling import DeepseekV41Ngram


PAD = 3
VOCAB = 64
MAX_NGRAM = 4
SEQ_LEN = 13  # several times the lookback, so a mistrimmed tail cannot pass


def make_config(**overrides) -> DeepseekV41NgramConfig:
    fields = dict(
        layer_ids=[0],
        num_embeddings=[1024],
        max_ngram_size=MAX_NGRAM,
        n_heads=2,
        head_dim=8,
        vocab_size=VOCAB,
        hash_vocab_size=VOCAB,
        compressed_vocab_size=32,
        pad_token_id=PAD,
    )
    fields.update(overrides)
    return DeepseekV41NgramConfig(**fields)


def make_module(config=None) -> DeepseekV41Ngram:
    """A module whose ``token_map`` actually distinguishes tokens.

    The buffer is registered as zeros and filled by the convert step, so a module
    built straight from a config collapses the whole vocabulary onto id 0 -- every
    n-gram would hash alike and every test here would pass on a broken lookback.
    """
    config = config or make_config()
    module = DeepseekV41Ngram(config)
    torch.manual_seed(0)
    module.token_map.copy_(torch.arange(config.vocab_size) % config.compressed_vocab_size)
    return module


def make_ids(seq_len=SEQ_LEN, seed=0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, VOCAB, (1, seq_len))


class TestIncrementalEqualsWhole:
    """The property the cache exists to preserve."""

    def test_token_by_token_matches_the_whole_sequence(self):
        module = make_module()
        input_ids = make_ids()
        expected = module.hash_ids(input_ids)

        cache = DeepseekV41NgramCache.from_config(module.config)
        rows = []
        for position in range(SEQ_LEN):
            token = input_ids[:, position : position + 1]
            rows.append(module.hash_ids(token, history=cache.history, start_pos=cache.start_pos))
            cache.advance(module.compress(token))

        assert torch.equal(torch.cat(rows, dim=1), expected)

    @pytest.mark.parametrize("split", [1, 3, 4, 5, SEQ_LEN - 1])
    def test_a_prompt_then_the_rest_matches(self, split):
        """A real generation prefills the prompt in one call and steps after it."""
        module = make_module()
        input_ids = make_ids()
        expected = module.hash_ids(input_ids)

        cache = DeepseekV41NgramCache.from_config(module.config)
        rows = [module.hash_ids(input_ids[:, :split], history=cache.history, start_pos=cache.start_pos)]
        cache.advance(module.compress(input_ids[:, :split]))
        for position in range(split, SEQ_LEN):
            token = input_ids[:, position : position + 1]
            rows.append(module.hash_ids(token, history=cache.history, start_pos=cache.start_pos))
            cache.advance(module.compress(token))

        assert torch.equal(torch.cat(rows, dim=1), expected)

    def test_an_image_span_still_blocks_across_the_boundary(self):
        """DEAD has to survive in the tail, or an n-gram hashes over an image.

        The mask is placed so the dead tokens land in the prompt and the queries
        that must not reach them land in the steps after it -- the blocking then
        depends on the cache carrying DEAD, not on anything the step can see.
        """
        module = make_module()
        input_ids = make_ids()
        token_mask = torch.ones(1, SEQ_LEN, dtype=torch.bool)
        token_mask[:, 4:6] = False
        expected = module.hash_ids(input_ids, token_mask)

        split = 6  # the dead pair lands at the tail of the prompt
        cache = DeepseekV41NgramCache.from_config(module.config)
        rows = [
            module.hash_ids(
                input_ids[:, :split], token_mask[:, :split], history=cache.history, start_pos=cache.start_pos
            )
        ]
        cache.advance(module.compress(input_ids[:, :split], token_mask[:, :split]))
        # The claim being tested: DEAD is in the tail the steps will read.
        assert (cache.history == DEAD).any()

        for position in range(split, SEQ_LEN):
            token = input_ids[:, position : position + 1]
            mask = token_mask[:, position : position + 1]
            rows.append(module.hash_ids(token, mask, history=cache.history, start_pos=cache.start_pos))
            cache.advance(module.compress(token, mask))

        assert torch.equal(torch.cat(rows, dim=1), expected)

    def test_the_helper_agrees_with_the_manual_loop(self):
        module = make_module()
        input_ids = make_ids()

        manual = DeepseekV41NgramCache.from_config(module.config)
        for position in range(SEQ_LEN):
            manual.advance(module.compress(input_ids[:, position : position + 1]))

        helper = DeepseekV41NgramCache.from_config(module.config)
        outputs = None
        for position in range(SEQ_LEN):
            outputs = ngram_lookup_step(module, input_ids[:, position : position + 1], helper)

        assert torch.equal(helper.history, manual.history)
        assert helper.start_pos == manual.start_pos == SEQ_LEN
        assert set(outputs) >= {"ngram_embeds", "ngram_compressed"}


class TestTheTrapTheCacheCloses:
    """Why this is a class and not two loose arguments at the call site."""

    def test_history_without_start_pos_would_corrupt_the_lookback(self):
        """``history=None, start_pos>0`` reads forward from index 0 unblocked.

        The blocking rule stops a walk that runs off the start of the sequence,
        and it decides that from ``start_pos``. Tell it the walk is at position 9
        while handing it no history and the walk stays unblocked, clamps to index
        0, and hashes the query token against itself. Nothing raises and the ids
        look ordinary. :class:`DeepseekV41NgramCache` cannot hold that pair.
        """
        module = make_module()
        input_ids = make_ids()
        position = 9
        token = input_ids[:, position : position + 1]

        correct = module.hash_ids(input_ids)[:, position : position + 1]
        corrupt = module.hash_ids(token, history=None, start_pos=position)
        assert not torch.equal(corrupt, correct)

        cache = DeepseekV41NgramCache.from_config(module.config)
        cache.advance(module.compress(input_ids[:, :position]))
        assert torch.equal(module.hash_ids(token, history=cache.history, start_pos=cache.start_pos), correct)


class TestTheCacheStaysSmall:
    """Constant-sized state is the whole point; a leak here is a slow OOM."""

    def test_the_tail_never_outgrows_the_lookback(self):
        module = make_module()
        input_ids = make_ids(seq_len=64, seed=1)
        cache = DeepseekV41NgramCache.from_config(module.config)
        for position in range(64):
            cache.advance(module.compress(input_ids[:, position : position + 1]))
            assert cache.history.shape[1] <= MAX_NGRAM - 1
        assert cache.history.shape[1] == MAX_NGRAM - 1
        assert cache.start_pos == 64

    def test_a_long_prompt_is_trimmed_in_one_go(self):
        module = make_module()
        cache = DeepseekV41NgramCache.from_config(module.config)
        cache.advance(module.compress(make_ids(seq_len=1000, seed=2)))
        assert cache.history.shape[1] == MAX_NGRAM - 1
        assert cache.start_pos == 1000

    def test_the_tail_keeps_the_newest_ids(self):
        """Trimmed from the correct end -- the oldest go, the newest stay."""
        module = make_module()
        input_ids = make_ids(seq_len=10, seed=3)
        cache = DeepseekV41NgramCache.from_config(module.config)
        cache.advance(module.compress(input_ids))
        assert torch.equal(cache.history, module.compress(input_ids)[:, -(MAX_NGRAM - 1) :])

    def test_a_zero_lookback_keeps_nothing(self):
        """``[-0:]`` is the identity slice, so the bound is computed explicitly."""
        cache = DeepseekV41NgramCache(lookback=0)
        cache.advance(torch.arange(5).unsqueeze(0))
        assert cache.history.shape[1] == 0
        assert cache.start_pos == 5

    def test_from_config_reads_the_lookback_off_the_ngram_size(self):
        assert DeepseekV41NgramCache.from_config(make_config()).lookback == MAX_NGRAM - 1

    def test_a_rejected_draft_rewinds_the_tail(self):
        """Batch-verify hashes the whole block, then restores tokens that missed."""
        module = make_module()
        cache = DeepseekV41NgramCache.from_config(module.config)
        first = module.compress(make_ids(seq_len=4, seed=4))
        extra = module.compress(make_ids(seq_len=2, seed=5))
        cache.advance(first)
        snap = cache.snapshot()
        cache.advance(extra)
        cache.restore(snap)
        assert cache.start_pos == 4
        assert torch.equal(cache.history, first[:, -(MAX_NGRAM - 1) :] if first.shape[1] > MAX_NGRAM - 1 else first)
