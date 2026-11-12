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

"""Tests for `GenerationPromptTokenizerTemplate`, its Gemma 4 variants, and the official sudoku template."""

import pytest

from veomni.data.chat_template import DiffusionGemmaSudokuOfficialTemplate, build_chat_template
from veomni.utils.constants import IGNORE_INDEX


class _StubTokenizer:
    """Renders `<|turn>{role}\\n{content}<turn|>\\n` turns, one id per whitespace token.

    Mirrors the shape of the DiffusionGemma checkpoint's template -- markers are single
    ids, the generation prompt opens the model turn, and each turn only appends -- without
    needing the 47GB repo on disk.
    """

    chat_template = "{# stub #}"

    #: Markers get ids of their own, mirroring a vocabulary that contains them.
    markers = {"<|turn>": 105, "<turn|>": 106, "\n": 107}

    #: Words are interned as they are first seen, so ids stay collision-free and stable
    #: across runs -- `hash()` is randomized per process and would make this flaky.
    _vocab: dict = {}
    bos_token = "<bos>"
    bos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        ids = [ord(char) for char in text]
        return ([2] if add_special_tokens else []) + ids

    def __init__(self, append_only: bool = True):
        self.append_only = append_only

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, return_dict=True, **kwargs):
        ids = []
        for message in messages:
            role = "model" if message["role"] == "assistant" else message["role"]
            ids += [self.markers["<|turn>"], self._word(role), self.markers["\n"]]
            ids += [self._word(word) for word in message["content"].split()]
            ids += [self.markers["<turn|>"], self.markers["\n"]]
        if add_generation_prompt:
            ids += [self.markers["<|turn>"], self._word("model"), self.markers["\n"]]
        if not self.append_only and not add_generation_prompt:
            ids = ids[::-1]
        return {"input_ids": ids}

    @classmethod
    def _word(cls, word: str) -> int:
        return cls._vocab.setdefault(word, 1000 + len(cls._vocab))


class _Gemma4StubTokenizer(_StubTokenizer):
    """Mimic Gemma 4's reasoning guard and non-thinking prompt rewrite."""

    chat_template = """{% if thinking_text and loop.index0 > ns_turn.last_user_idx and message.get('tool_calls') %}
{%- if not continue_same_model_turn -%}
        {{- '<|turn>' + role + '\\n' }}
    {%- endif -%}"""

    def __init__(self):
        super().__init__()
        self.enable_thinking_values = []

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt=False,
        enable_thinking=False,
        chat_template=None,
        **kwargs,
    ):
        self.enable_thinking_values.append(enable_thinking)
        rendered_messages = []
        training_template = chat_template and "and message.get('tool_calls')" not in chat_template
        no_thinking_training_template = chat_template and "<|channel>thought\\n<channel|>" in chat_template
        for message in messages:
            rendered = dict(message)
            if (
                training_template
                and not no_thinking_training_template
                and message["role"] == "assistant"
                and message.get("reasoning_content")
            ):
                rendered["content"] = message["reasoning_content"] + " " + message["content"]
            rendered_messages.append(rendered)
        encoded = {"input_ids": []}
        for message in rendered_messages:
            role = "model" if message["role"] == "assistant" else message["role"]
            encoded["input_ids"] += [self.markers["<|turn>"], self._word(role), self.markers["\n"]]
            if no_thinking_training_template and role == "model":
                encoded["input_ids"].append(9999)
            encoded["input_ids"] += [self._word(word) for word in message["content"].split()]
            encoded["input_ids"] += [self.markers["<turn|>"], self.markers["\n"]]
        if add_generation_prompt:
            encoded["input_ids"] += [self.markers["<|turn>"], self._word("model"), self.markers["\n"]]
        if add_generation_prompt and not enable_thinking:
            encoded["input_ids"].append(9999)
        return encoded


def _template(**kwargs):
    return build_chat_template("tokenizer_generation_prompt", _StubTokenizer(**kwargs))


def _supervised(encoded):
    return [i for i, label in zip(encoded["input_ids"], encoded["labels"]) if label != IGNORE_INDEX]


def test_only_the_assistant_turns_are_supervised():
    encoded = _template().encode_messages(
        [
            {"role": "user", "content": "u one"},
            {"role": "assistant", "content": "a one"},
        ]
    )
    tokenizer = _StubTokenizer()
    # The answer plus the terminator that ends the model's turn, and nothing before it.
    assert _supervised(encoded) == [
        tokenizer._word("a"),
        tokenizer._word("one"),
        tokenizer.markers["<turn|>"],
        tokenizer.markers["\n"],
    ]


def test_the_turn_terminator_is_supervised_so_the_model_learns_to_stop():
    """A model that never sees its own turn terminator never emits one, and generation
    then runs to the block ceiling on every prompt."""
    encoded = _template().encode_messages([{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}])
    assert _StubTokenizer.markers["<turn|>"] in _supervised(encoded)


def test_every_assistant_turn_of_a_multi_turn_conversation_is_supervised():
    encoded = _template().encode_messages(
        [
            {"role": "user", "content": "u one"},
            {"role": "assistant", "content": "a one"},
            {"role": "user", "content": "u two"},
            {"role": "assistant", "content": "a two"},
        ]
    )
    tokenizer = _StubTokenizer()
    supervised = _supervised(encoded)
    assert tokenizer._word("a") in supervised
    assert tokenizer._word("one") in supervised
    assert tokenizer._word("two") in supervised
    # The user turn between them stays masked.
    assert tokenizer._word("u") not in supervised


def test_labels_repeat_the_input_at_supervised_positions():
    encoded = _template().encode_messages(
        [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a b c"}]
    )
    assert all(label in (IGNORE_INDEX, token) for token, label in zip(encoded["input_ids"], encoded["labels"]))
    assert len(encoded["labels"]) == len(encoded["input_ids"]) == len(encoded["attention_mask"])


def test_a_system_prompt_is_not_supervised():
    encoded = _template().encode_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
    )
    assert _StubTokenizer()._word("sys") not in _supervised(encoded)


def test_an_explicit_loss_mask_overrides_the_role_default():
    encoded = _template().encode_messages(
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a", "loss_mask": 0},
        ]
    )
    assert _supervised(encoded) == []


def test_a_template_that_rewrites_earlier_turns_is_rejected():
    """Silently accepting one would shift the label mask onto the prompt."""
    with pytest.raises(ValueError, match="not append-only"):
        _template(append_only=False).encode_messages(
            [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
        )


def test_a_tokenizer_without_a_template_is_rejected():
    tokenizer = _StubTokenizer()
    tokenizer.chat_template = None
    with pytest.raises(ValueError, match="no chat template of its own"):
        build_chat_template("tokenizer_generation_prompt", tokenizer).encode_messages(
            [{"role": "user", "content": "u"}]
        )


def test_truncation_keeps_the_end_of_the_conversation():
    encoded = _template().encode_messages(
        [{"role": "user", "content": "u " * 20}, {"role": "assistant", "content": "a"}], max_seq_len=4
    )
    assert len(encoded["input_ids"]) == 4
    assert len(encoded["labels"]) == 4
    # The answer sits at the end, so truncation must not cut it away.
    assert _StubTokenizer().markers["<turn|>"] in encoded["input_ids"]


def test_the_exported_jinja_is_the_tokenizers_own():
    """The export writes this back onto the tokenizer, so returning a rewritten template
    here would ship a checkpoint whose template no longer matches training."""
    assert _template().get_jinja_template() == _StubTokenizer.chat_template


def test_gemma4_thinking_template_includes_reasoning_and_stays_append_only():
    tokenizer = _Gemma4StubTokenizer()
    encoded = build_chat_template("gemma4_thinking", tokenizer).encode_messages(
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "reasoning_content": "reason", "content": "a"},
        ]
    )

    assert _supervised(encoded)
    assert tokenizer._word("reason") in _supervised(encoded)
    assert tokenizer.enable_thinking_values
    assert all(tokenizer.enable_thinking_values)


def test_gemma4_no_thinking_matches_the_serving_prompt_and_stays_append_only():
    tokenizer = _Gemma4StubTokenizer()
    messages = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "reasoning_content": "private reason", "content": "a"},
    ]
    template = build_chat_template("gemma4_no_thinking", tokenizer)
    encoded = template.encode_messages(messages)
    serving_prompt = tokenizer.apply_chat_template(
        messages[:1], tokenize=True, add_generation_prompt=True, return_dict=True, enable_thinking=False
    )["input_ids"]

    assert encoded["input_ids"][: len(serving_prompt)] == serving_prompt
    assert encoded["labels"][: len(serving_prompt)] == [IGNORE_INDEX] * len(serving_prompt)
    assert tokenizer._word("a") in _supervised(encoded)
    assert tokenizer._word("private") not in encoded["input_ids"]
    assert tokenizer._word("reason") not in encoded["input_ids"]
    assert tokenizer.enable_thinking_values
    assert not any(tokenizer.enable_thinking_values)


def test_official_sudoku_template_has_exact_nemo_prompt_and_answer_boundary():
    tokenizer = _StubTokenizer()
    template = build_chat_template("diffusion_gemma_sudoku_official", tokenizer)
    messages = [
        {"role": "system", "content": template.SYSTEM},
        {"role": "user", "content": "0 1 2"},
        {"role": "assistant", "content": "3 4 5"},
    ]
    encoded = template.encode_messages(messages)
    prompt = f"<|turn>system\n{template.SYSTEM}<turn|>\n<|turn>user\n0 1 2<turn|>\n<|turn>model\n"
    prompt_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode("3 4 5<turn|>\n", add_special_tokens=False)
    assert encoded["input_ids"] == prompt_ids + answer_ids
    assert encoded["labels"] == [IGNORE_INDEX] * len(prompt_ids) + answer_ids
    assert 9999 not in encoded["input_ids"]


def test_official_sudoku_jinja_matches_nemo_for_prompt_and_full_turn():
    from jinja2 import Environment

    tokenizer = _StubTokenizer()
    template = build_chat_template("diffusion_gemma_sudoku_official", tokenizer)
    messages = [
        {"role": "system", "content": template.SYSTEM},
        {"role": "user", "content": "0 1 2"},
    ]
    expected = f"<bos><|turn>system\n{template.SYSTEM}<turn|>\n<|turn>user\n0 1 2<turn|>\n<|turn>model\n"
    rendered = (
        Environment()
        .from_string(template.get_jinja_template())
        .render(
            bos_token=tokenizer.bos_token,
            messages=messages,
            add_generation_prompt=True,
        )
    )

    assert rendered == expected
    assert rendered.endswith("<|turn>model\n")

    full = (
        Environment()
        .from_string(template.get_jinja_template())
        .render(
            bos_token=tokenizer.bos_token,
            messages=[*messages, {"role": "assistant", "content": "3 4 5"}],
            add_generation_prompt=False,
        )
    )
    assert full == expected + "3 4 5<turn|>\n"


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "system", "content": "wrong"}, {"role": "user", "content": "0 1 2"}],
        [{"role": "system", "content": DiffusionGemmaSudokuOfficialTemplate.SYSTEM}],
        [
            {"role": "system", "content": DiffusionGemmaSudokuOfficialTemplate.SYSTEM},
            {"role": "assistant", "content": "0 1 2"},
        ],
        [
            {"role": "system", "content": DiffusionGemmaSudokuOfficialTemplate.SYSTEM},
            {"role": "user", "content": "0 1 2"},
            {"role": "tool", "content": "3 4 5"},
        ],
    ],
)
def test_official_sudoku_jinja_refuses_template_drift(messages):
    from jinja2 import Environment

    tokenizer = _StubTokenizer()
    template = build_chat_template("diffusion_gemma_sudoku_official", tokenizer)

    def raise_exception(message):
        raise ValueError(message)

    environment = Environment()
    environment.globals["raise_exception"] = raise_exception
    with pytest.raises(ValueError):
        environment.from_string(template.get_jinja_template()).render(
            bos_token=tokenizer.bos_token,
            messages=messages,
            add_generation_prompt=len(messages) == 2,
        )


def test_the_role_header_is_supervised_only_by_the_plain_tokenizer_template():
    """`tokenizer` supervises every appended token; this variant starts after the generation prompt."""
    messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    model_word = _StubTokenizer._word("model")
    assert model_word not in _supervised(_template().encode_messages(messages))
    assert model_word in _supervised(build_chat_template("tokenizer", _StubTokenizer()).encode_messages(messages))
