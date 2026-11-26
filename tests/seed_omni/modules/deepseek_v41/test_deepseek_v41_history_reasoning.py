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

"""Historical-turn reasoning in the V4.1 wire format.

Serving does not replay earlier turns' ``reasoning_content``: the Anthropic bridge
between the agent harness and the model strips it, so every historical assistant
turn arrives with an empty ``</think>``. Training that only ever saw the full
reasoning therefore runs out of distribution from the second turn onward.

``history_reasoning_drop_rate`` mixes both shapes into one run: a fraction of the
samples compact their history the way serving does, the rest keep it the way the
released encoder renders it, and the turn under training always keeps its own.
"""

from __future__ import annotations

import pytest

from veomni.models.seed_omni.modules.deepseek_v41.text_encoder.chat_template import (
    ASSISTANT_SP_TOKEN,
    THINKING_END_TOKEN,
    THINKING_START_TOKEN,
    DeepseekV41ChatTemplate,
)
from veomni.models.seed_omni.utils.conversation import ConversationItem


TOOL = {"type": "function", "function": {"name": "bash", "parameters": {}}}


class _CharTokenizer:
    """One id per character; enough for the template, no vocabulary to load."""

    unk_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in text]}

    def convert_tokens_to_ids(self, token):
        return 7


def _template(**kwargs) -> DeepseekV41ChatTemplate:
    return DeepseekV41ChatTemplate(_CharTokenizer(), **kwargs)


def _tool_conversation(reasoning: str = "why") -> list[ConversationItem]:
    """system(tools) -> user -> assistant(history) -> user -> assistant(trained)."""
    return [
        ConversationItem(type="text", value="sys", role="system", meta={"tools": [TOOL]}),
        ConversationItem(type="text", value="q1", role="user"),
        ConversationItem(type="text", value="a1", role="assistant", meta={"reasoning_content": reasoning}),
        ConversationItem(type="text", value="q2", role="user"),
        ConversationItem(type="text", value="a2", role="assistant", meta={"reasoning_content": "final"}),
    ]


def _rendered(template: DeepseekV41ChatTemplate, sample) -> str:
    return "".join(str(item.value) for item in template.apply_chat_template(sample))


def test_history_reasoning_survives_when_the_drop_rate_is_zero():
    text = _rendered(_template(history_reasoning_drop_rate=0.0), _tool_conversation("why"))
    assert THINKING_START_TOKEN + "why" + THINKING_END_TOKEN in text


def test_history_reasoning_is_compacted_when_the_drop_rate_is_one():
    text = _rendered(_template(history_reasoning_drop_rate=1.0), _tool_conversation("why"))
    assert "why" not in text
    # The compacted turn closes the block its header never opened.
    assert ASSISTANT_SP_TOKEN + THINKING_END_TOKEN + "a1" in text


@pytest.mark.parametrize("rate", [0.1, 0.5, 0.9])
def test_the_turn_under_training_always_keeps_its_reasoning(rate):
    """The dropped reasoning is the *history*; the last assistant turn is the
    behaviour being trained and must never lose its block."""
    text = _rendered(_template(history_reasoning_drop_rate=rate), _tool_conversation("why"))
    assert "final" in text
    assert THINKING_START_TOKEN + "final" + THINKING_END_TOKEN in text


def test_the_choice_is_stable_for_the_same_sample():
    """A re-drawn mask would renumber the token stream on resume, under an
    unchanged sample order, so the decision has to come from the content."""
    template = _template(history_reasoning_drop_rate=0.5)
    first = _rendered(template, _tool_conversation("why"))
    for _ in range(5):
        assert _rendered(template, _tool_conversation("why")) == first


def test_the_choice_varies_across_samples():
    template = _template(history_reasoning_drop_rate=0.5)

    def conversation(marker: str):
        return [
            ConversationItem(type="text", value="sys", role="system", meta={"tools": [TOOL]}),
            ConversationItem(type="text", value=f"q1-{marker}", role="user"),
            ConversationItem(type="text", value="a1", role="assistant", meta={"reasoning_content": "why"}),
            ConversationItem(type="text", value="q2", role="user"),
            ConversationItem(type="text", value="a2", role="assistant", meta={"reasoning_content": "final"}),
        ]

    rendered = [_rendered(template, conversation(f"m{i}")) for i in range(24)]
    kept = [text for text in rendered if "why" in text]
    assert kept, "expected some samples to keep their history reasoning"
    assert len(kept) < len(rendered), "expected some samples to compact it"


def test_a_conversation_without_tools_compacts_either_way():
    """No tools means the released encoder's usual compaction; the drop rate is
    not consulted, so both settings must agree."""

    def conversation():
        return [
            ConversationItem(type="text", value="sys", role="system"),
            ConversationItem(type="text", value="q1", role="user"),
            ConversationItem(type="text", value="a1", role="assistant", meta={"reasoning_content": "why"}),
            ConversationItem(type="text", value="q2", role="user"),
            ConversationItem(type="text", value="a2", role="assistant", meta={"reasoning_content": "final"}),
        ]

    kept = _rendered(_template(history_reasoning_drop_rate=0.0), conversation())
    dropped = _rendered(_template(history_reasoning_drop_rate=1.0), conversation())
    assert kept == dropped
    assert "why" not in kept


@pytest.mark.parametrize("rate", [-0.1, 1.5, float("nan")])
def test_an_out_of_range_drop_rate_is_rejected(rate):
    with pytest.raises(ValueError, match="history_reasoning_drop_rate"):
        _template(history_reasoning_drop_rate=rate)
