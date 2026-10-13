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

"""DeepSeek-V4.1 chat template.

The released checkpoint ships **no** ``chat_template.json`` and its
``tokenizer_config.json`` has no jinja template: the wire format lives only in
``encoding/encoding.py``, a 979-line Python encoder over OpenAI-shaped messages.
This file ports that wire format to the SeedOmni carrier -- the markers, the DSML
tool-call tags, the numeric reasoning-effort prefix and the thinking-mode
transitions are reproduced literally, checked against ``encoding/test_encoding.py``
by ``tests/models/test_deepseek_v41_chat_template.py``.

Two deliberate differences from the released encoder, both consequences of the
carrier rather than choices about the format:

* **Images are items, not placeholders.** ``encoding.py`` rewrites ``<image>``
  tags into ``<｜deepseek_image｜>`` and leaves a downstream processor to expand
  each into its span. Here the sibling ``deepseek_v41_vision`` module has already
  produced the span's embeddings, so an image row is passed through verbatim and
  contributes no text tokens at all. The token sequence is identical either way.

* **Turns come from item roles.** ``encoding.py``'s per-message ``context`` /
  ``drop_thinking`` bookkeeping collapses to one question the carrier can answer
  directly: is this the last user turn? Reasoning content on earlier assistant
  turns is dropped exactly as ``_drop_thinking_messages`` does.

Tool definitions, tool calls and reasoning content ride on ``item.meta``
(``tools`` / ``tool_calls`` / ``reasoning_content``), which is where the data
layer puts anything that is not the item's own ``value``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ....utils.conversation import ConversationItem
from ...base.text_encoder.chat_template import TextEncoderChatTemplate


BOS_TOKEN = "<｜begin▁of▁sentence｜>"
EOS_TOKEN = "<｜end▁of▁sentence｜>"
THINKING_START_TOKEN = "<think>"
THINKING_END_TOKEN = "</think>"
DSML_TOKEN = "｜DSML｜"

SYSTEM_SP_TOKEN = "<｜System｜>"
USER_SP_TOKEN = "<｜User｜>"
ASSISTANT_SP_TOKEN = "<｜Assistant｜>"

IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"

# Leading-space tag names are not a typo: V4.1 renamed the V4 DSML tags to
# "<｜DSML｜ calls>" / "<｜DSML｜ invoke>" / "<｜DSML｜ parameter>", and the space is
# part of the tag.
TOOL_CALLS_BLOCK_NAME = " calls"
TOOL_CALL_TAG_NAME = " invoke"
TOOL_PARAMETER_TAG_NAME = " parameter"

TOOL_CALL_TEMPLATE = '<{dsml}{tag} name="{name}">\n{arguments}\n</{dsml}{tag}>'
TOOL_CALLS_TEMPLATE = "<{dsml}{block}>\n{tool_calls}\n</{dsml}{block}>"
TOOL_PARAMETER_TEMPLATE = '<{dsml}{tag} name="{key}" string="{is_str}">{value}</{dsml}{tag}>'
TOOL_OUTPUT_TEMPLATE = "<tool_result>{content}</tool_result>"

RESPONSE_FORMAT_TEMPLATE = (
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}"
)

REASONING_EFFORT_TEMPLATE = (
    "Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
)
REASONING_EFFORT_MAPPINGS: dict[str, int] = {"low": 50, "high": 75, "max": 100}
DEFAULT_REASONING_EFFORT = "high"

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml}{block}>" block like the following:

<{dsml}{block}>
<{dsml}{tag} name="$TOOL_NAME">
<{dsml}{param} name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml}{param}>
...
</{dsml}{tag}>
<{dsml}{tag} name="$TOOL_NAME2">
...
</{dsml}{tag}>
</{dsml}{block}>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {think_start}), you MUST output your complete reasoning inside {think_start}...{think_end} BEFORE any tool calls or final response.

Otherwise, output directly after {think_end} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


def to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(value, ensure_ascii=True)


def split_tool_name(name: str, namespace: str | None = None) -> tuple[str | None, str]:
    """Split ``namespace::name``, refusing a qualifier that contradicts ``namespace``."""
    prefix, separator, bare_name = name.partition("::")
    if separator:
        if namespace not in (None, prefix):
            raise ValueError(f"Conflicting tool namespaces: {namespace} != {prefix}")
        namespace, name = prefix, bare_name
    if "::" in name or (namespace is not None and "::" in namespace):
        raise ValueError(f"Tool name / namespace must not contain '::': {namespace}::{name}")
    return namespace, name


def tool_name_for_encoding(tool: dict[str, Any]) -> str:
    namespace = tool.get("namespace")
    if isinstance(namespace, dict):
        namespace = namespace["name"]
    namespace, name = split_tool_name(tool["name"], namespace)
    return name if namespace is None else f"{namespace}::{name}"


def tools_from_openai_format(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten OpenAI ``{"function": {...}}`` wrappers into namespace-qualified defs."""
    functions = []
    for tool in tools:
        function = dict(tool["function"])
        if tool.get("namespace") is not None:
            function["namespace"] = tool["namespace"]
        function["name"] = tool_name_for_encoding(function)
        namespace = function.pop("namespace", None)
        if isinstance(namespace, dict) and namespace.get("description"):
            function["description"] = namespace["description"] + "\n" + (function.get("description") or "")
        functions.append(function)
    return functions


def tool_calls_from_openai_format(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls = []
    for tool_call in tool_calls:
        function = tool_call["function"]
        namespace, name = split_tool_name(function["name"], tool_call.get("namespace") or function.get("namespace"))
        call = {"name": name, "arguments": function["arguments"]}
        if namespace is not None:
            call["namespace"] = namespace
        calls.append(call)
    return calls


def render_tools(tools: list[dict[str, Any]]) -> str:
    return TOOLS_TEMPLATE.format(
        tool_schemas="\n".join(to_json(tool) for tool in tools),
        dsml=DSML_TOKEN,
        block=TOOL_CALLS_BLOCK_NAME,
        tag=TOOL_CALL_TAG_NAME,
        param=TOOL_PARAMETER_TAG_NAME,
        think_start=THINKING_START_TOKEN,
        think_end=THINKING_END_TOKEN,
    )


def encode_arguments_to_dsml(tool_call: dict[str, Any]) -> str:
    """One ``<｜DSML｜ parameter>`` row per argument, JSON-encoding non-strings.

    Arguments arrive as a dict, a JSON string, or -- from some clients -- a
    double-encoded JSON string. The released encoder unwraps up to two levels and
    falls back to a single ``arguments`` key; reproduced here because a tool call
    that renders differently from training is invisible until the model answers
    with a malformed call.
    """
    arguments = tool_call["arguments"]
    if not isinstance(arguments, dict):
        for _ in range(2):
            if not isinstance(arguments, str):
                break
            try:
                arguments = json.loads(arguments)
            except (TypeError, ValueError):
                break
        if not isinstance(arguments, dict):
            arguments = {"arguments": tool_call["arguments"]}

    return "\n".join(
        TOOL_PARAMETER_TEMPLATE.format(
            dsml=DSML_TOKEN,
            tag=TOOL_PARAMETER_TAG_NAME,
            key=key,
            is_str="true" if isinstance(value, str) else "false",
            value=value if isinstance(value, str) else to_json(value),
        )
        for key, value in arguments.items()
    )


def render_reasoning_effort(effort: str | int | None) -> str:
    """The numeric effort prefix; only the first turn of a thinking-mode prompt gets it."""
    if effort is None:
        effort = DEFAULT_REASONING_EFFORT
    if isinstance(effort, str):
        if effort not in REASONING_EFFORT_MAPPINGS:
            raise ValueError(
                f"Invalid reasoning effort {effort!r}: expected an int in [1, 100] or one of "
                f"{list(REASONING_EFFORT_MAPPINGS)}."
            )
        effort = REASONING_EFFORT_MAPPINGS[effort]
    elif not isinstance(effort, int) or isinstance(effort, bool) or not 1 <= effort <= 100:
        raise ValueError(f"Invalid reasoning effort {effort!r}: expected an int in [1, 100].")
    return REASONING_EFFORT_TEMPLATE.format(budget=effort)


@dataclass(frozen=True)
class DeepseekV41ChatMarkers:
    """Wire-format markers -- fixed literals, not resolved from the tokenizer."""

    bos_token: str = BOS_TOKEN
    eos_token: str = EOS_TOKEN
    system_token: str = SYSTEM_SP_TOKEN
    user_token: str = USER_SP_TOKEN
    assistant_token: str = ASSISTANT_SP_TOKEN
    thinking_start_token: str = THINKING_START_TOKEN
    thinking_end_token: str = THINKING_END_TOKEN


@dataclass
class _Turn:
    """One role's consecutive items, as the template sees them."""

    role: str
    items: list[ConversationItem]


class DeepseekV41ChatTemplate(TextEncoderChatTemplate):
    """DeepSeek-V4.1 wire format over ``ConversationItem`` rows.

    ``thinking_mode`` and ``reasoning_effort`` are template-level defaults; a
    sample overrides them from the first item's ``meta``, which is how a dataset
    mixes chat and thinking samples in one run.
    """

    chat_markers: DeepseekV41ChatMarkers

    def __init__(
        self,
        tokenizer: Any,
        thinking_mode: str = "thinking",
        reasoning_effort: str | int | None = None,
    ):
        super().__init__(tokenizer)
        self.chat_markers = DeepseekV41ChatMarkers()
        if thinking_mode not in ("chat", "thinking"):
            raise ValueError(f"thinking_mode must be 'chat' or 'thinking', got {thinking_mode!r}")
        self.thinking_mode = thinking_mode
        self.reasoning_effort = reasoning_effort
        self.image_placeholder_id = self._optional_token_id(IMAGE_PLACEHOLDER)

    def _optional_token_id(self, token: str) -> int | None:
        """Token id, or ``None`` when this tokenizer revision spells the token differently."""
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        unk = getattr(self.tokenizer, "unk_token_id", None)
        if token_id is None or (unk is not None and token_id == unk):
            return None
        return int(token_id)

    # ── Turn assembly ─────────────────────────────────────────────────────────

    @staticmethod
    def _group_turns(sample: list[ConversationItem]) -> tuple[list[_Turn], list[ConversationItem]]:
        """Consecutive same-role items become one turn; ``is_dummy`` rows are set aside."""
        turns: list[_Turn] = []
        dummies: list[ConversationItem] = []
        for item in sample:
            if item.is_dummy:
                dummies.append(item)
                continue
            if turns and turns[-1].role == item.role:
                turns[-1].items.append(item)
            else:
                turns.append(_Turn(role=item.role, items=[item]))
        return turns, dummies

    @staticmethod
    def _last_user_turn(turns: list[_Turn]) -> int:
        """Index of the turn that triggers the assistant header.

        A mid-conversation system message counts as a user turn in V4.1 -- that is
        what ``find_last_user_index`` does, and it is why a system message appearing
        after turn 0 gets an assistant header of its own.
        """
        for index in range(len(turns) - 1, -1, -1):
            if turns[index].role == "user" or (turns[index].role == "system" and index > 0):
                return index
        return -1

    def _sample_options(self, sample: list[ConversationItem]) -> tuple[str, str | int | None]:
        """Per-sample ``(thinking_mode, reasoning_effort)``, from the first item's meta."""
        thinking_mode, effort = self.thinking_mode, self.reasoning_effort
        for item in sample:
            thinking_mode = item.meta.get("thinking_mode", thinking_mode)
            effort = item.meta.get("reasoning_effort", effort)
            break
        if thinking_mode not in ("chat", "thinking"):
            raise ValueError(f"thinking_mode must be 'chat' or 'thinking', got {thinking_mode!r}")
        return thinking_mode, effort

    # ── Templating ────────────────────────────────────────────────────────────

    def apply_chat_template(self, sample: list[ConversationItem]) -> list[ConversationItem]:
        turns, dummies = self._group_turns(sample)
        thinking_mode, effort = self._sample_options(sample)
        last_user = self._last_user_turn(turns)

        # A conversation that defines tools keeps every turn's reasoning: the model
        # has to see how earlier tool calls were arrived at, so the usual "drop
        # reasoning before the last user turn" compaction is switched off.
        drop_thinking = not any(turn.items[0].meta.get("tools") for turn in turns)

        out: list[ConversationItem] = []
        prefix = self.chat_markers.bos_token
        if thinking_mode == "thinking":
            prefix += SYSTEM_SP_TOKEN + render_reasoning_effort(effort)
        elif turns and turns[0].role == "system":
            prefix += SYSTEM_SP_TOKEN
        out.append(self._build_conversation_item("text", prefix, "user", loss_mask=0))

        for index, turn in enumerate(turns):
            if turn.role == "system":
                out.extend(self._render_system(turn, index))
            elif turn.role == "user":
                out.extend(self._render_user(turn))
            elif turn.role == "assistant":
                out.extend(self._render_assistant(turn, index, last_user, thinking_mode, drop_thinking))
            else:
                raise ValueError(f"DeepSeek-V4.1 text encoder got an unknown role {turn.role!r}")

            if self._emits_assistant_header(turns, index):
                out.append(
                    self._build_conversation_item(
                        "text",
                        self._assistant_header(thinking_mode, drop_thinking, index >= last_user),
                        "assistant",
                        loss_mask=0,
                    )
                )

        out.extend(dummies)
        return out

    @staticmethod
    def _emits_assistant_header(turns: list[_Turn], index: int) -> bool:
        """Whether this turn is followed by ``<｜Assistant｜>`` + a thinking transition.

        Two conditions, both from ``render_message``'s transition block: the turn
        must be one that prompts the assistant (a user turn, or a *mid-conversation*
        system turn), and it must actually be followed by an assistant turn -- or be
        the last turn, which is the inference case where the model is about to
        speak. A system turn followed by a user turn gets no header.
        """
        turn = turns[index]
        if not (turn.role == "user" or (turn.role == "system" and index > 0)):
            return False
        return index + 1 >= len(turns) or turns[index + 1].role == "assistant"

    def _assistant_header(self, thinking_mode: str, drop_thinking: bool, is_last_user: bool) -> str:
        """``<｜Assistant｜>`` plus the thinking transition.

        ``<think>`` opens a reasoning block the assistant is expected to fill;
        ``</think>`` closes one that was never opened, which is how an earlier turn
        whose reasoning was dropped stays well-formed. The two have to agree with
        :meth:`_render_assistant` exactly -- a header that opens a block the body
        never closes trains the model to keep reasoning forever.
        """
        header = ASSISTANT_SP_TOKEN
        if thinking_mode != "thinking":
            return header + THINKING_END_TOKEN
        if not drop_thinking or is_last_user:
            return header + THINKING_START_TOKEN
        return header + THINKING_END_TOKEN

    def _render_system(self, turn: _Turn, index: int) -> list[ConversationItem]:
        out: list[ConversationItem] = []
        # Turn 0's `<｜System｜>` is already in the prefix; a mid-conversation system
        # message carries its own.
        text = SYSTEM_SP_TOKEN if index > 0 else ""
        for item in turn.items:
            if item.type != "text":
                raise ValueError(f"system turns carry text only, got {item.type!r}")
            text += str(item.value)
        first = turn.items[0]
        tools = first.meta.get("tools")
        if tools:
            text += "\n\n" + render_tools(tools_from_openai_format(tools))
        response_format = first.meta.get("response_format")
        if response_format:
            text += "\n\n" + RESPONSE_FORMAT_TEMPLATE.format(schema=to_json(response_format))
        out.append(self._build_conversation_item("text", text, "user", loss_mask=0))
        return out

    def _render_user(self, turn: _Turn) -> list[ConversationItem]:
        out: list[ConversationItem] = [self._build_conversation_item("text", USER_SP_TOKEN, "user", loss_mask=0)]
        for item in turn.items:
            if item.type == "text":
                text = str(item.value)
                tool_result = item.meta.get("tool_result")
                if tool_result is not None:
                    text = TOOL_OUTPUT_TEMPLATE.format(content=tool_result)
                out.append(self._build_conversation_item("text", text, "user", loss_mask=0))
            elif item.type == "image":
                # Passed through verbatim: the vision module already produced this
                # item's whole span, delimiters included, so it contributes no text.
                out.append(item)
            else:
                raise ValueError(f"DeepSeek-V4.1 user turns support text/image, got {item.type!r}")
        return out

    def _render_assistant(
        self,
        turn: _Turn,
        index: int,
        last_user: int,
        thinking_mode: str,
        drop_thinking: bool,
    ) -> list[ConversationItem]:
        first = turn.items[0]
        content = "".join(str(item.value) for item in turn.items if item.type == "text")

        body = ""
        if thinking_mode == "thinking" and (not drop_thinking or index > last_user):
            # Mirrors the header: a turn whose reasoning survived closes its own
            # block here, and one whose reasoning was dropped had the closing tag
            # emitted by its header instead.
            body += str(first.meta.get("reasoning_content") or "") + THINKING_END_TOKEN
        body += content

        tool_calls = first.meta.get("tool_calls")
        if tool_calls:
            calls = tool_calls_from_openai_format(tool_calls)
            rendered = "\n".join(
                TOOL_CALL_TEMPLATE.format(
                    dsml=DSML_TOKEN,
                    tag=TOOL_CALL_TAG_NAME,
                    name=tool_name_for_encoding(call),
                    arguments=encode_arguments_to_dsml(call),
                )
                for call in calls
            )
            body += "\n\n" + TOOL_CALLS_TEMPLATE.format(
                dsml=DSML_TOKEN, tool_calls=rendered, block=TOOL_CALLS_BLOCK_NAME
            )

        if not first.meta.get("wo_eos", False):
            body += EOS_TOKEN
        return [self._build_conversation_item("text", body, "assistant", loss_mask=1)]

    def tokenize(self, parts: list[ConversationItem]) -> None:
        """The base tokenize, plus a surviving copy of the ids in ``meta``.

        ``encode`` overwrites each text item's ``value`` with its embedding, so
        after it runs the ids are gone from the carrier. The n-gram node needs
        them -- it hashes token ids, not embeddings -- and without this copy it
        would only be correct when the graph executor happened to schedule it
        before the text encoder. Both nodes depend on the vision node and on
        nothing else, so that ordering is not something the DAG guarantees.

        BAGEL's template stashes ids the same way and for the same reason.
        """
        super().tokenize(parts)
        for part in parts:
            if part.type == "text":
                part.meta["input_ids"] = part.value.detach().clone()

    @staticmethod
    def merge_text_embeds(parts: list[ConversationItem]) -> list[ConversationItem]:
        """The base merge, keeping the stashed ids in step with ``value``.

        The base concatenates ``value`` / ``labels`` / ``attention_mask``; the
        copy above has to travel with them. Leaving it out would not shorten
        anything or raise -- the merged item would simply carry the first
        segment's ids, and every n-gram past the join would be hashed from the
        wrong tokens.
        """
        merged = TextEncoderChatTemplate.merge_text_embeds(parts)
        for part in merged:
            if part.type == "text" and "input_ids" in part.meta:
                part.meta["input_ids"] = part.value.detach().clone()
        return merged

    def apply_generation_prompt(self, sample: list[ConversationItem]) -> list[ConversationItem]:
        """No-op: :meth:`apply_chat_template` already ends on the assistant header.

        V4.1 attaches the header to the user turn rather than opening the assistant
        one, so a templated prompt whose last turn is a user turn is already ready
        to decode. Appending a second header here would emit ``<｜Assistant｜>``
        twice.
        """
        return list(sample)


__all__ = [
    "ASSISTANT_SP_TOKEN",
    "BOS_TOKEN",
    "EOS_TOKEN",
    "IMAGE_PLACEHOLDER",
    "SYSTEM_SP_TOKEN",
    "THINKING_END_TOKEN",
    "THINKING_START_TOKEN",
    "USER_SP_TOKEN",
    "DeepseekV41ChatMarkers",
    "DeepseekV41ChatTemplate",
    "encode_arguments_to_dsml",
    "render_reasoning_effort",
    "render_tools",
]
