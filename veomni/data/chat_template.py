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


import json
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Union

import torch
import torch.nn.functional as F

from veomni.utils import logging

from ..utils.constants import IGNORE_INDEX, IMAGE_INPUT_INDEX, TYPE2INDEX
from ..utils.registry import Registry


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin


logger = logging.get_logger(__name__)

ROLE_SUPPORTED = ["system", "user", "assistant", "tool"]
CHAT_TEMPLATE_REGISTRY = Registry("ChatTemplate")


def add_mtp_labels(feature: Dict[str, torch.Tensor | List[int]], num_depths: int) -> None:
    """Add future-token labels to one un-packed sample when MTP is enabled."""
    if num_depths <= 0:
        return
    labels = feature["labels"]
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    feature["mtp_labels"] = torch.stack(
        [F.pad(labels, (0, depth + 2), value=IGNORE_INDEX)[..., depth + 2 :] for depth in range(num_depths)],
        dim=-2,
    )


def _get_mtp_num_hidden_layers(tokenizer_or_processor) -> int:
    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
    name_or_path = getattr(tokenizer, "name_or_path", None)
    if not name_or_path:
        name_or_path = getattr(tokenizer, "init_kwargs", {}).get("name_or_path")
    if not name_or_path:
        return 0
    config_path = Path(name_or_path) / "config.json"
    try:
        with config_path.open() as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError):
        return 0
    text_config = config.get("text_config", config)
    return int(text_config.get("mtp_num_hidden_layers", 0) or 0)


def build_chat_template(
    template_name: str,
    tokenizer_or_processor: Union["PreTrainedTokenizer", "ProcessorMixin"],
    **kwargs,
) -> "ChatTemplate":
    """Builds any registered template, text-only or multimodal.

    Multimodal templates need the processor, which carries both the tokenizer and
    the grid parameters they use. Text-only templates accept either and encode
    through the tokenizer, so callers can pass the processor whenever one loaded.

    ``kwargs`` reach the template constructor. No template currently declares
    one, so an unrecognised option raises there instead of being silently
    dropped.
    """
    return CHAT_TEMPLATE_REGISTRY[template_name](tokenizer_or_processor, **kwargs)


class ChatTemplate(ABC):
    """
    Abstract class for chat template.
    """

    def __init__(self, tokenizer_or_processor: Union["PreTrainedTokenizer", "ProcessorMixin"]) -> None:
        self.tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)

    @abstractmethod
    def encode_messages(self, messages: Sequence[Dict[str, str]], max_seq_len: int = 8192) -> Dict[str, List[int]]:
        """
        Encodes messages to a dictionary of input_ids, attention_mask, and labels.
        """
        ...

    def get_jinja_template(self) -> str:
        """This template's layout expressed as jinja, for whoever wants to serialise it.

        Optional, and read by nothing in the training or export path: training
        lays out tokens through :meth:`encode_messages`, and an export keeps the
        jinja the checkpoint shipped with rather than substituting a job's
        formatting choice. A layout that jinja cannot express — image
        placeholders sized from the processor's grid parameters — says so by
        leaving this alone.
        """
        return ""


@CHAT_TEMPLATE_REGISTRY.register("default")
class DefaultTemplate(ChatTemplate):
    def encode_messages(self, messages: Sequence[Dict[str, str]], max_seq_len: int = 8192) -> Dict[str, List[int]]:
        input_ids, attention_mask, labels = [], [], []
        for message in messages:
            content_str = message["role"].title() + ": " + message["content"].strip() + self.tokenizer.eos_token + "\n"
            content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
            input_ids += content_ids
            attention_mask += [1] * len(content_ids)
            if message["loss_mask"] == 1:
                labels += content_ids
            else:
                labels += [IGNORE_INDEX] * len(content_ids)

        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        model_inputs = {k: v[-max_seq_len:] for k, v in model_inputs.items()}
        return model_inputs

    def get_jinja_template(self) -> str:
        return (
            "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
            "{% for message in messages %}"
            "{{ message['role'].title() + ': ' + message['content'] | trim + eos_token + '\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}"
        )


@CHAT_TEMPLATE_REGISTRY.register("tokenizer")
class TokenizerTemplate(ChatTemplate):
    """Use a prefix-stable native chat template with assistant-only labels."""

    def _update_prefix_labels(self, previous_ids: List[int], current_ids: List[int], labels: List[int]) -> None:
        """Validate that adding a message preserved the previously rendered prefix."""
        previous_length = len(previous_ids)
        if current_ids[:previous_length] != previous_ids:
            raise ValueError(
                "The tokenizer chat template structurally rewrote an earlier conversation prefix; "
                "the generic tokenizer template requires prefix-stable rendering."
            )

    def encode_messages(self, messages: Sequence[Dict[str, str]], max_seq_len: int = 8192) -> Dict[str, List[int]]:
        input_ids: List[int] = []
        labels: List[int] = []
        previous_length = 0

        for end, message in enumerate(messages, start=1):
            encoded = self.tokenizer.apply_chat_template(
                messages[:end],
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
            )
            current_ids = encoded["input_ids"]
            current_length = len(current_ids)
            if current_length < previous_length:
                raise ValueError(
                    "The tokenizer chat template shortened the conversation after adding a message; "
                    "assistant-only loss masking requires monotonic message boundaries."
                )

            self._update_prefix_labels(input_ids, current_ids, labels)

            loss_mask = message.get("loss_mask", 1 if message["role"] == "assistant" else 0)
            new_ids = current_ids[previous_length:]
            labels.extend(new_ids if loss_mask == 1 else [IGNORE_INDEX] * len(new_ids))
            input_ids = current_ids
            previous_length = current_length

        input_ids = input_ids[-max_seq_len:]
        labels = labels[-max_seq_len:]
        model_inputs = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}
        return model_inputs

    def get_jinja_template(self) -> str:
        if not self.tokenizer.chat_template:
            raise ValueError("The tokenizer does not define a native chat template.")
        return self.tokenizer.chat_template


@CHAT_TEMPLATE_REGISTRY.register("gpt_oss")
class GptOssTokenizerTemplate(TokenizerTemplate):
    """GPT-OSS native template with its terminal assistant-token rewrite."""

    def __init__(self, tokenizer_or_processor: Union["PreTrainedTokenizer", "ProcessorMixin"]) -> None:
        super().__init__(tokenizer_or_processor)
        tokenizer = self.tokenizer
        self.return_token_id = tokenizer.convert_tokens_to_ids("<|return|>")
        self.end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
        if self.return_token_id == tokenizer.unk_token_id or self.end_token_id == tokenizer.unk_token_id:
            raise ValueError("The GPT-OSS chat template requires <|return|> and <|end|> tokenizer tokens.")

    def _update_prefix_labels(self, previous_ids: List[int], current_ids: List[int], labels: List[int]) -> None:
        previous_length = len(previous_ids)
        rewritten_positions = [index for index in range(previous_length) if previous_ids[index] != current_ids[index]]
        if not rewritten_positions:
            return

        is_terminal_rewrite = (
            rewritten_positions == [previous_length - 1]
            and len(current_ids) > previous_length
            and previous_ids[-1] == self.return_token_id
            and current_ids[previous_length - 1] == self.end_token_id
            and self.return_token_id not in current_ids[previous_length:]
        )
        if not is_terminal_rewrite:
            raise ValueError(
                "The GPT-OSS tokenizer chat template structurally rewrote an earlier conversation prefix; "
                "only the terminal <|return|>-to-<|end|> substitution is supported."
            )

        if labels[-1] != IGNORE_INDEX:
            labels[-1] = self.end_token_id


@CHAT_TEMPLATE_REGISTRY.register("llama2")
class Llama2Template(ChatTemplate):
    def encode_messages(self, messages: Sequence[Dict[str, str]], max_seq_len: int = 8192) -> Dict[str, List[int]]:
        input_ids, attention_mask, labels = [], [], []
        for message in messages:
            if message["role"] == "system":
                content_str = "<<SYS>>\n" + message["content"].strip() + "\n<</SYS>>\n\n"
            elif message["role"] == "user":
                content_str = self.tokenizer.bos_token + "[INST] " + message["content"].strip() + " [/INST]"
            elif message["role"] == "assistant":
                content_str = " " + message["content"].strip() + " " + self.tokenizer.eos_token
            elif message["role"] == "tool":
                content_str = self.tokenizer.bos_token + "[TOOL] " + message["content"].strip() + " [/TOOL]"
            else:
                raise ValueError(
                    f"Unknown role {message['role']}, should be one of {{system, user, assistant, tool}}."
                )

            content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
            input_ids += content_ids
            attention_mask += [1] * len(content_ids)
            if message["loss_mask"] == 1:
                labels += content_ids
            else:
                labels += [IGNORE_INDEX] * len(content_ids)

        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        model_inputs = {k: v[-max_seq_len:] for k, v in model_inputs.items()}
        return model_inputs

    def get_jinja_template(self) -> str:
        return (
            "{% if messages[0]['role'] == 'system' %}"
            "{{ '<<SYS>>\n' + messages[0]['content'] | trim + '\n<</SYS>>\n\n' }}"
            "{% set loop_messages = messages[1:] %}"
            "{% else %}"
            "{% set loop_messages = messages %}"
            "{% endif %}"
            "{% for message in loop_messages %}"
            "{% set content = message['content'] %}"
            "{% if message['role'] == 'user' %}"
            "{{ bos_token + '[INST] ' + content | trim + ' [/INST]' }}"
            "{% elif message['role'] == 'tool' %}"
            "{{ bos_token + '[TOOL] ' + content | trim + ' [/TOOL]' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ ' ' + content | trim + ' ' + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
        )


@CHAT_TEMPLATE_REGISTRY.register("tokenizer_generation_prompt")
class GenerationPromptTokenizerTemplate(ChatTemplate):
    """Drives the tokenizer's own chat template, supervising only what follows the generation prompt.

    Unlike ``tokenizer``, which supervises every token a message appends (the assistant
    role header included), this supervises exactly what a model produces after being
    given ``add_generation_prompt=True``, so the label span matches inference input ids.

    Labels come from a prefix diff rather than from a ``{% generation %}`` block, since
    most published templates do not mark one: encoding through turn ``k``'s generation
    prompt and then through turn ``k`` itself brackets exactly that turn's tokens. The
    supervised span deliberately includes the trailing turn terminator, so the model
    learns to end its turn. This assumes each turn only appends to the last encoding,
    which is checked rather than trusted -- a template that rewrites earlier turns (some
    reorder thinking blocks once a later turn arrives) would otherwise silently shift the
    label mask off the answer.
    """

    def encode_messages(self, messages: Sequence[Dict[str, str]], max_seq_len: int = 8192) -> Dict[str, List[int]]:
        if self.tokenizer.chat_template is None:
            raise ValueError(
                "The tokenizer has no chat template of its own, so "
                "`chat_template: tokenizer_generation_prompt` cannot be used. Pick one of the templates defined in veomni/data/chat_template.py instead."
            )

        input_ids = self._encode(messages)
        labels = [IGNORE_INDEX] * len(input_ids)
        for index, message in enumerate(messages):
            loss_mask = message.get("loss_mask", 1 if message["role"] == "assistant" else 0)
            if loss_mask != 1 or index == 0:
                continue
            prompt = self._encode(messages[:index], add_generation_prompt=True)
            through_turn = self._encode(messages[: index + 1])
            if input_ids[: len(through_turn)] != through_turn or through_turn[: len(prompt)] != prompt:
                raise ValueError(
                    f"Encoding the conversation through turn {index} is not a prefix of the full encoding, so the "
                    "turn's token span cannot be recovered. The tokenizer's chat template is not append-only and "
                    "this template class cannot mask its labels."
                )
            labels[len(prompt) : len(through_turn)] = input_ids[len(prompt) : len(through_turn)]

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }
        return {k: v[-max_seq_len:] for k, v in model_inputs.items()}

    def _encode(self, messages: Sequence[Dict[str, str]], add_generation_prompt: bool = False) -> List[int]:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt, return_dict=True
        )["input_ids"]

    def get_jinja_template(self) -> str:
        return self.tokenizer.chat_template


@CHAT_TEMPLATE_REGISTRY.register("gemma4_thinking")
class Gemma4ThinkingTokenizerTemplate(GenerationPromptTokenizerTemplate):
    """Render Gemma 4 reasoning fields into an append-only training transcript."""

    _reasoning_guard = "thinking_text and loop.index0 > ns_turn.last_user_idx and message.get('tool_calls')"

    def __init__(self, tokenizer: "PreTrainedTokenizer") -> None:
        super().__init__(tokenizer)
        template = tokenizer.chat_template
        if template is None or template.count(self._reasoning_guard) != 1:
            raise ValueError("Gemma 4 chat template does not contain the expected reasoning guard")
        # The serving template hides reasoning from earlier context and only emits
        # ``reasoning_content`` alongside tool calls. SFT must instead include the
        # reasoning field for ordinary math answers. This private rendering copy is
        # used only to build labels; ``save_pretrained`` retains the official template.
        self._training_chat_template = template.replace(self._reasoning_guard, "thinking_text")

    def _encode(self, messages: Sequence[Dict[str, str]], add_generation_prompt: bool = False) -> List[int]:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            enable_thinking=True,
            chat_template=self._training_chat_template,
        )["input_ids"]


@CHAT_TEMPLATE_REGISTRY.register("gemma4_no_thinking")
class Gemma4NoThinkingTokenizerTemplate(GenerationPromptTokenizerTemplate):
    """Match Gemma 4's no-thinking serving prompt while keeping SFT append-only.

    The official template appends an empty thought channel to a generation prompt when
    ``enable_thinking`` is false, but omits it when rendering an assistant turn that
    already has content. Training needs the former to be an exact prefix of the latter
    so label boundaries and inference input ids agree. This private rendering copy adds
    that same empty channel before each newly opened model turn and suppresses stored
    reasoning fields. ``get_jinja_template`` still exports the tokenizer's official
    serving template.
    """

    _reasoning_guard = "thinking_text and loop.index0 > ns_turn.last_user_idx and message.get('tool_calls')"
    _model_turn_anchor = """{%- if not continue_same_model_turn -%}
        {{- '<|turn>' + role + '\\n' }}
    {%- endif -%}"""
    _model_turn_replacement = """{%- if not continue_same_model_turn -%}
        {{- '<|turn>' + role + '\\n' }}
        {%- if role == 'model' -%}
            {{- '<|channel>thought\\n<channel|>' -}}
        {%- endif -%}
    {%- endif -%}"""

    def __init__(self, tokenizer: "PreTrainedTokenizer") -> None:
        super().__init__(tokenizer)
        template = tokenizer.chat_template
        if template is None or template.count(self._reasoning_guard) != 1:
            raise ValueError("Gemma 4 chat template does not contain the expected reasoning guard")
        if template.count(self._model_turn_anchor) != 1:
            raise ValueError("Gemma 4 chat template does not contain the expected model-turn anchor")
        self._training_chat_template = template.replace(self._reasoning_guard, "false").replace(
            self._model_turn_anchor, self._model_turn_replacement
        )

    def _encode(self, messages: Sequence[Dict[str, str]], add_generation_prompt: bool = False) -> List[int]:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            enable_thinking=False,
            chat_template=self._training_chat_template,
        )["input_ids"]


@CHAT_TEMPLATE_REGISTRY.register("diffusion_gemma_sudoku_official")
class DiffusionGemmaSudokuOfficialTemplate(ChatTemplate):
    """Exact single-turn boundary used by the aligned NeMo Sudoku SFT."""

    SYSTEM = (
        "Solve the following Sudoku puzzle. Empty cells are represented by 0. "
        "Output ONLY the solved puzzle immediately as a 9x9 grid of numbers separated by spaces. "
        "Do not include ####, explanations, or any other text."
    )

    def encode_messages(self, messages: Sequence[Dict[str, str]], max_seq_len: int = 8192) -> Dict[str, List[int]]:
        if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
            raise ValueError("The official Sudoku template requires exactly system, user, assistant turns")
        if messages[0].get("content", "").strip() != self.SYSTEM:
            raise ValueError("The Sudoku system instruction differs from the released SFT prompt")
        prompt = (
            f"<|turn>system\n{self.SYSTEM}<turn|>\n"
            f"<|turn>user\n{messages[1]['content'].strip()}<turn|>\n"
            "<|turn>model\n"
        )
        if self.tokenizer.bos_token_id is None:
            raise ValueError("The official Sudoku template requires a BOS token id")
        prompt_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(messages[2]["content"].strip() + "<turn|>\n", add_special_tokens=False)
        input_ids = prompt_ids + answer_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
        model_inputs = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}
        return {key: value[-max_seq_len:] for key, value in model_inputs.items()}

    def get_jinja_template(self) -> str:
        template = (
            "{% if messages | length not in [2, 3] %}"
            "{{ raise_exception('The aligned Sudoku template requires one system/user turn and an optional answer') }}"
            "{% endif %}"
            "{% if messages[0]['role'] != 'system' or messages[0]['content'] | trim != '__EXPECTED_SYSTEM__' %}"
            "{{ raise_exception('The Sudoku system instruction differs from the aligned NeMo prompt') }}"
            "{% endif %}"
            "{% if messages[1]['role'] != 'user' %}"
            "{{ raise_exception('The aligned Sudoku template requires a user puzzle after the system turn') }}"
            "{% endif %}"
            "{% if messages | length == 3 and messages[2]['role'] != 'assistant' %}"
            "{{ raise_exception('The aligned Sudoku template accepts only an assistant answer as the third turn') }}"
            "{% endif %}"
            "{{ bos_token }}"
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}<|turn>system\n"
            "{{ message['content'] | trim }}<turn|>\n"
            "{% elif message['role'] == 'user' %}<|turn>user\n"
            "{{ message['content'] | trim }}<turn|>\n"
            "{% elif message['role'] == 'assistant' %}<|turn>model\n"
            "{{ message['content'] | trim }}<turn|>\n"
            "{% else %}{{ raise_exception('Unsupported role: ' + message['role']) }}{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|turn>model\n{% endif %}"
        )
        return template.replace("__EXPECTED_SYSTEM__", self.SYSTEM)


def _encode_segments(
    tokenizer: "PreTrainedTokenizer", segments: Sequence[tuple], max_seq_len: int
) -> Dict[str, List[int]]:
    """Tokenize ``(text, is_target)`` segments one by one and concatenate them.

    The LLaDA checkpoints were fine-tuned on exactly this segmentation (header,
    content and terminator encoded separately), so encoding a turn in one go could
    merge BPE pieces across a boundary the model never saw merged.
    """
    input_ids, labels = [], []
    for text, is_target in segments:
        ids = tokenizer.encode(text, add_special_tokens=False)
        input_ids += ids
        labels += ids if is_target else [IGNORE_INDEX] * len(ids)
    model_inputs = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}
    return {k: v[-max_seq_len:] for k, v in model_inputs.items()}


def _is_target(message: Dict[str, str]) -> bool:
    return message.get("loss_mask", 1 if message["role"] == "assistant" else 0) == 1


@CHAT_TEMPLATE_REGISTRY.register("llada")
class LladaTemplate(ChatTemplate):
    """LLaDA-8B-Instruct / LLaDA-V: Llama-3 headers, BOS on the first message only.

    ``{bos}<|start_header_id|>{role}<|end_header_id|>\\n\\n{content}<|eot_id|>...``; an
    answer's content and its closing ``<|eot_id|>`` are the loss targets.
    """

    start_header = "<|start_header_id|>"
    end_header = "<|end_header_id|>"
    eot = "<|eot_id|>"

    def encode_messages(self, messages: Sequence[Dict[str, str]], max_seq_len: int = 8192) -> Dict[str, List[int]]:
        segments = []
        for index, message in enumerate(messages):
            header = self.start_header + message["role"] + self.end_header + "\n\n"
            if index == 0:
                header = self.tokenizer.bos_token + header
            target = _is_target(message)
            segments += [(header, False), (str(message["content"]).strip(), target), (self.eot, target)]
        return _encode_segments(self.tokenizer, segments, max_seq_len)

    def encode_multimodal_messages(
        self, messages: Sequence[Dict[str, Any]], image_seq_len: int
    ) -> Dict[str, List[int]]:
        """LLaDA-V: messages whose content is a list of ``text`` / ``image`` parts.

        Each image becomes ``image_seq_len`` ``IMAGE_INPUT_INDEX`` slots, and text that
        follows an image starts on a new line (LLaVA's ``<image>\\n{question}``). A message's
        content is trimmed at its two ends only, as the checkpoint's template trims the
        string it was rendered from. Nothing is truncated: cutting an image span would
        leave pixels without their slots, so the caller decides what to do with an
        over-long sample.
        """
        input_ids, labels = [], []

        def add_text(text: str, is_target: bool) -> None:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            input_ids.extend(ids)
            labels.extend(ids if is_target else [IGNORE_INDEX] * len(ids))

        for index, message in enumerate(messages):
            header = self.start_header + message["role"] + self.end_header + "\n\n"
            if index == 0:
                header = self.tokenizer.bos_token + header
            add_text(header, False)
            target = _is_target(message)
            parts = message["content"]
            if not isinstance(parts, list):
                parts = [{"type": "text", "text": str(parts)}]
            after_image = False
            for part_index, part in enumerate(parts):
                if part["type"] == "image":
                    if target:
                        raise ValueError("LLaDA-V supervises text only; an image cannot be part of an answer.")
                    input_ids.extend([IMAGE_INPUT_INDEX] * image_seq_len)
                    labels.extend([IGNORE_INDEX] * image_seq_len)
                    after_image = True
                    continue
                text = str(part["text"])
                if part_index == 0:
                    text = text.lstrip()
                if part_index == len(parts) - 1:
                    text = text.rstrip()
                if after_image and not text.startswith("\n"):
                    text = "\n" + text
                if text:
                    add_text(text, target)
                after_image = False
            add_text(self.eot, target)
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

    def get_jinja_template(self) -> str:
        if self.tokenizer.chat_template is not None:
            return self.tokenizer.chat_template
        return (
            "{% for message in messages %}"
            "{% if loop.index0 == 0 %}{{ bos_token }}{% endif %}"
            "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        )


@CHAT_TEMPLATE_REGISTRY.register("llada_moe")
class LladaMoeTemplate(ChatTemplate):
    """LLaDA-MoE-7B-A1B-Instruct: Bailing role markers, no BOS, text-only SFT subset.

    ``<role>SYSTEM</role>{system\\n?}detailed thinking off<|role_end|><role>HUMAN</role>{q}<|role_end|>``
    ``<role>ASSISTANT</role>{a}<|role_end|>``, as the checkpoint's own template renders it
    without tools or reasoning. A leading system message folds into the always-present
    preamble, and an answer's content plus its ``<|role_end|>`` is the loss target.
    """

    system_prefix = "<role>SYSTEM</role>"
    human_prefix = "<role>HUMAN</role>"
    assistant_prefix = "<role>ASSISTANT</role>"
    role_end = "<|role_end|>"
    thinking = "detailed thinking off"

    def encode_messages(self, messages: Sequence[Dict[str, str]], max_seq_len: int = 8192) -> Dict[str, List[int]]:
        body = list(messages)
        system_text = ""
        if body and body[0]["role"] == "system":
            system_text = str(body[0]["content"]) + "\n"
            body = body[1:]
        segments = [(self.system_prefix + system_text + self.thinking + self.role_end, False)]
        for message in body:
            if message["role"] == "assistant":
                segments += [
                    (self.assistant_prefix, False),
                    (str(message["content"]) + self.role_end, _is_target(message)),
                ]
            else:
                prefix = self.system_prefix if message["role"] == "system" else self.human_prefix
                segments.append((prefix + str(message["content"]) + self.role_end, False))
        return _encode_segments(self.tokenizer, segments, max_seq_len)

    def get_jinja_template(self) -> str:
        if self.tokenizer.chat_template is not None:
            return self.tokenizer.chat_template
        return (
            "{% set ns = namespace(system='') %}"
            "{% if messages and messages[0]['role'] == 'system' %}{% set ns.system = messages[0]['content'] + '\n' %}"
            "{% set body = messages[1:] %}{% else %}{% set body = messages %}{% endif %}"
            "{{ '<role>SYSTEM</role>' + ns.system + 'detailed thinking off<|role_end|>' }}"
            "{% for message in body %}"
            "{% if message['role'] == 'assistant' %}{{ '<role>ASSISTANT</role>' + message['content'] + '<|role_end|>' }}"
            "{% elif message['role'] == 'system' %}{{ '<role>SYSTEM</role>' + message['content'] + '<|role_end|>' }}"
            "{% else %}{{ '<role>HUMAN</role>' + message['content'] + '<|role_end|>' }}{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<role>ASSISTANT</role>' }}{% endif %}"
        )


@CHAT_TEMPLATE_REGISTRY.register("llada2_uni")
class Llada2UniTemplate(ChatTemplate):
    """LLaDA2.0-Uni image understanding, laid out as the checkpoint's ``understand_image`` prompt.

    ``<role>SYSTEM</role> {system} <role>HUMAN</role>{parts}<role>ASSISTANT</role>{a}<|endoftext|>``:
    role tags are concatenated with no separator, the system text (default
    ``understand_system``) is space-joined to the first turn's header, and consecutive
    messages of one role share a header. An image is
    ``<|image|><|reserved_token_{h}|><|reserved_token_{w}|><boi>`` + ``h*w`` VQ slots +
    ``<|/image|>``, ``(h, w)`` being its patch grid. Each segment is tokenized on its own,
    and an answer's text plus its ``<|endoftext|>`` are the loss targets.
    """

    role_open = "<role>"
    role_close = "</role>"
    eos = "<|endoftext|>"
    understand_system = "You are a multimodal understanding assistant."
    image_start = "<|image|>"
    image_end = "<|/image|>"
    boi = "<boi>"
    role_names = {"system": "SYSTEM", "user": "HUMAN", "assistant": "ASSISTANT", "tool": "OBSERVATION"}

    def grid_token_id(self, size: int) -> int | None:
        """``<|reserved_token_{size}|>``'s id, or ``None`` when the vocabulary has no such token."""
        token = f"<|reserved_token_{size}|>"
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None or self.tokenizer.convert_ids_to_tokens(token_id) != token:
            return None
        return token_id

    def encode_messages(self, messages: Sequence[Dict[str, str]], max_seq_len: int = 8192) -> Dict[str, List[int]]:
        tokenized = self.encode_multimodal_messages(messages, image_grids=[])
        return {key: value[:max_seq_len] for key, value in tokenized.items()}

    def encode_multimodal_messages(
        self, messages: Sequence[Dict[str, Any]], image_grids: Sequence[tuple[int, int]]
    ) -> Dict[str, List[int]]:
        """Messages whose content is a string or a list of ``text`` / ``image`` parts.

        ``image_grids`` holds each image's ``(h, w)`` in order of appearance; every grid
        side must have a ``<|reserved_token_*|>`` (see `grid_token_id`). Nothing is
        truncated, for the reason `LladaTemplate.encode_multimodal_messages` gives.
        """
        input_ids, labels = [], []

        def add_ids(ids: List[int], is_target: bool) -> None:
            input_ids.extend(ids)
            labels.extend(ids if is_target else [IGNORE_INDEX] * len(ids))

        def add_text(text: str, is_target: bool) -> None:
            add_ids(self.tokenizer.encode(text, add_special_tokens=False), is_target)

        def token_id(token: str) -> int:
            idx = self.tokenizer.convert_tokens_to_ids(token)
            if idx is None or self.tokenizer.convert_ids_to_tokens(idx) != token:
                raise ValueError(f"The tokenizer has no `{token}` token; is this the LLaDA2.0-Uni tokenizer?")
            return idx

        system = "".join(str(m["content"]) for m in messages if m["role"] == "system") or self.understand_system
        body = [m for m in messages if m["role"] != "system"]
        grids = iter(image_grids)
        previous_role, previous_target = None, False
        for message in body:
            role = message["role"]
            target = _is_target(message)
            if role != previous_role:
                header = self.role_open + self.role_names.get(role, role.upper()) + self.role_close
                if previous_role is None:
                    header = f"{self.role_open}SYSTEM{self.role_close} {system} {header}"
                if previous_role == "assistant":
                    add_text(self.eos, previous_target)
                add_text(header, False)
                previous_role = role
            previous_target = target
            parts = message["content"]
            if not isinstance(parts, list):
                parts = [{"type": "text", "text": str(parts)}]
            for part in parts:
                if part["type"] != "image":
                    add_text(str(part["text"]), target)
                    continue
                if target:
                    raise ValueError("LLaDA2.0-Uni supervises text only; an image cannot be part of an answer.")
                try:
                    height, width = next(grids)
                except StopIteration:
                    raise ValueError("The conversation has more image parts than image grids.") from None
                grid_ids = [self.grid_token_id(height), self.grid_token_id(width)]
                if None in grid_ids:
                    raise ValueError(f"No `<|reserved_token_*|>` encodes an image grid of {height}x{width}.")
                add_ids([token_id(self.image_start), *grid_ids, token_id(self.boi)], False)
                add_ids([IMAGE_INPUT_INDEX] * (height * width), False)
                add_ids([token_id(self.image_end)], False)
        if previous_role == "assistant":
            add_text(self.eos, previous_target)
        if next(grids, None) is not None:
            raise ValueError("The conversation has fewer image parts than image grids.")
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

    def get_jinja_template(self) -> str:
        return (
            "{% set ns = namespace(system='', prev='') %}"
            "{% for m in messages %}{% if m['role'] == 'system' %}{% set ns.system = ns.system + m['content'] %}"
            "{% endif %}{% endfor %}"
            "{% if not ns.system %}{% set ns.system = 'You are a multimodal understanding assistant.' %}{% endif %}"
            "{% set roles = {'user': 'HUMAN', 'assistant': 'ASSISTANT', 'tool': 'OBSERVATION'} %}"
            "{% for m in messages if m['role'] != 'system' %}"
            "{% if m['role'] != ns.prev %}"
            "{% if ns.prev == 'assistant' %}{{ '<|endoftext|>' }}{% endif %}"
            "{% if not ns.prev %}{{ '<role>SYSTEM</role> ' + ns.system + ' ' }}{% endif %}"
            "{{ '<role>' + roles.get(m['role'], m['role'] | upper) + '</role>' }}"
            "{% set ns.prev = m['role'] %}{% endif %}"
            "{{ m['content'] }}"
            "{% endfor %}"
            "{% if ns.prev == 'assistant' %}{{ '<|endoftext|>' }}{% endif %}"
            "{% if add_generation_prompt %}{{ '<role>ASSISTANT</role>' }}{% endif %}"
        )


@CHAT_TEMPLATE_REGISTRY.register("chatml")
class ChatmlTemplate(ChatTemplate):
    def encode_messages(self, messages: Sequence[Dict[str, str]], max_seq_len: int = 8192) -> Dict[str, List[int]]:
        input_ids, attention_mask, labels = [], [], []
        for message in messages:
            content_str = "<|im_start|>" + message["role"] + "\n" + message["content"].strip() + "<|im_end|>\n"
            content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
            input_ids += content_ids
            attention_mask += [1] * len(content_ids)

            if "loss_mask" in message:
                loss_mask = message["loss_mask"]
            else:
                loss_mask = 1 if message["role"] == "assistant" else 0
            if loss_mask == 1:
                labels += content_ids
            else:
                labels += [IGNORE_INDEX] * len(content_ids)

        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        model_inputs = {k: v[-max_seq_len:] for k, v in model_inputs.items()}
        return model_inputs

    def get_jinja_template(self) -> str:
        return (
            "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )


class MTPLabelMixin:
    def __init__(self, tokenizer_or_processor):
        super().__init__(tokenizer_or_processor)
        self.mtp_num_hidden_layers = _get_mtp_num_hidden_layers(tokenizer_or_processor)

    def encode_messages(self, *args, **kwargs):
        feature = super().encode_messages(*args, **kwargs)
        add_mtp_labels(feature, self.mtp_num_hidden_layers)
        return feature


@CHAT_TEMPLATE_REGISTRY.register("qwen3_5")
class Qwen3_5ChatTemplate(MTPLabelMixin, ChatmlTemplate):
    pass


class MultimodalChatTemplate(ChatTemplate):
    """Base for templates whose messages carry non-text content.

    ``encode_messages`` takes a different contract from the text-only base: it
    receives the per-modality token counts the processor derived from the actual
    pixels, because only then can the template expand a placeholder into the
    right number of pad tokens. Selecting one of these by name from a text-only
    trainer therefore fails at encode time, not at build time.

    Built from the processor rather than the tokenizer, since laying those
    placeholders out also needs the grid parameters the processor used.
    """

    def __init__(self, processor: "ProcessorMixin") -> None:
        super().__init__(processor.tokenizer)
        self.processor = processor

    @abstractmethod
    def encode_messages(
        self, messages: Sequence[Dict[str, str]], num_tokens: Dict[str, List[int]] = None, **kwargs
    ) -> Dict[str, List[int]]:
        """
        Encodes messages to a dictionary of input_ids, attention_mask, labels, and mm with mm_seqlens.
        """


class Qwen2VLTemplate(MultimodalChatTemplate):
    def __init__(self, processor: "ProcessorMixin") -> None:
        super().__init__(processor)
        self.image_pad = "<|image_pad|>"
        self.video_pad = "<|video_pad|>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_pad)
        self.video_token_id = self.tokenizer.convert_tokens_to_ids(self.video_pad)
        # A tokenizer that never learned a pad answers with its unk id, and
        # _tokenize_and_remap would then rewrite every unk in the batch into a
        # modality sentinel, marking positions that carry no pixels. A tokenizer
        # with no unk id answers None instead, which the remap leaves alone.
        unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
        missing_pads = [
            pad
            for pad, token_id in ((self.image_pad, self.image_token_id), (self.video_pad, self.video_token_id))
            if unk_token_id is not None and token_id == unk_token_id
        ]
        if missing_pads:
            raise ValueError(f"The Qwen-VL chat template requires the {' and '.join(missing_pads)} tokenizer tokens.")

        logger.info_rank0("Qwen2VLTemplate will not truncate sequence when longer than [max_seq_lens].")

    def image_pattern(self, token_num):
        return "<|vision_start|>" + self.image_pad * token_num + "<|vision_end|>"

    def video_pattern(self, token_num):
        return "<|vision_start|>" + self.video_pad * token_num + "<|vision_end|>"

    @staticmethod
    def _next_token_num(token_nums, modality: str) -> int:
        """Pull the next per-item token count, naming the modality when short.

        A bare ``StopIteration`` here surfaces inside a dataloader worker with no
        indication of which modality ran out of counts.
        """
        try:
            return next(token_nums)
        except StopIteration as e:
            raise ValueError(f"{modality.capitalize()} token number is missing for a {modality} input.") from e

    def _tokenize_and_remap(self, messages: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """Tokenize rendered messages and remap modality pads to TYPE2INDEX.

        Shared by every Qwen-VL variant: the subclasses differ only in how they
        render ``messages`` (system prompt, video timestamps), not in how the
        rendered text becomes ids. Keeping one copy means a change to the
        modality contract cannot land in one variant and miss the other.
        """
        input_ids, attention_mask, labels = [], [], []
        for message in messages:
            content_str = message["content"].strip()
            loss_mask = message["loss_mask"]
            message_ids = self.tokenizer.encode("<|im_start|>" + message["role"] + "\n", add_special_tokens=False)
            # The "<|im_start|>{role}\n" header is a fixed prompt prefix, never a training target.
            prefix_len = len(message_ids)

            if content_str:
                end_ids = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
                content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
                message_ids += content_ids + end_ids

            input_ids += message_ids
            attention_mask += [1] * len(message_ids)
            if loss_mask == 1:
                labels += [IGNORE_INDEX] * prefix_len + message_ids[prefix_len:]
            else:
                labels += [IGNORE_INDEX] * len(message_ids)

        tokenized_example = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}

        # Replace the Qwen image/video pad ids with VeOmni's modality sentinels,
        # which is what process_sample_qwen_vl turns into image_mask / video_mask.
        image_mask = tokenized_example["input_ids"] == self.image_token_id
        tokenized_example["input_ids"][image_mask] = TYPE2INDEX["input"]["image"]

        video_mask = tokenized_example["input_ids"] == self.video_token_id
        tokenized_example["input_ids"][video_mask] = TYPE2INDEX["input"]["video"]

        return tokenized_example

    @abstractmethod
    def encode_messages(self, messages: Sequence[Dict[str, str]], **kwargs) -> Dict[str, List[int]]:
        pass


@CHAT_TEMPLATE_REGISTRY.register("qwen2vl")
class Qwen2VLChatTemplate(Qwen2VLTemplate):
    system_prompt = "You are a helpful assistant."

    def _get_system_message(self):
        system_message = {
            "role": "system",
            "content": self.system_prompt,
            "loss_mask": 0,
        }
        return system_message

    def encode_messages(
        self, conversations: Sequence[Dict[str, str]], num_tokens: Dict[str, List[int]] = None, **kwargs
    ) -> Dict[str, List[int]]:
        if num_tokens is None:
            num_tokens = defaultdict(list)
        sys_msg = self._get_system_message()
        messages = [] if sys_msg is None else [sys_msg]
        # Read, not popped: the per-modality counts belong to the caller, and the
        # local iterators already give the one-shot consumption this needs.
        image_token_num_list = iter(num_tokens.get("image", []))
        video_token_num_list = iter(num_tokens.get("video", []))
        for message in conversations:
            role = message[0]
            content = ""
            for value in message[1:]:
                if value[0] == "text":
                    content += value[1]
                elif value[0] == "image":
                    content += self.image_pattern(self._next_token_num(image_token_num_list, "image"))
                elif value[0] == "video":
                    content += self.video_pattern(self._next_token_num(video_token_num_list, "video"))
                else:
                    raise ValueError(f"Unknown value type: {value[0]}")
            messages.append(
                {
                    "role": role,
                    "content": content,
                    "loss_mask": 1 if role == "assistant" else 0,
                }
            )

        return self._tokenize_and_remap(messages)


# Qwen2.5-VL shares Qwen2-VL's template; the decorator form takes one name per class.
CHAT_TEMPLATE_REGISTRY.register("qwen2_5vl", Qwen2VLChatTemplate)


@CHAT_TEMPLATE_REGISTRY.register("qwen3vl")
class Qwen3VLChatTemplate(Qwen2VLTemplate):
    def _calculate_timestamps(self, indices: List[int], video_fps: float, temporal_patch_size: int):
        """
        Replicates Qwen3-VL official logic: Pad -> Convert to Seconds -> Average.
        """
        # 1. Pad frame indices to be divisible by temporal_patch_size
        # Copied first: VideoMetadata.frames_indices is declared list[int], and
        # padding it in place would append duplicate frames to the caller's
        # metadata.
        indices = list(indices)
        if len(indices) % temporal_patch_size != 0:
            indices.extend([indices[-1]] * (temporal_patch_size - len(indices) % temporal_patch_size))

        # 2. Convert indices to timestamps (seconds)
        timestamps = [idx / video_fps for idx in indices]

        # 3. Merge by size and take the average of start/end timestamps for each chunk
        timestamps = [
            (timestamps[i] + timestamps[i + temporal_patch_size - 1]) / 2
            for i in range(0, len(timestamps), temporal_patch_size)
        ]
        return timestamps

    def encode_messages(
        self, conversations: Sequence[Dict[str, str]], num_tokens: Dict[str, List[int]] = None, **kwargs
    ) -> Dict[str, List[int]]:
        if num_tokens is None:
            num_tokens = defaultdict(list)
        messages = []
        image_token_num_list = iter(num_tokens.get("image", []))
        video_token_num_list = iter(num_tokens.get("video", []))

        # Retrieve video metadata iterator; ensures order matches video inputs in conversations
        video_metadata_list = iter(kwargs.get("video_metadata", []))

        for message in conversations:
            role = message[0]
            content = ""
            for value in message[1:]:
                if value[0] == "text":
                    content += value[1]
                elif value[0] == "image":
                    content += self.image_pattern(self._next_token_num(image_token_num_list, "image"))

                elif value[0] == "video":
                    total_video_tokens = self._next_token_num(video_token_num_list, "video")
                    # Read off the same video processor that derived the grid, so
                    # the two agree on the chunk count. Models reusing this
                    # template need not all patch time by the same amount. Read
                    # here so a processor with no video side is never touched.
                    temporal_patch_size = self.processor.video_processor.temporal_patch_size

                    # Get metadata for the current video
                    try:
                        v_meta = next(video_metadata_list)
                    except StopIteration as e:
                        raise ValueError("Video metadata is missing for a video input.") from e

                    # 1. Extract FPS (default to 2.0 if missing)
                    fps = v_meta.fps if v_meta.fps is not None else 2.0

                    # 2. Retrieve sampled frame indices
                    if hasattr(v_meta, "frames_indices") and v_meta.frames_indices is not None:
                        indices = v_meta.frames_indices
                        # Convert numpy array to list if necessary
                        if hasattr(indices, "tolist"):
                            indices = indices.tolist()
                        elif not isinstance(indices, list):
                            indices = list(indices)
                    else:
                        # Fallback: create indices based on total frame count
                        total_frames = v_meta.total_num_frames if v_meta.total_num_frames is not None else 16
                        indices = list(range(total_frames))

                    # 3. Calculate timestamps using the new logic
                    timestamps = self._calculate_timestamps(indices, fps, temporal_patch_size=temporal_patch_size)

                    # 4. Calculate visual tokens per time chunk
                    num_time_chunks = len(timestamps)
                    # The vision tower emits exactly total_video_tokens embeddings and
                    # the model scatters them onto the placeholders emitted below, so
                    # an uneven split would drop placeholders and misalign every
                    # visual feature after it. The chunk count is re-derived from
                    # frames_indices rather than taken from video_grid_thw, so this
                    # only stays exact while the two agree -- fail loudly when not.
                    if num_time_chunks == 0 or total_video_tokens % num_time_chunks != 0:
                        raise ValueError(
                            f"Cannot lay out {total_video_tokens} video tokens over {num_time_chunks} time "
                            f"chunks ({len(indices)} frame indices, temporal_patch_size={temporal_patch_size}): "
                            f"the token count must divide evenly across chunks."
                        )
                    tokens_per_chunk = total_video_tokens // num_time_chunks

                    # 5. Construct Qwen3-VL style video string
                    # Format: <t seconds><|vision_start|>...tokens...<|vision_end|>
                    video_str_buffer = ""
                    for t_val in timestamps:
                        video_str_buffer += f"<{float(t_val):.1f} seconds>"
                        video_str_buffer += "<|vision_start|>"
                        video_str_buffer += self.video_pad * tokens_per_chunk
                        video_str_buffer += "<|vision_end|>"

                    content += video_str_buffer

                else:
                    raise ValueError(f"Unknown value type: {value[0]}")

            messages.append(
                {
                    "role": role,
                    "content": content,
                    "loss_mask": 1 if role == "assistant" else 0,
                }
            )

        return self._tokenize_and_remap(messages)


@CHAT_TEMPLATE_REGISTRY.register("qwen3_5vl")
class Qwen3_5VLChatTemplate(MTPLabelMixin, Qwen3VLChatTemplate):
    pass
