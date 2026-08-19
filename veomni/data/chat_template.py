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


from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Sequence

from veomni.utils import logging

from ..utils.constants import IGNORE_INDEX
from ..utils.registry import Registry


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


logger = logging.get_logger(__name__)

ROLE_SUPPORTED = ["system", "user", "assistant", "tool"]
CHAT_TEMPLATE_REGISTRY = Registry("ChatTemplate")


def build_chat_template(template_name: str, tokenizer: "PreTrainedTokenizer") -> "ChatTemplate":
    return CHAT_TEMPLATE_REGISTRY[template_name](tokenizer)


class ChatTemplate(ABC):
    """
    Abstract class for chat template.
    """

    def __init__(self, tokenizer: "PreTrainedTokenizer") -> None:
        self.tokenizer = tokenizer

    def save_pretrained(self, output_dir: str) -> None:
        self.tokenizer.chat_template = self.get_jinja_template()
        try:
            self.tokenizer.save_pretrained(output_dir)
        except Exception:
            logger.warning("Failed to save tokenizer.")

    @abstractmethod
    def encode_messages(self, messages: Sequence[Dict[str, str]], max_seq_len: int = 8192) -> Dict[str, List[int]]:
        """
        Encodes messages to a dictionary of input_ids, attention_mask, and labels.
        """
        ...

    @abstractmethod
    def get_jinja_template(self) -> str:
        """
        Gets the jinja template for the chat template.
        """
        ...


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
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    def get_jinja_template(self) -> str:
        if not self.tokenizer.chat_template:
            raise ValueError("The tokenizer does not define a native chat template.")
        return self.tokenizer.chat_template


@CHAT_TEMPLATE_REGISTRY.register("gpt_oss")
class GptOssTokenizerTemplate(TokenizerTemplate):
    """GPT-OSS native template with its terminal assistant-token rewrite."""

    def __init__(self, tokenizer: "PreTrainedTokenizer") -> None:
        super().__init__(tokenizer)
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
