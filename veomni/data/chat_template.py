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

import torch

from veomni.utils import logging

from ..utils.registry import Registry
from .constants import IGNORE_INDEX


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


@CHAT_TEMPLATE_REGISTRY.register("Janus")
class JanusTemplate(ChatTemplate):
    def encode_messages(
        self, messages: Sequence[Dict[str, str]], max_seq_len: int = 8192, task_type: str = ""
    ) -> Dict[str, List[int]]:
        input_ids, attention_mask, labels = [], [], []
        images_seq_mask, images_emb_mask = [], []
        seps = ["\n\n", "<｜end▁of▁sentence｜>"]
        assitant_cnt = 0
        for idx, message in enumerate(messages):
            if message["content"] == "":
                content_str = message["role"] + ":"
            elif (
                "assistant" in message["role"]
                and "wikihow_generation" in task_type
                or "assistant" in message["role"]
                and "interleave_generation" in task_type
            ):
                prefix = "Assistant: " if assitant_cnt == 0 else ""
                suffix = seps[1] if idx + 1 == len(messages) else seps[0]
                content_str = prefix + message["content"].strip() + suffix
                assitant_cnt += 1
            elif "assistant" in message["role"]:
                content_str = "Assistant" + ": " + message["content"].strip() + seps[1]
            elif "user" in message["role"]:
                content_str = "User" + ": " + message["content"].strip() + seps[0]
            elif "system" in message["role"] and "wikihow_generation" in task_type:
                content_str = (
                    message["content"].strip()
                    + seps[0]
                    + "Please generate a step-by-step tutorial with images for the following question."
                    + seps[0]
                )
            elif "system" in message["role"]:
                content_str = message["content"].strip() + seps[0]
            if "system" in message["role"]:
                content_ids = self.tokenizer.encode(content_str)
            else:
                content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
            input_ids += content_ids
            attention_mask += [1] * len(content_ids)
            image_token_id = self.tokenizer.vocab.get("<image_placeholder>")
            content_ids_tensor = torch.tensor(content_ids)
            images_seq_mask += (content_ids_tensor == image_token_id).tolist()
            image_token_id = self.tokenizer.vocab.get("<image_placeholder>")
            num_image_tokens = torch.sum(content_ids_tensor == image_token_id).item()
            n_image = num_image_tokens // 576
            if n_image > 0:
                for j, n_image_tokens in enumerate([num_image_tokens]):
                    images_emb_mask.append([True] * n_image_tokens)

            if message["loss_mask"] == 1:
                if (
                    image_token_id in content_ids
                    and "wikihow_generation" not in task_type
                    and "interleave_generation" not in task_type
                ):
                    labels += [image_token_id if x == image_token_id else IGNORE_INDEX for x in content_ids]
                else:
                    labels += content_ids
            else:
                labels += [IGNORE_INDEX] * len(content_ids)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images_seq_mask": images_seq_mask,
            "images_emb_mask": images_emb_mask,
        }
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


@CHAT_TEMPLATE_REGISTRY.register("chatml")
class ChatmlTemplate(ChatTemplate):
    def encode_messages(self, messages: Sequence[Dict[str, str]], max_seq_len: int = 8192) -> Dict[str, List[int]]:
        input_ids, attention_mask, labels = [], [], []
        for message in messages:
            content_str = "<|im_start|>" + message["role"] + "\n" + message["content"].strip() + "<|im_end|>\n"
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
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )
