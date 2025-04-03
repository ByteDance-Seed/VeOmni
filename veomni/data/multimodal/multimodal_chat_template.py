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
from collections import defaultdict
from typing import Dict, List, Literal, Sequence

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from ...utils import logging
from ..chat_template import ChatmlTemplate, ChatTemplate
from ..constants import IGNORE_INDEX, TYPE2INDEX


logger = logging.get_logger(__name__)


class MultimodalChatTemplate(ChatTemplate):
    @abstractmethod
    def encode_messages(
        self, messages: Sequence[Dict[str, str]], mm_num_tokens: Dict[str, List[int]], max_seq_len: int
    ) -> Dict[str, List[int]]:
        """
        Encodes messages to a dictionary of input_ids, attention_mask, labels, and mm with mm_seqlens.
        """

    def get_jinja_template(self) -> str:
        return ""

    def mm_tokenize(
        self,
        mm_type: Literal["image", "video", "audio"],
        token_num: int = 1,
    ):
        raise NotImplementedError

    def tokenize(
        self,
        content_type: Literal["text", "image", "video", "audio"],
        content: str,
        token_num: int = 1,
    ) -> List:
        if content_type == "text":
            input_ids = self.tokenizer(content).input_ids
        else:
            input_ids = self.mm_tokenize(content_type, token_num)
        return input_ids


class DefaultTag(ABC):
    def mm_tokenize(
        self,
        mm_type: Literal["image", "video", "audio"],
        token_num: int = 1,
    ):
        return [TYPE2INDEX["input"][mm_type]] * token_num


class MMTag(ABC):
    def mm_tokenize(
        self,
        mm_type: Literal["image", "video", "audio"],
        token_num: int = 1,
    ):
        mm_start = f"[{mm_type.upper()}]"
        mm_end = f"[/{mm_type.upper()}]"
        mm_token = (
            self.tokenizer(mm_start).input_ids
            + [TYPE2INDEX["input"][mm_type]] * token_num
            + self.tokenizer(mm_end).input_ids
        )
        return mm_token


class PretrainTemplate(MultimodalChatTemplate):
    """
    Pretrain template for multimodal model.
    Text-to-Multimodal or Multimodal-to-Text only.
    """

    def encode_messages(
        self, messages: Sequence[Dict[str, str]], mm_num_tokens: Dict = defaultdict(lambda: [1]), **kwargs
    ) -> Dict[str, List[int]]:
        messages = messages[:2]
        assert messages[0][0] == "user"
        assert messages[1][0] == "assistant"
        messages = [message[1:] for message in messages]  # skip role
        mm = None
        for message in messages[0]:
            if message[0] != "text":
                mm = message[0]
                break

        converted_messages = []
        if mm is None:  # text to multimodal
            user_content = [messages[0][0]]
            assistant_content = []
            for message in messages[1]:
                if message[0] != "text":
                    assistant_content = [message]
                    mm = message[0]
                    break
        else:  # multimodal to text
            for message in messages[0]:
                if message[0] == mm:
                    user_content = [message]
                    break
            assistant_content = messages[1][:1]  # [] if eval

        converted_messages = [["user"] + user_content, ["assistant"] + assistant_content]
        mm_num_token = mm_num_tokens[mm][0]

        input_ids, labels = [], []
        for message in converted_messages:
            role = message[0]
            message = message[1:]
            if len(message) == 0:  # eval
                break

            output = self.tokenize(message[0][0], message[0][1], token_num=mm_num_token)

            if role == "user":
                labels += [IGNORE_INDEX] * len(output)
            else:
                output += [self.tokenizer.eos_token_id]
                labels += output

            input_ids += output

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)

        # mask multimodal label, set output_multimodal_token to input_ids
        input_mask = labels == IGNORE_INDEX
        for mm_type in mm_num_tokens.keys():
            mm_mask = input_ids == TYPE2INDEX["input"][mm_type]
            input_mm_mask = input_mask & mm_mask
            output_mm_mask = ~input_mask & mm_mask

            input_ids[input_mm_mask] = TYPE2INDEX["input"][mm_type]
            input_ids[output_mm_mask] = TYPE2INDEX["output"][mm_type]
            labels[output_mm_mask] = IGNORE_INDEX

        return {"input_ids": input_ids, "labels": labels, "attention_mask": torch.tensor([1] * len(input_ids))}


class SFTTemplate(MultimodalChatTemplate):
    def encode_messages(
        self, messages: Sequence[Dict[str, str]], mm_num_tokens: Dict[str, List[int]], **kwargs
    ) -> Dict[str, List[int]]:
        input_ids, labels = [], []
        mm_index = dict.fromkeys(mm_num_tokens.keys(), 0)
        for message_list in messages:
            role = message_list[0]
            message_list = message_list[1:]
            if len(message_list) == 0:  # eval
                break
            if role == "user":
                if message_list[0][0] == "text":
                    new_tuple = ("text", "[INST]" + message_list[0][1])
                    message_list[0] = new_tuple
                else:
                    message_list = [("text", "[INST]")] + message_list

                if message_list[-1][0] == "text":
                    new_tuple = ("text", message_list[-1][1] + "[/INST]")
                    message_list[-1] = new_tuple
                else:
                    message_list.append(("text", "[/INST]"))

            content_ids = []
            for message in message_list:
                content_type = message[0]
                content = message[1]
                if content_type != "text":
                    num_token = mm_num_tokens[content_type][mm_index[content_type]]
                    mm_index[content_type] += 1
                else:
                    num_token = None

                content_ids += self.tokenize(content_type, content, num_token)

            if role == "user":
                input_ids += content_ids
                labels += [IGNORE_INDEX] * len(content_ids)
            else:
                input_ids += content_ids + [self.tokenizer.eos_token_id]
                labels += content_ids + [self.tokenizer.eos_token_id]

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)

        # mask multimodal label, set output_multimodal_token to input_ids
        input_mask = labels == IGNORE_INDEX
        for mm_type in mm_num_tokens.keys():
            mm_mask = input_ids == TYPE2INDEX["input"][mm_type]
            input_mm_mask = input_mask & mm_mask
            output_mm_mask = ~input_mask & mm_mask

            input_ids[input_mm_mask] = TYPE2INDEX["input"][mm_type]
            input_ids[output_mm_mask] = TYPE2INDEX["output"][mm_type]
            labels[output_mm_mask] = IGNORE_INDEX

        return {"input_ids": input_ids, "labels": labels, "attention_mask": torch.tensor([1] * len(input_ids))}


class PlainTextTemplate(DefaultTag, PretrainTemplate):
    pass


class PlainTextnMMTagTemplate(MMTag, PretrainTemplate):
    pass


class ConversationTemplate(DefaultTag, SFTTemplate):
    pass


class ConversationMMTagTemplate(MMTag, SFTTemplate):
    pass


class Qwen2VLTemplate(MultimodalChatTemplate):
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__(tokenizer)
        self.image_pad = "<|image_pad|>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_pad)
        self.image_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.eos = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)

        logger.info_rank0("Qwen2VLTemplate will not truncate sequence when longer than [max_seq_lens].")

    def image_pattern(self, token_num):
        if self.template_type == "pretrain" or self.template_type == "conversation":
            return "<|vision_start|>" + self.image_pad * token_num + "<|vision_end|>"
        elif self.template_type == "pretrain_stg1":
            return self.image_pad * token_num
        else:
            raise ValueError(f"Unknown template type: {self.template_type}")

    @abstractmethod
    def encode_messages(self, messages: Sequence[Dict[str, str]]) -> Dict[str, List[int]]:
        pass


class Qwen2VLPretrainTemplate(Qwen2VLTemplate):
    template_type = "pretrain"

    def encode_messages(
        self, conversations: Sequence[Dict[str, str]], mm_num_tokens: Dict[str, List[int]]
    ) -> Dict[str, List[int]]:
        messages = []
        for message in conversations:
            role = message[0]
            content = ""
            for item in message[1:]:
                mm_type = item[0]
                if mm_type == "image":
                    content += self.image_pattern(mm_num_tokens[mm_type][0])
                    mm_num_tokens[mm_type] = mm_num_tokens[mm_type][1:]
                else:
                    content += item[1]
            messages.append(
                {
                    "role": role,
                    "content": content,
                    "loss_mask": 1 if role == "assistant" else 0,
                }
            )

        input_ids, attention_mask, labels = [], [], []
        for message in messages:
            content_str = message["content"].strip()
            content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
            input_ids += content_ids
            attention_mask += [1] * len(content_ids)
            if message["loss_mask"] == 1:
                labels += content_ids
            else:
                labels += [IGNORE_INDEX] * len(content_ids)
        input_ids += self.eos
        attention_mask += [1] * len(self.eos)
        labels += self.eos

        tokenized_example = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}

        # change qwen2vl_tokenized_image_id to seedomni_image_id
        image_mask = tokenized_example["input_ids"] == self.image_token_id
        input_mask = tokenized_example["labels"] == IGNORE_INDEX
        input_image_mask = image_mask & input_mask
        output_image_mask = image_mask & ~input_mask
        tokenized_example["input_ids"][input_image_mask] = TYPE2INDEX["input"]["image"]
        tokenized_example["input_ids"][output_image_mask] = TYPE2INDEX["output"]["image"]
        tokenized_example["labels"][output_image_mask] = IGNORE_INDEX  # the label will be filled in decoder.

        return tokenized_example


class Qwen2VLPretrainSTG1Template(Qwen2VLPretrainTemplate):
    template_type = "pretrain_stg1"


class Qwen2VLChatTemplate(Qwen2VLTemplate):
    template_type = "conversation"

    def encode_messages(
        self, conversations: Sequence[Dict[str, str]], mm_num_tokens: Dict[str, List[int]]
    ) -> Dict[str, List[int]]:
        image_index = 0
        messages = []
        token_num_list = mm_num_tokens.pop("image", [])
        assert len(mm_num_tokens) == 0
        for message in conversations:
            role = message[0]
            content = ""
            for value in message[1:]:
                if value[0] == "text":
                    content += value[1]
                else:
                    assert value[0] == "image"
                    content += self.image_pattern(token_num_list[image_index])
                    image_index += 1
            messages.append(
                {
                    "role": role,
                    "content": content,
                    "loss_mask": 1 if role == "assistant" else 0,
                }
            )

        input_ids, attention_mask, labels = [], [], []
        for message in messages:
            if message["content"] == "":  # eval
                content_str = "<|im_start|>" + message["role"] + "\n"
            else:
                content_str = "<|im_start|>" + message["role"] + "\n" + message["content"].strip() + "<|im_end|>\n"
            content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
            input_ids += content_ids
            attention_mask += [1] * len(content_ids)
            if message["loss_mask"] == 1:
                labels += content_ids
            else:
                labels += [IGNORE_INDEX] * len(content_ids)

        tokenized_example = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}

        # change qwen2vl_tokenized_image_id to seedomni_image_id
        image_mask = tokenized_example["input_ids"] == self.image_token_id
        input_mask = tokenized_example["labels"] == IGNORE_INDEX
        input_image_mask = image_mask & input_mask
        output_image_mask = image_mask & ~input_mask
        tokenized_example["input_ids"][input_image_mask] = TYPE2INDEX["input"]["image"]
        tokenized_example["input_ids"][output_image_mask] = TYPE2INDEX["output"]["image"]
        tokenized_example["labels"][output_image_mask] = IGNORE_INDEX  # the label will be filled in decoder.

        return tokenized_example


class Qwen2_5VLTemplate(MultimodalChatTemplate):
    def __init__(self, tokenizer: PreTrainedTokenizer, **kwargs) -> None:
        super().__init__(tokenizer)
        self.image_pad = "<|image_pad|>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_pad)
        self.image_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.eos = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)

        logger.info_rank0("Qwen2_5VLTemplate will not truncate sequence when longer than [max_seq_lens].")

    def image_pattern(self, token_num):
        if self.template_type == "conversation":
            return "<|vision_start|>" + self.image_pad * token_num + "<|vision_end|>"
        else:
            raise ValueError(f"Unknown template type: {self.template_type}")

    @abstractmethod
    def encode_messages(self, messages: Sequence[Dict[str, str]]) -> Dict[str, List[int]]:
        pass


class Qwen2_5VLChatTemplate(Qwen2_5VLTemplate):
    template_type = "conversation"
    system_prompt = "You are a helpful assistant."

    def _set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    def _get_system_mesage(self):
        system_message = {
            "role": "system",
            "content": self.system_prompt,
            "loss_mask": 0,
        }
        return system_message

    def encode_messages(
        self,
        conversations: Sequence[Dict[str, str]],
        mm_num_tokens: Dict[str, List[int]],
    ) -> Dict[str, List[int]]:
        image_index = 0
        messages = []
        token_num_list = mm_num_tokens.pop("image", [])
        assert len(mm_num_tokens) == 0
        # messages.append(self._get_system_mesage())
        for message in conversations:
            role = message[0]
            content = ""
            for value in message[1:]:
                if value[0] == "text":
                    content += value[1]
                else:
                    assert value[0] == "image"
                    content += self.image_pattern(token_num_list[image_index])
                    image_index += 1
            messages.append(
                {
                    "role": role,
                    "content": content,
                    "loss_mask": 1 if role == "assistant" else 0,
                }
            )

        input_ids, attention_mask, labels = [], [], []
        for message in messages:
            if message["content"] == "":  # eval
                content_str = "<|im_start|>" + message["role"] + "\n"
            else:
                content_str = "<|im_start|>" + message["role"] + "\n" + message["content"].strip() + "<|im_end|>\n"
            content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
            input_ids += content_ids
            attention_mask += [1] * len(content_ids)
            if message["loss_mask"] == 1:
                labels += content_ids
            else:
                labels += [IGNORE_INDEX] * len(content_ids)

        tokenized_example = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}

        # change qwen2vl_tokenized_image_id to seedomni_image_id
        image_mask = tokenized_example["input_ids"] == self.image_token_id
        input_mask = tokenized_example["labels"] == IGNORE_INDEX
        input_image_mask = image_mask & input_mask
        output_image_mask = image_mask & ~input_mask
        tokenized_example["input_ids"][input_image_mask] = TYPE2INDEX["input"]["image"]
        tokenized_example["input_ids"][output_image_mask] = TYPE2INDEX["output"]["image"]
        tokenized_example["labels"][output_image_mask] = IGNORE_INDEX  # the label will be filled in decoder.

        return tokenized_example


class JanusChatTemplate(ChatmlTemplate):
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__(tokenizer)
        self.image_pad = "<image_placeholder>"
        self.image_start_tag = "<begin_of_image>"
        self.image_end_tag = "<end_of_image>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_pad)
        self.image_start_id = self.tokenizer.convert_tokens_to_ids(self.image_start_tag)
        self.use_system_prompt = True
        self.system_prompt = (
            "You are a helpful language and vision assistant. "
            "You are able to understand the visual content that the user provides, "
            "and assist the user with a variety of tasks using natural language."
        )
        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.image_pad]})
        self.sep1 = "\n\n"
        self.sep2 = "<｜end▁of▁sentence｜>"  # eos
        self.eos = self.tokenizer.encode(self.sep2, add_special_tokens=False)

    def image_pattern(self, token_num):
        return self.image_start_tag + self.image_pad * token_num + self.image_end_tag

    def encode_messages(
        self, conversations: Sequence[Dict[str, str]], mm_num_tokens: Dict[str, List[int]], max_seq_len: int = 8192
    ) -> Dict[str, List[int]]:
        image_index = 0
        token_num_list = mm_num_tokens.pop("image")
        assert len(mm_num_tokens) == 0
        messages = []
        for i, message in enumerate(conversations):
            role = message[0]
            message = message[1:]
            content = ""
            for value in message:
                if value[0] == "text":
                    content += value[1]
                else:
                    assert value[0] == "image"
                    content += self.image_pattern(token_num_list[image_index])
                    image_index += 1
            messages.append(
                {
                    "role": role,
                    "content": content,
                    "loss_mask": 1 if role == "assistant" else 0,
                }
            )

        if self.use_system_prompt:
            input_ids = self.tokenizer.encode(self.system_prompt + self.sep1)
            attention_mask = [1] * len(input_ids)
            labels = [IGNORE_INDEX] * len(input_ids)
        else:
            input_ids = self.tokenizer.encode("")
            attention_mask = [1] * len(input_ids)
            labels = [IGNORE_INDEX] * len(input_ids)

        for i, message in enumerate(messages):
            if message["content"] == "":  # eval
                content_str = message["role"].capitalize() + ":"
            else:
                sep = self.sep2 if i % 2 == 1 else self.sep1
                content_str = message["role"].capitalize() + ": " + message["content"] + sep

            content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
            if len(input_ids) + len(content_ids) > max_seq_len:  # truncate
                break
            input_ids += content_ids
            attention_mask += [1] * len(content_ids)
            if message["loss_mask"] == 1:
                labels += content_ids
            else:
                labels += [IGNORE_INDEX] * len(content_ids)
        tokenized_example = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}

        # change qwen2vl_tokenized_image_id to seedomni_image_id
        image_mask = tokenized_example["input_ids"] == self.image_token_id
        input_mask = tokenized_example["labels"] == IGNORE_INDEX
        input_image_mask = image_mask & input_mask
        output_image_mask = image_mask & ~input_mask
        tokenized_example["input_ids"][input_image_mask] = TYPE2INDEX["input"]["image"]
        tokenized_example["input_ids"][output_image_mask] = TYPE2INDEX["output"]["image"]
        tokenized_example["labels"][output_image_mask] = IGNORE_INDEX  # the label will be filled in decoder.
        return tokenized_example


TEMPLATES = {
    "conversation_default": ConversationTemplate,
    "conversation_mmtag": ConversationMMTagTemplate,
    "plaintext_default": PlainTextTemplate,
    "plaintext_mmtag": PlainTextnMMTagTemplate,
    "qwen2vl": Qwen2VLChatTemplate,
    "qwen2vl_pretrain_stg1": Qwen2VLPretrainSTG1Template,
    "qwen2vl_pretrain": Qwen2VLPretrainTemplate,
    "qwen2_5vl": Qwen2_5VLChatTemplate,
    "janus": JanusChatTemplate,
}


def build_multimodal_chat_template(template_name: str, tokenizer: AutoTokenizer) -> "ChatTemplate":
    if template_name not in TEMPLATES:
        raise ValueError(f"Unknown chat template: {template_name}")

    return TEMPLATES[template_name](tokenizer)
