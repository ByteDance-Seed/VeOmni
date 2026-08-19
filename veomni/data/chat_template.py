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
from typing import TYPE_CHECKING, Dict, List, Sequence

import torch

from veomni.utils import logging

from ..utils.constants import IGNORE_INDEX, TYPE2INDEX
from ..utils.registry import Registry


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


logger = logging.get_logger(__name__)

ROLE_SUPPORTED = ["system", "user", "assistant", "tool"]
CHAT_TEMPLATE_REGISTRY = Registry("ChatTemplate")


def build_chat_template(
    template_name: str,
    tokenizer: "PreTrainedTokenizer",
    **kwargs,
) -> "ChatTemplate":
    """Builds any registered template, text-only or multimodal.

    ``kwargs`` reach the template constructor. No template currently declares
    one, so an unrecognised option raises there instead of being silently
    dropped.
    """
    return CHAT_TEMPLATE_REGISTRY[template_name](tokenizer, **kwargs)


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


class MultimodalChatTemplate(ChatTemplate):
    """Base for templates whose messages carry non-text content.

    ``encode_messages`` takes a different contract from the text-only base: it
    receives the per-modality token counts the processor derived from the actual
    pixels, because only then can the template expand a placeholder into the
    right number of pad tokens. Selecting one of these by name from a text-only
    trainer therefore fails at encode time, not at build time.
    """

    @abstractmethod
    def encode_messages(
        self, messages: Sequence[Dict[str, str]], num_tokens: Dict[str, List[int]] = None, **kwargs
    ) -> Dict[str, List[int]]:
        """
        Encodes messages to a dictionary of input_ids, attention_mask, labels, and mm with mm_seqlens.
        """

    def get_jinja_template(self) -> str:
        return ""


class Qwen2VLTemplate(MultimodalChatTemplate):
    def __init__(self, tokenizer: "PreTrainedTokenizer") -> None:
        super().__init__(tokenizer)
        self.image_pad = "<|image_pad|>"
        self.video_pad = "<|video_pad|>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_pad)
        self.video_token_id = self.tokenizer.convert_tokens_to_ids(self.video_pad)

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


@CHAT_TEMPLATE_REGISTRY.register("qwen3vl")
class Qwen3VLChatTemplate(Qwen2VLTemplate):
    # Fallback for callers that do not pass the video processor's value. Frames
    # are grouped into time chunks of this size, matching the processor's own
    # temporal patching, so a mismatch shifts every timestamp and chunk boundary.
    DEFAULT_TEMPORAL_PATCH_SIZE = 2

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
        # Supplied by the caller that ran the video processor, so the chunking
        # here matches the grid the processor derived. Models reusing this
        # template need not all patch time by the same amount.
        temporal_patch_size = kwargs.get("temporal_patch_size") or self.DEFAULT_TEMPORAL_PATCH_SIZE

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
                        video_str_buffer += "<|video_pad|>" * tokens_per_chunk
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


# Qwen2.5-VL shares Qwen2-VL's template; the decorator form takes one name per class.
CHAT_TEMPLATE_REGISTRY.register("qwen2_5vl", Qwen2VLChatTemplate)
