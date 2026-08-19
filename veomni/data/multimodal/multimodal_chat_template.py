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


import random
from abc import abstractmethod
from collections import defaultdict
from typing import Dict, List, Sequence

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from ...utils import logging
from ...utils.constants import IGNORE_INDEX, TYPE2INDEX
from ..chat_template import ChatTemplate


logger = logging.get_logger(__name__)


class MultimodalChatTemplate(ChatTemplate):
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
    def __init__(self, tokenizer: PreTrainedTokenizer, **kwargs) -> None:
        super().__init__(tokenizer)
        self.image_pad = "<|image_pad|>"
        self.video_pad = "<|video_pad|>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_pad)
        self.video_token_id = self.tokenizer.convert_tokens_to_ids(self.video_pad)
        self.image_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")  # 151652
        self.image_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")  # 151653
        self.eos = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)  # [151645, 198]
        self.bos = self.tokenizer.encode("<|im_start|>", add_special_tokens=False)

        logger.info_rank0("Qwen2VLTemplate will not truncate sequence when longer than [max_seq_lens].")

        self.cfg_ratio = kwargs.get("cfg_ratio", None)

    @property
    def _unconditioned_generation(self):
        return self.cfg_ratio and random.random() < self.cfg_ratio

    def image_pattern(self, token_num):
        return "<|vision_start|>" + self.image_pad * token_num + "<|vision_end|>"

    def video_pattern(self, token_num):
        return "<|vision_start|>" + self.video_pad * token_num + "<|vision_end|>"

    @abstractmethod
    def encode_messages(self, messages: Sequence[Dict[str, str]], **kwargs) -> Dict[str, List[int]]:
        pass


class Qwen2VLChatTemplate(Qwen2VLTemplate):
    system_prompt = "You are a helpful assistant."

    def _get_system_mesage(self):
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
        sys_msg = self._get_system_mesage()
        messages = [] if sys_msg is None else [sys_msg]
        data_type = ""
        image_token_num_list = iter(num_tokens.pop("image", []))
        video_token_num_list = iter(num_tokens.pop("video", []))
        for message in conversations:
            role = message[0]
            content = ""
            for value in message[1:]:
                if value[0] == "text":
                    content += value[1]
                elif value[0] == "image":
                    data_type = "t2i" if role == "assistant" else "i2t"
                    content += self.image_pattern(next(image_token_num_list))
                elif value[0] == "video":
                    content += self.video_pattern(next(video_token_num_list))
                else:
                    raise ValueError(f"Unknown value type: {value[0]}")
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
            loss_mask = message["loss_mask"]
            role = message["role"]
            message_ids = self.tokenizer.encode("<|im_start|>" + message["role"] + "\n", add_special_tokens=False)
            # The "<|im_start|>{role}\n" header is a fixed prompt prefix, never a training target.
            prefix_len = len(message_ids)

            if content_str:
                end_ids = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
                content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
                if (
                    role == "user" and data_type == "t2i" and self._unconditioned_generation
                ):  # unconditioned generation
                    message_ids += [self.tokenizer.pad_token_id] * len(content_ids) + end_ids
                else:
                    message_ids += content_ids + end_ids

            input_ids += message_ids
            attention_mask += [1] * len(message_ids)
            if loss_mask == 1:
                labels += [IGNORE_INDEX] * prefix_len + message_ids[prefix_len:]
            else:
                labels += [IGNORE_INDEX] * len(message_ids)

        tokenized_example = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}

        # change qwen2vl tokenized_image/video_id to seedomni_image/video_id
        image_mask = tokenized_example["input_ids"] == self.image_token_id
        input_mask = tokenized_example["labels"] == IGNORE_INDEX
        input_image_mask = image_mask & input_mask
        output_image_mask = image_mask & ~input_mask
        tokenized_example["input_ids"][input_image_mask] = TYPE2INDEX["input"]["image"]
        tokenized_example["input_ids"][output_image_mask] = TYPE2INDEX["output"]["image"]

        video_mask = tokenized_example["input_ids"] == self.video_token_id
        tokenized_example["input_ids"][video_mask] = TYPE2INDEX["input"]["video"]
        tokenized_example["labels"][output_image_mask] = IGNORE_INDEX  # the label will be filled in decoder.
        if data_type == "t2i":  # t2i doesn't train <|vision_start|>
            labels = tokenized_example["labels"]
            labels[labels == self.image_start_id] = IGNORE_INDEX
            tokenized_example["labels"] = labels

        return tokenized_example


class Qwen3VLChatTemplate(Qwen2VLTemplate):
    # Qwen3-VL default temporal_patch_size
    MERGE_SIZE = 2

    # ================= [New: Official Timestamp Calculation Logic] =================
    def _calculate_timestamps(self, indices: List[int], video_fps: float, merge_size: int = 2):
        """
        Replicates Qwen3-VL official logic: Pad -> Convert to Seconds -> Average.
        """
        # 1. Pad frame indices to be divisible by merge_size
        if len(indices) % merge_size != 0:
            indices.extend([indices[-1]] * (merge_size - len(indices) % merge_size))

        # 2. Convert indices to timestamps (seconds)
        timestamps = [idx / video_fps for idx in indices]

        # 3. Merge by size and take the average of start/end timestamps for each chunk
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps

    # ===============================================================================

    def encode_messages(
        self, conversations: Sequence[Dict[str, str]], num_tokens: Dict[str, List[int]] = None, **kwargs
    ) -> Dict[str, List[int]]:
        if num_tokens is None:
            num_tokens = defaultdict(list)
        messages = []
        data_type = ""
        image_token_num_list = iter(num_tokens.pop("image", []))
        video_token_num_list = iter(num_tokens.pop("video", []))

        # Retrieve video metadata iterator; ensures order matches video inputs in conversations
        video_metadata_list = iter(kwargs.get("video_metadata", []))

        for message in conversations:
            role = message[0]
            content = ""
            for value in message[1:]:
                if value[0] == "text":
                    content += value[1]
                elif value[0] == "image":
                    data_type = "t2i" if role == "assistant" else "i2t"
                    # Assumes self.image_pattern returns Qwen2-VL style image padding
                    content += self.image_pattern(next(image_token_num_list))

                elif value[0] == "video":
                    # --- [Core Modification: Video Timestamp Processing] ---
                    try:
                        total_video_tokens = next(video_token_num_list)
                    except StopIteration as e:
                        raise ValueError("Video token number is missing for a video input.") from e

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
                    timestamps = self._calculate_timestamps(indices, fps, merge_size=self.MERGE_SIZE)

                    # 4. Calculate visual tokens per time chunk
                    num_time_chunks = len(timestamps)

                    if num_time_chunks > 0:
                        tokens_per_chunk = total_video_tokens // num_time_chunks
                    else:
                        tokens_per_chunk = 0

                    # 5. Construct Qwen3-VL style video string
                    # Format: <t seconds><|vision_start|>...tokens...<|vision_end|>
                    video_str_buffer = ""
                    for t_val in timestamps:
                        video_str_buffer += f"<{float(t_val):.1f} seconds>"
                        video_str_buffer += "<|vision_start|>"  # self.vision_start_token
                        video_str_buffer += "<|video_pad|>" * tokens_per_chunk
                        video_str_buffer += "<|vision_end|>"

                    content += video_str_buffer
                    # --- [End Modification] ---

                else:
                    raise ValueError(f"Unknown value type: {value[0]}")

            messages.append(
                {
                    "role": role,
                    "content": content,
                    "loss_mask": 1 if role == "assistant" else 0,
                }
            )

        # Standard logic to convert messages to input_ids (kept largely unchanged)
        input_ids, attention_mask, labels = [], [], []
        for message in messages:
            content_str = message["content"].strip()
            loss_mask = message["loss_mask"]
            role = message["role"]
            message_ids = self.tokenizer.encode("<|im_start|>" + message["role"] + "\n", add_special_tokens=False)
            # The "<|im_start|>{role}\n" header is a fixed prompt prefix, never a training target.
            prefix_len = len(message_ids)

            if content_str:
                end_ids = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
                # Note: content_str now contains expanded timestamps and video pads
                content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)

                if role == "user" and data_type == "t2i" and self._unconditioned_generation:
                    message_ids += [self.tokenizer.pad_token_id] * len(content_ids) + end_ids
                else:
                    message_ids += content_ids + end_ids

            input_ids += message_ids
            attention_mask += [1] * len(message_ids)
            if loss_mask == 1:
                labels += [IGNORE_INDEX] * prefix_len + message_ids[prefix_len:]
            else:
                labels += [IGNORE_INDEX] * len(message_ids)

        tokenized_example = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}

        # ID replacement logic
        # Note: video_mask logic remains valid as self.video_token_id maps to <|video_pad|>
        image_mask = tokenized_example["input_ids"] == self.image_token_id
        input_mask = tokenized_example["labels"] == IGNORE_INDEX
        input_image_mask = image_mask & input_mask
        output_image_mask = image_mask & ~input_mask
        tokenized_example["input_ids"][input_image_mask] = TYPE2INDEX["input"]["image"]
        tokenized_example["input_ids"][output_image_mask] = TYPE2INDEX["output"]["image"]

        video_mask = tokenized_example["input_ids"] == self.video_token_id
        tokenized_example["input_ids"][video_mask] = TYPE2INDEX["input"]["video"]
        tokenized_example["labels"][output_image_mask] = IGNORE_INDEX

        if data_type == "t2i":
            labels = tokenized_example["labels"]
            labels[labels == self.image_start_id] = IGNORE_INDEX
            tokenized_example["labels"] = labels

        return tokenized_example


TEMPLATES = {
    "qwen2vl": Qwen2VLChatTemplate,
    "qwen3vl": Qwen3VLChatTemplate,
    "qwen2_5vl": Qwen2VLChatTemplate,  # same as qwen2vl
}


def build_multimodal_chat_template(template_name: str, tokenizer: AutoTokenizer, **kwargs) -> "ChatTemplate":
    if template_name not in TEMPLATES:
        raise ValueError(f"Unknown chat template: {template_name}")

    return TEMPLATES[template_name](tokenizer, **kwargs)
