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
        self, messages: Sequence[Dict[str, str]], num_tokens: Dict[str, List[int]] = defaultdict(list), **kwargs
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
        self, messages: Sequence[Dict[str, str]], num_tokens: Dict = defaultdict(list), **kwargs
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
        mm_num_token = num_tokens[mm][0]

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
        for mm_type in num_tokens.keys():
            mm_mask = input_ids == TYPE2INDEX["input"][mm_type]
            input_mm_mask = input_mask & mm_mask
            output_mm_mask = ~input_mask & mm_mask

            input_ids[input_mm_mask] = TYPE2INDEX["input"][mm_type]
            input_ids[output_mm_mask] = TYPE2INDEX["output"][mm_type]
            labels[output_mm_mask] = IGNORE_INDEX

        return {"input_ids": input_ids, "labels": labels, "attention_mask": torch.tensor([1] * len(input_ids))}


class SFTTemplate(MultimodalChatTemplate):
    def encode_messages(
        self, messages: Sequence[Dict[str, str]], num_tokens: Dict[str, List[int]] = defaultdict(list), **kwargs
    ) -> Dict[str, List[int]]:
        input_ids, labels = [], []
        mm_index = dict.fromkeys(num_tokens.keys(), 0)
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
                    num_token = num_tokens[content_type][mm_index[content_type]]
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
        for mm_type in num_tokens.keys():
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


class Qwen2VLPretrainTemplate(Qwen2VLTemplate):  # For Omni Only
    def encode_messages(
        self, conversations: Sequence[Dict[str, str]], num_tokens: Dict[str, List[int]] = defaultdict(list), **kwargs
    ) -> Dict[str, List[int]]:
        messages = []
        data_type = ""
        mm_num_tokens = {key: iter(item) for key, item in num_tokens.items()}
        for message in conversations:
            role = message[0]
            content = ""
            for item in message[1:]:
                mm_type = item[0]
                if mm_type == "image":
                    data_type = "t2i" if role == "assistant" else "i2t"
                    content += self.image_pattern(next(mm_num_tokens[mm_type]))
                elif mm_type == "video":
                    content += self.video_pattern(next(mm_num_tokens[mm_type]))
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
        input_ids += self.bos
        attention_mask += [1] * len(self.bos)
        labels += self.bos
        for message in messages:
            content_str = message["content"].strip()
            content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
            loss_mask = message["loss_mask"]
            if content_str == "":  # eval
                break
            if role == "user" and data_type == "t2i" and self._unconditioned_generation:  # unconditioned generation
                input_ids += [self.tokenizer.pad_token_id] * len(content_ids)
            else:
                input_ids += content_ids

            attention_mask += [1] * len(content_ids)
            if loss_mask == 1:
                labels += content_ids
                input_ids += self.eos
                attention_mask += [1] * len(self.eos)
                labels += self.eos
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

        if data_type == "t2i":  # t2i doesn't train <|vision_start|> and <|vision_end|>
            labels = tokenized_example["labels"]
            labels[~output_image_mask] = IGNORE_INDEX
            tokenized_example["labels"] = labels
        return tokenized_example


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
        self, conversations: Sequence[Dict[str, str]], num_tokens: Dict[str, List[int]] = defaultdict(list), **kwargs
    ) -> Dict[str, List[int]]:
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
                labels += message_ids
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
    system_prompt = "You are a helpful assistant."

    # Qwen3-VL default temporal_patch_size
    MERGE_SIZE = 2

    def _get_system_mesage(self):
        system_message = {
            "role": "system",
            "content": self.system_prompt,
            "loss_mask": 0,
        }
        return system_message

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
        self, conversations: Sequence[Dict[str, str]], num_tokens: Dict[str, List[int]] = defaultdict(list), **kwargs
    ) -> Dict[str, List[int]]:
        sys_msg = self._get_system_mesage()
        messages = [] if sys_msg is None else [sys_msg]
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
                    except StopIteration:
                        raise ValueError("Video token number is missing for a video input.")

                    # Get metadata for the current video
                    try:
                        v_meta = next(video_metadata_list)
                    except StopIteration:
                        raise ValueError("Video metadata is missing for a video input.")

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
                labels += message_ids
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


class Qwen25OmniChatTemplate(Qwen2VLChatTemplate):
    system_prompt = (
        "You are Qwen, a virtual human developed by the Qwen Team, "
        "Alibaba Group, capable of perceiving auditory and visual inputs, "
        "as well as generating text and speech."
    )

    def __init__(self, tokenizer: PreTrainedTokenizer, **kwargs) -> None:
        MultimodalChatTemplate.__init__(self, tokenizer)
        self.image_pad = "<|IMAGE|>"
        self.video_pad = "<|VIDEO|>"
        self.audio_pad = "<|AUDIO|>"
        self.vision_bos_token = "<|vision_bos|>"
        self.vision_eos_token = "<|vision_eos|>"
        self.audio_bos_token = "<|audio_bos|>"
        self.audio_eos_token = "<|audio_eos|>"

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_pad)  # 151655
        self.video_token_id = self.tokenizer.convert_tokens_to_ids(self.video_pad)  # 151656
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_pad)  # 151646

        self.vision_bos_id = self.tokenizer.convert_tokens_to_ids(self.vision_bos_token)  # 151652
        self.vision_eos_id = self.tokenizer.convert_tokens_to_ids(self.vision_eos_token)  # 151653
        self.audio_bos_id = self.tokenizer.convert_tokens_to_ids(self.audio_bos_token)  # 151647
        self.audio_eos_id = self.tokenizer.convert_tokens_to_ids(self.audio_eos_token)  # 151648

        self.bos = self.tokenizer.encode("<|im_start|>", add_special_tokens=False)
        self.eos = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)  # [151645, 198]

        # TODO: maybe customized these
        self.seconds_per_chunk = 2.0
        self.position_id_per_seconds = 25
        self.video_second_per_grid = 1.0

        logger.info_rank0("Qwen25OmniTemplate will not truncate sequence when longer than [max_seq_lens].")

    def image_pattern(self, token_num):
        return self.vision_bos_token + self.image_pad * token_num + self.vision_eos_token

    def get_chunked_index(self, token_indices, tokens_per_chunk):
        """Copied from processing_qwen2_5_omni.py"""

        def _iter():
            i, start_idx = 0, 0
            current_chunk = 1
            while i < len(token_indices):
                if token_indices[i] >= current_chunk * tokens_per_chunk:
                    yield (start_idx, i)
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield (start_idx, len(token_indices))

        return list(_iter())

    def video_pattern(
        self, video_token_num: torch.Tensor, audio_token_num: torch.Tensor, curr_video_grid_thw: torch.Tensor
    ):
        if audio_token_num == 0:  # no audio with this video
            return self.vision_bos_token + self.video_pad * video_token_num + self.vision_eos_token
        else:
            """Modified from processing_qwen2_5_omni.py
            """
            audio_token_indices = torch.arange(audio_token_num)
            merge_size = torch.sqrt(curr_video_grid_thw.prod() // video_token_num).int()
            height = (curr_video_grid_thw[1] // merge_size).item()
            width = (curr_video_grid_thw[2] // merge_size).item()
            video_token_indices = torch.arange(curr_video_grid_thw[0]).reshape(-1, 1, 1)
            video_token_indices = video_token_indices.expand(-1, height, width).reshape(-1)
            video_token_indices = video_token_indices * self.video_second_per_grid * self.position_id_per_seconds

            tokens_per_chunk = int(self.position_id_per_seconds * self.seconds_per_chunk)
            video_chunk_indexes = self.get_chunked_index(video_token_indices, tokens_per_chunk)
            audio_chunk_indexes = self.get_chunked_index(audio_token_indices, tokens_per_chunk)

            content = self.vision_bos_token + self.audio_bos_token
            for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                if j < len(video_chunk_indexes):
                    video_seq_length = video_chunk_indexes[j][1] - video_chunk_indexes[j][0]
                    content += self.video_pad * video_seq_length
                if j < len(audio_chunk_indexes):
                    audio_seq_length = audio_chunk_indexes[j][1] - audio_chunk_indexes[j][0]
                    content += self.audio_pad * audio_seq_length
            content += self.audio_eos_token + self.vision_eos_token
            return content

    def audio_pattern(self, token_num):
        return self.audio_bos_token + self.audio_pad * token_num + self.audio_eos_token

    def encode_messages(
        self, conversations: Sequence[Dict[str, str]], num_tokens: Dict[str, List[int]] = defaultdict(list), **kwargs
    ) -> Dict[str, List[int]]:
        sys_msg = self._get_system_mesage()
        messages = [] if sys_msg is None else [sys_msg]
        multimodal_num_tokens = {key: iter(item) for key, item in num_tokens.items()}  # image, video, audio

        video_grid_thw = kwargs.get("grid_thw", {}).get("video", None)
        video_grid_thw = iter(video_grid_thw) if video_grid_thw is not None else None

        for message in conversations:
            role = message[0]
            content = ""
            for value in message[1:]:
                if value[0] == "text":
                    content += value[1]
                elif value[0] == "image":
                    content += self.image_pattern(next(multimodal_num_tokens["image"]))
                elif value[0] == "video":
                    if video_grid_thw is None:
                        raise ValueError(
                            f"video_grid_thw: {video_grid_thw} is None. "
                            "Make sure your video processor outputs `grid_thw`."
                        )
                    content += self.video_pattern(
                        next(multimodal_num_tokens["video"]),
                        next(multimodal_num_tokens["audio"]),
                        curr_video_grid_thw=next(video_grid_thw),
                    )
                elif value[0] == "audio":
                    content += self.audio_pattern(next(multimodal_num_tokens["audio"]))
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
            if content_str:
                content_str = "<|im_start|>" + message["role"] + "\n" + content_str + "<|im_end|>\n"
            else:
                content_str = "<|im_start|>" + message["role"] + "\n"

            message_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
            input_ids += message_ids
            attention_mask += [1] * len(message_ids)
            if loss_mask == 1:
                labels += message_ids
            else:
                labels += [IGNORE_INDEX] * len(message_ids)

        tokenized_example = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}

        # change qwen25omni tokenized_image/video/audio_id to seedomni_image/video/audio_id
        input_mask = tokenized_example["labels"] == IGNORE_INDEX

        image_mask = tokenized_example["input_ids"] == self.image_token_id
        input_image_mask = image_mask & input_mask
        output_image_mask = image_mask & ~input_mask
        tokenized_example["input_ids"][input_image_mask] = TYPE2INDEX["input"]["image"]
        tokenized_example["input_ids"][output_image_mask] = TYPE2INDEX["output"]["image"]

        # no video/audio output currently
        video_mask = tokenized_example["input_ids"] == self.video_token_id
        tokenized_example["input_ids"][video_mask] = TYPE2INDEX["input"]["video"]

        audio_mask = tokenized_example["input_ids"] == self.audio_token_id
        tokenized_example["input_ids"][audio_mask] = TYPE2INDEX["input"]["audio"]

        tokenized_example["labels"][output_image_mask] = IGNORE_INDEX  # the label will be filled in decoder.
        return tokenized_example


class JanusChatTemplate(ChatmlTemplate):
    def __init__(self, tokenizer: PreTrainedTokenizer, use_system_prompt=True) -> None:
        super().__init__(tokenizer)
        self.image_pad = "<image_placeholder>"
        self.image_start_tag = "<begin_of_image>"
        self.image_end_tag = "<end_of_image>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_pad)
        self.image_start_id = self.tokenizer.convert_tokens_to_ids(self.image_start_tag)
        self.use_system_prompt = use_system_prompt
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
        self,
        conversations: Sequence[Dict[str, str]],
        num_tokens: Dict[str, List[int]] = defaultdict(list),
        max_seq_len: int = 8192,
        **kwargs,
    ) -> Dict[str, List[int]]:
        image_index = 0
        token_num_list = num_tokens.pop("image", [])
        messages = []
        use_system_prompt = False
        for i, message in enumerate(conversations):
            role = message[0]
            message = message[1:]
            content = ""
            for value in message:
                if value[0] == "text":
                    content += value[1]
                else:
                    use_system_prompt = True if role == "user" else use_system_prompt
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

        if use_system_prompt:
            input_ids = self.tokenizer.encode(self.system_prompt + self.sep1)
            attention_mask = [1] * len(input_ids)
            labels = [IGNORE_INDEX] * len(input_ids)
        else:
            input_ids = self.tokenizer.encode("")
            attention_mask = [1] * len(input_ids)
            labels = [IGNORE_INDEX] * len(input_ids)

        for i, message in enumerate(messages):
            role: str = message["role"]
            if role == "user":
                content_str = role.capitalize() + ": " + message["content"] + self.sep1
                content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
                input_ids += content_ids
                attention_mask += [1] * len(content_ids)
                labels += [IGNORE_INDEX] * len(content_ids)
            else:
                content_str = role.capitalize() + ":"
                content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
                input_ids += content_ids
                attention_mask += [1] * len(content_ids)
                labels += [IGNORE_INDEX] * len(content_ids)
                if message["content"]:
                    content_str = message["content"] + self.sep2
                    content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
                    input_ids += content_ids
                    attention_mask += [1] * len(content_ids)
                    labels += content_ids

        tokenized_example = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}

        image_mask = tokenized_example["input_ids"] == self.image_token_id
        input_mask = tokenized_example["labels"] == IGNORE_INDEX
        input_image_mask = image_mask & input_mask
        output_image_mask = image_mask & ~input_mask
        tokenized_example["input_ids"][input_image_mask] = TYPE2INDEX["input"]["image"]
        tokenized_example["input_ids"][output_image_mask] = TYPE2INDEX["output"]["image"]
        tokenized_example["labels"][output_image_mask] = IGNORE_INDEX  # the label will be filled in decoder.
        if not use_system_prompt:
            tokenized_example["labels"][tokenized_example["labels"] == self.eos[0]] = (
                IGNORE_INDEX  # eos seems not trained in Janus
            )
            tokenized_example["labels"][tokenized_example["labels"] == self.image_start_id] = (
                IGNORE_INDEX  # image_start_id seems not trained in Janus
            )
        return tokenized_example


class LlamaPretrainTemplate(MultimodalChatTemplate):  # For Omni Only
    def __init__(self, tokenizer: PreTrainedTokenizer, **kwargs) -> None:
        super().__init__(tokenizer)
        self.vision_bos_token = "<|vision_start|>"
        self.vision_eos_token = "<|vision_end|>"
        self.audio_bos_token = "<|audio_start|>"
        self.audio_eos_token = "<|audio_end|>"
        num_add = self.tokenizer.add_tokens(
            [self.vision_bos_token, self.vision_eos_token, self.audio_bos_token, self.audio_eos_token]
        )
        num_add += self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.add_token_num = num_add

        self.image_pad = "<|image_pad|>"
        self.video_pad = "<|video_pad|>"
        self.audio_pad = "<|audio_pad|>"
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.image_pad, self.video_pad, self.audio_pad]}
        )

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_pad)
        self.video_token_id = self.tokenizer.convert_tokens_to_ids(self.video_pad)
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_pad)

        self.vision_bos_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")  # 128256
        self.vision_eos_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")  # 128257
        self.audio_bos_id = self.tokenizer.convert_tokens_to_ids("<|audio_start|>")
        self.audio_eos_id = self.tokenizer.convert_tokens_to_ids("<|audio_end|>")

        self.pad_token_id = self.tokenizer.convert_tokens_to_ids("<|pad|>")  # 128258

        if self.add_token_num > 0:
            self.trained_embedding = [
                self.vision_bos_id,
                self.vision_eos_id,
                self.audio_bos_id,
                self.audio_eos_id,
                self.pad_token_id,
            ]

        # config for audio in video
        self.seconds_per_chunk = 2.0
        self.position_id_per_seconds = 25
        self.video_second_per_grid = 1.0

        self.eos = self.tokenizer.encode(self.tokenizer.eos_token, add_special_tokens=False)  # [128001]
        self.bos = self.tokenizer.encode(self.tokenizer.bos_token, add_special_tokens=False)  # [128000]
        self.cfg_ratio = kwargs.get("cfg_ratio", None)

    def image_pattern(self, token_num):
        return self.vision_bos_token + self.image_pad * token_num + self.vision_eos_token

    def get_chunked_index(self, token_indices, tokens_per_chunk):
        """Copied from processing_qwen2_5_omni.py"""

        def _iter():
            i, start_idx = 0, 0
            current_chunk = 1
            while i < len(token_indices):
                if token_indices[i] >= current_chunk * tokens_per_chunk:
                    yield (start_idx, i)
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield (start_idx, len(token_indices))

        return list(_iter())

    def video_pattern(
        self, video_token_num: torch.Tensor, audio_token_num: torch.Tensor, curr_video_grid_thw: torch.Tensor
    ):
        if audio_token_num == 0:  # no audio with this video
            return self.vision_bos_token + self.video_pad * video_token_num + self.vision_eos_token
        else:
            """Modified from processing_qwen2_5_omni.py
            """
            audio_token_indices = torch.arange(audio_token_num)
            merge_size = torch.sqrt(curr_video_grid_thw.prod() // video_token_num).int()
            height = (curr_video_grid_thw[1] // merge_size).item()
            width = (curr_video_grid_thw[2] // merge_size).item()
            video_token_indices = torch.arange(curr_video_grid_thw[0]).reshape(-1, 1, 1)
            video_token_indices = video_token_indices.expand(-1, height, width).reshape(-1)
            video_token_indices = video_token_indices * self.video_second_per_grid * self.position_id_per_seconds

            tokens_per_chunk = int(self.position_id_per_seconds * self.seconds_per_chunk)
            video_chunk_indexes = self.get_chunked_index(video_token_indices, tokens_per_chunk)
            audio_chunk_indexes = self.get_chunked_index(audio_token_indices, tokens_per_chunk)

            content = self.vision_bos_token + self.audio_bos_token
            for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                if j < len(video_chunk_indexes):
                    video_seq_length = video_chunk_indexes[j][1] - video_chunk_indexes[j][0]
                    content += self.video_pad * video_seq_length
                if j < len(audio_chunk_indexes):
                    audio_seq_length = audio_chunk_indexes[j][1] - audio_chunk_indexes[j][0]
                    content += self.audio_pad * audio_seq_length
            content += self.audio_eos_token + self.vision_eos_token
            return content

    def audio_pattern(self, token_num):
        return self.audio_bos_token + self.audio_pad * token_num + self.audio_eos_token

    @property
    def _unconditioned_generation(self):
        return self.cfg_ratio and random.random() < self.cfg_ratio

    def encode_messages(
        self, conversations: Sequence[Dict[str, str]], num_tokens: Dict[str, List[int]] = defaultdict(list), **kwargs
    ) -> Dict[str, List[int]]:
        messages = []
        multimodal_num_tokens = {key: iter(item) for key, item in num_tokens.items()}  # image, video, audio
        data_type = ""
        video_grid_thw = kwargs.get("grid_thw", {}).get("video", None)
        video_grid_thw = iter(video_grid_thw) if video_grid_thw is not None else None

        for message in conversations:
            role = message[0]
            content = ""
            for item in message[1:]:
                mm_type = item[0]
                if mm_type == "text":
                    content += item[1]
                elif mm_type == "image":
                    data_type = "t2i" if role == "assistant" else "i2t"
                    content += self.image_pattern(next(multimodal_num_tokens[mm_type]))
                elif mm_type == "video":
                    if video_grid_thw is None:
                        raise ValueError(
                            f"video_grid_thw: {video_grid_thw} is None. "
                            "Make sure your video processor outputs `grid_thw`."
                        )
                    content += self.video_pattern(
                        next(multimodal_num_tokens["video"]),
                        next(multimodal_num_tokens["audio"]),
                        curr_video_grid_thw=next(video_grid_thw),
                    )
                elif mm_type == "audio":
                    content += self.audio_pattern(next(multimodal_num_tokens["audio"]))
                else:
                    raise ValueError(f"Unknown value type: {item[0]}")
            messages.append(
                {
                    "role": role,
                    "content": content,
                    "loss_mask": 1 if role == "assistant" else 0,
                }
            )
        input_ids, attention_mask, labels = [], [], []

        input_ids += self.bos
        attention_mask += [1] * len(self.bos)
        labels += [IGNORE_INDEX] * len(self.bos)
        for message in messages:
            content_str = message["content"].strip()
            content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
            loss_mask = message["loss_mask"]
            if content_str == "":  # eval
                break
            if role == "user" and data_type == "t2i" and self._unconditioned_generation:  # unconditioned generation
                input_ids += [self.pad_token_id] * len(content_ids)
            else:
                input_ids += content_ids

            attention_mask += [1] * len(content_ids)
            if loss_mask == 1:
                labels += content_ids
                input_ids += self.eos
                attention_mask += [1] * len(self.eos)
                labels += self.eos
            else:
                labels += [IGNORE_INDEX] * len(content_ids)

        tokenized_example = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}

        # change to seedomni_image_id
        input_mask = tokenized_example["labels"] == IGNORE_INDEX
        image_mask = tokenized_example["input_ids"] == self.image_token_id

        input_image_mask = image_mask & input_mask
        output_image_mask = image_mask & ~input_mask
        tokenized_example["input_ids"][input_image_mask] = TYPE2INDEX["input"]["image"]
        tokenized_example["input_ids"][output_image_mask] = TYPE2INDEX["output"]["image"]

        # no video/audio output currently
        video_mask = tokenized_example["input_ids"] == self.video_token_id
        tokenized_example["input_ids"][video_mask] = TYPE2INDEX["input"]["video"]

        audio_mask = tokenized_example["input_ids"] == self.audio_token_id
        tokenized_example["input_ids"][audio_mask] = TYPE2INDEX["input"]["audio"]

        tokenized_example["labels"][output_image_mask] = IGNORE_INDEX  # the label will be filled in decoder.

        if data_type == "t2i":  # t2i doesn't train <|vision_start|> and <|vision_end|>
            labels = tokenized_example["labels"]
            labels[~output_image_mask] = IGNORE_INDEX
            tokenized_example["labels"] = labels
        return tokenized_example


class Qwen3MoeChatTemplate(Qwen25OmniChatTemplate):
    def __init__(self, tokenizer: PreTrainedTokenizer, **kwargs) -> None:
        MultimodalChatTemplate.__init__(self, tokenizer)
        self.image_pad = "<|IMAGE|>"
        self.video_pad = "<|VIDEO|>"
        self.audio_pad = "<|AUDIO|>"

        self.vision_bos_token = "<|vision_bos|>"
        self.vision_eos_token = "<|vision_eos|>"
        self.audio_bos_token = "<|audio_bos|>"
        self.audio_eos_token = "<|audio_eos|>"

        num_add = self.tokenizer.add_tokens(
            [self.vision_bos_token, self.vision_eos_token, self.audio_bos_token, self.audio_eos_token]
        )
        self.add_token_num = num_add

        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.image_pad, self.video_pad, self.audio_pad]}
        )

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_pad)
        self.video_token_id = self.tokenizer.convert_tokens_to_ids(self.video_pad)
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_pad)

        self.vision_bos_id = self.tokenizer.convert_tokens_to_ids(self.vision_bos_token)
        self.vision_eos_id = self.tokenizer.convert_tokens_to_ids(self.vision_eos_token)
        self.audio_bos_id = self.tokenizer.convert_tokens_to_ids(self.audio_bos_token)
        self.audio_eos_id = self.tokenizer.convert_tokens_to_ids(self.audio_eos_token)

        self.trained_embedding = []
        if self.add_token_num > 0:
            self.trained_embedding = [self.vision_bos_id, self.vision_eos_id, self.audio_bos_id, self.audio_eos_id]

        self.bos = self.tokenizer.encode("<|im_start|>", add_special_tokens=False)
        self.eos = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)

        self.seconds_per_chunk = 2.0
        self.position_id_per_seconds = 25
        self.video_second_per_grid = 1.0

        logger.info_rank0("Qwen3MoeTemplate will not truncate sequence when longer than [max_seq_lens].")

    def _get_system_mesage(self):
        return None


class SeedOssPretrainTemplate(LlamaPretrainTemplate):
    def __init__(self, tokenizer: PreTrainedTokenizer, **kwargs) -> None:
        super().__init__(tokenizer)
        self.vision_bos_token = "<|vision_start|>"
        self.vision_eos_token = "<|vision_end|>"
        self.audio_bos_token = "<|audio_start|>"
        self.audio_eos_token = "<|audio_end|>"
        num_add = self.tokenizer.add_tokens(
            [self.vision_bos_token, self.vision_eos_token, self.audio_bos_token, self.audio_eos_token]
        )
        num_add += self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.add_token_num = num_add

        self.image_pad = "<|image_pad|>"
        self.video_pad = "<|video_pad|>"
        self.audio_pad = "<|audio_pad|>"
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.image_pad, self.video_pad, self.audio_pad]}
        )

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_pad)
        self.video_token_id = self.tokenizer.convert_tokens_to_ids(self.video_pad)
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_pad)

        self.vision_bos_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_eos_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.audio_bos_id = self.tokenizer.convert_tokens_to_ids("<|audio_start|>")
        self.audio_eos_id = self.tokenizer.convert_tokens_to_ids("<|audio_end|>")

        self.pad_token_id = self.tokenizer.convert_tokens_to_ids("<|pad|>")

        if self.add_token_num > 0:
            self.trained_embedding = [
                self.vision_bos_id,
                self.vision_eos_id,
                self.audio_bos_id,
                self.audio_eos_id,
                self.pad_token_id,
            ]

        # config for audio in video
        self.seconds_per_chunk = 2.0
        self.position_id_per_seconds = 25
        self.video_second_per_grid = 1.0

        self.eos = self.tokenizer.encode(self.tokenizer.eos_token, add_special_tokens=False)
        self.bos = self.tokenizer.encode(self.tokenizer.bos_token, add_special_tokens=False)
        self.cfg_ratio = kwargs.get("cfg_ratio", None)


TEMPLATES = {
    "conversation_default": ConversationTemplate,
    "conversation_mmtag": ConversationMMTagTemplate,
    "plaintext_default": PlainTextTemplate,
    "plaintext_mmtag": PlainTextnMMTagTemplate,
    "qwen2vl": Qwen2VLChatTemplate,
    "qwen3vl": Qwen3VLChatTemplate,
    "qwen2vl_pretrain": Qwen2VLPretrainTemplate,
    "qwen2_5omni": Qwen25OmniChatTemplate,
    "qwen2_5vl": Qwen2VLChatTemplate,  # same as qwen2vl
    "janus": JanusChatTemplate,
    "llama": LlamaPretrainTemplate,
    "qwen3moe": Qwen3MoeChatTemplate,
    "seed_oss": SeedOssPretrainTemplate,
}


def build_multimodal_chat_template(template_name: str, tokenizer: AutoTokenizer, **kwargs) -> "ChatTemplate":
    if template_name not in TEMPLATES:
        raise ValueError(f"Unknown chat template: {template_name}")

    return TEMPLATES[template_name](tokenizer, **kwargs)
