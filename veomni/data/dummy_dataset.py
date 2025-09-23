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
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from ..utils import logging


logger = logging.get_logger(__name__)


class DummyTextDataset(Dataset):
    def __init__(self, size: int, seq_length: int):
        """
        Args:
            size (int): Nums of datasets
            seq_length (int, optional): seq_length
        """
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = 32768

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        input_ids = torch.randint(low=0, high=self.vocab_size, size=(self.seq_length,))
        attention_mask = torch.ones((self.seq_length,), dtype=torch.long)
        labels = input_ids.clone()
        return [{"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}]


class DummyQwenVLDataset(Dataset):
    def __init__(self, size: int, seq_length: int):
        """
        Args:
            size (int): Nums of datasets
            seq_length (int, optional): seq_length
        """
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = 32768

        image_token_num = 81
        image_t = 2

        self.text_seqlen = seq_length // 4
        video_seq_length = self.seq_length - self.text_seqlen - image_t * image_token_num
        video_t = video_seq_length // image_token_num

        self.image_size = [324 * image_t, 1176]
        self.image_grid_thw = torch.tensor([[1, 18, 18]] * image_t, dtype=torch.long)
        self.image_seqlen = image_t * image_token_num

        self.video_size = [324 * video_t, 1176]
        self.video_grid_thw = torch.tensor([[video_t, 18, 18]], dtype=torch.long)
        self.video_seqlen = video_t * image_token_num

        self.seq_length = self.text_seqlen + self.image_seqlen + self.video_seqlen
        mask = torch.zeros((self.seq_length,), dtype=torch.bool)
        self.image_mask = mask.clone()
        self.image_mask[: self.image_seqlen] = 1
        self.video_mask = mask.clone()
        self.video_mask[-self.video_seqlen :] = 1

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        input_ids = torch.randint(low=0, high=self.vocab_size, size=(self.seq_length,))
        attention_mask = torch.ones((self.seq_length,), dtype=torch.long)
        labels = input_ids.clone()
        position_ids = torch.arange(0, self.seq_length).unsqueeze(0).repeat(3, 1)
        pixel_values = torch.rand(self.image_size, dtype=torch.float32)
        pixel_values_videos = torch.rand(self.video_size, dtype=torch.float32)
        return [
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "position_ids": position_ids,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_mask": self.image_mask,
                "video_mask": self.video_mask,
                "image_grid_thw": self.image_grid_thw,
                "video_grid_thw": self.video_grid_thw,
            }
        ]


class DummyOmniDataset(Dataset):
    def __init__(self, size: int, seq_length: int):
        """
        Args:
            size (int): Nums of datasets
            seq_length (int, optional): seq_length
            dummy_data:
            [input_ids, input_image_token, input_audio_token, input_video_token, output_image_token]
        """
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = 32768

        input_image_token_num = 81
        input_image_t = 2
        self.input_image_size = [324 * input_image_t, 1176]
        self.input_image_grid_thw = torch.tensor([[1, 18, 18]] * input_image_t, dtype=torch.long)
        self.input_image_seq_length = input_image_t * input_image_token_num

        audio_token_num = 100
        audio_num = 2
        self.input_audio_size = [4 * audio_token_num * audio_num, 128]
        self.input_audio_feature_lengths = torch.tensor([4 * audio_token_num] * audio_num, dtype=torch.long)
        self.input_audio_seq_length = audio_num * audio_token_num

        output_image_token_num = 1024
        output_image_num = 1
        self.output_image_size = [output_image_num, 3, 256, 256]
        self.output_image_seq_length = output_image_num * output_image_token_num

        rest_seq_length = self.seq_length - (
            self.input_image_seq_length + self.input_audio_seq_length + self.output_image_seq_length
        )

        self.text_seq_length = rest_seq_length // 4
        self.video_seq_length = rest_seq_length - self.text_seq_length
        video_t = self.video_seq_length // input_image_token_num
        self.input_video_size = [324 * video_t, 1176]
        self.input_video_grid_thw = torch.tensor([[video_t, 18, 18]], dtype=torch.long)

        self.seq_length = (
            self.text_seq_length
            + self.input_image_seq_length
            + self.input_audio_seq_length
            + self.video_seq_length
            + self.output_image_seq_length
        )
        mask = torch.zeros((self.seq_length,), dtype=torch.bool)
        start_index = self.text_seq_length
        self.image_input_mask = mask.clone()
        self.image_input_mask[start_index : start_index + self.input_image_seq_length] = 1
        self.audio_input_mask = mask.clone()
        start_index += self.input_image_seq_length
        self.audio_input_mask[start_index : start_index + self.input_audio_seq_length] = 1
        self.video_input_mask = mask.clone()
        start_index += self.input_audio_seq_length
        self.video_input_mask[start_index : start_index + self.video_seq_length] = 1
        self.image_output_mask = mask.clone()
        start_index += self.video_seq_length
        self.image_output_mask[start_index : start_index + self.output_image_seq_length] = 1

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        input_ids = torch.randint(low=0, high=self.vocab_size, size=(self.seq_length,))
        attention_mask = torch.ones((self.seq_length,), dtype=torch.long)
        labels = input_ids.clone()
        position_ids = torch.arange(0, self.seq_length).unsqueeze(0).repeat(3, 1)
        image_input_features = torch.rand(self.input_image_size, dtype=torch.float32)
        audio_input_features = torch.rand(self.input_audio_size, dtype=torch.float32)
        video_input_features = torch.rand(self.input_video_size, dtype=torch.float32)
        image_output_features = torch.rand(self.output_image_size, dtype=torch.float32)
        return [
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "position_ids": position_ids,
                "image_input_features": image_input_features,
                "audio_input_features": audio_input_features,
                "video_input_features": video_input_features,
                "image_output_features": image_output_features,
                "image_input_mask": self.image_input_mask,
                "audio_input_mask": self.audio_input_mask,
                "video_input_mask": self.video_input_mask,
                "image_output_mask": self.image_output_mask,
                "image_input_grid_thw": self.input_image_grid_thw,
                "video_input_grid_thw": self.input_video_grid_thw,
                "audio_input_feature_lengths": self.input_audio_feature_lengths,
            }
        ]


def build_dummy_dataset(task_type: str, size: int, max_seq_len: int) -> "Dataset":
    if task_type == "text":
        return DummyTextDataset(size=size, seq_length=max_seq_len)
    elif task_type == "qwenvl":
        return DummyQwenVLDataset(size=size, seq_length=max_seq_len)
    elif task_type == "omni":
        return DummyOmniDataset(size=size, seq_length=max_seq_len)
    else:
        raise ValueError(f"Dummy dataset type ({task_type}) is not supported.")
