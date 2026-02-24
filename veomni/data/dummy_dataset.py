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
from ..utils.constants import IGNORE_INDEX


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
        self.vocab_size = 1024

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        input_ids = torch.randint(low=0, high=self.vocab_size, size=(self.seq_length,))
        attention_mask = torch.ones((self.seq_length,), dtype=torch.long)
        position_ids = torch.arange(0, self.seq_length)
        labels = input_ids.clone()
        labels[0] = IGNORE_INDEX
        return [
            {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "position_ids": position_ids}
        ]


class DummyQwenVLDataset(Dataset):
    def __init__(
        self, size: int, seq_length: int, patch_size: int = 14, temporal_patch_size: int = 2, merge_size: int = 2
    ):
        """
        Args:
            size (int): Nums of datasets
            seq_length (int, optional): seq_length
        """
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = 1024

        image_t = 2
        video_t = 10
        h, w = 4, 4
        channel_size = 3

        self.image_size = [image_t * h * w, patch_size * patch_size * temporal_patch_size * channel_size]
        self.image_grid_thw = torch.tensor([[1, h, w]] * image_t, dtype=torch.long)
        self.image_seqlen = h * w // (merge_size**2) * image_t

        self.video_size = [video_t * h * w, patch_size * patch_size * temporal_patch_size * channel_size]
        self.video_grid_thw = torch.tensor([[video_t, h, w]], dtype=torch.long)
        self.video_seqlen = h * w // (merge_size**2) * video_t

        self.text_seqlen = self.seq_length - self.image_seqlen - self.video_seqlen

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
        labels[0] = IGNORE_INDEX
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


class DummyQwenOmniDataset(Dataset):
    def __init__(
        self, size: int, seq_length: int, patch_size: int = 14, temporal_patch_size: int = 2, merge_size: int = 2
    ):
        """
        Args:
            size (int): Nums of datasets
            seq_length (int, optional): seq_length
            dummy_data:
            [input_ids, input_image_token, input_audio_token, input_video_token, output_image_token]
        """
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = 1024

        h, w = 4, 4
        image_t = 2
        video_t = 10
        channel_size = 3

        self.image_size = [image_t * h * w, patch_size * patch_size * temporal_patch_size * channel_size]
        self.image_grid_thw = torch.tensor([[1, h, w]] * image_t, dtype=torch.long)
        self.image_seqlen = h * w // (merge_size**2) * image_t

        audio_token_num = 100
        audio_num = 2
        self.audio_size = [4 * audio_token_num * audio_num, 128]
        self.audio_feature_lengths = torch.tensor([4 * audio_token_num] * audio_num, dtype=torch.long)
        self.audio_seq_length = audio_num * audio_token_num
        self.feature_attention_mask = torch.ones((audio_num, 4 * audio_token_num), dtype=torch.long)

        rest_seq_length = self.seq_length - (self.image_seqlen + self.audio_seq_length)

        self.video_size = [video_t * h * w, patch_size * patch_size * temporal_patch_size * channel_size]
        self.video_grid_thw = torch.tensor([[video_t, h, w]], dtype=torch.long)
        self.video_seqlen = h * w // (merge_size**2) * video_t

        self.text_seqlen = rest_seq_length - self.video_seqlen

        self.seq_length = self.text_seqlen + self.image_seqlen + self.audio_seq_length + self.video_seqlen
        mask = torch.zeros((self.seq_length,), dtype=torch.bool)
        start_index = self.text_seqlen
        self.image_mask = mask.clone()
        self.image_mask[start_index : start_index + self.image_seqlen] = 1
        self.audio_mask = mask.clone()
        start_index += self.image_seqlen
        self.audio_mask[start_index : start_index + self.audio_seq_length] = 1
        self.video_mask = mask.clone()
        start_index += self.audio_seq_length
        self.video_mask[start_index : start_index + self.video_seqlen] = 1

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        input_ids = torch.randint(low=0, high=self.vocab_size, size=(self.seq_length,))
        attention_mask = torch.ones((self.seq_length,), dtype=torch.long)
        labels = input_ids.clone()
        labels[0] = IGNORE_INDEX
        position_ids = torch.arange(0, self.seq_length).unsqueeze(0).repeat(3, 1)
        image_features = torch.rand(self.image_size, dtype=torch.float32)
        audio_features = torch.rand(self.audio_size, dtype=torch.float32)
        video_features = torch.rand(self.video_size, dtype=torch.float32)
        return [
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "position_ids": position_ids,
                "pixel_values": image_features,
                "input_features": audio_features,
                "pixel_values_videos": video_features,
                "image_mask": self.image_mask,
                "audio_mask": self.audio_mask,
                "video_mask": self.video_mask,
                "image_grid_thw": self.image_grid_thw,
                "video_grid_thw": self.video_grid_thw,
                "audio_feature_lengths": self.audio_feature_lengths,
                "feature_attention_mask": self.feature_attention_mask,
            }
        ]


class DummyQwen3OmniMoeDataset(DummyQwenOmniDataset):
    def __init__(
        self, size: int, seq_length: int, patch_size: int = 14, temporal_patch_size: int = 2, merge_size: int = 2
    ):
        super().__init__(size, seq_length, patch_size, temporal_patch_size, merge_size)
        self.audio_seq_length = self._get_feat_extract_output_lengths(self.audio_seq_length * 4)
        self.seq_length = self.text_seqlen + self.image_seqlen + self.audio_seq_length + self.video_seqlen
        mask = torch.zeros((self.seq_length,), dtype=torch.bool)
        start_index = self.text_seqlen
        self.image_mask = mask.clone()
        self.image_mask[start_index : start_index + self.image_seqlen] = 1
        self.audio_mask = mask.clone()
        start_index += self.image_seqlen
        self.audio_mask[start_index : start_index + self.audio_seq_length] = 1
        self.video_mask = mask.clone()
        start_index += self.audio_seq_length
        self.video_mask[start_index : start_index + self.video_seqlen] = 1

    def _get_feat_extract_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """

        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return output_lengths


class DummyUGDataset(Dataset):
    def __init__(
        self, size: int, seq_length: int, patch_size: int = 14, temporal_patch_size: int = 2, merge_size: int = 2
    ):
        """
        Args:
            size (int): Nums of datasets
            seq_length (int, optional): seq_length
            dummy_data:
            [input_ids, input_image_token, input_audio_token, input_video_token, output_image_token]
        """
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = 1024

        h, w = 18, 18
        image_t = 2
        video_t = 10
        channel_size = 3

        self.input_image_size = [image_t * h * w, patch_size * patch_size * temporal_patch_size * channel_size]
        self.input_image_grid_thw = torch.tensor([[1, h, w]] * image_t, dtype=torch.long)
        self.input_image_seq_length = image_t * h * w // (merge_size**2)

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

        self.input_video_size = [video_t * h * w, patch_size * patch_size * temporal_patch_size * channel_size]
        self.input_video_grid_thw = torch.tensor([[video_t, h, w]], dtype=torch.long)
        self.video_seq_length = video_t * h * w // (merge_size**2)

        self.text_seq_length = rest_seq_length - self.video_seq_length

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
    elif task_type == "qwen2vl":
        return DummyQwenVLDataset(size=size, seq_length=max_seq_len, patch_size=14)
    elif task_type == "qwen3vl":
        return DummyQwenVLDataset(size=size, seq_length=max_seq_len, patch_size=16)
    elif task_type == "qwen2omni":
        return DummyQwenOmniDataset(size=size, seq_length=max_seq_len, patch_size=14)
    elif task_type == "qwen3omni":
        return DummyQwen3OmniMoeDataset(size=size, seq_length=max_seq_len, patch_size=16)
    elif task_type == "ug":
        return DummyUGDataset(size=size, seq_length=max_seq_len)
    else:
        raise ValueError(f"Dummy dataset type ({task_type}) is not supported.")
