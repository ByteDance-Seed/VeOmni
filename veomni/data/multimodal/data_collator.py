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


from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_collate

from ...distributed.parallel_state import get_parallel_state
from ..constants import IGNORE_INDEX
from ..data_collator import DataCollator, pos2culen


@dataclass
class OmniSequenceShardCollator(DataCollator):
    """
    Data collator to chunk inputs according to sequence parallelism.
    """

    rmpad_with_pos_ids: bool = False

    # features to slice sequence dimension
    sp_slice_features: Dict[str, int] = field(
        default_factory=lambda: {
            "input_ids": -1,
            "labels": -1,
            "pixel_values": 0,
            "pixel_values_videos": 0,
        },
        metadata={"help": "features to slice sequence dimension."},
    )

    # features to padding sequence dimension
    padding_features: Dict[str, int] = field(
        default_factory=lambda: {
            "input_ids": 0,
            "attention_mask": 0,
            "labels": IGNORE_INDEX,
            "pixel_values": 0,
            "pixel_values_videos": 0,
            "position_ids": 0,
            "image_mask": False,
            "video_mask": False,
            "audio_mask": False,
        },
        metadata={"help": "features to padding sequence dimension."},
    )

    # padding scale for padding features
    padding_scale: Dict[str, int] = field(
        default_factory=lambda: {"pixel_values": 4}, metadata={"help": "padding scale for padding features."}
    )

    def __post_init__(self):
        self.sp_size = get_parallel_state().sp_size
        self.sp_rank = get_parallel_state().sp_rank

        # if rmpad_with_pos_ids, attention_mask will be set to 1 for flash attention in transformers
        # refer to https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L641
        # `if not (attention_mask == 0.0).any()`, ``attention_mask`` will be set to None
        # so we need to set attention_mask to 1 for padding features

        if self.rmpad_with_pos_ids:
            self.padding_features["attention_mask"] = 1

    def sp_slice(self, feature: torch.Tensor, dim: int = -1) -> Dict[str, "torch.Tensor"]:
        seq_length = feature.size(dim)
        sp_chunk_size = (seq_length + self.sp_size - 1) // self.sp_size
        return feature.narrow(dim, self.sp_rank * sp_chunk_size, sp_chunk_size)

    def sp_padding(
        self, tensor: "torch.Tensor", dim: int = -1, pad_value: int = 0, pad_scale: int = 1
    ) -> "torch.Tensor":
        """
        Pads a tensor with pad_length to aligns tensor with sp size.
        """
        seq_length = tensor.size(dim)
        scale_sp_size = self.sp_size * pad_scale

        sp_chunk_size = (seq_length + scale_sp_size - 1) // scale_sp_size
        pad_size = sp_chunk_size * scale_sp_size - seq_length
        if pad_size == 0:
            return tensor

        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_size
        pad = torch.full(pad_shape, fill_value=pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat((tensor, pad), dim=dim)

    def __call__(self, batch: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        # shift labels
        labels = batch["labels"][..., 1:].contiguous()
        labels = F.pad(labels, (0, 1), "constant", IGNORE_INDEX)

        if "position_ids" in batch.keys():
            cu_seqlens = pos2culen(batch["position_ids"])
            labels[:, cu_seqlens[1:-1] - 1] = IGNORE_INDEX
        elif "cu_seqlens" in batch.keys():
            labels = labels.view(-1)
            labels[batch["cu_seqlens"][1:-1] - 1] = IGNORE_INDEX

        batch["labels"] = labels

        # padding to sp size
        for key in batch.keys():
            if key in self.padding_features.keys():
                batch[key] = self.sp_padding(
                    batch[key],
                    dim=self.sp_slice_features.get(key, -1),
                    pad_value=self.padding_features[key],
                    pad_scale=self.padding_scale.get(key, 1),
                )
        # sp slice
        for key in batch.keys():
            if key in self.sp_slice_features.keys():
                batch[key] = self.sp_slice(batch[key], dim=self.sp_slice_features[key])

        return batch


@dataclass
class OmniDataCollatorWithPadding(DataCollator):
    """
    Data collator to padding for omni dataset.
    Args:
        concat_features: features to concat in batch.
        padding_features: features to padding in batch, keys are feature names, values are padding values.
    Example:
        >>> from veomni.data import OmniDataCollatorWithPadding
        >>> collator = OmniDataCollatorWithPadding(
                concat_features={
                    "pixel_values": 0,
                    "pixel_values_videos": 0,
                    "image_grid_hw": 0,
                    "image_grid_thw": 0,
                },
                padding_features={
                    "input_ids": 0,
                    "attention_mask": 0,
                    "labels": IGNORE_INDEX,
                    "position_ids": 0,
                    "image_mask": False,
                }
        )
    """

    concat_features: Dict[str, int] = field(
        default_factory=lambda: {
            "pixel_values": 0,
            "pixel_values_videos": 0,
            "image_grid_hw": 0,
            "image_grid_thw": 0,
            "video_grid_thw": 0,
        },
        metadata={"help": "features to concat in batch."},
    )

    padding_features: Dict[str, int] = field(
        default_factory=lambda: {
            "input_ids": 0,
            "attention_mask": 0,
            "labels": IGNORE_INDEX,
            "position_ids": 0,
            "image_mask": False,
            "video_mask": False,
            "audio_mask": False,
        },
        metadata={"help": "features to padding in batch, keys are feature names, values are padding values."},
    )

    def __call__(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = defaultdict(list)
        for feature in features:
            for key in feature.keys():
                batch[key].append(feature[key])

        for key in batch.keys():
            if key in self.concat_features:
                batch[key] = torch.cat(batch[key], dim=self.concat_features[key])
            elif key in self.padding_features.keys():
                pad_list = batch[key]
                pad_value = self.padding_features.get(key, 0)
                if key == "position_ids" and len(batch[key][0].shape) == 2:
                    # For multimodal rope 2d/3d List[(dim, length)] -> List[(length, dim)]
                    # Others: List[(length)]
                    pad_list = [item.transpose(0, 1) for item in batch[key]]
                batch[key] = pad_sequence(pad_list, batch_first=True, padding_value=pad_value)
                if key == "position_ids" and len(batch[key][0].shape) == 2:
                    batch[key] = batch[key].transpose(1, 2)  # (bs, length, dim) -> (bs, dim, length)
            else:
                batch[key] = default_collate(batch[key])

        return batch


@dataclass
class OmniDataCollatorWithPacking(DataCollator):
    """
    Data collator to packing for omni dataset.
    Args:
        packing_features: features to packing in batch.
        concat_features: features to concat in batch.
    Example:
        >>> from veomni.data import OmniDataCollatorWithPacking
        >>> collator = OmniDataCollatorWithPacking(
                packing_features=[
                    "input_ids",
                    "attention_mask",
                    "labels",
                    "position_ids",
                    "image_input_mask",
                    "image_output_mask",
                ],
                concat_features=[
                    "image_input_features",
                    "image_input_grid_thw",
                    "image_output_features",
                    "image_output_grid_thw",
                ],
            )
    """

    packing_features: List = field(
        default_factory=lambda: [
            "input_ids",
            "attention_mask",
            "labels",
            "position_ids",
            "image_mask",
            "video_mask",
            "audio_mask",
        ],
        metadata={"help": "features to packing in batch."},
    )

    concat_features: List = field(
        default_factory=lambda: [
            "pixel_values",
            "pixel_values_videos",
            "image_grid_hw",
            "image_grid_thw",
            "video_grid_thw",
        ],
        metadata={"help": "features to concat in batch."},
    )

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        batch = {}
        keys = {key for feature in features for key in feature.keys()}
        for input_name in keys:
            if input_name in self.packing_features:
                batch[input_name] = torch.cat(
                    [feature[input_name] for feature in features if input_name in feature], dim=-1
                ).unsqueeze(0)
            elif input_name in self.concat_features:
                batch[input_name] = torch.cat(
                    [feature[input_name] for feature in features if input_name in feature], dim=0
                )
            else:
                batch[input_name] = default_collate(
                    [feature[input_name] for feature in features if input_name in feature]
                )

        return batch
