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
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_collate

from ..data.constants import IGNORE_INDEX
from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from ..utils.seqlen_pos_transform_utils import pos2culen, prepare_fa_kwargs_from_position_ids


logger = logging.get_logger(__name__)

MODALITY = ["image", "video", "audio"]


@dataclass
class DataCollateInfo:
    pack_dim: int = field(
        default=0,
        metadata={"help": "Dim to pack in batch. Default is 0. If -1, pack in last dim and unsqueeze(0)"},
    )
    pad_value: int = field(
        default=None,
        metadata={"help": "Pad value of a sequence in batch. concat instead of pad if None. Default is None"},
    )
    sp_slice: bool = field(
        default=False,
        metadata={"help": "Whether to sp slice in batch. Default is False"},
    )
    sp_pad_value: int = field(
        default=None,
        metadata={"help": "sp_pad value of a sequence in batch. concat instead of pad if None. Default is None"},
    )
    sp_pad_scale: int = field(
        default=1,
        metadata={"help": "sp_pad scale of a sequence in batch. Default is 1"},
    )

    def __post_init__(self):
        assert self.pack_dim is not None, "pack_dim must be specified"
        if self.sp_slice:
            assert self.sp_pad_value is not None and self.sp_pad_scale is not None, (
                "sp_pad_value and sp_pad_scale must be specified when sp_slice is True"
            )

        assert (self.sp_pad_value is None) == (self.sp_pad_scale is None), (
            "sp_pad_value and sp_pad_scale must be specified together or None"
        )


class Preforward:
    # pack_dim , pad_value, sp_slice, sp_pad_value, sp_pad_scale
    default_info = {
        "input_ids": (-1, 0, True, 0, 1),
        "labels": (-1, IGNORE_INDEX, True, IGNORE_INDEX, 1),
        "attention_mask": (-1, 1, False, 1, 1),
        "position_ids": (-1, 0, False, 0, 1),
        "pixel_values": (0, None, True, 0, 4),
        "pixel_values_videos": (0, None, True, 0, 4),
        "image_mask": (-1, 0, False, 0, 1),
        "video_mask": (-1, 0, False, 0, 1),
        "audio_mask": (-1, 0, False, 0, 1),
        "image_grid_hw": (0, None, False, None, None),
        "image_grid_thw": (0, None, False, None, None),
        "video_grid_thw": (0, None, False, None, None),
    }

    def __init__(
        self,
        rmpad_with_pos_ids: bool = False,
        data_collate_info: Dict[str, Union[DataCollateInfo, tuple, Dict]] = {},
        attn_implementation: str = "flash_attention_2",
    ) -> None:
        self.preforward_pipeline = []
        self.rmpad_with_pos_ids = rmpad_with_pos_ids
        self.collate_infos: Dict[str, DataCollateInfo] = {}

        full_info = self.default_info.copy()
        full_info.update(data_collate_info)

        for name, params in full_info.items():
            if isinstance(params, DataCollateInfo):
                self.collate_infos[name] = params
            elif isinstance(params, dict):
                self.collate_infos[name] = DataCollateInfo(**params)
            elif isinstance(params, tuple):
                self.collate_infos[name] = DataCollateInfo(*params)

        """attention_mask always pad 1 except when `attn_implementation=eager`
        VeOmni sp slice `input_ids` & `labels` while keeps the full sequence of `attention_mask`. This leads to wrong behavior of `create_causal_mask` in transformers.
        `create_causal_mask` will slice the `attention_mask` to `attention_mask[-len(input_ids):]`.
        refer to https://github.com/huggingface/transformers/blob/bdc85cb85c8772d37aa29ce447860b44d7fad6ef/src/transformers/masking_utils.py#L770
        So VeOmni make sure attention_mask is all_ones when using flash_attn, and precalculate the position_ids & cu_seqlens & max_seqlens.
        As eager attention not supports sp, we pad `attention_mask` with 0.
        """
        if attn_implementation == "eager":
            self.collate_infos["attention_mask"].pad_value = 0
        assert self.collate_infos["position_ids"].sp_slice is False, (
            "position_ids should be sp sliced after precompute fa kwargs"
        )

        self.preforward_pipeline.append(PrecomputePositionIDs())

        if self.rmpad_with_pos_ids:
            self.preforward_pipeline.append(PackingPreforward(self.collate_infos))
        else:
            self.preforward_pipeline.append(PaddingPreforward(self.collate_infos))
        if get_parallel_state().sp_enabled:
            self.preforward_pipeline.append(SequenceParallelPreforward(self.collate_infos))

        if attn_implementation == "flash_attention_2" or attn_implementation == "flash_attention_3":
            self.preforward_pipeline.append(PrecomputeFlashAttenKwargs())

        logger.info_rank0(self.log_collate_infos())

    def __call__(self, micro_batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        for preforward_func in self.preforward_pipeline:
            micro_batch = preforward_func(micro_batch)
        return micro_batch

    def log_collate_infos(self) -> None:
        sample_info = next(iter(self.collate_infos.values()))
        fields = list(asdict(sample_info).keys())

        header = ["name"] + fields

        row_format = "{:<25}" + "{:<18}" * len(fields)

        log_str = ""
        log_str += "\n" + "=" * (25 + 18 * len(fields)) + "\n"
        log_str += "Sequence Parallel Collate Configuration\n"
        log_str += "-" * (25 + 18 * len(fields)) + "\n"

        log_str += row_format.format(*header) + "\n"
        log_str += "-" * (25 + 18 * len(fields)) + "\n"

        for name, info in self.collate_infos.items():
            row_data = [name] + [str(getattr(info, f)) for f in fields]
            log_str += row_format.format(*row_data) + "\n"

        log_str += "=" * (25 + 18 * len(fields)) + "\n"
        return log_str


class PrecomputePositionIDs:
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        for feature in features:
            if "position_ids" not in feature:
                # default position_ids is 0 ~ seq_len - 1 for text models
                feature["position_ids"] = torch.arange(feature["input_ids"].size(-1), dtype=torch.int64)
        return features


class PrecomputeFlashAttenKwargs:
    def __call__(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        position_ids = features["position_ids"]
        if position_ids.dim() == 3:  # bs, dim, seq_len
            position_ids = position_ids[:, 0, :]
        (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(
            position_ids.contiguous()
        )
        features["cu_seq_lens_q"] = cu_seq_lens_q
        features["cu_seq_lens_k"] = cu_seq_lens_k
        features["max_length_q"] = max_length_q
        features["max_length_k"] = max_length_k
        return features


class PackingPreforward:
    def __init__(self, collate_infos: Dict[str, DataCollateInfo]) -> None:
        self.collate_infos = collate_infos

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = defaultdict(list)
        for feature in features:
            for key in feature.keys():
                batch[key].append(feature[key])

        for key in batch.keys():
            collate_info: DataCollateInfo = self.collate_infos.get(key, None)
            if collate_info is None:
                try:
                    if key.split("_")[0] in MODALITY:
                        batch[key] = torch.cat(batch[key], dim=0)
                    else:
                        batch[key] = default_collate(batch[key])
                except Exception:
                    # use List of tensor, for example: num, height, width, c in different resolution
                    pass
            else:
                pack_dim = collate_info.pack_dim
                batch[key] = torch.cat(batch[key], dim=pack_dim)
                if pack_dim == -1:
                    batch[key] = batch[key].unsqueeze(0)

        return batch


class PaddingPreforward:
    def __init__(self, collate_infos: Dict[str, DataCollateInfo]) -> None:
        self.collate_infos = collate_infos

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = defaultdict(list)
        for feature in features:
            for key in feature.keys():
                batch[key].append(feature[key])

        for key in batch.keys():
            collate_info: DataCollateInfo = self.collate_infos.get(key, None)

            if collate_info is None:
                try:
                    if key.split("_")[0] in MODALITY:
                        batch[key] = torch.cat(batch[key], dim=0)
                    else:
                        batch[key] = default_collate(batch[key])
                except Exception:
                    # use List of tensor, for example: num, height, width, c in different resolution
                    pass
            else:
                if collate_info.pad_value is None:
                    # concat instead of pad
                    pack_dim = collate_info.pack_dim
                    batch[key] = torch.cat(batch[key], dim=pack_dim)
                else:
                    pad_list = batch[key]
                    pad_value = collate_info.pad_value

                    """ For multimodal 1d/3d position_ids:
                    1. List[(dim, length)] -> List[(length, dim)]
                    2. Pad multimodal position_ids to max length -> torch.tensor (bs, max_length, dim)
                    3. transpose -> torch.tensor (bs, dim, max_length)
                    Others: List[(length)]
                    """
                    if key == "position_ids" and len(batch["position_ids"][0].shape) == 2:
                        pad_list = [item.transpose(0, 1) for item in batch[key]]

                    batch[key] = pad_sequence(pad_list, batch_first=True, padding_value=pad_value)

                    if key == "position_ids" and len(batch[key][0].shape) == 2:
                        batch[key] = batch[key].transpose(1, 2)

        return batch


class SequenceParallelPreforward:
    def __init__(self, collate_infos: Dict[str, DataCollateInfo]):
        self.sp_size = get_parallel_state().sp_size
        self.sp_rank = get_parallel_state().sp_rank

        self.collate_infos = collate_infos

    def sp_slice(self, key: str, feature: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if isinstance(feature, list):
            assert dim == 0, f"Only support dim=0 for {key} as it is a List"
            seq_length = len(feature)
            sp_chunk_size = seq_length // self.sp_size
            return feature[self.sp_rank * sp_chunk_size : (self.sp_rank + 1) * sp_chunk_size]
        else:
            seq_length = feature.size(dim)
            sp_chunk_size = seq_length // self.sp_size
            return feature.narrow(dim, self.sp_rank * sp_chunk_size, sp_chunk_size)

    def sp_padding(
        self,
        key: str,
        feature: Union[torch.Tensor, List[torch.Tensor]],
        dim: int = -1,
        pad_value: int = 0,
        pad_scale: int = 1,
    ) -> torch.Tensor:
        if isinstance(feature, List):
            assert dim == 0, f"Only support dim=0 for {key} as {key} is a List of Tensor"
            seq_length = len(feature)
        else:
            seq_length = feature.size(dim)

        scale_sp_size = self.sp_size * pad_scale
        sp_chunk_size = (seq_length + scale_sp_size - 1) // scale_sp_size
        pad_size = sp_chunk_size * scale_sp_size - seq_length
        if pad_size == 0:
            return feature

        if isinstance(feature, List):
            # if feature is uncatable, pad pad_size num feature[-1] to the List
            feature += [feature[-1]] * pad_size
            return feature
        else:
            pad_shape = list(feature.shape)
            pad_shape[dim] = pad_size
            pad = torch.full(pad_shape, fill_value=pad_value, dtype=feature.dtype, device=feature.device)
            return torch.cat((feature, pad), dim=dim)

    def __call__(self, batch: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # shift labels
        labels = batch["labels"][..., 1:].contiguous()
        labels = F.pad(labels, (0, 1), "constant", IGNORE_INDEX)

        cu_seqlens = pos2culen(batch["position_ids"])

        if labels.size(0) != 1:  # padding
            bs = labels.size(0)
            labels = labels.view(-1)
            labels[cu_seqlens[:-1] - 1] = IGNORE_INDEX
            labels = labels.view(bs, -1)  # align shape with input_ids to align sp_pad & sp_slice
        else:
            labels[:, cu_seqlens[1:-1] - 1] = IGNORE_INDEX

        batch["labels"] = labels

        for key in batch.keys():
            collate_info: DataCollateInfo = self.collate_infos.get(key, None)
            if collate_info is None:
                continue
            pack_dim = collate_info.pack_dim
            sp_slice = collate_info.sp_slice
            sp_pad_value = collate_info.sp_pad_value
            sp_pad_scale = collate_info.sp_pad_scale
            if sp_pad_value is not None:
                # sp padding
                batch[key] = self.sp_padding(
                    key,
                    batch[key],
                    dim=pack_dim,
                    pad_value=sp_pad_value,
                    pad_scale=sp_pad_scale,
                )

            if sp_slice:
                # sp slice
                batch[key] = self.sp_slice(key, batch[key], dim=pack_dim)

        return batch
