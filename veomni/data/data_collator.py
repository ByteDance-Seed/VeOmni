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
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from transformers.modeling_outputs import ModelOutput

from ..distributed.parallel_state import get_parallel_state
from ..distributed.sequence_parallel import gather_outputs
from ..utils import logging
from ..utils.constants import IGNORE_INDEX, MODALITY
from ..utils.seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids, valid_seqlens_from_cu_seqlens


logger = logging.get_logger(__name__)


def add_flash_attention_kwargs_from_position_ids(
    batch: Dict[str, "torch.Tensor"],
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Calculate and add Flash Attention kwargs (cu_seq_lens and max_length) from position_ids.

    Pass down already computed cu_seq_lens and max_length as the HF transformers
    FlashAttentionKwargs naming so that it can be used without recomputation every layer.
    HF model code would handle the pass down of those kwargs for us.
    Note that the recomputation would cause host->device sync which hurts performance and
    stability due to CPU instability.

    Args:
        batch: The batch dictionary containing position_ids. Will be modified in-place to add
               cu_seq_lens_q, cu_seq_lens_k, max_length_q, and max_length_k.

    Returns:
        Tuple of (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k) for additional use.
    """
    position_ids = batch["position_ids"]
    if position_ids.dim() == 3:  # bs, dim, seq_len
        position_ids = position_ids[:, 0, :]
    (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(position_ids)

    batch["cu_seq_lens_q"] = cu_seq_lens_q
    batch["cu_seq_lens_k"] = cu_seq_lens_k
    batch["max_length_q"] = max_length_q
    batch["max_length_k"] = max_length_k

    return cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k


@dataclass
class DataCollateInfo:
    pack_dim: int = field(
        default=0,
        metadata={"help": "Dim to pack in batch. Default is 0. If -1, pack in last dim and unsqueeze(0)"},
    )
    sp_slice: bool = field(
        default=False,
        metadata={"help": "Whether to sp slice in batch. Default is False"},
    )
    sp_pad_value: int = field(
        default=None,
        metadata={"help": "sp_pad value of a sequence in batch. Not pad if None. Default is None"},
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


# pack_dim, sp_slice, sp_pad_value, sp_pad_scale
DEFAULT_DATA_COLLATE_INFO: Dict[str, DataCollateInfo] = {
    "input_ids": DataCollateInfo(-1, True, 0, 1),
    "labels": DataCollateInfo(-1, True, IGNORE_INDEX, 1),
    "attention_mask": DataCollateInfo(-1, False, 1, 1),
    "position_ids": DataCollateInfo(-1, False, 0, 1),
    "pixel_values": DataCollateInfo(0, True, 0, 4),
    "pixel_values_videos": DataCollateInfo(0, True, 0, 4),
    "image_mask": DataCollateInfo(-1, False, 0, 1),
    "video_mask": DataCollateInfo(-1, False, 0, 1),
    "image_grid_hw": DataCollateInfo(0, False, None, None),
    "image_grid_thw": DataCollateInfo(0, False, None, None),
    "video_grid_thw": DataCollateInfo(0, False, None, None),
}


@dataclass
class DataCollator(ABC):
    """
    Used in dataloader as a collate_fn.
    """

    @abstractmethod
    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        """
        Converts a list of features to batched tensor dict.
        """
        ...


@dataclass
class NoopDataCollator(DataCollator):
    """
    Data collator with no operation, used when collating in preforward.
    """

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> List[Dict[str, "torch.Tensor"]]:
        return features


@dataclass
class UnpackDataCollator(DataCollator):
    """
    Data collator to unpack examples, used in dynamic batch dataloader.
    """

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        return features[0]


@dataclass
class MakeMicroBatchCollator(DataCollator):
    """
    Data collator to build micro batches, used in mapping dataloader.
    """

    num_micro_batch: int
    internal_data_collator: "DataCollator"

    def __call__(self, features: Sequence[Tuple[Dict[str, "torch.Tensor"]]]) -> List[Dict[str, "torch.Tensor"]]:
        micro_batch_size = len(features) // self.num_micro_batch
        for i in range(len(features)):
            features[i] = features[i][0]  # 1-to-N inverse transform

        micro_batches = []
        for i in range(0, len(features), micro_batch_size):
            micro_batches.append(self.internal_data_collator(features[i : i + micro_batch_size]))

        return micro_batches


@dataclass
class PrecomputePositionIDsCollator(DataCollator):
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        for feature in features:
            if "position_ids" not in feature:
                # default position_ids is 0 ~ seq_len - 1 for text models
                feature["position_ids"] = torch.arange(feature["input_ids"].size(-1), dtype=torch.int64)
        return features


@dataclass
class PackingCollator(DataCollator):
    collate_infos: Dict[str, DataCollateInfo] = field(default_factory=lambda: DEFAULT_DATA_COLLATE_INFO.copy())
    pad_to_length: int = False
    seq_classification: bool = (
        False  # whether the training task is sequence classification, if true, do not mask boundary labels
    )

    def __post_init__(self):
        self.sp_enabled = get_parallel_state().sp_enabled

    def pad_feature_to_length(
        self,
        feature: Union[torch.Tensor, List[torch.Tensor]],
        dim: int = -1,
        pad_value: int = 0,
        pad_size: int = 0,
    ) -> torch.Tensor:
        pad_shape = list(feature.shape)
        pad_shape[dim] = pad_size
        pad = torch.full(pad_shape, fill_value=pad_value, dtype=feature.dtype, device=feature.device)
        return torch.cat((feature, pad), dim=dim)

    def pad_batch_to_length(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        seq_len = batch["input_ids"].shape[-1]
        assert seq_len <= self.pad_to_length, "pad_to_length must be >= packed sequence length."

        pad_len = self.pad_to_length - seq_len
        if pad_len == 0:
            return batch

        keys_to_pad = ["input_ids", "attention_mask", "labels", "position_ids"]
        for key in keys_to_pad:
            if key in batch:
                batch[key] = self.pad_feature_to_length(
                    batch[key],
                    dim=self.collate_infos[key].pack_dim,
                    pad_value=self.collate_infos[key].sp_pad_value,
                    pad_size=pad_len,
                )
        return batch

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

                # first token of packed sequence must be IGNORE_INDEX
                if key == "labels" and not self.seq_classification:
                    for i in range(1, len(batch[key])):
                        batch[key][i][0] = IGNORE_INDEX

                batch[key] = torch.cat(batch[key], dim=pack_dim)
                if pack_dim == -1:
                    batch[key] = batch[key].unsqueeze(0)

        if self.pad_to_length:
            batch = self.pad_batch_to_length(batch)

        if not self.sp_enabled:
            add_flash_attention_kwargs_from_position_ids(batch)
        return batch


@dataclass
class SequenceParallelCollator(DataCollator):
    collate_infos: Dict[str, DataCollateInfo] = field(default_factory=lambda: DEFAULT_DATA_COLLATE_INFO.copy())
    seq_classification: bool = (
        False  # whether the training task is sequence classification, if true, do not shift labels
    )

    def __post_init__(self):
        self.sp_size = get_parallel_state().sp_size
        self.sp_rank = get_parallel_state().sp_rank

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
        if not self.seq_classification:
            # shift labels
            labels = batch["labels"][..., 1:].contiguous()
            labels = F.pad(labels, (0, 1), "constant", IGNORE_INDEX)
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

            if sp_slice and key != "position_ids":  # position_ids should be sp sliced after precompute fa kwargs
                # sp slice
                batch[key] = self.sp_slice(key, batch[key], dim=pack_dim)

        add_flash_attention_kwargs_from_position_ids(batch)

        batch["position_ids"] = self.sp_slice(
            "position_ids", batch["position_ids"], dim=self.collate_infos["position_ids"].pack_dim
        )

        return batch


@dataclass
class MainCollator(DataCollator):
    data_collate_info: Dict[str, Union[DataCollateInfo, tuple, Dict]] = field(default_factory=lambda: {})
    pad_to_length: bool = False
    seq_classification: bool = False

    """
    Data collator pipeline with a unified collate info.

    Args:
        data_collate_info:
            User config to override the default collate info.
        pad_to_length:
            Whether to pad sequence to a fixed length. Default is False.
        seq_classification:
            If True, sequence classification task. Default is False.
    """

    def __post_init__(self):
        self.preforward_pipeline = []
        self.collate_infos: Dict[str, DataCollateInfo] = {}

        full_info = DEFAULT_DATA_COLLATE_INFO.copy()
        full_info.update(self.data_collate_info)

        for name, params in full_info.items():
            if isinstance(params, DataCollateInfo):
                self.collate_infos[name] = params
            elif isinstance(params, dict):
                self.collate_infos[name] = DataCollateInfo(**params)
            elif isinstance(params, tuple):
                self.collate_infos[name] = DataCollateInfo(*params)

        """attention_mask always pad 1
        VeOmni sp slice `input_ids` & `labels` while keeps the full sequence of `attention_mask`. This leads to wrong behavior of `create_causal_mask` in transformers.
        `create_causal_mask` will slice the `attention_mask` to `attention_mask[-len(input_ids):]`.
        refer to https://github.com/huggingface/transformers/blob/bdc85cb85c8772d37aa29ce447860b44d7fad6ef/src/transformers/masking_utils.py#L770
        So VeOmni make sure attention_mask is all_ones when using flash_attn, and precalculate the position_ids & cu_seqlens & max_seqlens.
        """
        assert self.collate_infos["attention_mask"].sp_pad_value == 1

        self.preforward_pipeline.append(PrecomputePositionIDsCollator())
        self.preforward_pipeline.append(
            PackingCollator(
                collate_infos=self.collate_infos,
                pad_to_length=self.pad_to_length,
                seq_classification=self.seq_classification,
            )
        )
        if get_parallel_state().sp_enabled:
            self.preforward_pipeline.append(
                SequenceParallelCollator(collate_infos=self.collate_infos, seq_classification=self.seq_classification)
            )
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
        log_str += "Main Collate Configuration\n"
        log_str += "-" * (25 + 18 * len(fields)) + "\n"

        log_str += row_format.format(*header) + "\n"
        log_str += "-" * (25 + 18 * len(fields)) + "\n"

        for name, info in self.collate_infos.items():
            row_data = [name] + [str(getattr(info, f)) for f in fields]
            log_str += row_format.format(*row_data) + "\n"

        log_str += "=" * (25 + 18 * len(fields)) + "\n"
        return log_str


@dataclass
class PostCollator(DataCollator):
    def __init__(self):
        self.postforward_pipeline = []
        self.compute_seqlens_func = SeqlensComputePostCollator()
        self.postforward_pipeline.append(PackingPostCollator())

    def __call__(self, outputs: ModelOutput, micro_batch: Dict[str, torch.Tensor]):
        seq_lens = self.compute_seqlens_func(micro_batch)
        for postforward_func in self.postforward_pipeline:
            outputs = postforward_func(outputs, seq_lens)
        return outputs


@dataclass
class SeqlensComputePostCollator(DataCollator):
    def __call__(self, micro_batch: Dict[str, torch.Tensor]):
        seq_lens = valid_seqlens_from_cu_seqlens(micro_batch["cu_seq_lens_q"]).tolist()
        return seq_lens


@dataclass
class PackingPostCollator(DataCollator):
    def __call__(self, outputs: ModelOutput, seq_lens):
        logits = outputs.logits
        if get_parallel_state().sp_enabled:
            logits = gather_outputs(logits, gather_dim=0, group=get_parallel_state().sp_group)
            logits = logits[: sum(seq_lens)]  # remove sp padding
        logits_list = logits.split(seq_lens, dim=0)
        outputs.logits = logits_list
        return outputs
