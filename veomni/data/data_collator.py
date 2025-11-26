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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_collate

from ..distributed.parallel_state import get_parallel_state
from ..utils.seqlen_pos_transform_utils import len2culen, pos2culen, prepare_fa_kwargs_from_position_ids
from .constants import IGNORE_INDEX


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
    (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(
        batch["position_ids"]
    )

    batch["cu_seq_lens_q"] = cu_seq_lens_q
    batch["cu_seq_lens_k"] = cu_seq_lens_k
    batch["max_length_q"] = max_length_q
    batch["max_length_k"] = max_length_k

    return cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k


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


class CollatePipeline:
    def __init__(self, data_collators: Optional[Union[Callable, List[Callable]]] = None):
        """
        Args:
            data_collators: a list of data collators or a single data collator
        """

        if not isinstance(data_collators, (list, tuple)):
            data_collators = [data_collators]
        self.data_collators = data_collators

    def __call__(self, batch: Sequence[Dict[str, Any]]):
        """
        process data batch through data collators.

        Args:
            batch: the original input data batch

        Returns:
            batch: the processed data batch

        """
        for data_collator in self.data_collators:
            batch = data_collator(batch)
        return batch


@dataclass
class DataCollatorWithPadding(DataCollator):
    """
    Data collator with padding.
    """

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        batch = defaultdict(list)

        # batching features
        for feature in features:
            for key in feature.keys():
                batch[key].append(feature[key])

        for key in batch.keys():
            # process padding features
            if key in ["input_ids", "attention_mask", "position_ids", "images_seq_mask"]:
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=0)
            elif key in ["labels", "labels_image"]:
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=IGNORE_INDEX)
            else:
                batch[key] = default_collate(batch[key])

        return batch


@dataclass
class DataCollatorWithPacking(DataCollator):
    """
    Data collator with packing.
    """

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        seqlens = torch.tensor([len(feature["input_ids"]) for feature in features], dtype=torch.long)
        batch = {"cu_seqlens": len2culen(seqlens)}
        for input_name in features[0].keys():
            if input_name in ("input_ids", "attention_mask", "labels"):
                batch[input_name] = torch.cat([feature[input_name] for feature in features])
            else:
                batch[input_name] = default_collate([feature[input_name] for feature in features])

        return batch


@dataclass
class DataCollatorWithPositionIDs(DataCollator):
    """
    Data collator with packing by position ids.
    """

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        batch = {}
        for input_name in features[0].keys():
            if input_name in ("input_ids", "attention_mask", "labels", "position_ids"):
                batch[input_name] = torch.cat([feature[input_name] for feature in features], dim=-1).unsqueeze(0)
            else:
                batch[input_name] = default_collate([feature[input_name] for feature in features])

        if "position_ids" not in batch:
            batch["position_ids"] = torch.cat(
                [torch.arange(len(feature["input_ids"])) for feature in features]
            ).unsqueeze(0)

        # cu_seq_lens_q should equal to cu_seq_lens_k and max_length_q should equal to max_length_k
        if not get_parallel_state().sp_enabled:
            # We only enter here to pass down cu_seqlens and max_length when sequence parallelism is not enabled.
            # When sp_enabled is True, position_ids will be padded later, so we calculate them after padding
            cu_seq_lens_q, _, _, _ = add_flash_attention_kwargs_from_position_ids(batch)
        else:
            # Still need cu_seq_lens_q for label masking even when sp_enabled
            (cu_seq_lens_q, _), (_, _) = prepare_fa_kwargs_from_position_ids(batch["position_ids"])

        if "labels" in batch:
            batch["labels"][:, cu_seq_lens_q[1:-1]] = IGNORE_INDEX

        return batch


@dataclass
class NoopDataCollator(DataCollator):
    """
    Data collator with no operation, used in dynamic batch dataloader at main process.
    """

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> List[Dict[str, "torch.Tensor"]]:
        return features


@dataclass
class UnpackDataCollator(DataCollator):
    """
    Data collator to unpack examples, used in dynamic batch dataloader at worker process.
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
class TextSequenceShardCollator(DataCollator):
    """
    Data collator to chunk inputs according to sequence parallelism.
    Args:
        rmpad: whether the samples is packing or not.
        rmpad_with_pos_ids: whether the samples is packing by position ids or not.
        pad_token_id: the id of the padding token.
    """

    rmpad: bool
    rmpad_with_pos_ids: bool
    pad_token_id: int = 0

    def __post_init__(self):
        self.sp_size = get_parallel_state().sp_size
        self.sp_rank = get_parallel_state().sp_rank

    def sp_slice(self, tensor: "torch.Tensor", dim: int = -1) -> "torch.Tensor":
        """
        Slices a tensor along the specified dimension for sequence parallelism.
        """
        seq_length = tensor.size(dim)
        sp_chunk_size = (seq_length + self.sp_size - 1) // self.sp_size
        return tensor.narrow(dim, self.sp_rank * sp_chunk_size, sp_chunk_size)

    def sp_padding(
        self, tensor: "torch.Tensor", dim: int = -1, pad_value: int = 0, pad_length: int = 0, sequential: bool = False
    ) -> "torch.Tensor":
        """
        Pads a tensor with pad_length to aligns tensor with sp size.
        """
        if pad_length == 0:
            return tensor

        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_length
        # For position_ids to create one single sequence for all padded tokens
        if sequential:
            # seq: [pad_length]
            seq = torch.arange(pad_length, device=tensor.device, dtype=tensor.dtype)

            # We want to broadcast seq along every dimension except `dim`.
            # view_shape: [1, 1, ..., pad_length(at dim), ..., 1]  (ndim entries)
            view_shape = [1] * tensor.ndim
            view_shape[dim] = pad_length

            # seq.view(view_shape): [1, 1, ..., pad_length, ..., 1]
            # expand to pad_shape:   [s0, s1, ..., pad_length, ..., s{n-1}]
            pad = seq.view(view_shape).expand(pad_shape)
        else:
            pad = torch.full(pad_shape, fill_value=pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat((tensor, pad), dim=dim)

    def __call__(self, batch: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        input_ids = batch.pop("input_ids")
        labels = batch.pop("labels")[..., 1:].contiguous()  # shift labels
        labels = F.pad(labels, (0, 1), "constant", IGNORE_INDEX)

        if self.rmpad_with_pos_ids:  # mask the last token of each sequence
            cu_seqlens = pos2culen(batch["position_ids"])
            labels[:, cu_seqlens[1:-1] - 1] = IGNORE_INDEX
        elif self.rmpad:
            labels = labels.view(-1)
            labels[batch["cu_seqlens"][1:-1] - 1] = IGNORE_INDEX
        else:
            if "position_ids" not in batch:  # we should calculate the position ids before chunking
                batch["position_ids"] = torch.arange(0, input_ids.size(-1)).unsqueeze(0)

        # sp padding
        seq_length = input_ids.size(-1)
        sp_chunk_size = (seq_length + self.sp_size - 1) // self.sp_size
        pad_length = sp_chunk_size * self.sp_size - seq_length

        input_ids = self.sp_padding(input_ids, dim=-1, pad_value=self.pad_token_id, pad_length=pad_length)
        labels = self.sp_padding(labels, dim=-1, pad_value=IGNORE_INDEX, pad_length=pad_length)

        if self.rmpad_with_pos_ids:
            batch["attention_mask"] = self.sp_padding(
                batch["attention_mask"], dim=-1, pad_value=1, pad_length=pad_length
            )
        else:
            batch["attention_mask"] = self.sp_padding(
                batch["attention_mask"], dim=-1, pad_value=0, pad_length=pad_length
            )

        if self.rmpad:
            if pad_length > 0:
                batch["cu_seqlens"] = F.pad(
                    batch["cu_seqlens"], (0, 1), "constant", batch["cu_seqlens"][-1].item() + pad_length
                )
        else:
            # For position_ids to create one single sequence for all padded tokens by pass sequential=True
            batch["position_ids"] = self.sp_padding(
                batch["position_ids"], dim=-1, pad_value=0, pad_length=pad_length, sequential=True
            )

        # sp slice
        batch["input_ids"] = self.sp_slice(input_ids, dim=-1)
        batch["labels"] = self.sp_slice(labels, dim=-1)

        # Calculate these info from position_ids here when SP_enable to use padded position_ids
        if not self.rmpad:
            add_flash_attention_kwargs_from_position_ids(batch)

        return batch
