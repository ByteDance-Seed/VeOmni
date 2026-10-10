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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from transformers.modeling_outputs import ModelOutput

from ..distributed.parallel_state import get_parallel_state
from ..distributed.sequence_parallel import gather_outputs
from ..utils import logging
from ..utils.constants import IGNORE_INDEX, MODALITY
from ..utils.seqlen_pos_transform_utils import (
    coalesce_tail_padding_cu_seqlens,
    prepare_fa_kwargs_from_position_ids,
    valid_seqlens_from_cu_seqlens,
)


# A model-provided hook that derives ``multimodal_metadata`` from a packed +
# SP-padded batch. It mirrors ``get_position_id_func``: a picklable callable
# (``partial`` over a module-level patchgen helper closed over config constants,
# never an ``nn.Module``) so it survives shipping to DataLoader workers.
# Signature: ``fn(batch: dict, sp_pad: dict[str, int]) -> None`` — mutates
# ``batch`` in place, writing ``batch["multimodal_metadata"]``.
MetadataCollateFunc = Callable[[Dict[str, Any], Dict[str, int]], None]


logger = logging.get_logger(__name__)

_LINEAR_ATTN_TAIL_PADDING_LENGTH = "_linear_attn_tail_padding_length"


def add_flash_attention_kwargs_from_position_ids(
    batch: Dict[str, "torch.Tensor"],
    linear_attn_tail_padding_length: int = 0,
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
               cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k, and
               linear_attn_cu_seq_lens_q. When ``linear_attn_tail_padding_length > 0``,
               also stores ``tail_padding_length`` (0-d ``torch.int32`` tensor) so
               downstream valid-seqlens helpers can strip the coalesced pad segment and
               distributed/PP stages preserve the metadata.
        linear_attn_tail_padding_length: Number of known padding tokens appended at the tail by
               collators (``pad_to_length`` / SP pad). ``position_ids == 0`` encodes each pad
               token as its own 1-token boundary; both FlashAttention and linear-attention
               cu-seqlens coalesce that tail into one segment. Ascend NPU
               ``npu_fusion_attention`` SIGSEGVs on the uncoalesced per-token form.

    Returns:
        Tuple of (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k) for additional use.
    """
    position_ids = batch["position_ids"]
    if position_ids.dim() == 3:  # bs, dim, seq_len
        position_ids = position_ids[:, 0, :]
    (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(position_ids)

    if linear_attn_tail_padding_length > 0:
        cu_seq_lens_q = coalesce_tail_padding_cu_seqlens(cu_seq_lens_q, linear_attn_tail_padding_length)
        cu_seq_lens_k = cu_seq_lens_q
        seqlens = cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]
        max_length_q = int(seqlens.max().item()) if seqlens.numel() else 0
        max_length_k = max_length_q
        # Keep as a 0-d tensor so PP / distributed stages preserve the metadata.
        batch["tail_padding_length"] = torch.tensor(
            linear_attn_tail_padding_length,
            dtype=torch.int32,
            device=cu_seq_lens_q.device,
        )

    batch["cu_seq_lens_q"] = cu_seq_lens_q
    batch["cu_seq_lens_k"] = cu_seq_lens_k
    batch["max_length_q"] = max_length_q
    batch["max_length_k"] = max_length_k
    batch["linear_attn_cu_seq_lens_q"] = cu_seq_lens_q

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
    "mtp_labels": DataCollateInfo(-1, True, IGNORE_INDEX, 1),
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
    # Model-provided hook (see ``MetadataCollateFunc``). ``None`` for text
    # models / pipelines without multimodal metadata — then this is a no-op.
    metadata_collate_func: Optional[MetadataCollateFunc] = None

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

        keys_to_pad = []
        for key in self.collate_infos.keys():
            if self.collate_infos[key].pack_dim == -1:
                keys_to_pad.append(key)

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

        linear_attn_tail_padding_length = 0
        if self.pad_to_length:
            input_ids_len_before = batch["input_ids"].shape[-1]
            batch = self.pad_batch_to_length(batch)
            linear_attn_tail_padding_length = max(0, batch["input_ids"].shape[-1] - input_ids_len_before)

        if not self.sp_enabled:
            add_flash_attention_kwargs_from_position_ids(batch, linear_attn_tail_padding_length)
            # No SP downstream → no sp-pad. Hand the packed batch to the
            # model-provided hook (if any), which derives ``multimodal_metadata``
            # from the packed ``*_grid_thw`` tensors using its own config. When
            # SP is enabled this is deferred to ``SequenceParallelCollator`` so
            # the hook sees the SP-padded batch + per-modality pad counts.
            if self.metadata_collate_func is not None:
                self.metadata_collate_func(batch, {"pixel_values": 0, "pixel_values_videos": 0})
        elif linear_attn_tail_padding_length:
            batch[_LINEAR_ATTN_TAIL_PADDING_LENGTH] = linear_attn_tail_padding_length
        return batch


@dataclass
class SequenceParallelCollator(DataCollator):
    collate_infos: Dict[str, DataCollateInfo] = field(default_factory=lambda: DEFAULT_DATA_COLLATE_INFO.copy())
    seq_classification: bool = (
        False  # whether the training task is sequence classification, if true, do not shift labels
    )
    # Model-provided hook (see ``MetadataCollateFunc``). ``None`` for text
    # models / pipelines without multimodal metadata — then this is a no-op.
    metadata_collate_func: Optional[MetadataCollateFunc] = None

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

        linear_attn_tail_padding_length = int(batch.pop(_LINEAR_ATTN_TAIL_PADDING_LENGTH, 0))

        # Track sp_pad sizes for pixel_values{,_videos} so the ViT metadata
        # ``cu_seqlens`` can be extended with the sp-pad tail entry (mirrors
        # how the text-side cu_seq_lens picks up sp-pad via the position_ids==0
        # convention in ``add_flash_attention_kwargs_from_position_ids``).
        vit_sp_pad: Dict[str, int] = {"pixel_values": 0, "pixel_values_videos": 0}

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
                pre_pad_len = len(batch[key]) if isinstance(batch[key], list) else batch[key].size(pack_dim)
                batch[key] = self.sp_padding(
                    key,
                    batch[key],
                    dim=pack_dim,
                    pad_value=sp_pad_value,
                    pad_scale=sp_pad_scale,
                )
                post_pad_len = len(batch[key]) if isinstance(batch[key], list) else batch[key].size(pack_dim)
                if key in vit_sp_pad:
                    vit_sp_pad[key] = post_pad_len - pre_pad_len
                if key == "position_ids":
                    linear_attn_tail_padding_length += post_pad_len - pre_pad_len

            if sp_slice and key != "position_ids":  # position_ids should be sp sliced after precompute fa kwargs
                # sp slice
                batch[key] = self.sp_slice(key, batch[key], dim=pack_dim)

        add_flash_attention_kwargs_from_position_ids(batch, linear_attn_tail_padding_length)

        batch["position_ids"] = self.sp_slice(
            "position_ids", batch["position_ids"], dim=self.collate_infos["position_ids"].pack_dim
        )

        # Hand the SP-padded batch + per-modality sp-pad patch counts to the
        # model-provided hook, which derives ``multimodal_metadata`` (cu_seqlens,
        # window cu_seqlens, …) using its own config — including the sp-pad tail.
        # No-op for text models / third-party pipelines without a hook.
        if self.metadata_collate_func is not None:
            self.metadata_collate_func(batch, vit_sp_pad)

        return batch


@dataclass
class MainCollator(DataCollator):
    data_collate_info: Dict[str, Union[DataCollateInfo, tuple, Dict]] = field(default_factory=lambda: {})
    pad_to_length: bool = False
    seq_classification: bool = False
    metadata_collate_func: Optional[MetadataCollateFunc] = None

    """
    Data collator pipeline with a unified collate info.

    Args:
        data_collate_info:
            User config to override the default collate info.
        pad_to_length:
            Whether to pad sequence to a fixed length. Default is False.
        seq_classification:
            If True, sequence classification task. Default is False.
        metadata_collate_func:
            Optional model-provided hook (``model.get_metadata_collate_func()``)
            that derives ``multimodal_metadata`` from the packed + SP-padded
            batch. ``None`` for text models. See ``MetadataCollateFunc``.
    """

    def __post_init__(self):
        """Build the ordered collation pipeline and merge model-specific field rules."""
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
                metadata_collate_func=self.metadata_collate_func,
            )
        )
        if get_parallel_state().sp_enabled:
            self.preforward_pipeline.append(
                SequenceParallelCollator(
                    collate_infos=self.collate_infos,
                    seq_classification=self.seq_classification,
                    metadata_collate_func=self.metadata_collate_func,
                )
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
        tail_padding_length = micro_batch.get("tail_padding_length")
        seq_lens = valid_seqlens_from_cu_seqlens(
            micro_batch["cu_seq_lens_q"],
            tail_padding_length=int(tail_padding_length) if tail_padding_length is not None else None,
        ).tolist()
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


def _diffusion_gemma_num_canvases(feature: Dict[str, "torch.Tensor"]) -> int:
    decoder_input_ids = feature["decoder_input_ids"]
    if decoder_input_ids.ndim == 1:
        return 1
    if decoder_input_ids.ndim == 2:
        return int(decoder_input_ids.shape[0])
    raise ValueError(
        "`decoder_input_ids` must be `(canvas_length,)` or `(num_canvases, canvas_length)`, "
        f"got shape {tuple(decoder_input_ids.shape)}."
    )


def _as_canvas_stack(tensor: "torch.Tensor") -> "torch.Tensor":
    return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor


def _diffusion_gemma_parts(
    feature: Dict[str, "torch.Tensor"],
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Canonicalize one feature to ``(ids, prompt_mask, canvases, labels, weight, context)``.

    Shared by the padded and the packed path so a shape the one rejects cannot slip
    through the other.
    """
    ids = feature["input_ids"]
    length = int(ids.numel())
    prompt_mask = feature.get("encoder_prompt_mask", torch.zeros_like(ids, dtype=torch.bool))
    if int(prompt_mask.numel()) != length:
        raise ValueError(
            f"`encoder_prompt_mask` must be the same length as `input_ids` ({length}), got {int(prompt_mask.numel())}."
        )
    canvases = _as_canvas_stack(feature["decoder_input_ids"])
    canvas_labels = _as_canvas_stack(feature["labels"])
    canvas_weight = _as_canvas_stack(feature["loss_weight"])
    if canvas_labels.shape != canvases.shape or canvas_weight.shape != canvases.shape:
        raise ValueError(
            "`labels` and `loss_weight` must match `decoder_input_ids` "
            f"{tuple(canvases.shape)}, got {tuple(canvas_labels.shape)} and {tuple(canvas_weight.shape)}."
        )
    context = _as_canvas_stack(feature["decoder_context_mask"])
    if int(context.shape[-1]) != length:
        raise ValueError(
            f"`decoder_context_mask` must be the same length as `input_ids` ({length}), got {int(context.shape[-1])}."
        )
    if int(context.shape[0]) != int(canvases.shape[0]):
        raise ValueError(
            "`decoder_context_mask` must have one row per canvas "
            f"({int(canvases.shape[0])}), got {int(context.shape[0])}."
        )
    return ids, prompt_mask.bool(), canvases, canvas_labels, canvas_weight, context


def _pad_canvas_stack(tensor: "torch.Tensor", max_canvases: int, value: float | int) -> "torch.Tensor":
    extra = max_canvases - int(tensor.shape[0])
    if extra <= 0:
        return tensor
    return F.pad(tensor, (0, 0, 0, extra), value=value)


@dataclass
class DiffusionGemmaCollator(DataCollator):
    """Batching collator for DiffusionGemma's encoder-prompt / decoder-canvas layout.

    Does not pack the way ``MainCollator`` does, and cannot: that packing concatenates
    samples into one row and encodes boundaries in ``position_ids``, which assumes a
    single sequence dimension shared by inputs and labels. DiffusionGemma has two
    different ones -- a variable length prompt consumed by the encoder and a fixed
    ``canvas_length`` canvas consumed by the decoder, related only through the KV cache.
    So by default prompts are padded to the longest in the batch and canvases -- already
    a fixed length from the data transform -- are simply stacked.

    ``pack_samples`` packs the axis that *can* be packed. The encoder rows concatenate
    with their boundaries in ``position_ids``, as ``MainCollator`` marks them; the
    canvases go onto the canvas axis, where a canvas is still one segment in its own
    row and sliding attention is undisturbed. See ``_packed_batch``.

    Prompts are **left**-padded. Each decoder canvas continues after the last encoder
    position visible to it. Left padding also keeps real tokens at the physical end of
    the cache, which matters when generation uses a truncating sliding cache: right
    padding would leave empty slots in that retained tail and evict valid prefix keys.

    Also emits ``decoder_attention_mask`` over ``[encoder | canvas]``. The encoder half
    is ``decoder_context_mask``, which every feature must carry: 1 on the prompt and
    the clean canvases *before* the draw, 0 on the draw and everything after, so the
    decoder cannot read the answer it is reconstructing. A missing key is an error --
    defaulting to all-ones would leak the clean canvas into the decoder.

    ``encoder_prompt_mask`` is left-padded with False (padding is not the prompt). Samples
    that omit it are treated as having no prompt span, so ``mask_prompt_ar_loss`` does not
    change their objective.

    With images, ``mm_token_type_ids`` is left-padded alongside the prompt (padding is
    tagged as text), while ``pixel_values`` and ``image_position_ids`` are concatenated
    along the image dimension. Those two need no padding: the processor already emits a
    fixed ``max_patches`` per image regardless of the source resolution and marks unused
    patches with ``(-1, -1)`` in ``image_position_ids``.

    ``compute_all_canvas_df_loss`` stacks ``K`` canvases on one feature as ``(K, C)``.
    Those stay stacked as ``(B, K, C)`` here -- the encoder is **not** repeated.
    The model runs the encoder once and ``config.canvas_batching`` decides whether the
    decoder loops over the K canvases or folds them into its batch, sharing one KV
    cache either way. Dummy canvases pad a mixed-K batch to ``max K`` with
    ``IGNORE_INDEX`` / zero weight.     ``num_supervised_examples`` still counts
    original features so the trainer divides by the number of examples.
    """

    pad_token_id: int = 0
    pack_samples: bool = False
    sp_size: int = 1

    def __post_init__(self):
        # Taken as a field rather than read off `get_parallel_state()` here, because a
        # collator is pickled into the dataloader workers and the global state does not
        # travel with it. The trainer, which builds this in the main process, passes it.
        if self.sp_size > 1 and not self.pack_samples:
            raise ValueError(
                f"`sp_size` is {self.sp_size} but `pack_samples` is off. Sequence parallelism shards "
                "the encoder row, and the padded path has one row per sample with its length set by "
                "the longest prompt in the batch -- there is no single row to shard, and the sdpa mask "
                "those rows need is quadratic in the length SP was turned on to grow. Set "
                "`canvas_batching: flat_seq_pack`."
            )

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        if not features:
            raise ValueError("DiffusionGemmaCollator received an empty feature list")

        if self.pack_samples:
            return self._packed_batch(features)

        num_supervised_examples = sum(1 for feature in features if float(feature["loss_weight"].sum()) > 0)
        max_canvases = max(_diffusion_gemma_num_canvases(feature) for feature in features)
        prompt_len = max(int(f["input_ids"].numel()) for f in features)
        input_ids, attention_mask, encoder_prompt_mask, mm_token_type_ids = [], [], [], []
        decoder_input_ids, labels, loss_weight, decoder_context = [], [], [], []
        for feature in features:
            ids, prompt_mask, canvases, canvas_labels, canvas_weight, context = _diffusion_gemma_parts(feature)
            pad = prompt_len - int(ids.numel())
            input_ids.append(F.pad(ids, (pad, 0), value=self.pad_token_id))
            attention_mask.append(F.pad(torch.ones_like(ids), (pad, 0), value=0))
            encoder_prompt_mask.append(F.pad(prompt_mask, (pad, 0), value=False))
            if "mm_token_type_ids" in feature:
                mm_token_type_ids.append(F.pad(feature["mm_token_type_ids"], (pad, 0), value=0))
            context = torch.stack([F.pad(row, (pad, 0), value=0) for row in context])
            decoder_input_ids.append(_pad_canvas_stack(canvases, max_canvases, 0))
            labels.append(_pad_canvas_stack(canvas_labels, max_canvases, IGNORE_INDEX))
            loss_weight.append(_pad_canvas_stack(canvas_weight, max_canvases, 0))
            decoder_context.append(_pad_canvas_stack(context, max_canvases, 0))

        stacked_canvases = torch.stack(decoder_input_ids)
        stacked_context = torch.stack(decoder_context).bool()
        if max_canvases == 1:
            stacked_canvases = stacked_canvases.squeeze(1)
            stacked_labels = torch.stack(labels).squeeze(1)
            stacked_weight = torch.stack(loss_weight).squeeze(1)
            stacked_context = stacked_context.squeeze(1)
        else:
            stacked_labels = torch.stack(labels)
            stacked_weight = torch.stack(loss_weight)
        canvas_length = stacked_canvases.shape[-1]
        batch = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "encoder_prompt_mask": torch.stack(encoder_prompt_mask),
            "decoder_input_ids": stacked_canvases,
            "labels": stacked_labels,
            "loss_weight": stacked_weight,
            "decoder_attention_mask": F.pad(stacked_context, (0, canvas_length), value=True),
            "num_supervised_examples": torch.tensor(num_supervised_examples, dtype=torch.long),
        }

        if mm_token_type_ids:
            if len(mm_token_type_ids) != len(features):
                raise ValueError(
                    "Some samples carry `mm_token_type_ids` and some do not; the encoder needs the tag "
                    "for the whole batch or none of it."
                )
            batch["mm_token_type_ids"] = torch.stack(mm_token_type_ids)
        for key in ("pixel_values", "image_position_ids"):
            tensors = [f[key] for f in features if key in f]
            if tensors:
                batch[key] = torch.cat(tensors, dim=0)
        return batch

    def _packed_batch(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        """One row for the whole batch: encoder tokens concatenated, canvases stacked.

        Boundaries ride in ``position_ids``, which restarts at 0 on each sample -- the
        convention ``prepare_fa_kwargs_from_position_ids`` reads and the one the model
        turns into ``cu_seqlens`` for its varlen encoder kernels. So nothing is padded:
        neither the prompts to the longest in the batch, nor the canvas counts to the
        largest ``K``. A sample's cost is its own token count rather than the batch's
        worst case, which is what lets a token budget hold across ragged data.

        ``sp_size > 1`` adds the one exception: fewer than ``sp_size`` tokens at the end,
        so the row divides across the sequence-parallel ranks that will shard it.

        Each canvas keeps its own row on the canvas axis, with the encoder half of
        ``decoder_attention_mask`` placed at its own sample's offset and False
        everywhere else. That interval is how the model recovers which sample a canvas
        belongs to (``canvas_owner_segments``), so it also has to be the thing that
        stops a canvas reading a neighbour's prompt.
        """
        lengths: List[int] = []
        ids_parts, position_parts, prompt_parts = [], [], []
        canvas_parts, label_parts, weight_parts, context_parts = [], [], [], []
        for feature in features:
            for key in ("mm_token_type_ids", "pixel_values", "image_position_ids"):
                if key in feature:
                    raise ValueError(
                        f"`pack_samples` cannot pack `{key}`: the packed encoder runs varlen attention "
                        "kernels that carry no image block boundaries, so an image span would be "
                        "attended as text. Pack text-only data, or leave `pack_samples` off."
                    )
            ids, prompt_mask, canvases, canvas_labels, canvas_weight, context = _diffusion_gemma_parts(feature)
            lengths.append(int(ids.numel()))
            ids_parts.append(ids)
            position_parts.append(torch.arange(lengths[-1], dtype=torch.long, device=ids.device))
            prompt_parts.append(prompt_mask)
            canvas_parts.append(canvases)
            label_parts.append(canvas_labels)
            weight_parts.append(canvas_weight)
            context_parts.append(context)

        # Sequence parallelism shards this row, so it has to divide. The tail is its own
        # segment -- a fresh restart in `position_ids` -- rather than an extension of the
        # last sample, which would hand that sample `pad_size` extra keys to attend to
        # and change its answer. `attention_mask` is 0 there so the AR term can tell the
        # tail from data, and the canvas half of `decoder_attention_mask` is False so no
        # canvas can read it.
        #
        # Padded here rather than sliced here: the slice happens in the encoder, which is
        # the one place that knows the boundaries of the whole row. `cu_seqlens` cannot be
        # recovered from a shard, and a full-length boundary vector travelling beside a
        # sliced `position_ids` is a pair with no local invariant to check it by.
        #
        # One consequence to know about: being its own segment, the tail adds one to the
        # segment count the row reports, so a caller supplying `encoder_ar_row_weight`
        # would have to supply one more weight under SP than without it. Nothing hits
        # this today -- this collator never emits that key, and `encoder_ar_targets`
        # clamps a zero-count segment -- but the number is `sp_size`-dependent, which is
        # not obvious from either end.
        pad_size = -sum(lengths) % self.sp_size if self.sp_size > 1 else 0
        if pad_size:
            template = ids_parts[0]
            ids_parts.append(torch.full((pad_size,), self.pad_token_id, dtype=template.dtype, device=template.device))
            position_parts.append(torch.arange(pad_size, dtype=torch.long, device=template.device))
            prompt_parts.append(torch.zeros(pad_size, dtype=prompt_parts[0].dtype, device=template.device))

        total = sum(lengths) + pad_size
        rows, start = [], 0
        for length, context in zip(lengths, context_parts):
            rows.append(F.pad(context.bool(), (start, total - start - length), value=False))
            start += length
        packed_context = torch.cat(rows, dim=0)
        canvas_length = int(canvas_parts[0].shape[-1])
        template = ids_parts[0]
        attention_mask = torch.ones(1, total, dtype=template.dtype, device=template.device)
        attention_mask[:, total - pad_size :] = 0
        return {
            "input_ids": torch.cat(ids_parts).unsqueeze(0),
            "position_ids": torch.cat(position_parts).unsqueeze(0),
            "attention_mask": attention_mask,
            "encoder_prompt_mask": torch.cat(prompt_parts).unsqueeze(0),
            "decoder_input_ids": torch.cat(canvas_parts).unsqueeze(0),
            "labels": torch.cat(label_parts).unsqueeze(0),
            "loss_weight": torch.cat(weight_parts).unsqueeze(0),
            "decoder_attention_mask": F.pad(packed_context, (0, canvas_length), value=True).unsqueeze(0),
            "num_supervised_examples": torch.tensor(
                sum(1 for feature in features if float(feature["loss_weight"].sum()) > 0), dtype=torch.long
            ),
        }
