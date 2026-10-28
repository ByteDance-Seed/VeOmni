# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Opt-in DPO-specific evidence around the ordinary packing collator."""

import hashlib
from dataclasses import dataclass
from typing import Any

import torch

from ..utils.constants import IGNORE_INDEX
from ..utils.dpo_provenance import (
    DPO_APPROVED_KEY,
    DPO_FEATURE_KEY,
    DPO_POSTCHECK_KEY,
    DPO_PROOF_KEY,
    DPOFeatureSegments,
    DPOInvalidProof,
    DPOPackedProof,
    callback_metadata_digests,
    packed_batch_digest,
)
from . import data_collator as collator_module
from .data_collator import (
    DEFAULT_DATA_COLLATE_INFO,
    MainCollator,
    PackingCollator,
    PrecomputePositionIDsCollator,
    SequenceParallelCollator,
)
from .dataset import get_length_fn


def _source_id(feature: dict[str, Any], extra_keys: tuple[str, ...]) -> tuple[bool, str | None]:
    digest = hashlib.sha256()
    found = False
    for key in dict.fromkeys(("ds_idx", "source_name", *extra_keys)):
        if key not in feature:
            continue
        value = feature[key]
        if isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.numel() == 1:
            value = value.item()
        if type(value) not in (int, str):
            return False, None
        found = True
        digest.update(f"{key}:{type(value).__name__}:{value!r};".encode("utf-8"))
    return True, digest.hexdigest() if found else None


def _input_digest(features: list[dict[str, Any]], source_keys: tuple[str, ...]) -> str | None:
    """Hash transformed CPU tensors before packing or SP slicing; do not infer raw dataset identity."""
    digest = hashlib.sha256()
    allowed = {"input_ids", "labels", "position_ids", "attention_mask", "cur_token_num", *source_keys}
    for feature in features:
        if any(key not in allowed for key in feature):
            return None
        for key in ("input_ids", "labels", "position_ids", "attention_mask"):
            tensor = feature.get(key)
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.ndim != 1:
                return None
        try:
            digest.update(bytes.fromhex(packed_batch_digest(feature, source_keys, feature=True)))
        except (TypeError, ValueError):
            return None
    return digest.hexdigest()


@dataclass
class DPOProvenanceCollator:
    """Carry producer-proven segment lengths without modifying model inputs.

    Mixed legacy or unsupported layouts are marked invalid for coordinated
    rejection; we never infer padding from token IDs or masks.
    """

    collator: Any
    require_ds_idx: bool = False

    def __post_init__(self) -> None:
        state = collator_module.get_parallel_state()
        self.sp_size = state.sp_size if state.sp_enabled else 1
        self.sp_rank = state.sp_rank if state.sp_enabled else 0

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if any(
            any(key in feature for key in (DPO_PROOF_KEY, DPO_APPROVED_KEY, DPO_POSTCHECK_KEY)) for feature in features
        ):
            raise ValueError("DPO input contains a reserved packed-provenance key")
        sp_size, sp_rank = self.sp_size, self.sp_rank
        if type(self.collator) is MainCollator:
            expected_types = (
                (PrecomputePositionIDsCollator, PackingCollator, SequenceParallelCollator)
                if sp_size > 1
                else (PrecomputePositionIDsCollator, PackingCollator)
            )
            if tuple(type(item) for item in self.collator.preforward_pipeline) != expected_types:
                raise ValueError("DPO provenance requires the standard collator pipeline")
            packer = self.collator.preforward_pipeline[1]
            if (
                packer.collate_infos is not self.collator.collate_infos
                or packer.pad_to_length != self.collator.pad_to_length
                or packer.seq_classification != self.collator.seq_classification
                or packer.metadata_collate_func is not self.collator.metadata_collate_func
                or packer.sp_enabled != (sp_size > 1)
            ):
                raise ValueError("DPO provenance requires the standard packing configuration")
            if sp_size > 1:
                slicer = self.collator.preforward_pipeline[2]
                if (
                    slicer.collate_infos is not self.collator.collate_infos
                    or slicer.seq_classification != self.collator.seq_classification
                    or slicer.metadata_collate_func is not self.collator.metadata_collate_func
                    or slicer.sp_size != sp_size
                    or slicer.sp_rank != sp_rank
                ):
                    raise ValueError("DPO provenance requires the standard sequence-parallel configuration")
        metadata = [feature.get(DPO_FEATURE_KEY) for feature in features]
        known = [isinstance(item, DPOFeatureSegments) for item in metadata]
        # Never remove a same-named field belonging to an external producer.
        clean = []
        for feature, is_known in zip(features, known):
            sample = {k: v for k, v in feature.items() if not (is_known and k == DPO_FEATURE_KEY)}
            if isinstance(sample.get("labels"), torch.Tensor):
                sample["labels"] = sample["labels"].clone()
            clean.append(sample)
        if not any(known):
            batch = self.collator(clean)
            batch[DPO_PROOF_KEY] = DPOInvalidProof("DPO feature provenance is missing")
            return batch
        can_prove = all(known)
        source_keys = tuple(
            dict.fromkeys(
                ("ds_idx", "source_name", *(getattr(self.collator, "channel_loss_source_id_keys", ()) or ()))
            )
        )
        model_keys = {
            "input_ids",
            "labels",
            "position_ids",
            "attention_mask",
            "cu_seq_lens_q",
            "cu_seq_lens_k",
            "linear_attn_cu_seq_lens_q",
            "packed_sequence_slices",
            "max_length_q",
            "max_length_k",
            "attention_mask_is_all_ones",
            "tail_padding_length",
            "use_cache",
            "return_log_probs",
        }
        if model_keys.intersection(source_keys):
            raise ValueError("DPO source metadata keys must not overlap model inputs")
        can_prove = can_prove and not any(
            any(key in feature for feature in clean) and not all(key in feature for feature in clean)
            for key in source_keys
        )
        has_ds_idx = all("ds_idx" in feature for feature in clean)
        can_prove = can_prove and (has_ds_idx or not self.require_ds_idx)
        ds_indices: list[int] = []
        if can_prove and has_ds_idx:
            for feature in clean:
                value = feature["ds_idx"]
                if isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.numel() == 1:
                    value = value.item()
                if type(value) is not int or value < 0:
                    can_prove = False
                    break
                ds_indices.append(value)
        input_digest = _input_digest(clean, source_keys) if can_prove else None
        can_prove = can_prove and input_digest is not None
        segment_lengths: list[int] = []
        counts: list[int] = []
        sources: list[str | None] = []
        if can_prove:
            for feature, item in zip(clean, metadata):
                ids = feature.get("input_ids")
                source_valid, source = _source_id(
                    feature, tuple(getattr(self.collator, "channel_loss_source_id_keys", ()) or ())
                )
                if (
                    not isinstance(ids, torch.Tensor)
                    or ids.ndim != 1
                    or type(item.producer_contract_version) is not int
                    or item.producer_contract_version != 1
                    or type(item.feature_digest) is not str
                    or item.feature_digest
                    != packed_batch_digest(
                        feature,
                        feature=True,
                        include_keys=("input_ids", "labels", "position_ids", "attention_mask"),
                    )
                    or type(item.chosen) is not int
                    or type(item.rejected) is not int
                    or item.chosen <= 0
                    or item.rejected <= 0
                    or item.chosen + item.rejected != ids.numel()
                    or any(
                        feature[key].dtype not in (torch.int32, torch.int64)
                        for key in ("input_ids", "labels", "position_ids")
                    )
                    or any(feature[key].shape != ids.shape for key in ("labels", "position_ids", "attention_mask"))
                    or not bool(feature["attention_mask"].eq(1).all())
                    or not bool((feature["input_ids"] >= 0).all())
                    or not bool(((feature["labels"] >= 0) | (feature["labels"] == IGNORE_INDEX)).all())
                    or not bool(
                        ((feature["labels"] == IGNORE_INDEX) | (feature["labels"] == feature["input_ids"])).all()
                    )
                    or not bool((feature["labels"][: item.chosen] != IGNORE_INDEX).any())
                    or not bool((feature["labels"][item.chosen :] != IGNORE_INDEX).any())
                    or (item.chosen > 1 and not bool((feature["labels"][1 : item.chosen] != IGNORE_INDEX).any()))
                    or (item.rejected > 1 and not bool((feature["labels"][item.chosen + 1 :] != IGNORE_INDEX).any()))
                    or not bool((feature["position_ids"] >= 0).all())
                    or not source_valid
                ):
                    can_prove = False
                    break
                segment_lengths.extend((item.chosen, item.rejected))
                counts.append(get_length_fn(feature))
                sources.append(source)

        batch = self.collator(clean)
        if any(key in batch for key in (DPO_PROOF_KEY, DPO_APPROVED_KEY, DPO_POSTCHECK_KEY)):
            batch[DPO_PROOF_KEY] = DPOInvalidProof("reserved packed-proof key already exists")
            return batch
        if not can_prove:
            batch[DPO_PROOF_KEY] = DPOInvalidProof("missing or invalid feature segments")
            return batch
        if has_ds_idx:
            packed_ds = batch.get("ds_idx")
            if isinstance(packed_ds, torch.Tensor) and packed_ds.device.type == "cpu" and packed_ds.ndim == 1:
                packed_ds = packed_ds.tolist()
            if not isinstance(packed_ds, list) or packed_ds != ds_indices:
                batch[DPO_PROOF_KEY] = DPOInvalidProof("packed source IDs do not match feature order")
                return batch
        if (
            type(self.collator) is not MainCollator
            or self.collator.metadata_collate_func is not None
            or self.collator.seq_classification
        ):
            batch[DPO_PROOF_KEY] = DPOInvalidProof(
                "custom collator, metadata hook, or classification layout is not proven"
            )
            return batch
        infos = self.collator.collate_infos
        if any(
            infos[name] != DEFAULT_DATA_COLLATE_INFO[name]
            for name in ("input_ids", "labels", "position_ids", "attention_mask")
        ):
            batch[DPO_PROOF_KEY] = DPOInvalidProof("custom token packing or label padding is not proven")
            return batch
        if any(infos.get(name) != DEFAULT_DATA_COLLATE_INFO.get(name) for name in ("cur_token_num", *source_keys)):
            batch[DPO_PROOF_KEY] = DPOInvalidProof("custom source metadata collation is not proven")
            return batch
        cu = batch.get("cu_seq_lens_q")
        if not isinstance(cu, torch.Tensor) or cu.device.type != "cpu" or cu.ndim != 1:
            batch[DPO_PROOF_KEY] = DPOInvalidProof("packed CU is unavailable on CPU")
            return batch
        actual = cu.tolist()
        expected = [0]
        for length in segment_lengths:
            expected.append(expected[-1] + length)
        tail = batch.get("tail_padding_length")
        tail = int(tail) if tail is not None else 0
        expected_full = expected + [expected[-1] + tail] if tail > 0 else expected
        if tail < 0 or actual != expected_full:
            batch[DPO_PROOF_KEY] = DPOInvalidProof("packed CU does not match producer segments and collator tail")
            return batch
        batch[DPO_PROOF_KEY] = DPOPackedProof(
            tuple(segment_lengths),
            tuple(counts),
            tail,
            tuple(sources),
            input_digest,
            packed_batch_digest(batch, source_keys),
            packed_batch_digest(batch, source_keys, after_callbacks=True),
            packed_batch_digest(batch, global_only=True),
            tuple(ds_indices),
            source_keys,
            callback_metadata_digests(batch, source_keys),
            sp_size,
            sp_rank,
            producer_contract_version=1,
        )
        return batch
