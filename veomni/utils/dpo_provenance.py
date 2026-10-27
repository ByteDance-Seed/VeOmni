# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Private, CPU-resident evidence for opt-in DPO packed-segment accounting."""

import hashlib
from dataclasses import dataclass
from typing import Any

import torch


DPO_FEATURE_KEY = "_veomni_dpo_feature_segments"
DPO_PROOF_KEY = "_veomni_dpo_packed_proof"
DPO_APPROVED_KEY = "_veomni_dpo_approved_proof"
DPO_POSTCHECK_KEY = "_veomni_dpo_post_callback_proof"
DPO_CALLBACK_METADATA_KEYS = frozenset(
    {"ds_idx", "source_name", "cur_token_num", "_veomni_channel_loss_source_provenance"}
)


@dataclass(frozen=True)
class DPOFeatureSegments:
    """Physical lengths from the built-in DPO producer, not padding guesses."""

    chosen: int
    rejected: int


@dataclass(frozen=True)
class DPOPackedProof:
    """A checked packed layout; feature counts use the loader's token convention."""

    segment_lengths: tuple[int, ...]
    feature_token_counts: tuple[int, ...]
    collator_tail: int
    source_ids: tuple[str | int | None, ...]
    input_digest: str
    packed_digest: str
    forward_digest: str
    global_digest: str
    source_indices: tuple[int, ...] = ()
    source_keys: tuple[str, ...] = ()
    callback_metadata_digests: tuple[tuple[str, str], ...] = ()
    layout_sp_size: int = 1
    layout_sp_rank: int = 0


@dataclass(frozen=True)
class DPOInvalidProof:
    """Inconsistent evidence; the trainer must coordinate rejection across ranks."""

    reason: str


def packed_batch_digest(
    batch: dict[str, Any],
    source_keys: tuple[str, ...] = (),
    *,
    after_callbacks: bool = False,
    global_only: bool = False,
    feature: bool = False,
    include_keys: tuple[str, ...] | None = None,
) -> str:
    """Fingerprint the local CPU batch so later mutation cannot reuse its evidence."""
    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise ValueError("DPO packed provenance tensors must be on CPU")
            digest.update(f"tensor:{value.dtype}:{tuple(value.shape)}:".encode())
            digest.update(memoryview(value.contiguous().reshape(-1).view(torch.uint8).numpy()))
        elif isinstance(value, dict):
            if any(type(key) is not str for key in value):
                raise ValueError("DPO packed provenance has non-string keys")
            digest.update(f"dict:{type(value).__module__}.{type(value).__qualname__}:".encode())
            for key in sorted(value):
                update(key)
                update(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(f"{type(value).__name__}:{len(value)}:".encode())
            for item in value:
                update(item)
        elif value is None or type(value) in (bool, int, float, str):
            digest.update(f"{type(value).__name__}:{value!r};".encode())
        else:
            raise ValueError("DPO packed provenance has an unsupported value")

    if global_only:
        for key in (
            "cu_seq_lens_q",
            "cu_seq_lens_k",
            "linear_attn_cu_seq_lens_q",
            "packed_sequence_slices",
            "max_length_q",
            "max_length_k",
            "attention_mask_is_all_ones",
            "tail_padding_length",
        ):
            update(key)
            update(batch.get(key))
        return digest.hexdigest()

    required = ("input_ids", "labels", "position_ids", "attention_mask")
    if not feature:
        required += ("cu_seq_lens_q",)
    for key in required:
        value = batch.get(key)
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise ValueError(f"DPO packed {key} must be a CPU tensor")
    if any(type(key) is not str for key in batch):
        raise ValueError("DPO packed batch has non-string keys")
    update(source_keys)
    ignored = {DPO_PROOF_KEY, DPO_APPROVED_KEY, DPO_POSTCHECK_KEY}
    if after_callbacks:
        ignored.update(("ds_idx", "source_name", "cur_token_num", "_veomni_channel_loss_source_provenance"))
        ignored.update(source_keys)
    for key in sorted(batch):
        if key in ignored or (include_keys is not None and key not in include_keys):
            continue
        update(key)
        update(batch[key])
    return digest.hexdigest()


def callback_metadata_digests(batch: dict[str, Any], source_keys: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """Fingerprint metadata that standard callbacks may remove, but not add or change."""
    return tuple(
        (key, packed_batch_digest(batch, source_keys, include_keys=(key,)))
        for key in sorted(DPO_CALLBACK_METADATA_KEYS | set(source_keys))
        if key in batch
    )
