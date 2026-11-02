# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Local validation of DPO packed-segment provenance."""

from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist

from ..utils.constants import IGNORE_INDEX
from ..utils.dpo_provenance import (
    DPO_AGREED_KEY,
    DPO_APPROVED_KEY,
    DPO_PROOF_KEY,
    DPOInvalidProof,
    DPOPackedProof,
)
from ..utils.seqlen_pos_transform_utils import packed_sequence_slices_from_cu_seqlens


def _validate_proof(value: Any) -> None:
    if isinstance(value, DPOInvalidProof):
        raise ValueError(f"invalid DPO packed-segment evidence: {value.reason}")
    if not isinstance(value, DPOPackedProof):
        raise ValueError("DPO packed-segment evidence is missing or has an unsupported type")
    if type(value.producer_contract_version) is not int or value.producer_contract_version != 1:
        raise ValueError("DPO packed evidence predates the required unpadded producer contract")
    if isinstance(value.segment_lengths, tuple) and any(
        type(length) is int and length == 1 for length in value.segment_lengths
    ):
        raise ValueError("DPO segments need a causal prediction target; add a BOS token or nonempty context")
    if (
        type(value.segment_lengths) is not tuple
        or not value.segment_lengths
        or len(value.segment_lengths) % 2
        or any(type(length) is not int or length <= 0 for length in value.segment_lengths)
        or type(value.feature_token_counts) is not tuple
        or len(value.feature_token_counts) * 2 != len(value.segment_lengths)
        or any(type(count) is not int or count < 0 for count in value.feature_token_counts)
        or type(value.collator_tail) is not int
        or value.collator_tail < 0
        or type(value.source_indices) is not tuple
        or len(value.source_indices) not in (0, len(value.feature_token_counts))
        or any(type(index) is not int or index < 0 for index in value.source_indices)
        or type(value.source_keys) is not tuple
        or any(type(key) is not str for key in value.source_keys)
        or type(value.layout_sp_size) is not int
        or value.layout_sp_size < 1
        or type(value.layout_sp_rank) is not int
        or not 0 <= value.layout_sp_rank < value.layout_sp_size
    ):
        raise ValueError("DPO packed-segment evidence is malformed")


def _validate_batch(
    batch: dict[str, Any],
    proof: DPOPackedProof,
    source_count: int | None,
    vocab_size: int | None,
    source_catalog: tuple[str, ...] | None,
) -> None:
    for key in ("input_ids", "labels", "position_ids", "attention_mask", "cu_seq_lens_q"):
        value = batch.get(key)
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise ValueError(f"DPO packed {key} must be a CPU tensor")
    ids, labels = batch["input_ids"], batch["labels"]
    if ids.ndim != 2 or ids.shape[0] != 1 or labels.shape != ids.shape:
        raise ValueError("DPO input and label shapes must agree")
    if not bool((ids >= 0).all()) or not bool(((labels >= 0) | (labels == IGNORE_INDEX)).all()):
        raise ValueError("DPO packed input or label contains an invalid negative token ID")
    if vocab_size is not None:
        if type(vocab_size) is not int or vocab_size <= 0:
            raise ValueError("DPO model vocabulary size is invalid")
        if not bool((ids < vocab_size).all()) or not bool(((labels < vocab_size) | (labels == IGNORE_INDEX)).all()):
            raise ValueError("DPO input or label token ID is outside the model vocabulary")
    cu = batch["cu_seq_lens_q"]
    expected = [0]
    for length in proof.segment_lengths:
        expected.append(expected[-1] + length)
    if proof.collator_tail:
        expected.append(expected[-1] + proof.collator_tail)
    if cu.ndim != 1 or cu.tolist() != expected:
        raise ValueError("DPO packed sequence boundaries changed after collation")
    for key in ("cu_seq_lens_k", "linear_attn_cu_seq_lens_q"):
        value = batch.get(key)
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or not torch.equal(value, cu):
            raise ValueError("DPO attention or GDN sequence boundaries are inconsistent")
    maximum = max(end - start for start, end in zip(expected, expected[1:]))
    if batch.get("max_length_q") != maximum or batch.get("max_length_k") != maximum:
        raise ValueError("DPO maximum sequence lengths are inconsistent")
    if batch.get("packed_sequence_slices") != packed_sequence_slices_from_cu_seqlens(cu):
        raise ValueError("DPO packed sequence slices are inconsistent")
    if not bool(batch["attention_mask"].eq(1).all()) or batch.get("attention_mask_is_all_ones") is not True:
        raise ValueError("DPO packed attention mask must contain only ones")
    if proof.source_indices:
        values = batch.get("ds_idx")
        if isinstance(values, torch.Tensor) and values.device.type == "cpu" and values.ndim == 1:
            values = values.tolist()
        if not isinstance(values, list) or tuple(values) != proof.source_indices:
            raise ValueError("DPO source IDs changed after collation")
    if source_count is not None:
        if type(source_count) is not int or source_count <= 0:
            raise ValueError("DPO multisource configuration has no valid sources")
        if len(proof.source_indices) != len(proof.feature_token_counts):
            raise ValueError("DPO multisource feature IDs are missing")
        if any(index >= source_count for index in proof.source_indices):
            raise ValueError("DPO source ID is outside the configured source list")
        if source_catalog is not None and "source_name" in batch:
            names = batch["source_name"]
            if isinstance(names, str):
                names = [names]
            if not isinstance(names, (list, tuple)) or tuple(names) != tuple(
                source_catalog[index] for index in proof.source_indices
            ):
                raise ValueError("DPO source names disagree with the configured source IDs")


def _next_local_dpo_microbatches(
    data_iterator: Iterator[Any],
    policy_state: Any,
    reference_state: Any,
    global_step: int,
    epoch: int = 0,
    source_count: Callable[[], int | None] | None = None,
    source_catalog: Callable[[], tuple[str, ...] | None] | None = None,
    vocab_size: Callable[[], int | None] | None = None,
) -> list[dict[str, Any]]:
    """Validate producer-proven segment lengths before pair-count reduction.

    Sequence-parallel ranks must receive the same deterministic input, as for
    the other training objectives. Callback consumers may remove metadata but
    must not mutate or reorder model inputs. Features must likewise remain
    unchanged between the producer and collator; token contents are not hashed.
    """
    del global_step, epoch
    batches = next(data_iterator)
    if not isinstance(batches, list) or not batches or not all(isinstance(batch, dict) for batch in batches):
        raise ValueError("DPO optimizer step requires a nonempty list of microbatches")
    sp_size = policy_state.sp_size if policy_state.sp_enabled else 1
    sp_rank = policy_state.sp_rank if policy_state.sp_enabled else 0
    if (reference_state.sp_size if reference_state.sp_enabled else 1) != sp_size or (
        reference_state.sp_rank if reference_state.sp_enabled else 0
    ) != sp_rank:
        raise ValueError("DPO policy/reference sequence-parallel layouts disagree")
    if sp_size > 1:
        if not dist.is_initialized() or policy_state.sp_group is None or reference_state.sp_group is None:
            raise ValueError("DPO sequence parallelism requires initialized policy and reference groups")
        if dist.get_process_group_ranks(policy_state.sp_group) != dist.get_process_group_ranks(
            reference_state.sp_group
        ):
            raise ValueError("DPO policy/reference sequence-parallel groups disagree")
    count = source_count() if source_count is not None else None
    catalog = source_catalog() if source_catalog is not None else None
    if catalog is not None:
        if type(catalog) is not tuple or not catalog or any(type(name) is not str for name in catalog):
            raise ValueError("DPO multisource catalog is invalid")
        if count is not None and count != len(catalog):
            raise ValueError("DPO source count disagrees with its catalog")
        count = len(catalog)
    vocabulary = vocab_size() if vocab_size is not None else None
    for batch in batches:
        if DPO_APPROVED_KEY in batch or DPO_AGREED_KEY in batch:
            raise ValueError("DPO microbatch contains a reserved approved-provenance key")
        proof = batch.get(DPO_PROOF_KEY)
        _validate_proof(proof)
        if proof.layout_sp_size != sp_size or proof.layout_sp_rank != sp_rank:
            raise ValueError("DPO collator SP topology disagrees with the model process group")
        _validate_batch(batch, proof, count, vocabulary, catalog)
        batch[DPO_APPROVED_KEY] = proof
    return batches


def next_dpo_microbatches(
    data_iterator: Iterator[Any],
    policy_state: Any,
    reference_state: Any,
    global_step: int,
    epoch: int = 0,
    source_count: Callable[[], int | None] | None = None,
    source_catalog: Callable[[], tuple[str, ...] | None] | None = None,
    vocab_size: Callable[[], int | None] | None = None,
) -> list[dict[str, Any]]:
    try:
        return _next_local_dpo_microbatches(
            data_iterator, policy_state, reference_state, global_step, epoch, source_count, source_catalog, vocab_size
        )
    except StopIteration:
        raise
    except Exception as exc:
        raise RuntimeError(f"DPO metadata failed: {exc}") from exc
