# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Opt-in rank agreement for DPO packed-segment proofs."""

import hashlib
from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist

from ..utils.constants import IGNORE_INDEX
from ..utils.dpo_provenance import (
    DPO_APPROVED_KEY,
    DPO_POSTCHECK_KEY,
    DPO_PROOF_KEY,
    DPOInvalidProof,
    DPOPackedProof,
    callback_metadata_digests,
    packed_batch_digest,
)
from ..utils.seqlen_pos_transform_utils import packed_sequence_slices_from_cu_seqlens


_PROTOCOL_VERSION = 1


def _group_digest(group: tuple[int, ...]) -> str:
    return hashlib.sha256(repr(group).encode()).hexdigest()


def _members(state: Any, rank: int) -> tuple[int, ...]:
    if not state.sp_enabled:
        return (rank,)
    if not dist.is_initialized():
        raise RuntimeError("sequence-parallel DPO proof requires an initialized process group")
    members = tuple(dist.get_process_group_ranks(state.sp_group))
    if len(members) != state.sp_size or not 0 <= state.sp_rank < len(members) or members[state.sp_rank] != rank:
        raise RuntimeError("DPO sequence-parallel group metadata is inconsistent")
    return members


def _proof_data(value: Any) -> str:
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
        or type(value.source_ids) is not tuple
        or len(value.source_ids) != len(value.feature_token_counts)
        or any(source is not None and (type(source) is not str or len(source) != 64) for source in value.source_ids)
        or type(value.input_digest) is not str
        or len(value.input_digest) != 64
        or any(char not in "0123456789abcdef" for char in value.input_digest)
        or type(value.packed_digest) is not str
        or len(value.packed_digest) != 64
        or any(char not in "0123456789abcdef" for char in value.packed_digest)
        or type(value.forward_digest) is not str
        or len(value.forward_digest) != 64
        or any(char not in "0123456789abcdef" for char in value.forward_digest)
        or type(value.global_digest) is not str
        or len(value.global_digest) != 64
        or any(char not in "0123456789abcdef" for char in value.global_digest)
        or type(value.source_indices) is not tuple
        or len(value.source_indices) not in (0, len(value.feature_token_counts))
        or any(type(index) is not int or index < 0 for index in value.source_indices)
        or type(value.source_keys) is not tuple
        or any(type(key) is not str for key in value.source_keys)
        or type(value.callback_metadata_digests) is not tuple
        or any(
            type(entry) is not tuple
            or len(entry) != 2
            or type(entry[0]) is not str
            or type(entry[1]) is not str
            or len(entry[1]) != 64
            for entry in value.callback_metadata_digests
        )
        or type(value.layout_sp_size) is not int
        or value.layout_sp_size < 1
        or type(value.layout_sp_rank) is not int
        or not 0 <= value.layout_sp_rank < value.layout_sp_size
    ):
        raise ValueError("DPO packed-segment evidence is malformed")
    data = (
        value.segment_lengths,
        value.feature_token_counts,
        value.collator_tail,
        value.source_ids,
        value.input_digest,
        value.global_digest,
        value.source_indices,
        value.source_keys,
        value.layout_sp_size,
    )
    # Do not send O(number of packed samples) metadata through WORLD.  Each
    # local layout is checked in full before hashing; ranks exchange fixed-size
    # digests and a microbatch count instead.
    return hashlib.sha256(repr(data).encode()).hexdigest()


def _validate_batch(
    batch: dict[str, Any],
    proof: DPOPackedProof,
    source_count: int | None,
    vocab_size: int | None,
    source_catalog: tuple[str, ...] | None,
) -> None:
    if packed_batch_digest(batch, proof.source_keys) != proof.packed_digest:
        raise ValueError("DPO batch changed after its segment evidence was produced")
    if callback_metadata_digests(batch, proof.source_keys) != proof.callback_metadata_digests:
        raise ValueError("DPO callback metadata evidence is inconsistent")
    if packed_batch_digest(batch, global_only=True) != proof.global_digest:
        raise ValueError("DPO global attention metadata evidence is inconsistent")
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


def _validate_packets(packets: list[dict[str, Any]]) -> None:
    """Use identical gathered primitive packets to make the same decision on every rank."""
    world_size = len(packets)
    for rank, packet in enumerate(packets):
        if packet["version"] != _PROTOCOL_VERSION or packet["rank"] != rank:
            raise RuntimeError("DPO provenance protocol or global rank disagrees")
    statuses = {packet["status"] for packet in packets}
    if statuses == {"eof"}:
        if len({(packet["step"], packet["epoch"]) for packet in packets}) != 1:
            raise RuntimeError("DPO ranks disagree on step or epoch at end of data")
        raise StopIteration
    if statuses != {"ok"}:
        details = [
            (packet["rank"], packet["status"], packet["error"]) for packet in packets if packet["status"] != "ok"
        ]
        raise RuntimeError(f"DPO data retrieval or metadata failed on at least one rank: {details}")
    if len({packet["step"] for packet in packets}) != 1:
        raise RuntimeError("DPO ranks disagree on global step")
    if len({packet["epoch"] for packet in packets}) != 1:
        raise RuntimeError("DPO ranks disagree on epoch")
    if len({packet["phase"] for packet in packets}) != 1:
        raise RuntimeError("DPO ranks disagree on preflight phase")
    if (
        len({(packet["source_count"], packet["source_catalog_digest"], packet["vocab_size"]) for packet in packets})
        != 1
    ):
        raise RuntimeError("DPO ranks disagree on source catalog or model vocabulary")
    batch_counts = {packet["batch_count"] for packet in packets}
    if len(batch_counts) != 1:
        raise RuntimeError("DPO ranks disagree on the number of microbatches")

    groups: dict[str, list[dict[str, Any]]] = {}
    for packet in packets:
        group_hash, size, slot = packet["group_digest"], packet["group_size"], packet["group_slot"]
        if (
            type(group_hash) is not str
            or len(group_hash) != 64
            or type(size) is not int
            or not 0 < size <= world_size
            or type(slot) is not int
            or not 0 <= slot < size
        ):
            raise RuntimeError("DPO sequence-parallel group descriptor is invalid")
        groups.setdefault(group_hash, []).append(packet)
    for group_hash, members in groups.items():
        size = members[0]["group_size"]
        if len(members) != size or any(member["group_size"] != size for member in members):
            raise RuntimeError("DPO sequence-parallel group membership disagrees")
        ordered_ranks: list[int | None] = [None] * size
        for member in members:
            slot = member["group_slot"]
            if ordered_ranks[slot] is not None:
                raise RuntimeError("DPO sequence-parallel group has duplicate rank slots")
            ordered_ranks[slot] = member["rank"]
        if _group_digest(tuple(ordered_ranks)) != group_hash:
            raise RuntimeError("DPO sequence-parallel group membership disagrees")
        if any(member["proof_digest"] != members[0]["proof_digest"] for member in members):
            raise RuntimeError("DPO sequence-parallel ranks disagree on packed inputs or provenance")


def next_dpo_microbatches(
    data_iterator: Iterator[Any],
    policy_state: Any,
    reference_state: Any,
    global_step: int,
    epoch: int = 0,
    source_count: Callable[[], int | None] | None = None,
    source_catalog: Callable[[], tuple[str, ...] | None] | None = None,
    vocab_size: Callable[[], int | None] | None = None,
    after_callbacks: bool = False,
    expected_proofs: tuple[DPOPackedProof, ...] | None = None,
) -> list[dict[str, Any]]:
    """Read and agree on proofs before callbacks or either model enters forward.

    This function is only called when explicitly enabled. All participating
    ranks must use the same opt-in setting. Hard process failures still rely on
    the distributed launcher's failure handling and collective timeout.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    batches = None
    step, epoch_value = 0, 0
    count, vocabulary, catalog_hash = None, None, None
    status, error = "ok", ""
    try:
        step, epoch_value = int(global_step), int(epoch)
        batches = next(data_iterator)
        if not isinstance(batches, list) or not batches or not all(isinstance(batch, dict) for batch in batches):
            raise ValueError("expected a non-empty list of DPO microbatch dictionaries")
        if after_callbacks and (
            expected_proofs is None
            or len(batches) != len(expected_proofs)
            or any(batch.get(DPO_APPROVED_KEY) != proof for batch, proof in zip(batches, expected_proofs))
        ):
            raise ValueError("DPO microbatch count or order changed during callbacks")
        policy_group = _members(policy_state, rank)
        reference_group = _members(reference_state, rank)
        if policy_group != reference_group:
            raise ValueError("DPO policy/reference sequence-parallel groups disagree")
        group_digest = _group_digest(policy_group)
        group_size = len(policy_group)
        group_slot = policy_group.index(rank)
        count = source_count() if source_count is not None else None
        catalog = source_catalog() if source_catalog is not None else None
        if catalog is not None:
            if type(catalog) is not tuple or not catalog or any(type(name) is not str for name in catalog):
                raise ValueError("DPO multisource catalog is invalid")
            if count is not None and count != len(catalog):
                raise ValueError("DPO source count disagrees with its catalog")
            count = len(catalog)
            catalog_hash = hashlib.sha256(repr(catalog).encode()).hexdigest()
        vocabulary = vocab_size() if vocab_size is not None else None
        proof_hash = hashlib.sha256()
        for batch in batches:
            if not after_callbacks and (DPO_APPROVED_KEY in batch or DPO_POSTCHECK_KEY in batch):
                raise ValueError("DPO microbatch contains a reserved approved-provenance key")
            proof = batch.get(DPO_PROOF_KEY)
            proof_data = _proof_data(proof)
            if isinstance(proof, DPOPackedProof):
                if proof.layout_sp_size != group_size or proof.layout_sp_rank != group_slot:
                    raise ValueError("DPO collator SP topology disagrees with the model process group")
                if after_callbacks:
                    if batch.get(DPO_APPROVED_KEY) != proof or DPO_POSTCHECK_KEY in batch:
                        raise ValueError("DPO approved evidence changed before callbacks completed")
                    if packed_batch_digest(batch, proof.source_keys, after_callbacks=True) != proof.forward_digest:
                        raise ValueError("DPO model input changed during a callback")
                    original_metadata = dict(proof.callback_metadata_digests)
                    if any(
                        original_metadata.get(key) != fingerprint
                        for key, fingerprint in callback_metadata_digests(batch, proof.source_keys)
                    ):
                        raise ValueError("DPO source metadata was added or changed during a callback")
                else:
                    _validate_batch(batch, proof, count, vocabulary, catalog)
            proof_hash.update(bytes.fromhex(proof_data))
        proof_digest = proof_hash.hexdigest()
        batch_count = len(batches)
    except StopIteration:
        status, group_digest, group_size, group_slot, proof_digest, batch_count = "eof", "", 0, 0, "", 0
    except Exception as exc:
        status, error, group_digest, group_size, group_slot, proof_digest, batch_count = (
            "error",
            f"{type(exc).__name__}: {exc}"[:512],
            "",
            0,
            0,
            "",
            0,
        )

    packet = {
        "version": _PROTOCOL_VERSION,
        "rank": rank,
        "step": step,
        "epoch": epoch_value,
        "phase": "after_callbacks" if after_callbacks else "before_callbacks",
        "source_count": count,
        "source_catalog_digest": catalog_hash,
        "vocab_size": vocabulary,
        "status": status,
        "error": error,
        "group_digest": group_digest,
        "group_size": group_size,
        "group_slot": group_slot,
        "proof_digest": proof_digest,
        "batch_count": batch_count,
    }
    if world_size > 1:
        packets: list[dict[str, Any]] = [None] * world_size
        dist.all_gather_object(packets, packet, group=dist.group.WORLD)
    else:
        packets = [packet]
    _validate_packets(packets)
    for batch in batches:
        proof = batch.get(DPO_PROOF_KEY)
        if isinstance(proof, DPOPackedProof):
            batch[DPO_POSTCHECK_KEY if after_callbacks else DPO_APPROVED_KEY] = proof
    return batches
