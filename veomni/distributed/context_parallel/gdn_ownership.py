# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Validated chunk ownership and reversible routing for lossless GDN context parallelism.

The physical input layout is the per-sample zigzag layout produced by Ring-CP
sharding. Gated DeltaNet, however, must scan each 64-token native chunk on one
rank and pass recurrent state between consecutive owners. This module builds a
deterministic, host-side plan that moves valid tokens from the physical layout
to contiguous chunk owners and back. Ring padding is deliberately omitted from
the wire and restored as zeros by the inverse route.

The planner is hardware agnostic. It contains no collectives, accelerator
imports, environment-variable selectors, or experiment-specific identities.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any, Sequence


GDN_NATIVE_CHUNK_SIZE = 64
_PLAN_SCHEMA = "veomni.gdn_lossless_cp/v1"
_ALLOCATION_POLICY = "fill_low_rank_contiguous"


def _require_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer (bool is not allowed), got {type(value).__name__}")
    return int(value)


def _require_int_sequence(name: str, values: Sequence[Any]) -> tuple[int, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{name} must be a tuple or list, got {type(values).__name__}")
    return tuple(_require_int(f"{name}[{index}]", value) for index, value in enumerate(values))


def _ceil_to_multiple(value: int, multiple: int) -> int:
    if value < 0:
        raise ValueError(f"value must be non-negative, got {value}")
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return ((value + multiple - 1) // multiple) * multiple


@dataclass(frozen=True)
class GdnOwnedChunk:
    """One native GDN chunk in global valid-token coordinates."""

    sample_index: int
    chunk_index: int
    global_start: int
    global_end: int
    owner_rank: int

    @property
    def length(self) -> int:
        return self.global_end - self.global_start


@dataclass(frozen=True)
class GdnRankSample:
    """One sample's valid-token ownership on a context-parallel rank."""

    sample_index: int
    global_start: int
    global_end: int
    local_start: int
    local_end: int
    is_active: bool
    is_bos_owner: bool
    predecessor_rank: int | None
    successor_rank: int | None
    halo_source_rank: int | None

    @property
    def length(self) -> int:
        return self.local_end - self.local_start


@dataclass(frozen=True)
class GdnRouteSpan:
    """A contiguous valid-token route from one physical rank to one owner."""

    sample_index: int
    source_rank: int
    source_start: int
    source_end: int
    destination_rank: int
    destination_start: int
    destination_end: int
    global_start: int
    global_end: int

    @property
    def length(self) -> int:
        return self.global_end - self.global_start


@dataclass(frozen=True)
class GdnCopySpan:
    """A local copy used to pack or unpack an all-to-all buffer."""

    source_start: int
    destination_start: int
    length: int


@dataclass(frozen=True)
class GdnRankPlan:
    """Executable routing metadata for one context-parallel rank."""

    rank: int
    source_token_count: int
    source_cu_seqlens: tuple[int, ...]
    owned_token_count: int
    owned_cu_seqlens: tuple[int, ...]
    samples: tuple[GdnRankSample, ...]
    forward_input_splits: tuple[int, ...]
    forward_output_splits: tuple[int, ...]
    inverse_input_splits: tuple[int, ...]
    inverse_output_splits: tuple[int, ...]
    forward_pack_spans: tuple[GdnCopySpan, ...]
    forward_unpack_spans: tuple[GdnCopySpan, ...]
    inverse_pack_spans: tuple[GdnCopySpan, ...]
    inverse_unpack_spans: tuple[GdnCopySpan, ...]

    @property
    def predecessor_rank(self) -> int | None:
        return _unique_peer(self.samples, "predecessor_rank")

    @property
    def successor_rank(self) -> int | None:
        return _unique_peer(self.samples, "successor_rank")

    @property
    def halo_source_rank(self) -> int | None:
        return _unique_peer(self.samples, "halo_source_rank")

    @property
    def bos_owner_flags(self) -> tuple[bool, ...]:
        return tuple(sample.is_bos_owner for sample in self.samples)

    @property
    def active_flags(self) -> tuple[bool, ...]:
        return tuple(sample.is_active for sample in self.samples)


@dataclass(frozen=True)
class GdnLosslessPlan:
    """Canonical ownership and route plan shared by every CP rank."""

    schema: str
    chunk_size: int
    cp_size: int
    ulysses_size: int
    allocation_policy: str
    valid_lengths: tuple[int, ...]
    ring_physical_lengths: tuple[int, ...]
    chunks: tuple[GdnOwnedChunk, ...]
    routes: tuple[GdnRouteSpan, ...]
    ranks: tuple[GdnRankPlan, ...]
    ownership_hash: str
    route_hash: str
    plan_hash: str

    def rank_plan(self, rank: int) -> GdnRankPlan:
        rank = _require_int("rank", rank)
        if not 0 <= rank < self.cp_size:
            raise ValueError(f"rank {rank} is outside [0, {self.cp_size})")
        return self.ranks[rank]


def _unique_peer(samples: Sequence[GdnRankSample], field_name: str) -> int | None:
    peers = {
        getattr(sample, field_name)
        for sample in samples
        if sample.is_active and getattr(sample, field_name) is not None
    }
    if not peers:
        return None
    if len(peers) != 1:
        raise ValueError(f"packed samples require one {field_name}, got {sorted(peers)}")
    return int(next(iter(peers)))


def _chunk_owners(num_chunks: int, cp_size: int) -> list[int]:
    base, remainder = divmod(num_chunks, cp_size)
    owners: list[int] = []
    for rank in range(cp_size):
        owners.extend([rank] * (base + (1 if rank < remainder else 0)))
    if len(owners) != num_chunks:
        raise RuntimeError("chunk allocation did not cover every chunk")
    return owners


def _hash_payload(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _ownership_payload(
    *,
    chunk_size: int,
    cp_size: int,
    valid_lengths: Sequence[int],
    chunks: Sequence[GdnOwnedChunk],
    rank_samples: Sequence[Sequence[GdnRankSample]],
) -> dict[str, Any]:
    return {
        "allocation_policy": _ALLOCATION_POLICY,
        "chunk_size": chunk_size,
        "cp_size": cp_size,
        "valid_lengths": list(valid_lengths),
        "chunks": [asdict(chunk) for chunk in chunks],
        "rank_samples": [[asdict(sample) for sample in samples] for samples in rank_samples],
    }


def _route_payload(
    *,
    cp_size: int,
    ulysses_size: int,
    ring_lengths: Sequence[int],
    routes: Sequence[GdnRouteSpan],
    ranks: Sequence[GdnRankPlan],
) -> dict[str, Any]:
    return {
        "cp_size": cp_size,
        "ulysses_size": ulysses_size,
        "ring_physical_lengths": list(ring_lengths),
        "routes": [asdict(route) for route in routes],
        "ranks": [
            {
                "rank": rank.rank,
                "source_token_count": rank.source_token_count,
                "source_cu_seqlens": rank.source_cu_seqlens,
                "owned_token_count": rank.owned_token_count,
                "owned_cu_seqlens": rank.owned_cu_seqlens,
                "forward_input_splits": rank.forward_input_splits,
                "forward_output_splits": rank.forward_output_splits,
                "inverse_input_splits": rank.inverse_input_splits,
                "inverse_output_splits": rank.inverse_output_splits,
                "forward_pack_spans": [asdict(span) for span in rank.forward_pack_spans],
                "forward_unpack_spans": [asdict(span) for span in rank.forward_unpack_spans],
                "inverse_pack_spans": [asdict(span) for span in rank.inverse_pack_spans],
                "inverse_unpack_spans": [asdict(span) for span in rank.inverse_unpack_spans],
            }
            for rank in ranks
        ],
    }


def _build_chunks_and_samples(
    valid_lengths: tuple[int, ...], cp_size: int, chunk_size: int
) -> tuple[list[GdnOwnedChunk], list[list[GdnRankSample]]]:
    chunks: list[GdnOwnedChunk] = []
    owned_ranges: list[list[tuple[int, int] | None]] = [[None] * cp_size for _ in valid_lengths]

    for sample_index, valid_length in enumerate(valid_lengths):
        num_chunks = (valid_length + chunk_size - 1) // chunk_size
        owners = _chunk_owners(num_chunks, cp_size)
        for chunk_index, owner_rank in enumerate(owners):
            start = chunk_index * chunk_size
            end = min(start + chunk_size, valid_length)
            chunks.append(GdnOwnedChunk(sample_index, chunk_index, start, end, owner_rank))
            previous = owned_ranges[sample_index][owner_rank]
            owned_ranges[sample_index][owner_rank] = (start, end) if previous is None else (previous[0], end)

    rank_samples: list[list[GdnRankSample]] = [[] for _ in range(cp_size)]
    local_cursors = [0] * cp_size
    for sample_index in range(len(valid_lengths)):
        active_ranks = [rank for rank in range(cp_size) if owned_ranges[sample_index][rank] is not None]
        for rank in range(cp_size):
            owned_range = owned_ranges[sample_index][rank]
            active = owned_range is not None
            global_start, global_end = owned_range if owned_range is not None else (0, 0)
            local_start = local_cursors[rank]
            local_end = local_start + global_end - global_start
            local_cursors[rank] = local_end
            active_index = active_ranks.index(rank) if active else -1
            predecessor = active_ranks[active_index - 1] if active_index > 0 else None
            successor = active_ranks[active_index + 1] if 0 <= active_index < len(active_ranks) - 1 else None
            rank_samples[rank].append(
                GdnRankSample(
                    sample_index=sample_index,
                    global_start=global_start,
                    global_end=global_end,
                    local_start=local_start,
                    local_end=local_end,
                    is_active=active,
                    is_bos_owner=active_index == 0,
                    predecessor_rank=predecessor,
                    successor_rank=successor,
                    halo_source_rank=predecessor,
                )
            )
    return chunks, rank_samples


def _physical_intervals(
    rank: int, ring_length: int, sample_offset: int, cp_size: int
) -> tuple[tuple[int, int, int], ...]:
    if ring_length == 0:
        return ()
    half = ring_length // (2 * cp_size)
    first_global = rank * half
    second_global = (2 * cp_size - 1 - rank) * half
    return (
        (first_global, first_global + half, sample_offset),
        (second_global, second_global + half, sample_offset + half),
    )


def _build_routes(
    *,
    valid_lengths: tuple[int, ...],
    ring_lengths: tuple[int, ...],
    rank_samples: Sequence[Sequence[GdnRankSample]],
    cp_size: int,
) -> list[GdnRouteSpan]:
    source_cursors = [0] * cp_size
    routes: list[GdnRouteSpan] = []
    for sample_index, (valid_length, ring_length) in enumerate(zip(valid_lengths, ring_lengths)):
        for source_rank in range(cp_size):
            intervals = _physical_intervals(source_rank, ring_length, source_cursors[source_rank], cp_size)
            for global_lo, global_hi, local_lo in intervals:
                valid_hi = min(global_hi, valid_length)
                if valid_hi <= global_lo:
                    continue
                for destination_rank in range(cp_size):
                    sample = rank_samples[destination_rank][sample_index]
                    if not sample.is_active:
                        continue
                    start = max(global_lo, sample.global_start)
                    end = min(valid_hi, sample.global_end)
                    if start >= end:
                        continue
                    routes.append(
                        GdnRouteSpan(
                            sample_index=sample_index,
                            source_rank=source_rank,
                            source_start=local_lo + start - global_lo,
                            source_end=local_lo + end - global_lo,
                            destination_rank=destination_rank,
                            destination_start=sample.local_start + start - sample.global_start,
                            destination_end=sample.local_start + end - sample.global_start,
                            global_start=start,
                            global_end=end,
                        )
                    )
            source_cursors[source_rank] += ring_length // cp_size
    return sorted(
        routes,
        key=lambda route: (
            route.sample_index,
            route.global_start,
            route.source_rank,
            route.destination_rank,
        ),
    )


def _route_count_matrix(routes: Sequence[GdnRouteSpan], cp_size: int) -> list[list[int]]:
    counts = [[0] * cp_size for _ in range(cp_size)]
    for route in routes:
        counts[route.source_rank][route.destination_rank] += route.length
    return counts


def _assert_contiguous_destination(spans: Sequence[GdnCopySpan], total: int, label: str) -> None:
    cursor = 0
    for span in sorted(spans, key=lambda item: item.destination_start):
        if span.length <= 0 or span.destination_start != cursor:
            raise ValueError(f"{label} does not form a contiguous destination at {cursor}")
        cursor += span.length
    if cursor != total:
        raise ValueError(f"{label} covers {cursor} rows, expected {total}")


def _assert_contiguous_source(spans: Sequence[GdnCopySpan], total: int, label: str) -> None:
    cursor = 0
    for span in sorted(spans, key=lambda item: item.source_start):
        if span.length <= 0 or span.source_start != cursor:
            raise ValueError(f"{label} does not form a contiguous source at {cursor}")
        cursor += span.length
    if cursor != total:
        raise ValueError(f"{label} covers {cursor} rows, expected {total}")


def _build_rank_plan(
    *,
    rank: int,
    cp_size: int,
    ring_lengths: tuple[int, ...],
    samples: tuple[GdnRankSample, ...],
    routes: Sequence[GdnRouteSpan],
    count_matrix: Sequence[Sequence[int]],
) -> GdnRankPlan:
    source_lengths = [ring_length // cp_size for ring_length in ring_lengths]
    source_cu = [0]
    owned_cu = [0]
    for source_length, sample in zip(source_lengths, samples):
        source_cu.append(source_cu[-1] + source_length)
        owned_cu.append(owned_cu[-1] + sample.length)

    forward_input = tuple(count_matrix[rank])
    forward_output = tuple(count_matrix[source][rank] for source in range(cp_size))
    inverse_input = forward_output
    inverse_output = forward_input

    forward_order = sorted(
        routes,
        key=lambda item: (
            item.source_rank,
            item.destination_rank,
            item.sample_index,
            item.source_start,
        ),
    )
    inverse_order = sorted(
        routes,
        key=lambda item: (
            item.destination_rank,
            item.source_rank,
            item.sample_index,
            item.destination_start,
        ),
    )

    forward_pack: list[GdnCopySpan] = []
    cursor = 0
    for route in forward_order:
        if route.source_rank == rank:
            forward_pack.append(GdnCopySpan(route.source_start, cursor, route.length))
            cursor += route.length

    forward_unpack: list[GdnCopySpan] = []
    receive_cursor = 0
    for source_rank in range(cp_size):
        peer_cursor = 0
        for route in forward_order:
            if route.source_rank == source_rank and route.destination_rank == rank:
                forward_unpack.append(GdnCopySpan(receive_cursor + peer_cursor, route.destination_start, route.length))
                peer_cursor += route.length
        receive_cursor += peer_cursor

    inverse_pack: list[GdnCopySpan] = []
    cursor = 0
    for route in inverse_order:
        if route.destination_rank == rank:
            inverse_pack.append(GdnCopySpan(route.destination_start, cursor, route.length))
            cursor += route.length

    inverse_unpack: list[GdnCopySpan] = []
    receive_cursor = 0
    for forward_destination in range(cp_size):
        peer_cursor = 0
        for route in inverse_order:
            if route.source_rank == rank and route.destination_rank == forward_destination:
                inverse_unpack.append(GdnCopySpan(receive_cursor + peer_cursor, route.source_start, route.length))
                peer_cursor += route.length
        receive_cursor += peer_cursor

    return GdnRankPlan(
        rank=rank,
        source_token_count=source_cu[-1],
        source_cu_seqlens=tuple(source_cu),
        owned_token_count=owned_cu[-1],
        owned_cu_seqlens=tuple(owned_cu),
        samples=samples,
        forward_input_splits=forward_input,
        forward_output_splits=forward_output,
        inverse_input_splits=inverse_input,
        inverse_output_splits=inverse_output,
        forward_pack_spans=tuple(forward_pack),
        forward_unpack_spans=tuple(forward_unpack),
        inverse_pack_spans=tuple(inverse_pack),
        inverse_unpack_spans=tuple(inverse_unpack),
    )


def _validate_copy_spans(rank: GdnRankPlan) -> None:
    _assert_contiguous_destination(rank.forward_pack_spans, sum(rank.forward_input_splits), "forward pack")
    _assert_contiguous_destination(rank.forward_unpack_spans, rank.owned_token_count, "forward unpack")
    _assert_contiguous_destination(rank.inverse_pack_spans, sum(rank.inverse_input_splits), "inverse pack")
    _assert_contiguous_source(rank.inverse_unpack_spans, sum(rank.inverse_output_splits), "inverse unpack")
    previous_end = 0
    for span in sorted(rank.inverse_unpack_spans, key=lambda item: item.destination_start):
        if span.destination_start < previous_end or span.destination_start + span.length > rank.source_token_count:
            raise ValueError(f"inverse route overlaps or exceeds physical rows on rank {rank.rank}")
        previous_end = span.destination_start + span.length


def _validate_rank_types(rank_index: int, rank: GdnRankPlan, sample_count: int, cp_size: int) -> None:
    _require_int(f"ranks[{rank_index}].source_token_count", rank.source_token_count)
    _require_int(f"ranks[{rank_index}].owned_token_count", rank.owned_token_count)
    _require_int_sequence(f"ranks[{rank_index}].source_cu_seqlens", rank.source_cu_seqlens)
    _require_int_sequence(f"ranks[{rank_index}].owned_cu_seqlens", rank.owned_cu_seqlens)
    for field_name in (
        "forward_input_splits",
        "forward_output_splits",
        "inverse_input_splits",
        "inverse_output_splits",
    ):
        values = _require_int_sequence(f"ranks[{rank_index}].{field_name}", getattr(rank, field_name))
        if len(values) != cp_size or any(value < 0 for value in values):
            raise ValueError(f"rank {rank_index} {field_name} must contain {cp_size} non-negative values")
    for sample_index, sample in enumerate(rank.samples):
        for field_name in (
            "sample_index",
            "global_start",
            "global_end",
            "local_start",
            "local_end",
        ):
            _require_int(f"ranks[{rank_index}].samples[{sample_index}].{field_name}", getattr(sample, field_name))
        for field_name in ("is_active", "is_bos_owner"):
            if not isinstance(getattr(sample, field_name), bool):
                raise TypeError(f"ranks[{rank_index}].samples[{sample_index}].{field_name} must be bool")
        for field_name in ("predecessor_rank", "successor_rank", "halo_source_rank"):
            peer = getattr(sample, field_name)
            if peer is not None:
                peer = _require_int(f"ranks[{rank_index}].samples[{sample_index}].{field_name}", peer)
                if not 0 <= peer < cp_size:
                    raise ValueError(f"rank {rank_index} sample {sample_index} has an out-of-range {field_name}")
    if len(rank.samples) != sample_count:
        raise ValueError(f"rank {rank_index} has the wrong number of sample views")
    for field_name in (
        "forward_pack_spans",
        "forward_unpack_spans",
        "inverse_pack_spans",
        "inverse_unpack_spans",
    ):
        for span_index, span in enumerate(getattr(rank, field_name)):
            for coordinate in ("source_start", "destination_start", "length"):
                value = _require_int(
                    f"ranks[{rank_index}].{field_name}[{span_index}].{coordinate}",
                    getattr(span, coordinate),
                )
                if value < 0 or (coordinate == "length" and value == 0):
                    raise ValueError(f"rank {rank_index} {field_name}[{span_index}] has an invalid {coordinate}")


def _physical_coordinate(global_index: int, ring_length: int, cp_size: int) -> tuple[int, int]:
    half = ring_length // (2 * cp_size)
    chunk_index, chunk_offset = divmod(global_index, half)
    if chunk_index < cp_size:
        return chunk_index, chunk_offset
    rank = 2 * cp_size - 1 - chunk_index
    return rank, half + chunk_offset


def validate_gdn_lossless_plan(plan: GdnLosslessPlan) -> None:
    """Fail closed on types, coverage, topology, route geometry, and hashes."""
    if plan.schema != _PLAN_SCHEMA:
        raise ValueError(f"unsupported GDN lossless plan schema {plan.schema!r}")
    chunk_size = _require_int("chunk_size", plan.chunk_size)
    cp_size = _require_int("cp_size", plan.cp_size)
    ulysses_size = _require_int("ulysses_size", plan.ulysses_size)
    if chunk_size != GDN_NATIVE_CHUNK_SIZE:
        raise ValueError(f"chunk_size must be {GDN_NATIVE_CHUNK_SIZE}, got {chunk_size}")
    if cp_size < 1 or ulysses_size < 1:
        raise ValueError("cp_size and ulysses_size must be positive")
    if plan.allocation_policy != _ALLOCATION_POLICY:
        raise ValueError(f"unsupported allocation policy {plan.allocation_policy!r}")
    valid_lengths = _require_int_sequence("valid_lengths", plan.valid_lengths)
    ring_lengths = _require_int_sequence("ring_physical_lengths", plan.ring_physical_lengths)
    if len(valid_lengths) != len(ring_lengths):
        raise ValueError("valid and ring length vectors must have the same size")
    ring_divisor = 2 * cp_size * ulysses_size
    for index, (valid, ring) in enumerate(zip(valid_lengths, ring_lengths)):
        if valid < 0 or ring != _ceil_to_multiple(valid, ring_divisor):
            raise ValueError(f"sample {index} has invalid valid/ring lengths {valid}/{ring}")

    if len(plan.ranks) != cp_size:
        raise ValueError(f"rank plan count {len(plan.ranks)} does not match cp_size {cp_size}")
    for rank_index, rank in enumerate(plan.ranks):
        if _require_int(f"ranks[{rank_index}].rank", rank.rank) != rank_index:
            raise ValueError("rank plans must be ordered by rank")
        _validate_rank_types(rank_index, rank, len(valid_lengths), cp_size)
        if len(rank.source_cu_seqlens) != len(valid_lengths) + 1:
            raise ValueError(f"rank {rank_index} has invalid source CU metadata")
        if len(rank.owned_cu_seqlens) != len(valid_lengths) + 1:
            raise ValueError(f"rank {rank_index} has invalid owned CU metadata")
        if rank.source_cu_seqlens[0] != 0 or rank.source_cu_seqlens[-1] != rank.source_token_count:
            raise ValueError(f"rank {rank_index} source CU does not bind its token count")
        if rank.owned_cu_seqlens[0] != 0 or rank.owned_cu_seqlens[-1] != rank.owned_token_count:
            raise ValueError(f"rank {rank_index} owned CU does not bind its token count")
        for sample_index, sample in enumerate(rank.samples):
            if sample.sample_index != sample_index:
                raise ValueError(f"rank {rank_index} sample ordinals are not canonical")
            if (sample.local_start, sample.local_end) != (
                rank.owned_cu_seqlens[sample_index],
                rank.owned_cu_seqlens[sample_index + 1],
            ):
                raise ValueError(f"rank {rank_index} sample {sample_index} is not bound to owned CU")
            if sample.length != sample.global_end - sample.global_start:
                raise ValueError(f"rank {rank_index} sample {sample_index} local/global lengths differ")
            if sample.is_active != (sample.length > 0):
                raise ValueError(f"rank {rank_index} sample {sample_index} active flag is inconsistent")
            if sample.halo_source_rank != sample.predecessor_rank:
                raise ValueError(f"rank {rank_index} sample {sample_index} halo predecessor differs")
        _validate_copy_spans(rank)

    expected_chunks: list[GdnOwnedChunk] = []
    for sample_index, valid_length in enumerate(valid_lengths):
        num_chunks = (valid_length + chunk_size - 1) // chunk_size
        owners = _chunk_owners(num_chunks, cp_size)
        for chunk_index, owner in enumerate(owners):
            expected_chunks.append(
                GdnOwnedChunk(
                    sample_index,
                    chunk_index,
                    chunk_index * chunk_size,
                    min((chunk_index + 1) * chunk_size, valid_length),
                    owner,
                )
            )
    for chunk_index, chunk in enumerate(plan.chunks):
        for field_name in ("sample_index", "chunk_index", "global_start", "global_end", "owner_rank"):
            _require_int(f"chunks[{chunk_index}].{field_name}", getattr(chunk, field_name))
    if tuple(expected_chunks) != plan.chunks:
        raise ValueError("owned chunks do not match the canonical allocation")

    for sample_index, valid_length in enumerate(valid_lengths):
        active = [rank.samples[sample_index] for rank in plan.ranks if rank.samples[sample_index].is_active]
        if valid_length == 0:
            if active:
                raise ValueError(f"empty sample {sample_index} has an active owner")
            continue
        if not active or active[0].global_start != 0 or active[-1].global_end != valid_length:
            raise ValueError(f"sample {sample_index} ownership does not cover its valid range")
        for index, sample in enumerate(active):
            if index and active[index - 1].global_end != sample.global_start:
                raise ValueError(f"sample {sample_index} ownership has a gap or overlap")
            owner_rank = next(rank.rank for rank in plan.ranks if rank.samples[sample_index] is sample)
            expected_predecessor = (
                None
                if index == 0
                else next(rank.rank for rank in plan.ranks if rank.samples[sample_index] is active[index - 1])
            )
            expected_successor = (
                None
                if index + 1 == len(active)
                else next(rank.rank for rank in plan.ranks if rank.samples[sample_index] is active[index + 1])
            )
            if sample.is_bos_owner != (index == 0):
                raise ValueError(f"sample {sample_index} has an invalid BOS owner on rank {owner_rank}")
            if sample.predecessor_rank != expected_predecessor or sample.successor_rank != expected_successor:
                raise ValueError(f"sample {sample_index} has an invalid state chain on rank {owner_rank}")

    count_matrix = _route_count_matrix(plan.routes, cp_size)
    for rank in plan.ranks:
        expected_input = tuple(count_matrix[rank.rank])
        expected_output = tuple(count_matrix[source][rank.rank] for source in range(cp_size))
        if rank.forward_input_splits != expected_input or rank.forward_output_splits != expected_output:
            raise ValueError(f"rank {rank.rank} forward split tables do not match routes")
        if rank.inverse_input_splits != expected_output or rank.inverse_output_splits != expected_input:
            raise ValueError(f"rank {rank.rank} inverse split tables are not the forward transpose")

    for route_index, route in enumerate(plan.routes):
        for field_name in (
            "sample_index",
            "source_rank",
            "source_start",
            "source_end",
            "destination_rank",
            "destination_start",
            "destination_end",
            "global_start",
            "global_end",
        ):
            _require_int(f"routes[{route_index}].{field_name}", getattr(route, field_name))
        if route.length <= 0:
            raise ValueError(f"route {route_index} has non-positive length")
        if route.source_end - route.source_start != route.length:
            raise ValueError(f"route {route_index} source length differs")
        if route.destination_end - route.destination_start != route.length:
            raise ValueError(f"route {route_index} destination length differs")
        if not 0 <= route.source_rank < cp_size or not 0 <= route.destination_rank < cp_size:
            raise ValueError(f"route {route_index} has an out-of-range rank")
        if not 0 <= route.sample_index < len(valid_lengths):
            raise ValueError(f"route {route_index} has an out-of-range sample")
        if not 0 <= route.global_start < route.global_end <= valid_lengths[route.sample_index]:
            raise ValueError(f"route {route_index} is outside the sample's valid range")
        sample = plan.ranks[route.destination_rank].samples[route.sample_index]
        if not (sample.global_start <= route.global_start < route.global_end <= sample.global_end):
            raise ValueError(f"route {route_index} is outside destination ownership")
        expected_destination = sample.local_start + route.global_start - sample.global_start
        if route.destination_start != expected_destination:
            raise ValueError(f"route {route_index} has an invalid destination coordinate")
        ring_length = ring_lengths[route.sample_index]
        source_sample_start = plan.ranks[route.source_rank].source_cu_seqlens[route.sample_index]
        for global_coordinate, source_coordinate in (
            (route.global_start, route.source_start),
            (route.global_end - 1, route.source_end - 1),
        ):
            expected_rank, expected_local = _physical_coordinate(global_coordinate, ring_length, cp_size)
            if expected_rank != route.source_rank or source_sample_start + expected_local != source_coordinate:
                raise ValueError(f"route {route_index} has an invalid physical source coordinate")

    for sample_index, valid_length in enumerate(valid_lengths):
        intervals = sorted(
            (route.global_start, route.global_end) for route in plan.routes if route.sample_index == sample_index
        )
        cursor = 0
        for start, end in intervals:
            if start != cursor:
                raise ValueError(f"sample {sample_index} routes have a gap or overlap at {cursor}")
            cursor = end
        if cursor != valid_length:
            raise ValueError(f"sample {sample_index} routes cover {cursor}, expected {valid_length}")

    rank_samples = [rank.samples for rank in plan.ranks]
    ownership_hash = _hash_payload(
        _ownership_payload(
            chunk_size=chunk_size,
            cp_size=cp_size,
            valid_lengths=valid_lengths,
            chunks=plan.chunks,
            rank_samples=rank_samples,
        )
    )
    route_hash = _hash_payload(
        _route_payload(
            cp_size=cp_size,
            ulysses_size=ulysses_size,
            ring_lengths=ring_lengths,
            routes=plan.routes,
            ranks=plan.ranks,
        )
    )
    if plan.ownership_hash != ownership_hash or plan.route_hash != route_hash:
        raise ValueError("GDN ownership or route hash does not match the plan contents")
    plan_hash = _hash_payload(
        {
            "schema": plan.schema,
            "ownership_hash": ownership_hash,
            "route_hash": route_hash,
        }
    )
    if plan.plan_hash != plan_hash:
        raise ValueError("GDN lossless plan hash does not match the plan contents")


def build_gdn_lossless_plan(
    valid_lengths: Sequence[int],
    *,
    cp_size: int,
    ulysses_size: int = 1,
    chunk_size: int = GDN_NATIVE_CHUNK_SIZE,
) -> GdnLosslessPlan:
    """Build a canonical lossless GDN ownership and route plan.

    ``valid_lengths`` are the unpadded global packed-sample lengths. The input
    physical layout may include per-sample Ring padding to
    ``2 * cp_size * ulysses_size``; padding is dropped before the all-to-all.
    """
    lengths = _require_int_sequence("valid_lengths", valid_lengths)
    cp_size = _require_int("cp_size", cp_size)
    ulysses_size = _require_int("ulysses_size", ulysses_size)
    chunk_size = _require_int("chunk_size", chunk_size)
    if cp_size < 1 or ulysses_size < 1:
        raise ValueError("cp_size and ulysses_size must be positive")
    if chunk_size != GDN_NATIVE_CHUNK_SIZE:
        raise ValueError(f"chunk_size must be {GDN_NATIVE_CHUNK_SIZE}, got {chunk_size}")
    if any(length < 0 for length in lengths):
        raise ValueError(f"valid_lengths must be non-negative, got {lengths}")

    ring_lengths = tuple(_ceil_to_multiple(length, 2 * cp_size * ulysses_size) for length in lengths)
    chunks, rank_sample_lists = _build_chunks_and_samples(lengths, cp_size, chunk_size)
    routes = _build_routes(
        valid_lengths=lengths,
        ring_lengths=ring_lengths,
        rank_samples=rank_sample_lists,
        cp_size=cp_size,
    )
    count_matrix = _route_count_matrix(routes, cp_size)
    ranks = tuple(
        _build_rank_plan(
            rank=rank,
            cp_size=cp_size,
            ring_lengths=ring_lengths,
            samples=tuple(rank_sample_lists[rank]),
            routes=routes,
            count_matrix=count_matrix,
        )
        for rank in range(cp_size)
    )
    ownership_hash = _hash_payload(
        _ownership_payload(
            chunk_size=chunk_size,
            cp_size=cp_size,
            valid_lengths=lengths,
            chunks=chunks,
            rank_samples=rank_sample_lists,
        )
    )
    route_hash = _hash_payload(
        _route_payload(
            cp_size=cp_size,
            ulysses_size=ulysses_size,
            ring_lengths=ring_lengths,
            routes=routes,
            ranks=ranks,
        )
    )
    plan_hash = _hash_payload(
        {
            "schema": _PLAN_SCHEMA,
            "ownership_hash": ownership_hash,
            "route_hash": route_hash,
        }
    )
    plan = GdnLosslessPlan(
        schema=_PLAN_SCHEMA,
        chunk_size=chunk_size,
        cp_size=cp_size,
        ulysses_size=ulysses_size,
        allocation_policy=_ALLOCATION_POLICY,
        valid_lengths=lengths,
        ring_physical_lengths=ring_lengths,
        chunks=tuple(chunks),
        routes=tuple(routes),
        ranks=ranks,
        ownership_hash=ownership_hash,
        route_hash=route_hash,
        plan_hash=plan_hash,
    )
    validate_gdn_lossless_plan(plan)
    return plan


__all__ = [
    "GDN_NATIVE_CHUNK_SIZE",
    "GdnCopySpan",
    "GdnLosslessPlan",
    "GdnOwnedChunk",
    "GdnRankPlan",
    "GdnRankSample",
    "GdnRouteSpan",
    "build_gdn_lossless_plan",
    "validate_gdn_lossless_plan",
]
