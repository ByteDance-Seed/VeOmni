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
"""Bounded host-only layout reuse for the compact QSA indexer."""

import functools
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HostQSAMetadata:
    counts: tuple[int, ...]
    flat_metadata: tuple[int, ...]
    section_sizes: tuple[int, ...]


@dataclass(frozen=True)
class PreparedQSAMetadata:
    """Integer tensors owned by one model forward, including its recomputation."""

    layout: tuple[int, int, int, int, int]
    counts: tuple[int, ...]
    local_batch_ids: torch.Tensor
    local_block_starts: torch.Tensor
    block_segment_ids: torch.Tensor
    block_starts: torch.Tensor
    local_segment_ids: torch.Tensor
    local_segment_starts: torch.Tensor


@functools.lru_cache(maxsize=2)
def cached_host_metadata(
    segments: tuple[tuple[int, int, int], ...], compress_ratio: int, local_seq_len: int, rank: int, world_size: int
):
    """Cache immutable integer layouts, including all packed-boundary and rank inputs."""
    # Bucket once, preserving the previous owner-major and within-owner order.
    # Only Python integers and immutable tuples survive in this bounded cache.
    owned = [[] for _ in range(world_size)]
    for segment_id, (batch_idx, segment_start, segment_end) in enumerate(segments):
        for start in range(segment_start, segment_end - compress_ratio + 1, compress_ratio):
            owned[start // local_seq_len].append((segment_id, batch_idx, start))
    owned_blocks = owned[rank]
    gathered_blocks = tuple(block for bucket in owned for block in bucket)
    sections = (
        tuple(batch_idx for _, batch_idx, _ in owned_blocks),
        tuple(start for _, _, start in owned_blocks),
        tuple(segment_id for segment_id, _, _ in gathered_blocks),
        tuple(start for _, _, start in gathered_blocks),
    )
    return HostQSAMetadata(
        tuple(len(bucket) for bucket in owned),
        tuple(value for section in sections for value in section),
        tuple(len(section) for section in sections),
    )


def prepare_qsa_metadata(
    batch_size: int,
    local_seq_len: int,
    compress_ratio: int,
    rank: int,
    world_size: int,
    device: torch.device,
    cu_seq_lens_q: torch.Tensor | None = None,
    packed_seq_lens: list[int] | tuple[int, ...] | None = None,
) -> PreparedQSAMetadata:
    """Prepare rank-local metadata once; direct indexer calls use the same path."""
    global_seq_len = local_seq_len * world_size
    if cu_seq_lens_q is None:
        segments = tuple((batch_idx, 0, global_seq_len) for batch_idx in range(batch_size))
    else:
        if packed_seq_lens is None:
            boundaries = cu_seq_lens_q.tolist()
        else:
            boundaries = [0]
            for length in packed_seq_lens:
                boundaries.append(boundaries[-1] + length)
        if not boundaries or boundaries[0] != 0 or boundaries[-1] != batch_size * global_seq_len:
            raise ValueError(
                "Qwen4-Exp compact QSA requires cu_seq_lens_q to cover the complete padded batch; "
                f"got endpoints {boundaries[:1]}..{boundaries[-1:]}, expected 0..{batch_size * global_seq_len}."
            )
        segments = []
        for flat_start, flat_end in zip(boundaries, boundaries[1:]):
            if flat_end <= flat_start:
                raise ValueError("Qwen4-Exp compact QSA requires strictly increasing cu_seq_lens_q.")
            batch_idx = flat_start // global_seq_len
            if (flat_end - 1) // global_seq_len != batch_idx:
                raise ValueError("Qwen4-Exp compact QSA packed samples may not cross the batch dimension.")
            segments.append((batch_idx, flat_start % global_seq_len, flat_end - batch_idx * global_seq_len))
        segments = tuple(segments)

    host = cached_host_metadata(segments, compress_ratio, local_seq_len, rank, world_size)
    flat = torch.tensor(host.flat_metadata, dtype=torch.long, device=device)
    local_batch_ids, local_block_starts, block_segment_ids, block_starts = flat.split(host.section_sizes)
    local_segment_ids = torch.empty((batch_size, local_seq_len), dtype=torch.long, device=device)
    local_segment_starts = torch.empty_like(local_segment_ids)
    local_start = rank * local_seq_len
    for segment_id, (batch_idx, start, end) in enumerate(segments):
        begin = max(start, local_start) - local_start
        finish = min(end, local_start + local_seq_len) - local_start
        if begin < finish:
            local_segment_ids[batch_idx, begin:finish] = segment_id
            local_segment_starts[batch_idx, begin:finish] = start
    return PreparedQSAMetadata(
        (batch_size, local_seq_len, compress_ratio, rank, world_size),
        host.counts,
        local_batch_ids,
        local_block_starts,
        block_segment_ids,
        block_starts,
        local_segment_ids,
        local_segment_starts,
    )
