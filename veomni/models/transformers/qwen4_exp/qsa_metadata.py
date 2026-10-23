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


@functools.lru_cache(maxsize=2)
def cached_host_metadata(
    segments: tuple[tuple[int, int, int], ...], compress_ratio: int, local_seq_len: int, rank: int, world_size: int
):
    """Cache immutable integer layouts, including all packed-boundary and rank inputs."""
    # Preserve exact owner-major block ordering. Only Python integers and
    # immutable tuples survive the call; GPU allocations are unchanged.
    blocks = tuple(
        (segment_id, batch_idx, start)
        for segment_id, (batch_idx, segment_start, segment_end) in enumerate(segments)
        for start in range(segment_start, segment_end - compress_ratio + 1, compress_ratio)
    )
    owned_blocks = tuple(block for block in blocks if block[2] // local_seq_len == rank)
    gathered_blocks = tuple(
        block for owner in range(world_size) for block in blocks if block[2] // local_seq_len == owner
    )
    meta_sections = (
        tuple(sum(start // local_seq_len == owner for _, _, start in blocks) for owner in range(world_size)),
        tuple(batch_idx for _, batch_idx, _ in owned_blocks),
        tuple(start for _, _, start in owned_blocks),
        tuple(segment_id for segment_id, _, _ in gathered_blocks),
        tuple(start for _, _, start in gathered_blocks),
    )
    return blocks, owned_blocks, gathered_blocks, meta_sections
