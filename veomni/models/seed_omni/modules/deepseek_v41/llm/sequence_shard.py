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

"""One rank's rows of a packed sequence split across the sequence-parallel group.

V4.1 attention is multi-query: 64 query heads read one 512-wide KV head. So
sequence parallelism keeps everything query-side on the rank's own contiguous
rows -- the projections, the indexer, the sparse attention and its teacher,
the MoE, mHC and Engram -- and gathers only the KV side to the full row: the
window KV, and the compressor projections whose groups pool across it. That
moves one KV head where a Ulysses head all-to-all would move 64 query heads
twice, and every row reads the same number of keys, so equal row counts are
equal work.

The compression plans, the window indices and the compressed-history
visibility are built from the full ``position_ids`` held here, exactly as an
unsharded forward builds them; each consumer then narrows to its rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from ......distributed.sequence_parallel import gather_outputs


@dataclass(frozen=True)
class SequenceShard:
    """Rows ``[start, end)`` of a row of ``full_len`` tokens, ``valid_len`` of them real.

    ``position_ids`` is the full ``[B, full_len]`` row. ``full_len`` is a
    multiple of ``size``; the tail past ``valid_len`` is padding, laid out as
    one extra packed sample so no real token can attend to it.
    """

    group: dist.ProcessGroup
    rank: int
    size: int
    position_ids: torch.Tensor
    valid_len: int

    def __post_init__(self) -> None:
        full_len = self.position_ids.shape[1]
        if full_len % self.size:
            raise ValueError(f"a {full_len}-token row does not split into {self.size} equal shards; pad it first")
        if not 0 < self.valid_len <= full_len:
            raise ValueError(f"valid_len={self.valid_len} is outside the {full_len}-token row")

    @property
    def full_len(self) -> int:
        return int(self.position_ids.shape[1])

    @property
    def local_len(self) -> int:
        return self.full_len // self.size

    @property
    def start(self) -> int:
        return self.rank * self.local_len

    @property
    def end(self) -> int:
        return self.start + self.local_len

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, local_len, ...]`` -> ``[B, full_len, ...]``.

        The backward sums every rank's gradient before slicing, because each
        rank's queries read a different part of the gathered row.
        """
        return gather_outputs(x, gather_dim=1, group=self.group)

    def local_rows(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's rows of a full-row tensor, along dim 1."""
        return x.narrow(1, self.start, self.local_len)

    def valid_rows(self, device: torch.device) -> torch.Tensor:
        """``[local_len]`` bool: which of this rank's rows are real tokens."""
        return torch.arange(self.start, self.end, device=device) < self.valid_len


__all__ = ["SequenceShard"]
