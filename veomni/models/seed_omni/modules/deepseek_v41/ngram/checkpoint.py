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

"""Padding the n-gram tables into their shardable row counts, at load.

A table's row count is a sum of the drawn primes -- 384006168 and 384016682 in
the release, the second not divisible even by 8 -- and a row-sharded table needs
a count its group size divides. The module therefore allocates a little more
than the checkpoint holds, and the gap is zero-filled here as the tensors stream
in, rather than baked into the artifact by the offline converter.

Doing it at load is what keeps the 91.6 GiB tables from ever being held whole.
Appending zero rows on dim 0 commutes with a ``Shard(0)`` split, which is what
``is_dim0_zero_pad`` declares: the streaming loader then reads only a rank's own
slice of the real rows and zero-fills whatever tail falls past them.
"""

import math
import re
from typing import List, Optional

import torch

from .....checkpoint_tensor_loading import ConvertedCheckpointTensor


#: ``tables.0.weight``, ``tables.1.weight_scale`` -- the row-sharded tensors and
#: nothing else. ``token_map`` is a per-vocabulary lookup, not a table, and is
#: replicated rather than sharded.
_TABLE_TENSOR = re.compile(r"^tables\.(?P<index>\d+)\.(weight|weight_scale)$")


class NgramTablePadding:
    """Zero-pads the two tables' dim 0 up to the counts the module allocated."""

    def __init__(self, padded_num_embeddings: List[int]) -> None:
        self.padded_num_embeddings = list(padded_num_embeddings)

    def can_handle(self, name: str) -> bool:
        return _TABLE_TENSOR.match(name) is not None

    def convert(self, name: str, tensor: torch.Tensor) -> Optional[ConvertedCheckpointTensor]:
        match = _TABLE_TENSOR.match(name)
        if match is None:
            return None

        index = int(match.group("index"))
        if index >= len(self.padded_num_embeddings):
            raise ValueError(
                f"{name} is table {index}, but this config declares only {len(self.padded_num_embeddings)} tables."
            )

        padded_rows = self.padded_num_embeddings[index]
        rows = tensor.shape[0]
        if rows > padded_rows:
            raise ValueError(
                f"{name} has {rows} rows, more than the {padded_rows} the module allocated. The config's "
                "`num_embeddings` describes a different table than this checkpoint holds, and every lookup "
                "would read a row belonging to another bucket."
            )
        if rows < padded_rows:
            padding = tensor.new_zeros((padded_rows - rows, *tensor.shape[1:]))
            tensor = torch.cat((tensor, padding), dim=0)
        return ConvertedCheckpointTensor(name=name, tensor=tensor)

    def is_dim0_zero_pad(self, name: str) -> bool:
        """True for both tensors: appending zero rows is the whole transform.

        Safe to declare because the padded rows cannot be addressed. A hash id
        is ``rolling % prime + offset`` and the offsets sum to the real row
        count, so every id lands below it -- unlike a padded MoE expert, which
        the router would happily select.
        """
        return self.can_handle(name)

    def finalize(self) -> List[ConvertedCheckpointTensor]:
        """Nothing is buffered: each tensor is padded and emitted on arrival."""
        return []


def create_ngram_table_padding(model) -> Optional[NgramTablePadding]:
    """The converter for a :class:`DeepseekV41Ngram`, or ``None`` if it needs none."""
    config = model.config
    if list(config.padded_num_embeddings) == list(config.num_embeddings):
        return None
    return NgramTablePadding(config.padded_num_embeddings)


def padded_rows(rows: int, divisor: int) -> int:
    """``rows`` rounded up to a multiple of ``divisor``."""
    return math.ceil(rows / divisor) * divisor


__all__ = ["NgramTablePadding", "create_ngram_table_padding", "padded_rows"]
