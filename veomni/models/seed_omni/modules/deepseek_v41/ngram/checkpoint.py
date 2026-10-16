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
from typing import Dict, List, Optional

import torch

from .....checkpoint_tensor_loading import ConvertedCheckpointTensor
from ..quant_scheme import quantized_storage_dtype
from .quantized import dequantize_rows


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


class NgramTableDequant(NgramTablePadding):
    """Released fp8 rows and their scales -> the padded bf16 tables training allocates.

    A trainable module cannot hold fp8 tables, so ``weight_quantization: null``
    allocates bf16 ones and the scales are spent here. Dequantizing a row reads
    only that row and its own scales, which is what ``dim0_rowwise_companion``
    declares: the streaming loader then reads a rank's slice of both and never
    holds the 196.6 B tables whole -- in bf16 that would be 393 GB per rank.

    A bf16 checkpoint carries no ``weight_scale``; its tables are only padded.
    """

    def __init__(self, padded_num_embeddings: List[int]) -> None:
        super().__init__(padded_num_embeddings)
        self._pending: Dict[int, Dict[str, torch.Tensor]] = {}

    def dim0_rowwise_companion(self, name: str) -> Optional[str]:
        match = _TABLE_TENSOR.match(name)
        if match is None or match.group(2) != "weight":
            return None
        return f"tables.{match.group('index')}.weight_scale"

    def convert_rows(self, name: str, rows: torch.Tensor, companion_rows: torch.Tensor) -> torch.Tensor:
        if rows.shape[0] == 0:
            return torch.empty(rows.shape, dtype=torch.bfloat16)
        return dequantize_rows(rows, companion_rows)

    def is_dim0_zero_pad(self, name: str) -> bool:
        """Only a bf16 checkpoint's table, which has no scale to pair with."""
        return self.can_handle(name) and name.endswith(".weight")

    def convert(self, name: str, tensor: torch.Tensor) -> Optional[ConvertedCheckpointTensor]:
        match = _TABLE_TENSOR.match(name)
        if match is None:
            return None
        index, part = int(match.group("index")), match.group(2)
        if part == "weight" and tensor.dtype != torch.float8_e4m3fn:
            return super().convert(name, tensor)
        pending = self._pending.setdefault(index, {})
        pending[part] = tensor
        if len(pending) < 2:
            return None
        del self._pending[index]
        weight = self.convert_rows(f"tables.{index}.weight", pending["weight"], pending["weight_scale"])
        return super().convert(f"tables.{index}.weight", weight)

    def finalize(self) -> List[ConvertedCheckpointTensor]:
        if self._pending:
            unpaired = {index: sorted(parts) for index, parts in self._pending.items()}
            raise ValueError(f"n-gram tables missing their weight or weight_scale: {unpaired}")
        return []


def create_ngram_table_converter(model) -> Optional[NgramTablePadding]:
    """The converter for a :class:`DeepseekV41Ngram`, or ``None`` if it needs none."""
    config = model.config
    if quantized_storage_dtype(config) is None:
        return NgramTableDequant(config.padded_num_embeddings)
    if list(config.padded_num_embeddings) == list(config.num_embeddings):
        return None
    return NgramTablePadding(config.padded_num_embeddings)


def padded_rows(rows: int, divisor: int) -> int:
    """``rows`` rounded up to a multiple of ``divisor``."""
    return math.ceil(rows / divisor) * divisor


__all__ = ["NgramTableDequant", "NgramTablePadding", "create_ngram_table_converter", "padded_rows"]
