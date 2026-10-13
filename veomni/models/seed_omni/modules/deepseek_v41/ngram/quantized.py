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

"""The n-gram tables, holding the released fp8 rows instead of bf16 ones.

These two tables are 196.6B of the model's parameters, so this is the single
biggest thing fp8 buys: 196 GB rather than 393 GB, which is what makes room for
everything else on eight H20s.

A table is quantized differently from a projection, and for a reason. A dense
weight's rows share an output space, so they can share a scale over a square
tile; a table's rows are unrelated memories, so each row carries its own eight
scales -- one per 32 columns of the 256-wide row.

Nothing is dequantized in bulk. A lookup gathers the handful of rows it needs
and dequantizes those, so the fp8 table stays fp8 in memory for the whole run.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ......distributed.emb_parallel import AllToAllEmbedding, ShardedEmbedding
from ......distributed.parallel_state import get_parallel_state
from ..dequant import dequant_fp8_blockwise
from ..quant_scheme import QUANT_BLOCK_SIZE, SCALE_DTYPE


def dequantize_rows(rows: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Gathered fp8 rows plus their block scales -> bf16, for any leading shape.

    ``dequant_fp8_blockwise`` reads the block shape off a 2-D weight and its
    2-D scale, and a lookup's result is ``[batch, positions, columns]``. The
    leading dimensions are folded away and restored rather than special-cased,
    which also keeps ``[n_ids, columns]`` working for the parallel path.
    """
    columns = rows.shape[-1]
    dequantized = dequant_fp8_blockwise(rows.reshape(-1, columns), scales.reshape(-1, scales.shape[-1]))
    return dequantized.view(*rows.shape[:-1], columns)


class DeepseekV41NgramTable(ShardedEmbedding):
    """One n-gram memory table, optionally over the released fp8 rows.

    ``quant_dtype=None`` is an ordinary :class:`ShardedEmbedding`. The converted
    checkpoint keeps ``weight_quantization``, so both train and infer stay on
    the fp8 tables.

    Quantized, the weight is built through ``_weight`` rather than by letting
    ``nn.Embedding`` allocate and initialize one. That is not a micro
    optimization: these tables are 384M rows, so the default path would
    allocate 393 GB of fp32 and then try to fill it from a normal distribution
    before any of it could be replaced.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        quant_dtype: Optional[torch.dtype] = None,
        device=None,
    ) -> None:
        if quant_dtype is None:
            super().__init__(num_embeddings, embedding_dim, device=device)
            return

        if embedding_dim % QUANT_BLOCK_SIZE:
            raise ValueError(
                f"A {embedding_dim}-wide row is not a whole number of {QUANT_BLOCK_SIZE}-wide scale "
                "blocks, so its scales would not line up with its columns."
            )
        super().__init__(
            num_embeddings,
            embedding_dim,
            _weight=torch.empty(num_embeddings, embedding_dim, dtype=quant_dtype, device=device),
            _freeze=True,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim // QUANT_BLOCK_SIZE,
                dtype=SCALE_DTYPE,
                device=device,
            ),
            requires_grad=False,
        )

    @property
    def is_quantized(self) -> bool:
        return getattr(self, "weight_scale", None) is not None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Gather ``ids``, vocab-parallel when the plan split this table over ``emb``.

        A quantized table is two tensors, the fp8 rows and their block scales.
        Both are dispatched over the same group with the same ids, so each lands
        on the rank owning that row, and the pair is reassembled in input order
        before being dequantized locally. Shipping fp8 rows and scales rather
        than dequantizing first is also the cheaper exchange -- 257 bytes a row
        against the 512 a bf16 row would cost.
        """
        if not self.is_quantized:
            return super().forward(ids)
        if not self._is_split():
            return dequantize_rows(F.embedding(ids, self.weight), F.embedding(ids, self.weight_scale))
        self._check_own_fsdp_unit()
        group = get_parallel_state().extra_parallel_group("emb")
        # int8 for transport: the collectives are byte moves either way, and fp8
        # dtypes are not something every backend will accept.
        rows = AllToAllEmbedding.apply(group, ids, self.weight.view(torch.int8)).view(torch.float8_e4m3fn)
        scales = AllToAllEmbedding.apply(group, ids, self.weight_scale.view(torch.int8)).view(SCALE_DTYPE)
        return dequantize_rows(rows, scales)


__all__ = ["DeepseekV41NgramTable", "dequantize_rows"]
