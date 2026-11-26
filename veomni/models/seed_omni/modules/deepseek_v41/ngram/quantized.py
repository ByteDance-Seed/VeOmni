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
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ......distributed.emb_parallel import AllToAllEmbedding, ShardedEmbedding
from ......distributed.parallel_state import get_parallel_state
from ..dequant import dequant_fp8_blockwise
from ..quant_scheme import QUANT_BLOCK_SIZE, SCALE_DTYPE


_DEFAULT_ALL_TO_ALL_APPLY = AllToAllEmbedding.apply


def _quantized_all_to_all_lookup(
    group: Optional[dist.ProcessGroup],
    ids: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch ``ids`` once over ``group`` and return both ``(rows, scales)`` on ``ids.device``."""
    if AllToAllEmbedding.apply is not _DEFAULT_ALL_TO_ALL_APPLY:
        rows = AllToAllEmbedding.apply(group, ids, weight.view(torch.int8)).view(torch.float8_e4m3fn)
        scales = AllToAllEmbedding.apply(group, ids, weight_scale.view(torch.int8)).view(SCALE_DTYPE)
        return rows, scales

    emb_size = 1 if group is None else dist.get_world_size(group)
    emb_rank = 0 if group is None else dist.get_rank(group)
    sharded = emb_size > 1
    vocab_size_per_rank, row_dim = weight.shape
    scale_dim = weight_scale.shape[1]
    packed_dim = row_dim + scale_dim

    raw_shape = ids.shape
    input_flat = ids.reshape(-1)
    num_input_ids = input_flat.shape[0]
    vocab_size = vocab_size_per_rank * emb_size

    in_range = (input_flat >= 0) & (input_flat < vocab_size)
    id_rank = torch.where(in_range, input_flat // vocab_size_per_rank, 0)
    full_rank_index = torch.argsort(id_rank, stable=True)
    send_counts = torch.zeros(emb_size, dtype=torch.int64, device=id_rank.device)
    send_counts.scatter_add_(0, id_rank.to(torch.int64), torch.ones_like(id_rank, dtype=torch.int64))
    send_meta = torch.stack([send_counts, (~in_range).any().long().expand(emb_size)], dim=1)

    if sharded:
        recv_meta = torch.empty_like(send_meta)
        dist.all_to_all_single(recv_meta, send_meta, group=group)
    else:
        recv_meta = send_meta
    send_meta_list, recv_meta_list = torch.stack([send_meta, recv_meta]).tolist()
    if any(bad for _, bad in recv_meta_list):
        bad_ids = input_flat[~in_range]
        span = f"[{int(bad_ids.min())}, {int(bad_ids.max())}]" if bad_ids.numel() > 0 else "all in range"
        raise RuntimeError(
            f"AllToAllEmbedding: token ids must lie in [0, {vocab_size}); "
            f"some rank of the emb group is out of range (this rank's offending ids: {span})."
        )
    send_rank_count_list = [count for count, _ in send_meta_list]
    receive_rank_count_list = [count for count, _ in recv_meta_list]

    send_ids = input_flat[full_rank_index].contiguous()
    if sharded:
        recv_ids = torch.empty(sum(receive_rank_count_list), dtype=input_flat.dtype, device=input_flat.device)
        dist.all_to_all_single(
            recv_ids,
            send_ids,
            output_split_sizes=receive_rank_count_list,
            input_split_sizes=send_rank_count_list,
            group=group,
        )
    else:
        recv_ids = send_ids

    local_indices = recv_ids - emb_rank * vocab_size_per_rank
    if weight.device != local_indices.device:
        local_indices = local_indices.to(weight.device)
    rows_local = F.embedding(local_indices, weight.view(torch.int8))
    scales_local = F.embedding(local_indices, weight_scale.view(torch.int8))
    packed_local = torch.cat([rows_local, scales_local], dim=-1)
    if packed_local.device != input_flat.device:
        packed_local = packed_local.to(input_flat.device, non_blocking=True)

    if sharded:
        packed_recv = torch.empty(num_input_ids, packed_dim, dtype=torch.int8, device=input_flat.device)
        dist.all_to_all_single(
            packed_recv,
            packed_local.contiguous(),
            output_split_sizes=send_rank_count_list,
            input_split_sizes=receive_rank_count_list,
            group=group,
        )
    else:
        packed_recv = packed_local

    output = torch.empty(num_input_ids, packed_dim, dtype=torch.int8, device=input_flat.device)
    if num_input_ids > 0:
        output[full_rank_index] = packed_recv
    rows = output[:, :row_dim].contiguous().view(torch.float8_e4m3fn).view(*raw_shape, row_dim)
    scales = output[:, row_dim:].contiguous().view(SCALE_DTYPE).view(*raw_shape, scale_dim)
    return rows, scales


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
            if self.weight.device != ids.device:
                ids_local = ids.to(self.weight.device)
                rows = F.embedding(ids_local, self.weight).to(ids.device, non_blocking=True)
                scales = F.embedding(ids_local, self.weight_scale).to(ids.device, non_blocking=True)
                return dequantize_rows(rows, scales)
            return dequantize_rows(F.embedding(ids, self.weight), F.embedding(ids, self.weight_scale))
        self._check_own_fsdp_unit()
        group = get_parallel_state().extra_parallel_group("emb")
        rows, scales = _quantized_all_to_all_lookup(group, ids, self.weight, self.weight_scale)
        return dequantize_rows(rows, scales)


__all__ = ["DeepseekV41NgramTable", "dequantize_rows"]
