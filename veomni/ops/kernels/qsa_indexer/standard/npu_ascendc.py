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

"""Ascend CANN LightningIndexerV2 QSA block selection (``cann_ops_transformer``)."""

from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor


def _ops():
    from cann_ops_transformer import ops

    return ops


# Shape-keyed caches hold device memory, so they are bounded: variable-length
# training observes unbounded distinct shapes over a run. Recent shapes repeat
# across the QSA layers of a step, so the LRU working set stays hot and an
# eviction only costs one rebuild.
_CACHE_MAXSIZE = 128

# LightningIndexerV2 accepts topk in [1, 2048] and, above that, only the
# 1024-multiples up to 8192 (Atlas A2/A3); other widths are rejected host-side.
_TOPK_CONTIGUOUS_MAX = 2048
_TOPK_LARGE = (3072, 4096, 5120, 6144, 7168, 8192)


@lru_cache(maxsize=_CACHE_MAXSIZE)
def _uniform_weights(device: str, batch: int, seq_len: int, query_heads: int, head_dim: int) -> Tensor:
    # Uniform fp32 weights 1/sqrt(head_dim) reproduce the eager ReLU-sum score.
    return torch.full((batch, seq_len, query_heads), head_dim**-0.5, dtype=torch.float32, device=device)


@lru_cache(maxsize=_CACHE_MAXSIZE)
def _bsnd_plan(
    device: str,
    query_heads: int,
    head_dim: int,
    width: int,
    batch: int,
    seq_len: int,
    key_count: int,
    compress_ratio: int,
) -> tuple[object, Tensor | None]:
    # The BSND plan depends only on shapes (residual is seq_len % compress_ratio
    # repeated ``batch`` times), so it is cached across layers. Returns
    # ``(metadata, cmp_residual)``.
    residual = (
        None
        if compress_ratio == 1
        else torch.full((batch,), seq_len % compress_ratio, dtype=torch.int32, device=device)
    )
    metadata = _ops().lightning_indexer_metadata(
        query_heads,
        1,
        head_dim,
        width,
        cmp_residual_k=residual,
        batch_size=batch,
        max_seqlen_q=seq_len,
        max_seqlen_k=key_count,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=3,
        cmp_ratio=compress_ratio,
    )
    return metadata, residual


def _lightning_bsnd_call(
    queries: Tensor,
    keys: Tensor,
    width: int,
    compress_ratio: int,
    device,
) -> Tensor:
    """One BSND LightningIndexerV2 call; ``[rows, width]`` int32 block ids."""
    batch, seq_len, query_heads, head_dim = queries.shape
    key_count = keys.shape[1]
    if not (1 <= width <= _TOPK_CONTIGUOUS_MAX or width in _TOPK_LARGE):
        raise ValueError(
            f"QSA selection width {width} is outside the CANN LightningIndexerV2 topk set "
            "(1..2048, or a 1024-multiple up to 8192); use qsa_indexer_implementation='eager'."
        )
    weights = _uniform_weights(str(device), batch, seq_len, query_heads, head_dim)
    metadata, residual = _bsnd_plan(
        str(device), query_heads, head_dim, width, batch, seq_len, key_count, compress_ratio
    )
    indices, _ = _ops().lightning_indexer(
        queries.contiguous(),
        keys.contiguous(),
        weights,
        int(width),
        metadata=metadata,
        cmp_residual_k=residual,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=3,
        cmp_ratio=int(compress_ratio),
        return_value=0,
    )
    return indices.reshape(-1, width)


def qsa_lightning_selected_token_ids(
    index_queries: Tensor,
    pooled_keys: Tensor,
    segment_lens: list[int] | tuple[int, ...],
    *,
    compress_ratio: int,
    block_topk: int,
) -> Tensor:
    """Select top-k blocks per query and expand them to token lanes.

    ``index_queries`` is ``[B, S, H, D]`` (``B=1`` for a packed batch),
    ``pooled_keys`` is ``[1, blocks, D]``, and ``segment_lens`` lists the
    per-segment query lengths. Returns ``[B*S, width*cr]`` int32 token lanes
    in eager loop order — LOCAL ids, ``-1`` where a slot is unused, with
    ``width = min(block_topk, blocks)``.
    """
    batch, query_length, query_heads, head_dim = index_queries.shape
    flat_keys = pooled_keys.reshape(-1, head_dim)
    total_blocks = flat_keys.shape[0]
    segment_block_counts = [length // compress_ratio for length in segment_lens]
    if (
        index_queries.device.type != "npu"
        or index_queries.dtype not in (torch.float16, torch.bfloat16)
        or pooled_keys.dtype != index_queries.dtype
        or pooled_keys.device != index_queries.device
        or not 1 <= query_heads <= 64
        or head_dim != 128
        or compress_ratio not in (1, 4, 128)
        or block_topk < 1
        or sum(segment_block_counts) != total_blocks
    ):
        raise ValueError("QSA shape is unsupported by CANN LightningIndexerV2")

    total_tokens = batch * query_length
    width = min(block_topk, total_blocks)
    device = index_queries.device
    if width < 1:  # no complete block anywhere: the eager empty contract
        return torch.empty((total_tokens, 0), dtype=torch.int32, device=device)
    if min(segment_lens) == max(segment_lens):
        # Equal-length batch: one B=batch call.
        segment_count = len(segment_lens)
        segment_len = segment_lens[0]
        segment_blocks = segment_block_counts[0]
        call_width = min(block_topk, segment_blocks)
        indices = _lightning_bsnd_call(
            index_queries.reshape(segment_count, segment_len, query_heads, head_dim),
            flat_keys.reshape(segment_count, segment_blocks, 1, head_dim),
            call_width,
            compress_ratio,
            device,
        )
        if call_width < width:  # topk spans more blocks than one batch holds
            block_ids = torch.full((total_tokens, width), -1, dtype=torch.int32, device=device)
            block_ids[:, :call_width] = indices
        else:
            block_ids = indices
    else:
        # Packed flat batch with varying segment lengths: one B=1 call per segment.
        block_ids = torch.full((total_tokens, width), -1, dtype=torch.int32, device=device)
        token_start = 0
        block_start = 0
        for index, segment_len in enumerate(segment_lens):
            segment_blocks = segment_block_counts[index]
            call_width = min(block_topk, segment_blocks)
            if call_width:  # a segment with no complete block keeps its -1 rows
                indices = _lightning_bsnd_call(
                    index_queries[:, token_start : token_start + segment_len],
                    flat_keys[block_start : block_start + segment_blocks].reshape(1, segment_blocks, 1, head_dim),
                    call_width,
                    compress_ratio,
                    device,
                )
                block_ids[token_start : token_start + segment_len, :call_width] = indices
            token_start += segment_len
            block_start += segment_blocks
    # Expand each block id into its cr token lanes (eager loop order; -1 pad survives).
    # int32 throughout: NPU int64 elementwise is a slow path and this chain is [T, width, cr].
    valid = block_ids >= 0
    safe_ids = block_ids.clamp(min=0)
    block_offsets = torch.arange(compress_ratio, device=device, dtype=torch.int32)
    lanes = safe_ids[..., None] * compress_ratio + block_offsets
    return lanes.masked_fill(~valid[..., None], -1).flatten(-2)


def wrapper(
    q: Tensor,
    pooled_keys: Tensor,
    *,
    cu_seq_lens_q: Tensor | None = None,
    packed_seq_lens: list[int] | tuple[int, ...] | None = None,
    compress_ratio: int,
    block_topk: int,
) -> Tensor:
    """Select QSA blocks with the CANN Lightning Indexer.

    *q* is the rotated queries ``[B, S, H, D]`` (``B=1`` for a packed batch),
    *pooled_keys* the pooled blocks ``[1, blocks, D]``, and *cu_seq_lens_q* /
    *packed_seq_lens* the packed layout (both ``None`` means equal-length).
    Returns the pre-tail ``[B, S, K]`` int32 token lanes, ``-1`` where a slot
    is unused. Out-of-contract shapes and sequence parallelism raise; use
    ``qsa_indexer_implementation='eager'``.
    """
    from veomni.distributed.parallel_state import get_parallel_state

    parallel_state = get_parallel_state()
    if bool(getattr(parallel_state, "ulysses_enabled", False)) or bool(getattr(parallel_state, "cp_enabled", False)):
        raise RuntimeError(
            "qsa_indexer 'npu_ascendc' does not support sequence parallelism; "
            "use qsa_indexer_implementation='eager'"
        )
    batch_size, local_seq_len = q.shape[0], q.shape[1]

    if cu_seq_lens_q is None:
        selected = qsa_lightning_selected_token_ids(
            q,
            pooled_keys,
            [local_seq_len] * batch_size,
            compress_ratio=compress_ratio,
            block_topk=block_topk,
        )
        return selected.view(batch_size, local_seq_len, -1)

    if packed_seq_lens is not None:
        segment_lens = [int(value) for value in packed_seq_lens]
    else:
        boundaries = cu_seq_lens_q.tolist()
        segment_lens = [end - start for start, end in zip(boundaries, boundaries[1:])]
    if sum(segment_lens) != batch_size * local_seq_len:
        raise ValueError(
            "Qwen4-Exp compact QSA CANN indexer needs segment lengths that cover the batch; "
            f"got {segment_lens} for batch_size={batch_size}, local_seq_len={local_seq_len}."
        )
    if packed_seq_lens is not None and batch_size != 1:
        raise ValueError(
            "Qwen4-Exp compact QSA CANN indexer takes packed_seq_lens only with batch_size=1; "
            f"got batch_size={batch_size}."
        )
    if batch_size > 1 and min(segment_lens) != max(segment_lens):
        raise ValueError(
            "Qwen4-Exp compact QSA CANN indexer needs one shared segment length when batch_size>1 "
            f"(packed varlen requires batch_size=1); got {segment_lens} for batch_size={batch_size}."
        )
    selected = qsa_lightning_selected_token_ids(
        q,
        pooled_keys,
        segment_lens,
        compress_ratio=compress_ratio,
        block_topk=block_topk,
    )
    # Rebase LOCAL ids to flat; -1 slots must survive.
    cu_q_host = [0]
    for length in segment_lens:
        cu_q_host.append(cu_q_host[-1] + length)
    cu_seq_q = torch.tensor(cu_q_host, dtype=torch.int32, device=q.device)
    total_tokens = batch_size * local_seq_len
    token_segment = torch.searchsorted(
        cu_seq_q, torch.arange(total_tokens, device=q.device, dtype=cu_seq_q.dtype), right=True
    ) - 1
    segment_starts = cu_seq_q[:-1].index_select(0, token_segment).view(total_tokens, 1)
    selected = torch.where(selected >= 0, selected + segment_starts, selected)
    return selected.view(1, local_seq_len, -1)
