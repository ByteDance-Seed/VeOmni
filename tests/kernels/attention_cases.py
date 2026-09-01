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
# See the License for the specific language governing limitations
# under the License.

"""Shared attention mask cases and the MATH SDPA numerical reference."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from veomni.kernels.mask import MagiAttentionMask


_BAGEL_MODES = ("causal", "noise", "full", "causal")


def bagel_span_splits(sequence_length: int) -> list[int]:
    if sequence_length % 4 != 0:
        raise ValueError("bagel mixed visibility requires a sequence length divisible by 4")
    quarter = sequence_length // 4
    return [quarter, quarter, quarter, sequence_length - 3 * quarter]


def bagel_dense_mask(sequence_length: int, device: torch.device | str) -> torch.Tensor:
    visible = torch.zeros((sequence_length, sequence_length), device=device, dtype=torch.bool)
    clean_spans: list[tuple[int, int]] = []
    span_start = 0
    for length, mode in zip(bagel_span_splits(sequence_length), _BAGEL_MODES, strict=True):
        span_end = span_start + length
        for clean_start, clean_end in clean_spans:
            visible[span_start:span_end, clean_start:clean_end] = True
        if mode == "causal":
            visible[span_start:span_end, span_start:span_end].fill_(True).tril_()
        else:
            visible[span_start:span_end, span_start:span_end] = True
        if mode != "noise":
            clean_spans.append((span_start, span_end))
        span_start = span_end
    return visible.unsqueeze(0).unsqueeze(0).contiguous()


def bagel_magi_mask(sequence_length: int, device: torch.device | str) -> MagiAttentionMask:
    q_ranges: list[list[int]] = []
    k_ranges: list[list[int]] = []
    attn_types: list[int] = []
    clean_spans: list[tuple[int, int]] = []
    span_start = 0
    for length, mode in zip(bagel_span_splits(sequence_length), _BAGEL_MODES, strict=True):
        span_end = span_start + length
        for clean_start, clean_end in clean_spans:
            q_ranges.append([span_start, span_end])
            k_ranges.append([clean_start, clean_end])
            attn_types.append(0)
        q_ranges.append([span_start, span_end])
        k_ranges.append([span_start, span_end])
        attn_types.append(1 if mode == "causal" else 0)
        if mode != "noise":
            clean_spans.append((span_start, span_end))
        span_start = span_end
    return MagiAttentionMask.from_ranges(
        torch.tensor(q_ranges, device=device, dtype=torch.int32),
        torch.tensor(k_ranges, device=device, dtype=torch.int32),
        torch.tensor(attn_types, device=device, dtype=torch.int32),
    )


def bagel_flex_mask(sequence_length: int, device: torch.device | str) -> BlockMask:
    quarter = sequence_length // 4

    def mask_mod(batch_idx, head_idx, query_idx, key_idx):
        first = (query_idx < quarter) & (key_idx <= query_idx)
        noise = (query_idx >= quarter) & (query_idx < 2 * quarter) & (key_idx < 2 * quarter)
        full = (
            (query_idx >= 2 * quarter)
            & (query_idx < 3 * quarter)
            & ((key_idx < quarter) | ((key_idx >= 2 * quarter) & (key_idx < 3 * quarter)))
        )
        last = (query_idx >= 3 * quarter) & (
            (key_idx < quarter)
            | ((key_idx >= 2 * quarter) & (key_idx < 3 * quarter))
            | ((key_idx >= 3 * quarter) & (key_idx <= query_idx))
        )
        return first | noise | full | last

    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=device,
        BLOCK_SIZE=128,
    )


def dense_mask(mask_case: str, sequence_length: int, device: torch.device | str) -> torch.Tensor:
    if mask_case == "causal":
        return torch.ones((1, 1, sequence_length, sequence_length), device=device, dtype=torch.bool).tril_()
    if mask_case == "full":
        return torch.ones((1, 1, sequence_length, sequence_length), device=device, dtype=torch.bool)
    if mask_case == "bagel_mixed":
        return bagel_dense_mask(sequence_length, device)
    raise ValueError(f"unsupported mask case: {mask_case}")


def magi_mask(mask_case: str, sequence_length: int, device: torch.device | str) -> MagiAttentionMask:
    if mask_case == "causal":
        return MagiAttentionMask.from_ranges(
            torch.tensor([[0, sequence_length]], device=device),
            torch.tensor([[0, sequence_length]], device=device),
            torch.tensor([1], device=device),
        )
    if mask_case == "full":
        return MagiAttentionMask.from_ranges(
            torch.tensor([[0, sequence_length]], device=device),
            torch.tensor([[0, sequence_length]], device=device),
        )
    if mask_case == "bagel_mixed":
        return bagel_magi_mask(sequence_length, device)
    raise ValueError(f"unsupported mask case: {mask_case}")


def flex_mask(mask_case: str, sequence_length: int, device: torch.device | str) -> BlockMask:
    if mask_case == "causal":
        return create_block_mask(
            lambda batch_idx, head_idx, query_idx, key_idx: query_idx >= key_idx,
            B=None,
            H=None,
            Q_LEN=sequence_length,
            KV_LEN=sequence_length,
            device=device,
            BLOCK_SIZE=128,
        )
    if mask_case == "bagel_mixed":
        return bagel_flex_mask(sequence_length, device)
    raise ValueError(f"unsupported flex mask case: {mask_case}")


def flex_visible(block_mask: BlockMask, q_len: int, kv_len: int) -> torch.Tensor:
    query_idx = torch.arange(q_len)
    key_idx = torch.arange(kv_len)
    return block_mask.mask_mod(0, 0, query_idx[:, None], key_idx[None, :])


def materialize_magi_mask(attention_mask: MagiAttentionMask, sequence_length: int) -> torch.Tensor:
    visible = torch.zeros((sequence_length, sequence_length), dtype=torch.bool)
    attn_types = (
        torch.zeros(attention_mask.q_ranges.shape[0], dtype=torch.int32)
        if attention_mask.attn_type_map is None
        else attention_mask.attn_type_map.cpu()
    )
    for q_range, k_range, attn_type in zip(
        attention_mask.q_ranges.cpu(),
        attention_mask.k_ranges.cpu(),
        attn_types,
        strict=True,
    ):
        q_start, q_end = q_range.tolist()
        k_start, k_end = k_range.tolist()
        slice_mask = torch.ones((q_end - q_start, k_end - k_start), dtype=torch.bool)
        if int(attn_type) == 1:
            slice_mask.tril_()
        visible[q_start:q_end, k_start:k_end] |= slice_mask
    return visible.unsqueeze(0).unsqueeze(0)


def math_sdpa_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dense: torch.Tensor,
    *,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=dense,
            dropout_p=0.0,
            scale=scaling,
            enable_gqa=True,
        ).transpose(1, 2)

    repeat_count = query.shape[1] // key.shape[1]
    expanded_key = key.repeat_interleave(repeat_count, dim=1)
    logits = torch.einsum("bhqd,bhkd->bhqk", query.float(), expanded_key.float()) * scaling
    lse = torch.logsumexp(logits.masked_fill(~dense, -torch.inf), dim=-1)
    return output, lse


def clone_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        query.detach().clone().requires_grad_(True),
        key.detach().clone().requires_grad_(True),
        value.detach().clone().requires_grad_(True),
    )
