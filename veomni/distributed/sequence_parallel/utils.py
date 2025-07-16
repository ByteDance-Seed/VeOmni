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


from typing import List, Tuple

import torch
from torch import Tensor
from torch.distributed import ProcessGroup

from .comm import (
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_rank,
    get_ulysses_sequence_parallel_world_size,
)


def unpadding_tensor_for_seqeunce_parallel(x: Tensor, dim: int, unpadded_dim_size: int, group: ProcessGroup = None):
    """
    A func to remove the padding part of the tensor based on its original shape
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return x
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    if unpadded_dim_size % sp_world == 0:
        return x
    padding_size = sp_world - (unpadded_dim_size % sp_world)
    assert (padding_size + unpadded_dim_size) % sp_world == 0
    return unpad_tensor(x, dim=dim, padding_size=padding_size)


def padding_tensor_for_seqeunce_parallel(x: Tensor, dim: int, group: ProcessGroup = None) -> Tensor:
    """
    A func to remove the padding part of the tensor based on its original shape
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return x
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    dim_size = x.shape[dim]
    if dim_size % sp_world:
        padding_size = sp_world - (dim_size % sp_world)
        x = pad_tensor(x, dim, padding_size)
    return x


def pad_tensor(x: Tensor, dim: int, padding_size: int, padding_value: int = 0) -> Tensor:
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.full(shape, padding_value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


def unpad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(0, -padding_size)
    return x[slc]


def remove_last_rank_padding(x: Tensor, dim: int, unpad_dim_size: int, group: ProcessGroup = None) -> Tensor:
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return x
    sp_rank = get_ulysses_sequence_parallel_rank(group)
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    if unpad_dim_size % sp_world == 0 and sp_rank + 1 != sp_world:
        return x
    pad = sp_world - (unpad_dim_size % sp_world)
    assert (pad + x.shape[dim]) % sp_world == 0
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(0, -pad)
    return x[slc]


def has_overlap(x1, x2, y1, y2) -> Tuple[bool, int]:
    """
    A func to judge if two intervals have overlaps, and return the length of overlaps
    """
    max_value = max(x1, y1)
    min_value = min(x2, y2)
    return max_value < min_value, min_value - max_value


def all2all_splits(image_lens: List, image_lens_per_rank: List, sp_size: int, sp_rank: int) -> Tuple[List, List]:
    """
    A func to generate splits for all2all communication
    """
    assert sum(image_lens) == sum(image_lens_per_rank)
    num_images = len(image_lens)
    sp_step = (num_images + sp_size - 1) // sp_size
    in_splits, out_splits = [0 for _ in range(sp_size)], [0 for _ in range(sp_size)]
    cu_seqlens = [0] + [sum(image_lens_per_rank[: i + 1]) for i in range(sp_size)]
    rank = 0
    num_tokens = 0
    for image_idx, image_lens in enumerate(image_lens):
        src_rank = image_idx // sp_step
        tokens_split = []
        for rank in range(sp_size):
            overlap, overlap_len = has_overlap(
                num_tokens, num_tokens + image_lens, cu_seqlens[rank], cu_seqlens[rank + 1]
            )
            if overlap:
                tokens_split.append(overlap_len)
                if rank == sp_rank:
                    out_splits[src_rank] += overlap_len
                if src_rank == sp_rank:
                    in_splits[rank] += overlap_len
        assert sum(tokens_split) == image_lens

        num_tokens += image_lens

    return in_splits, out_splits


def vlm_images_a2a_meta(
    sp_rank: int, sp_size: int, image_lens: List, image_masks: torch.Tensor
) -> Tuple[List, List, torch.Tensor]:
    """
    A func to generate metadata for all2all communication after we balance the computaion in vision encoder
    Usually we will split the batches of images for vision encoder in sp group. However, before we feed images
    tokens into language model, we need to use all2all communication to gather necessary tokens into the current rank.
    """
    assert sum(image_lens) == image_masks.sum().item(), (
        f"The sum of image_lens must be equal to the number of tokens, {image_lens} vs {image_masks.sum().item()}"
    )
    seq_len = image_masks.shape[1]
    step = (seq_len + sp_size - 1) // sp_size
    sequence_per_rank = [min(step * (i + 1), seq_len) - min(step * i, seq_len) for i in range(sp_size)]
    mask_per_rank = image_masks.split(sequence_per_rank, dim=1)
    image_lens_per_rank = [mask_per_rank[i].sum().item() for i in range(sp_size)]
    in_splits, out_splits = all2all_splits(image_lens, image_lens_per_rank, sp_size, sp_rank)
    local_image_masks = mask_per_rank[sp_rank]
    return in_splits, out_splits, local_image_masks
