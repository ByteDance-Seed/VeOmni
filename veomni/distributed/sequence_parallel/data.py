from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup

from ...data.constants import IGNORE_INDEX
from .comm import get_ulysses_sequence_parallel_group, get_unified_sequence_parallel_group
from .ulysses import _Gather, _Slice
from .utils import pad_tensor, unpadding_tensor_for_seqeunce_parallel


def slice_input_tensor(
    x: Tensor,
    dim: int,
    padding: bool = True,
    padding_value: int = 0,
    group: ProcessGroup = None,
) -> Tensor:
    """
    A func to slice the input sequence in sequence parallel
    """
    group = get_unified_sequence_parallel_group() if group is None else group
    if not group:
        return x
    sp_rank = dist.get_rank(group)
    sp_world = dist.get_world_size(group)
    dim_size = x.shape[dim]
    unit = (dim_size + sp_world - 1) // sp_world
    if padding and dim_size % sp_world:
        padding_size = sp_world - (dim_size % sp_world)
        x = pad_tensor(x, dim, padding_size, padding_value)
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(unit * sp_rank, unit * (sp_rank + 1))
    return x[slc].contiguous()


def slice_input_tensor_scale_grad(
    x: Tensor,
    dim: int,
    group: ProcessGroup = None,
    scale_grad=True,
):
    """
    A func to gather the outputs for the model result in sequence parallel
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return x
    x = _Slice.apply(group, x, dim, scale_grad)
    return x


def gather_outputs(
    x: Tensor,
    gather_dim: int,
    padding_dim: Optional[int] = None,
    unpad_dim_size: Optional[int] = None,
    scale_grad=True,
    group: ProcessGroup = None,
):
    """
    A func to gather the outputs for the model result in sequence parallel
    """
    group = get_unified_sequence_parallel_group() if group is None else group
    if not group:
        return x
    x = _Gather.apply(group, x, gather_dim, scale_grad)
    if padding_dim is not None:
        x = unpadding_tensor_for_seqeunce_parallel(x, padding_dim, unpad_dim_size, group)
    return x


def slice_position_embedding(position_embeddings: tuple, dim: int = 1, sp_group: dist.ProcessGroup = None):
    """
    Forward hook for LlamaRotaryEmbedding to apply Ulysses tensor slicing.

    Args:
        position_embeddings: Input tensors to the forward method
        dim: The dimension to slice
        sp_group: The sequence parallel group
    Returns:
        Modified (cos, sin) tuple with slicing applied if ulysses is enabled
    """
    if sp_group is not None:
        cos, sin = position_embeddings
        cos = slice_input_tensor(cos, dim=dim, padding=False, group=sp_group)
        sin = slice_input_tensor(sin, dim=dim, padding=False, group=sp_group)
        return (cos, sin)
    return position_embeddings


def sequence_parallel_preprocess(
    input_ids: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    sp_group: Optional[ProcessGroup] = None,
):
    """
    Preprocess input_ids and labels for sequence parallel training.

    Args:
        input_ids: Input token ids
        labels: Label token ids
        position_ids: Position ids
        attention_mask: Attention mask
        cu_seqlens: Cumulative sequence lengths

    Returns:
        Preprocessed input_ids, labels, position_ids, attention_mask, cu_seqlens
    """
    if sp_group is not None:
        sp_size = dist.get_world_size(sp_group)
        padding_size = (sp_size - (input_ids.shape[-1] % sp_size)) % sp_size

        # Slice input_ids among sequence parallel group
        input_ids = slice_input_tensor(input_ids, dim=-1, padding=True, padding_value=0, group=sp_group)

        # Slice labels among sequence parallel group
        if labels is not None:
            labels = labels[..., 1:].contiguous()  # shift labels
            labels = F.pad(labels, (0, 1), "constant", IGNORE_INDEX)  # pad to the same length as input_ids
            labels = slice_input_tensor(labels, dim=-1, padding=True, padding_value=IGNORE_INDEX, group=sp_group)

        # Padding position_ids
        if position_ids is not None:
            position_ids = pad_tensor(position_ids, dim=-1, padding_size=padding_size, padding_value=0)

        # Padding attention_mask
        if attention_mask is not None:
            attn_mask_padding_value = 1 if position_ids is not None else 0
            attention_mask = pad_tensor(
                attention_mask, dim=-1, padding_size=padding_size, padding_value=attn_mask_padding_value
            )

        # Padding cu_seqlens
        if cu_seqlens is not None:
            cu_seqlens_padding_value = cu_seqlens[-1].item() + padding_size
            cu_seqlens = pad_tensor(
                cu_seqlens, dim=-1, padding_size=padding_size, padding_value=cu_seqlens_padding_value
            )

    return input_ids, labels, position_ids, attention_mask, cu_seqlens
