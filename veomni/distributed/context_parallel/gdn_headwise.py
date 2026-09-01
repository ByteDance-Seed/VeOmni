# Copyright 2026 ByteDance Ltd. and/or its affiliates
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

"""Lossless head-parallel GDN over the flattened CP x Ulysses group.

Full-attention layers keep their Ring/Hybrid CP layout.  Linear-attention
layers instead exchange all five GDN projections once, gather the canonical
packed sequence, and shard heads across the same flattened sequence-parallel
group.  This is mathematically the ordinary head-parallel GDN computation;
there is no recurrent-state approximation or duplicated local recurrence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from .sharding import balanced_cp_restore, balanced_cp_to_rank_major


@dataclass(frozen=True)
class _HeadwiseTensorSpec:
    ndim: int
    sequence_dim: int
    head_dim: int
    total_heads: int
    local_heads: int
    tail_shape: tuple[int, ...]
    local_width: int


@dataclass(frozen=True)
class GdnHeadwiseLayout:
    """Validated layout shared by the forward and inverse packed A2A."""

    cp_size: int
    world_size: int
    rank: int
    batch_size: int
    valid_cu_seqlens: tuple[int, ...]
    valid_lengths: tuple[int, ...]
    padded_lengths: tuple[int, ...]
    local_lengths: tuple[int, ...]
    local_sequence_length: int
    total_valid_tokens: int
    total_padded_tokens: int
    input_specs: tuple[_HeadwiseTensorSpec, ...]
    dtype: torch.dtype
    device_type: str

    @property
    def ulysses_size(self) -> int:
        return self.world_size // self.cp_size


class _EqualSplitAllToAll(torch.autograd.Function):
    """Autograd-safe equal-split ``all_to_all_single``.

    Applying the same exchange to the output gradient is the exact transpose
    of the forward permutation.  Keeping this primitive private prevents a
    silent fallback to list ``all_to_all`` / HcclAlltoAllV.
    """

    @staticmethod
    def forward(ctx, tensor: Tensor, group: ProcessGroup) -> Tensor:
        ctx.group = group
        send = tensor.contiguous()
        output = torch.empty_like(send)
        dist.all_to_all_single(output, send, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        # ``empty_like`` preserves the input memory format by default.  The
        # gradient produced by the head concatenation is often a strided view;
        # using that layout as a collective output buffer corrupts the A2A
        # transpose on Gloo and HCCL.  Both collective buffers must be dense.
        send = grad_output.contiguous()
        grad_input = torch.empty_like(send)
        dist.all_to_all_single(grad_input, send, group=ctx.group)
        return grad_input, None


def _require_coordinate(value: int, *, name: str, upper: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 1 and upper is None:
        raise ValueError(f"{name} must be positive")
    if upper is not None and not 0 <= value < upper:
        raise ValueError(f"{name} must be in [0, {upper}), got {value}")
    return value


def _cu_points(cu_seqlens: Tensor | Sequence[int]) -> tuple[int, ...]:
    if isinstance(cu_seqlens, Tensor):
        if cu_seqlens.device.type != "cpu":
            raise ValueError("GDN headwise plan requires host cu_seqlens; device-to-host synchronization is forbidden")
        if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("GDN headwise cu_seqlens must be a one-dimensional int32/int64 tensor")
        raw_points = cu_seqlens.tolist()
    else:
        raw_points = list(cu_seqlens)
    points: list[int] = []
    for index, point in enumerate(raw_points):
        if isinstance(point, bool) or not isinstance(point, Integral):
            raise TypeError(f"cu_seqlens[{index}] must be an integer")
        points.append(int(point))
    if not points or points[0] != 0:
        raise ValueError("cu_seqlens must start at zero")
    if any(right < left for left, right in zip(points, points[1:])):
        raise ValueError("cu_seqlens must be non-decreasing")
    return tuple(points)


def _normalize_dims(tensor: Tensor, sequence_dim: int, head_dim: int) -> tuple[int, int]:
    if tensor.ndim < 3:
        raise ValueError(f"GDN headwise inputs must have at least three dimensions, got {tensor.ndim}")
    sequence_dim %= tensor.ndim
    head_dim %= tensor.ndim
    if sequence_dim == head_dim or sequence_dim == 0 or head_dim == 0:
        raise ValueError("batch, sequence, and head dimensions must be distinct")
    return sequence_dim, head_dim


def _local_layout(
    inputs: tuple[Tensor, ...],
    *,
    cu_seqlens: Tensor | Sequence[int],
    cp_size: int,
    world_size: int,
    rank: int,
    sequence_dim: int,
    head_dim: int,
) -> GdnHeadwiseLayout:
    if len(inputs) != 5:
        raise ValueError("GDN headwise packing requires exactly q, k, v, b, and a")
    cp_size = _require_coordinate(cp_size, name="cp_size")
    if world_size % cp_size:
        raise ValueError(f"world_size ({world_size}) must be divisible by cp_size ({cp_size})")
    _require_coordinate(rank, name="rank", upper=world_size)
    points = _cu_points(cu_seqlens)
    valid_lengths = tuple(right - left for left, right in zip(points, points[1:]))
    sample_multiple = 2 * world_size
    padded_lengths = tuple(
        ((length + sample_multiple - 1) // sample_multiple) * sample_multiple for length in valid_lengths
    )
    local_lengths = tuple(length // world_size for length in padded_lengths)
    local_sequence_length = sum(local_lengths)

    first = inputs[0]
    batch_size = int(first.size(0))
    dtype = first.dtype
    device_type = first.device.type
    specs: list[_HeadwiseTensorSpec] = []
    for index, tensor in enumerate(inputs):
        current_sequence_dim, current_head_dim = _normalize_dims(tensor, sequence_dim, head_dim)
        if int(tensor.size(0)) != batch_size:
            raise ValueError(f"input {index} batch size differs from q")
        if int(tensor.size(current_sequence_dim)) != local_sequence_length:
            raise ValueError(
                f"input {index} local sequence length {int(tensor.size(current_sequence_dim))} "
                f"does not match packed metadata {local_sequence_length}"
            )
        if tensor.dtype != dtype or tensor.device.type != device_type:
            raise ValueError("all GDN headwise inputs must share dtype and device type")
        canonical = tensor.movedim((current_sequence_dim, current_head_dim), (1, 2))
        total_heads = int(canonical.size(2))
        if total_heads <= 0 or total_heads % world_size:
            raise ValueError(
                f"input {index} head count ({total_heads}) must be positive and divisible by SP size ({world_size})"
            )
        tail_shape = tuple(int(size) for size in canonical.shape[3:])
        tail_width = math.prod(tail_shape) if tail_shape else 1
        local_heads = total_heads // world_size
        specs.append(
            _HeadwiseTensorSpec(
                ndim=tensor.ndim,
                sequence_dim=current_sequence_dim,
                head_dim=current_head_dim,
                total_heads=total_heads,
                local_heads=local_heads,
                tail_shape=tail_shape,
                local_width=local_heads * tail_width,
            )
        )
    if specs[0].total_heads != specs[1].total_heads:
        raise ValueError("q and k must have the same head count")
    if specs[2].total_heads != specs[3].total_heads or specs[2].total_heads != specs[4].total_heads:
        raise ValueError("v, b, and a must have the same head count")
    return GdnHeadwiseLayout(
        cp_size=cp_size,
        world_size=world_size,
        rank=rank,
        batch_size=batch_size,
        valid_cu_seqlens=points,
        valid_lengths=valid_lengths,
        padded_lengths=padded_lengths,
        local_lengths=local_lengths,
        local_sequence_length=local_sequence_length,
        total_valid_tokens=sum(valid_lengths),
        total_padded_tokens=sum(padded_lengths),
        input_specs=tuple(specs),
        dtype=dtype,
        device_type=device_type,
    )


def compile_gdn_headwise_layout(
    inputs: tuple[Tensor, ...],
    *,
    cu_seqlens: Tensor | Sequence[int],
    group: ProcessGroup,
    cp_size: int,
    sequence_dim: int = 1,
    head_dim: int = 2,
) -> GdnHeadwiseLayout:
    """Compile one symmetric plan and make validation failures collective."""

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before compiling a GDN headwise plan")
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    local_layout: GdnHeadwiseLayout | None = None
    local_error: str | None = None
    try:
        local_layout = _local_layout(
            inputs,
            cu_seqlens=cu_seqlens,
            cp_size=cp_size,
            world_size=world_size,
            rank=rank,
            sequence_dim=sequence_dim,
            head_dim=head_dim,
        )
    except Exception as error:  # noqa: BLE001 - all ranks must report before any payload collective
        local_error = f"{type(error).__name__}: {error}"

    signature = None
    if local_layout is not None:
        signature = (
            local_layout.cp_size,
            local_layout.world_size,
            local_layout.batch_size,
            local_layout.valid_cu_seqlens,
            local_layout.padded_lengths,
            tuple(
                (
                    spec.ndim,
                    spec.sequence_dim,
                    spec.head_dim,
                    spec.total_heads,
                    spec.tail_shape,
                )
                for spec in local_layout.input_specs
            ),
            str(local_layout.dtype),
            local_layout.device_type,
        )
    gathered: list[tuple[str | None, object | None] | None] = [None] * world_size
    dist.all_gather_object(gathered, (local_error, signature), group=group)
    errors = [f"rank {index}: {item[0]}" for index, item in enumerate(gathered) if item is not None and item[0]]
    if errors:
        raise RuntimeError("GDN headwise plan rejected: " + "; ".join(errors))
    signatures = [item[1] for item in gathered if item is not None]
    if len(signatures) != world_size or len(set(signatures)) != 1:
        raise RuntimeError(f"GDN headwise plan signatures differ across ranks: {signatures}")
    if local_layout is None:
        raise RuntimeError("GDN headwise plan compilation produced no local layout")
    return local_layout


def _assert_live_layout(layout: GdnHeadwiseLayout, group: ProcessGroup) -> None:
    if dist.get_world_size(group) != layout.world_size or dist.get_rank(group) != layout.rank:
        raise RuntimeError("GDN headwise layout is bound to a different process group")


def _rank_major_to_sample_major(tensor: Tensor, layout: GdnHeadwiseLayout) -> Tensor:
    rank_blocks = tensor.split(layout.local_sequence_length, dim=1)
    per_sample: list[list[Tensor]] = [[] for _ in layout.local_lengths]
    for rank_block in rank_blocks:
        for sample_index, piece in enumerate(rank_block.split(layout.local_lengths, dim=1)):
            per_sample[sample_index].append(piece)
    pieces = [torch.cat(sample_pieces, dim=1) for sample_pieces in per_sample]
    return torch.cat(pieces, dim=1) if pieces else tensor.narrow(1, 0, 0)


def _sample_major_to_rank_major(tensor: Tensor, layout: GdnHeadwiseLayout) -> Tensor:
    samples = tensor.split(layout.padded_lengths, dim=1)
    per_rank: list[list[Tensor]] = [[] for _ in range(layout.world_size)]
    for sample, local_length in zip(samples, layout.local_lengths):
        for rank, piece in enumerate(sample.split(local_length, dim=1)):
            per_rank[rank].append(piece)
    rank_blocks = [torch.cat(pieces, dim=1) for pieces in per_rank]
    return torch.cat(rank_blocks, dim=1) if rank_blocks else tensor.narrow(1, 0, 0)


def _restore_and_compact(tensor: Tensor, layout: GdnHeadwiseLayout) -> Tensor:
    sample_major = _rank_major_to_sample_major(tensor, layout)
    samples = sample_major.split(layout.padded_lengths, dim=1)
    valid_samples: list[Tensor] = []
    for sample, valid_length in zip(samples, layout.valid_lengths):
        canonical = balanced_cp_restore(sample, cp_size=layout.cp_size, dim=1)
        valid_samples.append(canonical.narrow(1, 0, valid_length))
    return torch.cat(valid_samples, dim=1) if valid_samples else sample_major.narrow(1, 0, 0)


def _expand_and_to_rank_major(tensor: Tensor, layout: GdnHeadwiseLayout) -> Tensor:
    samples = tensor.split(layout.valid_lengths, dim=1)
    padded_samples: list[Tensor] = []
    for sample, valid_length, padded_length in zip(samples, layout.valid_lengths, layout.padded_lengths):
        if padded_length > valid_length:
            shape = list(sample.shape)
            shape[1] = padded_length - valid_length
            sample = torch.cat((sample, sample.new_zeros(shape)), dim=1)
        padded_samples.append(balanced_cp_to_rank_major(sample, cp_size=layout.cp_size, dim=1))
    sample_major = torch.cat(padded_samples, dim=1) if padded_samples else tensor.narrow(1, 0, 0)
    return _sample_major_to_rank_major(sample_major, layout)


def prepare_gdn_headwise_inputs(
    inputs: tuple[Tensor, ...],
    *,
    cu_seqlens: Tensor | Sequence[int] | None = None,
    group: ProcessGroup,
    cp_size: int | None = None,
    sequence_dim: int = 1,
    head_dim: int = 2,
    layout: GdnHeadwiseLayout | None = None,
) -> tuple[tuple[Tensor, ...], GdnHeadwiseLayout]:
    """Gather packed sequence and scatter all GDN heads in one A2A."""

    if layout is None:
        if cu_seqlens is None or cp_size is None:
            raise ValueError("cu_seqlens and cp_size are required when no compiled layout is supplied")
        layout = compile_gdn_headwise_layout(
            inputs,
            cu_seqlens=cu_seqlens,
            group=group,
            cp_size=cp_size,
            sequence_dim=sequence_dim,
            head_dim=head_dim,
        )
    _assert_live_layout(layout, group)
    if len(inputs) != len(layout.input_specs):
        raise ValueError("GDN headwise input count differs from the compiled layout")

    canonical_inputs: list[Tensor] = []
    for tensor, spec in zip(inputs, layout.input_specs):
        canonical = tensor.movedim((spec.sequence_dim, spec.head_dim), (1, 2)).contiguous()
        expected_shape = (layout.batch_size, layout.local_sequence_length, spec.total_heads, *spec.tail_shape)
        if (
            tuple(canonical.shape) != expected_shape
            or tensor.dtype != layout.dtype
            or tensor.device.type != layout.device_type
        ):
            raise RuntimeError("live GDN headwise input differs from its compiled layout")
        canonical_inputs.append(canonical)

    if layout.total_padded_tokens == 0:
        prepared = tuple(
            canonical.narrow(1, 0, 0)
            .narrow(2, layout.rank * spec.local_heads, spec.local_heads)
            .movedim((1, 2), (spec.sequence_dim, spec.head_dim))
            .contiguous()
            for canonical, spec in zip(canonical_inputs, layout.input_specs)
        )
        return prepared, layout

    packed_features: list[Tensor] = []
    for canonical, spec in zip(canonical_inputs, layout.input_specs):
        # Expose the destination head rank as the equal-split leading
        # dimension.  This replaces ``5 * world_size`` device-side slice
        # copies with one packed concatenation per GDN layer.
        destination_major = canonical.reshape(
            layout.batch_size,
            layout.local_sequence_length,
            layout.world_size,
            spec.local_heads,
            *spec.tail_shape,
        ).permute(2, 0, 1, 3, *range(4, 4 + len(spec.tail_shape)))
        packed_features.append(
            destination_major.reshape(
                layout.world_size * layout.batch_size,
                layout.local_sequence_length,
                spec.local_width,
            )
        )
    send = torch.cat(packed_features, dim=-1).contiguous()
    received = _EqualSplitAllToAll.apply(send, group)
    rank_major = (
        received.reshape(layout.world_size, layout.batch_size, layout.local_sequence_length, -1)
        .permute(1, 0, 2, 3)
        .reshape(layout.batch_size, layout.total_padded_tokens, -1)
        .contiguous()
    )
    compact = _restore_and_compact(rank_major, layout)

    prepared: list[Tensor] = []
    offset = 0
    for spec in layout.input_specs:
        piece = compact.narrow(-1, offset, spec.local_width)
        offset += spec.local_width
        canonical = piece.reshape(layout.batch_size, layout.total_valid_tokens, spec.local_heads, *spec.tail_shape)
        prepared.append(canonical.movedim((1, 2), (spec.sequence_dim, spec.head_dim)).contiguous())
    return tuple(prepared), layout


def restore_gdn_headwise_output(
    output: Tensor,
    *,
    layout: GdnHeadwiseLayout,
    group: ProcessGroup,
    sequence_dim: int = 1,
    head_dim: int = 2,
) -> Tensor:
    """Invert the headwise exchange and restore each rank's physical tokens."""

    _assert_live_layout(layout, group)
    sequence_dim, head_dim = _normalize_dims(output, sequence_dim, head_dim)
    canonical = output.movedim((sequence_dim, head_dim), (1, 2)).contiguous()
    if int(canonical.size(0)) != layout.batch_size or int(canonical.size(1)) != layout.total_valid_tokens:
        raise ValueError("GDN headwise output does not match the compiled batch/sequence layout")
    local_heads = int(canonical.size(2))
    tail_shape = tuple(int(size) for size in canonical.shape[3:])
    if local_heads <= 0:
        raise ValueError("GDN headwise output must have a positive local head count")

    if layout.total_padded_tokens == 0:
        full_shape = (layout.batch_size, 0, local_heads * layout.world_size, *tail_shape)
        restored = canonical.new_empty(full_shape) + canonical.sum() * 0
        return restored.movedim((1, 2), (sequence_dim, head_dim)).contiguous()

    rank_major = _expand_and_to_rank_major(canonical, layout)
    rank_blocks = rank_major.split(layout.local_sequence_length, dim=1)
    send = torch.cat(
        [block.reshape(layout.batch_size, layout.local_sequence_length, -1) for block in rank_blocks], dim=0
    ).contiguous()
    received = _EqualSplitAllToAll.apply(send, group)
    restored = (
        received.reshape(layout.world_size, layout.batch_size, layout.local_sequence_length, local_heads, *tail_shape)
        .permute(1, 2, 0, 3, *range(4, 4 + len(tail_shape)))
        .reshape(
            layout.batch_size,
            layout.local_sequence_length,
            local_heads * layout.world_size,
            *tail_shape,
        )
        .contiguous()
    )
    return restored.movedim((1, 2), (sequence_dim, head_dim)).contiguous()


__all__ = [
    "GdnHeadwiseLayout",
    "compile_gdn_headwise_layout",
    "prepare_gdn_headwise_inputs",
    "restore_gdn_headwise_output",
]
