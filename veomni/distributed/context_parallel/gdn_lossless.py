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

"""Lossless Gated DeltaNet context-parallel collectives.

The module executes a validated :class:`GdnLosslessPlan` with three autograd
contracts:

* a reversible variable-size all-to-all between physical zigzag tokens and
  native-chunk owners;
* recurrent-state point-to-point transfer whose backward runs in the reverse
  direction; and
* causal-convolution halo transfer with the same reverse-gradient contract.

All collectives are expressed with ``torch.distributed`` and do not import an
accelerator runtime. Empty ranks still participate in all-to-all collectives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Sequence

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from .gdn_ownership import GdnCopySpan, GdnLosslessPlan, GdnRankPlan, build_gdn_lossless_plan
from .gdn_runtime import GdnCpOperation, GdnCpPhase, GdnCpRuntimeObserver


@dataclass(frozen=True)
class GdnLosslessRuntimePlan:
    """A validated global plan bound to one live process-group rank."""

    global_plan: GdnLosslessPlan
    local: GdnRankPlan

    @property
    def cp_size(self) -> int:
        return self.global_plan.cp_size

    @property
    def cp_rank(self) -> int:
        return self.local.rank

    @property
    def plan_hash(self) -> str:
        return self.global_plan.plan_hash

    @property
    def owned_cu_seqlens(self) -> tuple[int, ...]:
        return self.local.owned_cu_seqlens


def compile_gdn_lossless_runtime_plan(
    valid_lengths: Sequence[int],
    *,
    cp_group: ProcessGroup,
    ulysses_size: int = 1,
) -> GdnLosslessRuntimePlan:
    """Build the same plan on every rank and reject identity drift together."""
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before compiling a GDN CP plan")
    cp_size = dist.get_world_size(cp_group)
    cp_rank = dist.get_rank(cp_group)
    local_plan: GdnLosslessPlan | None = None
    local_error: str | None = None
    try:
        local_plan = build_gdn_lossless_plan(
            valid_lengths,
            cp_size=cp_size,
            ulysses_size=ulysses_size,
        )
    except Exception as error:  # noqa: BLE001 - coordinated fail-closed reporting
        local_error = f"{type(error).__name__}: {error}"

    local_identity = None if local_plan is None else local_plan.plan_hash
    gathered: list[tuple[str | None, str | None] | None] = [None] * cp_size
    dist.all_gather_object(gathered, (local_error, local_identity), group=cp_group)
    errors = [f"rank {rank}: {item[0]}" for rank, item in enumerate(gathered) if item is not None and item[0]]
    if errors:
        raise RuntimeError("GDN lossless plan rejected: " + "; ".join(errors))
    identities = [item[1] for item in gathered if item is not None]
    if len(identities) != cp_size or len(set(identities)) != 1:
        raise RuntimeError(f"GDN lossless plan identity differs across ranks: {identities}")
    if local_plan is None:
        raise RuntimeError("GDN lossless plan compilation produced no local plan")
    return GdnLosslessRuntimePlan(global_plan=local_plan, local=local_plan.rank_plan(cp_rank))


def _assert_live_plan(plan: GdnLosslessRuntimePlan, group: ProcessGroup) -> None:
    if dist.get_world_size(group) != plan.cp_size or dist.get_rank(group) != plan.cp_rank:
        raise RuntimeError("GDN lossless plan is bound to a different process group")


def _flatten_sequence(tensor: Tensor, sequence_dim: int) -> tuple[Tensor, int, tuple[int, ...]]:
    if tensor.ndim == 0:
        raise ValueError("GDN repartition requires a sequence dimension")
    normalized_dim = sequence_dim % tensor.ndim
    moved = tensor.movedim(normalized_dim, 0).contiguous()
    feature_shape = tuple(int(size) for size in moved.shape[1:])
    feature_width = math.prod(feature_shape) if feature_shape else 1
    if feature_width <= 0:
        raise ValueError(f"GDN repartition requires a positive feature width, got {feature_shape}")
    return moved.reshape(int(moved.size(0)), feature_width), normalized_dim, feature_shape


def _restore_sequence(flat: Tensor, sequence_dim: int, feature_shape: tuple[int, ...]) -> Tensor:
    moved = flat.reshape((int(flat.size(0)),) + feature_shape)
    return moved.movedim(0, sequence_dim).contiguous() if sequence_dim else moved.contiguous()


def _copy_to_buffer(flat: Tensor, spans: Sequence[GdnCopySpan], rows: int, *, zero_fill: bool) -> Tensor:
    output = flat.new_zeros((rows, int(flat.size(1)))) if zero_fill else flat.new_empty((rows, int(flat.size(1))))
    for span in spans:
        output.narrow(0, span.destination_start, span.length).copy_(flat.narrow(0, span.source_start, span.length))
    return output


def _all_to_all(
    send: Tensor,
    *,
    input_splits: Sequence[int],
    output_splits: Sequence[int],
    group: ProcessGroup,
) -> Tensor:
    output = send.new_empty((sum(output_splits), int(send.size(1))))
    dist.all_to_all_single(
        output,
        send.contiguous(),
        output_split_sizes=list(output_splits),
        input_split_sizes=list(input_splits),
        group=group,
    )
    return output


def _physical_to_owned_flat(flat: Tensor, plan: GdnLosslessRuntimePlan, group: ProcessGroup) -> Tensor:
    local = plan.local
    send = _copy_to_buffer(flat, local.forward_pack_spans, sum(local.forward_input_splits), zero_fill=False)
    receive = _all_to_all(
        send,
        input_splits=local.forward_input_splits,
        output_splits=local.forward_output_splits,
        group=group,
    )
    return _copy_to_buffer(receive, local.forward_unpack_spans, local.owned_token_count, zero_fill=True)


def _owned_to_physical_flat(flat: Tensor, plan: GdnLosslessRuntimePlan, group: ProcessGroup) -> Tensor:
    local = plan.local
    send = _copy_to_buffer(flat, local.inverse_pack_spans, sum(local.inverse_input_splits), zero_fill=False)
    receive = _all_to_all(
        send,
        input_splits=local.inverse_input_splits,
        output_splits=local.inverse_output_splits,
        group=group,
    )
    return _copy_to_buffer(receive, local.inverse_unpack_spans, local.source_token_count, zero_fill=True)


class _PhysicalToOwned(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        tensor: Tensor,
        plan: GdnLosslessRuntimePlan,
        group: ProcessGroup,
        sequence_dim: int,
        observer: GdnCpRuntimeObserver | None,
    ) -> Tensor:
        _assert_live_plan(plan, group)
        flat, normalized_dim, feature_shape = _flatten_sequence(tensor, sequence_dim)
        if int(flat.size(0)) != plan.local.source_token_count:
            raise ValueError(f"physical rows {flat.size(0)} do not match plan rows {plan.local.source_token_count}")
        ctx.plan = plan
        ctx.group = group
        ctx.sequence_dim = normalized_dim
        ctx.feature_shape = feature_shape
        ctx.observer = observer
        if observer is not None:
            observer.enter(GdnCpOperation.OWNERSHIP_A2A, GdnCpPhase.FORWARD)
        try:
            output = _physical_to_owned_flat(flat, plan, group)
        except Exception:
            if observer is not None:
                observer.error(GdnCpOperation.OWNERSHIP_A2A, GdnCpPhase.FORWARD)
            raise
        if observer is not None:
            observer.exit(GdnCpOperation.OWNERSHIP_A2A, GdnCpPhase.FORWARD)
        return _restore_sequence(output, normalized_dim, feature_shape)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None, None, None]:
        flat, _, _ = _flatten_sequence(grad_output.contiguous(), ctx.sequence_dim)
        if int(flat.size(0)) != ctx.plan.local.owned_token_count:
            raise ValueError("owned gradient rows do not match the GDN lossless plan")
        if ctx.observer is not None:
            ctx.observer.enter(GdnCpOperation.OWNERSHIP_A2A, GdnCpPhase.BACKWARD)
        try:
            grad_input = _owned_to_physical_flat(flat, ctx.plan, ctx.group)
        except Exception:
            if ctx.observer is not None:
                ctx.observer.error(GdnCpOperation.OWNERSHIP_A2A, GdnCpPhase.BACKWARD)
            raise
        if ctx.observer is not None:
            ctx.observer.exit(GdnCpOperation.OWNERSHIP_A2A, GdnCpPhase.BACKWARD)
        return _restore_sequence(grad_input, ctx.sequence_dim, ctx.feature_shape), None, None, None, None


class _OwnedToPhysical(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        tensor: Tensor,
        plan: GdnLosslessRuntimePlan,
        group: ProcessGroup,
        sequence_dim: int,
        observer: GdnCpRuntimeObserver | None,
    ) -> Tensor:
        _assert_live_plan(plan, group)
        flat, normalized_dim, feature_shape = _flatten_sequence(tensor, sequence_dim)
        if int(flat.size(0)) != plan.local.owned_token_count:
            raise ValueError(f"owned rows {flat.size(0)} do not match plan rows {plan.local.owned_token_count}")
        ctx.plan = plan
        ctx.group = group
        ctx.sequence_dim = normalized_dim
        ctx.feature_shape = feature_shape
        ctx.observer = observer
        if observer is not None:
            observer.enter(GdnCpOperation.OWNERSHIP_A2A, GdnCpPhase.FORWARD)
        try:
            output = _owned_to_physical_flat(flat, plan, group)
        except Exception:
            if observer is not None:
                observer.error(GdnCpOperation.OWNERSHIP_A2A, GdnCpPhase.FORWARD)
            raise
        if observer is not None:
            observer.exit(GdnCpOperation.OWNERSHIP_A2A, GdnCpPhase.FORWARD)
        return _restore_sequence(output, normalized_dim, feature_shape)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None, None, None]:
        flat, _, _ = _flatten_sequence(grad_output.contiguous(), ctx.sequence_dim)
        if int(flat.size(0)) != ctx.plan.local.source_token_count:
            raise ValueError("physical gradient rows do not match the GDN lossless plan")
        if ctx.observer is not None:
            ctx.observer.enter(GdnCpOperation.OWNERSHIP_A2A, GdnCpPhase.BACKWARD)
        try:
            grad_input = _physical_to_owned_flat(flat, ctx.plan, ctx.group)
        except Exception:
            if ctx.observer is not None:
                ctx.observer.error(GdnCpOperation.OWNERSHIP_A2A, GdnCpPhase.BACKWARD)
            raise
        if ctx.observer is not None:
            ctx.observer.exit(GdnCpOperation.OWNERSHIP_A2A, GdnCpPhase.BACKWARD)
        return _restore_sequence(grad_input, ctx.sequence_dim, ctx.feature_shape), None, None, None, None


def physical_to_owned(
    tensor: Tensor,
    *,
    plan: GdnLosslessRuntimePlan,
    cp_group: ProcessGroup,
    sequence_dim: int = 1,
    observer: GdnCpRuntimeObserver | None = None,
) -> Tensor:
    """Move valid physical tokens to native-chunk owners with inverse autograd."""
    return _PhysicalToOwned.apply(tensor, plan, cp_group, sequence_dim, observer)


def owned_to_physical(
    tensor: Tensor,
    *,
    plan: GdnLosslessRuntimePlan,
    cp_group: ProcessGroup,
    sequence_dim: int = 1,
    observer: GdnCpRuntimeObserver | None = None,
) -> Tensor:
    """Return owned valid tokens to the physical layout and restore zero padding."""
    return _OwnedToPhysical.apply(tensor, plan, cp_group, sequence_dim, observer)


def _group_global_rank(group: ProcessGroup, local_rank: int) -> int:
    ranks = dist.get_process_group_ranks(group)
    if not 0 <= local_rank < len(ranks):
        raise ValueError(f"local rank {local_rank} is outside process group size {len(ranks)}")
    return int(ranks[local_rank])


class _StateReceive(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        template: Tensor,
        participation: Tensor,
        predecessor: int | None,
        group: ProcessGroup,
        observer: GdnCpRuntimeObserver | None,
    ) -> Tensor:
        ctx.predecessor = predecessor
        ctx.group = group
        ctx.observer = observer
        ctx.participation_dtype = participation.dtype
        if predecessor is None:
            ctx.received = False
            output = torch.zeros_like(template)
        else:
            ctx.received = True
            output = torch.empty_like(template)
            if observer is not None:
                observer.enter(GdnCpOperation.STATE_P2P_RECV, GdnCpPhase.FORWARD, peer_rank=predecessor)
            try:
                dist.recv(output, src=_group_global_rank(group, predecessor), group=group)
            except Exception:
                if observer is not None:
                    observer.error(GdnCpOperation.STATE_P2P_RECV, GdnCpPhase.FORWARD, peer_rank=predecessor)
                raise
            if observer is not None:
                observer.exit(GdnCpOperation.STATE_P2P_RECV, GdnCpPhase.FORWARD, peer_rank=predecessor)
        return output + participation.to(dtype=output.dtype) * 0

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[None, Tensor, None, None, None]:
        if ctx.received:
            if ctx.observer is not None:
                ctx.observer.enter(GdnCpOperation.STATE_P2P_RECV, GdnCpPhase.BACKWARD, peer_rank=ctx.predecessor)
            try:
                dist.send(
                    grad_output.contiguous(),
                    dst=_group_global_rank(ctx.group, ctx.predecessor),
                    group=ctx.group,
                )
            except Exception:
                if ctx.observer is not None:
                    ctx.observer.error(GdnCpOperation.STATE_P2P_RECV, GdnCpPhase.BACKWARD, peer_rank=ctx.predecessor)
                raise
            if ctx.observer is not None:
                ctx.observer.exit(GdnCpOperation.STATE_P2P_RECV, GdnCpPhase.BACKWARD, peer_rank=ctx.predecessor)
        participation_grad = grad_output.new_zeros((), dtype=ctx.participation_dtype)
        return None, participation_grad, None, None, None


class _StateSend(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        state: Tensor,
        successor: int | None,
        group: ProcessGroup,
        observer: GdnCpRuntimeObserver | None,
    ) -> Tensor:
        ctx.successor = successor
        ctx.group = group
        ctx.observer = observer
        ctx.sent = successor is not None
        if ctx.sent:
            if observer is not None:
                observer.enter(GdnCpOperation.STATE_P2P_SEND, GdnCpPhase.FORWARD, peer_rank=successor)
            try:
                dist.send(state.contiguous(), dst=_group_global_rank(group, successor), group=group)
            except Exception:
                if observer is not None:
                    observer.error(GdnCpOperation.STATE_P2P_SEND, GdnCpPhase.FORWARD, peer_rank=successor)
                raise
            if observer is not None:
                observer.exit(GdnCpOperation.STATE_P2P_SEND, GdnCpPhase.FORWARD, peer_rank=successor)
        return state

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None, None]:
        if not ctx.sent:
            return grad_output, None, None, None
        downstream = torch.empty_like(grad_output)
        if ctx.observer is not None:
            ctx.observer.enter(GdnCpOperation.STATE_P2P_SEND, GdnCpPhase.BACKWARD, peer_rank=ctx.successor)
        try:
            dist.recv(downstream, src=_group_global_rank(ctx.group, ctx.successor), group=ctx.group)
        except Exception:
            if ctx.observer is not None:
                ctx.observer.error(GdnCpOperation.STATE_P2P_SEND, GdnCpPhase.BACKWARD, peer_rank=ctx.successor)
            raise
        if ctx.observer is not None:
            ctx.observer.exit(GdnCpOperation.STATE_P2P_SEND, GdnCpPhase.BACKWARD, peer_rank=ctx.successor)
        return grad_output + downstream, None, None, None


def receive_initial_state(
    *,
    plan: GdnLosslessRuntimePlan,
    cp_group: ProcessGroup,
    state_template: Tensor,
    participation: Tensor,
    observer: GdnCpRuntimeObserver | None = None,
) -> Tensor:
    """Receive the predecessor state and zero BOS or inactive sample slots."""
    if state_template.ndim == 0 or int(state_template.size(0)) != len(plan.local.samples):
        raise ValueError("state template leading dimension must equal the packed sample count")
    received = _StateReceive.apply(
        state_template,
        participation,
        plan.local.predecessor_rank,
        cp_group,
        observer,
    )
    active_non_bos = [sample.is_active and not sample.is_bos_owner for sample in plan.local.samples]
    mask_shape = (len(active_non_bos),) + (1,) * (received.ndim - 1)
    mask = received.new_tensor(active_non_bos, dtype=received.dtype).reshape(mask_shape)
    return received * mask


def send_final_state(
    state: Tensor,
    *,
    plan: GdnLosslessRuntimePlan,
    cp_group: ProcessGroup,
    observer: GdnCpRuntimeObserver | None = None,
) -> Tensor:
    """Send final recurrent state to the next owner.

    The returned tensor must remain in the loss graph. Use
    :func:`attach_state_dependency` when the model output does not otherwise
    depend on the final state.
    """
    return _StateSend.apply(state, plan.local.successor_rank, cp_group, observer)


def attach_state_dependency(output: Tensor, state: Tensor) -> Tensor:
    """Keep state-transfer backward live without changing numerical output."""
    if state.numel() == 0:
        return output
    return output + state[(0,) * state.ndim].to(dtype=output.dtype) * 0


def make_state_participation(*tensors: Tensor) -> Tensor:
    """Build an O(1), numerically zero autograd edge from live GDN inputs."""
    if not tensors:
        raise ValueError("make_state_participation requires at least one tensor")
    if any(tensor.numel() == 0 for tensor in tensors):
        return sum((tensor.sum() * 0 for tensor in tensors), tensors[0].new_zeros(()))
    participation = tensors[0][(0,) * tensors[0].ndim] * 0
    for tensor in tensors[1:]:
        participation = participation + tensor[(0,) * tensor.ndim] * 0
    return participation


def make_state_template(query: Tensor, value: Tensor, cu_seqlens: Tensor) -> Tensor:
    """Allocate the fp32 recurrent-state layout expected by GDN kernels."""
    if query.ndim != 4 or value.ndim != 4:
        raise ValueError("query and value must have [B, T, H, D] layout")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 1:
        raise ValueError("cu_seqlens must be a non-empty one-dimensional tensor")
    return torch.zeros(
        int(cu_seqlens.numel() - 1),
        int(query.size(2)),
        int(query.size(3)),
        int(value.size(3)),
        device=query.device,
        dtype=torch.float32,
    )


def aligned_gdn_cu_seqlens(cu_seqlens: Sequence[int], *, chunk_size: int = 32) -> list[int]:
    """Return host CU points after per-segment GDN chunk alignment."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral):
        raise TypeError("chunk_size must be an integer")
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    points: list[int] = []
    for index, point in enumerate(cu_seqlens):
        if isinstance(point, bool) or not isinstance(point, Integral):
            raise TypeError(f"cu_seqlens[{index}] must be an integer")
        points.append(int(point))
    if not points or points[0] != 0 or any(end < start for start, end in zip(points, points[1:])):
        raise ValueError("cu_seqlens must start at zero and be monotonic")
    aligned_points = [0]
    for start, end in zip(points, points[1:]):
        length = end - start
        aligned_points.append(aligned_points[-1] + ((length + chunk_size - 1) // chunk_size) * chunk_size)
    return aligned_points


def align_gdn_varlen_chunks(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    cu_seqlens: Tensor,
    *,
    cu_seqlens_list: Sequence[int] | None = None,
    chunk_size: int = 32,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    """Pad each owned segment for kernels that return state only on full chunks.

    ``value``, ``g``, and ``beta`` pads are zero, so a pad token preserves the
    recurrent state. Query and key pads repeat the last real token to avoid a
    zero-vector normalization. The final index selects real outputs again.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have [B, T, H, D] layout")
    if g.ndim != 3 or beta.ndim != 3:
        raise ValueError("g and beta must have [B, T, H] layout")
    if int(query.size(0)) != 1:
        raise ValueError("packed GDN chunk alignment requires batch size one")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if cu_seqlens_list is None:
        points = [int(point) for point in cu_seqlens.detach().cpu().tolist()]
    else:
        points = []
        for index, point in enumerate(cu_seqlens_list):
            if isinstance(point, bool) or not isinstance(point, Integral):
                raise TypeError(f"cu_seqlens_list[{index}] must be an integer")
            points.append(int(point))
        if len(points) != int(cu_seqlens.numel()):
            raise ValueError("host cu_seqlens_list and device cu_seqlens must have the same number of boundaries")
    if not points or points[0] != 0 or any(end < start for start, end in zip(points, points[1:])):
        raise ValueError("cu_seqlens must start at zero and be monotonic")
    lengths = [end - start for start, end in zip(points, points[1:])]
    if sum(lengths) != int(query.size(1)):
        raise ValueError("cu_seqlens do not cover the GDN token dimension")
    padded_points = aligned_gdn_cu_seqlens(points, chunk_size=chunk_size)
    padded_lengths = [end - start for start, end in zip(padded_points, padded_points[1:])]
    if padded_lengths == lengths:
        return query, key, value, g, beta, cu_seqlens, None

    padded_tokens = sum(padded_lengths)
    real_indices: list[int] = []
    source_cursor = 0
    destination_cursor = 0
    for length, padded_length in zip(lengths, padded_lengths):
        real_indices.extend(range(destination_cursor, destination_cursor + length))
        source_cursor += length
        destination_cursor += padded_length
    if source_cursor != int(query.size(1)):
        raise RuntimeError("GDN chunk alignment did not consume every input token")
    index = torch.tensor(real_indices, dtype=torch.long, device=query.device)

    def padded_zeros(tensor: Tensor) -> Tensor:
        shape = list(tensor.shape)
        shape[1] = padded_tokens
        return tensor.new_zeros(shape)

    query_padded = padded_zeros(query)
    key_padded = padded_zeros(key)
    value_padded = padded_zeros(value)
    g_padded = padded_zeros(g)
    beta_padded = padded_zeros(beta)
    for destination, source in (
        (query_padded, query),
        (key_padded, key),
        (value_padded, value),
        (g_padded, g),
        (beta_padded, beta),
    ):
        destination.index_copy_(1, index, source)

    source_cursor = 0
    destination_cursor = 0
    for length, padded_length in zip(lengths, padded_lengths):
        padding = padded_length - length
        if padding:
            if length == 0:
                raise ValueError("an empty GDN segment cannot have non-zero chunk padding")
            query_fill = query[:, source_cursor + length - 1 : source_cursor + length].expand(
                -1, padding, *([-1] * (query.ndim - 2))
            )
            key_fill = key[:, source_cursor + length - 1 : source_cursor + length].expand(
                -1, padding, *([-1] * (key.ndim - 2))
            )
            query_padded[:, destination_cursor + length : destination_cursor + padded_length] = query_fill
            key_padded[:, destination_cursor + length : destination_cursor + padded_length] = key_fill
        source_cursor += length
        destination_cursor += padded_length

    padded_cu = torch.tensor(padded_points, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    return query_padded, key_padded, value_padded, g_padded, beta_padded, padded_cu, index


def unpad_gdn_varlen_output(output: Tensor, real_indices: Tensor | None) -> Tensor:
    """Remove chunk-alignment pads from a GDN output."""
    return output if real_indices is None else output.index_select(1, real_indices)


class _HaloReceive(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        template: Tensor,
        participation: Tensor,
        source: int | None,
        group: ProcessGroup,
        observer: GdnCpRuntimeObserver | None,
    ) -> Tensor:
        ctx.source = source
        ctx.group = group
        ctx.observer = observer
        ctx.received = source is not None
        if ctx.received:
            output = torch.empty_like(template)
            if observer is not None:
                observer.enter(GdnCpOperation.HALO_P2P_RECV, GdnCpPhase.FORWARD, peer_rank=source)
            try:
                dist.recv(output, src=_group_global_rank(group, source), group=group)
            except Exception:
                if observer is not None:
                    observer.error(GdnCpOperation.HALO_P2P_RECV, GdnCpPhase.FORWARD, peer_rank=source)
                raise
            if observer is not None:
                observer.exit(GdnCpOperation.HALO_P2P_RECV, GdnCpPhase.FORWARD, peer_rank=source)
        else:
            output = torch.zeros_like(template)
        return output + participation.to(dtype=output.dtype) * 0

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[None, None, None, None, None]:
        if ctx.received:
            if ctx.observer is not None:
                ctx.observer.enter(GdnCpOperation.HALO_P2P_RECV, GdnCpPhase.BACKWARD, peer_rank=ctx.source)
            try:
                dist.send(
                    grad_output.contiguous(),
                    dst=_group_global_rank(ctx.group, ctx.source),
                    group=ctx.group,
                )
            except Exception:
                if ctx.observer is not None:
                    ctx.observer.error(GdnCpOperation.HALO_P2P_RECV, GdnCpPhase.BACKWARD, peer_rank=ctx.source)
                raise
            if ctx.observer is not None:
                ctx.observer.exit(GdnCpOperation.HALO_P2P_RECV, GdnCpPhase.BACKWARD, peer_rank=ctx.source)
        return None, None, None, None, None


class _HaloSend(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        payload: Tensor,
        destination: int | None,
        group: ProcessGroup,
        observer: GdnCpRuntimeObserver | None,
    ) -> Tensor:
        ctx.destination = destination
        ctx.group = group
        ctx.observer = observer
        ctx.sent = destination is not None
        if ctx.sent:
            if observer is not None:
                observer.enter(GdnCpOperation.HALO_P2P_SEND, GdnCpPhase.FORWARD, peer_rank=destination)
            try:
                dist.send(payload.contiguous(), dst=_group_global_rank(group, destination), group=group)
            except Exception:
                if observer is not None:
                    observer.error(GdnCpOperation.HALO_P2P_SEND, GdnCpPhase.FORWARD, peer_rank=destination)
                raise
            if observer is not None:
                observer.exit(GdnCpOperation.HALO_P2P_SEND, GdnCpPhase.FORWARD, peer_rank=destination)
        return payload

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None, None]:
        if not ctx.sent:
            return grad_output, None, None, None
        downstream = torch.empty_like(grad_output)
        if ctx.observer is not None:
            ctx.observer.enter(GdnCpOperation.HALO_P2P_SEND, GdnCpPhase.BACKWARD, peer_rank=ctx.destination)
        try:
            dist.recv(downstream, src=_group_global_rank(ctx.group, ctx.destination), group=ctx.group)
        except Exception:
            if ctx.observer is not None:
                ctx.observer.error(GdnCpOperation.HALO_P2P_SEND, GdnCpPhase.BACKWARD, peer_rank=ctx.destination)
            raise
        if ctx.observer is not None:
            ctx.observer.exit(GdnCpOperation.HALO_P2P_SEND, GdnCpPhase.BACKWARD, peer_rank=ctx.destination)
        return grad_output + downstream, None, None, None


def exchange_conv_halo(
    owned: Tensor,
    *,
    plan: GdnLosslessRuntimePlan,
    cp_group: ProcessGroup,
    kernel_size: int,
    sequence_dim: int = 1,
    observer: GdnCpRuntimeObserver | None = None,
) -> tuple[Tensor, Tensor]:
    """Prepend each owned sample with its predecessor's causal-conv halo."""
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}")
    sequence_dim %= owned.ndim
    lengths = [sample.length for sample in plan.local.samples]
    if sum(lengths) != int(owned.size(sequence_dim)):
        raise ValueError("owned tensor length does not match the GDN plan")
    if kernel_size == 1 or plan.local.owned_token_count == 0:
        cu = owned.new_tensor(plan.local.owned_cu_seqlens, dtype=torch.int32)
        return owned, cu

    halo_width = kernel_size - 1
    sample_tensors = list(owned.split(lengths, dim=sequence_dim))
    payloads: list[Tensor] = []
    for sample in sample_tensors:
        sample_length = int(sample.size(sequence_dim))
        if sample_length >= halo_width:
            payload = sample.narrow(sequence_dim, sample_length - halo_width, halo_width)
        else:
            pad_shape = list(sample.shape)
            pad_shape[sequence_dim] = halo_width - sample_length
            payload = torch.cat((sample.new_zeros(pad_shape), sample), dim=sequence_dim)
        payloads.append(payload.contiguous())
    payload = torch.cat(payloads, dim=sequence_dim).contiguous()
    participation = payload[(0,) * payload.ndim] if payload.numel() else owned.new_zeros(())
    received = _HaloReceive.apply(payload, participation, plan.local.halo_source_rank, cp_group, observer)
    sent = _HaloSend.apply(payload, plan.local.successor_rank, cp_group, observer)
    received_parts = list(received.split([halo_width] * len(sample_tensors), dim=sequence_dim))
    with_halo = torch.cat(
        [torch.cat((halo, sample), dim=sequence_dim) for halo, sample in zip(received_parts, sample_tensors)],
        dim=sequence_dim,
    ).contiguous()
    with_halo = with_halo + sent[(0,) * sent.ndim].to(dtype=with_halo.dtype) * 0
    halo_lengths = [length + halo_width for length in lengths]
    cu_with_halo = owned.new_tensor([0] + halo_lengths, dtype=torch.int32).cumsum(0)
    return with_halo, cu_with_halo


def trim_conv_halo(
    tensor: Tensor,
    *,
    plan: GdnLosslessRuntimePlan,
    kernel_size: int,
    sequence_dim: int = 1,
) -> Tensor:
    """Remove the per-sample prefix added by :func:`exchange_conv_halo`."""
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}")
    if kernel_size == 1 or plan.local.owned_token_count == 0:
        return tensor
    sequence_dim %= tensor.ndim
    halo_width = kernel_size - 1
    halo_lengths = [sample.length + halo_width for sample in plan.local.samples]
    if sum(halo_lengths) != int(tensor.size(sequence_dim)):
        raise ValueError("halo tensor length does not match the GDN plan")
    samples = tensor.split(halo_lengths, dim=sequence_dim)
    return torch.cat(
        [sample.narrow(sequence_dim, halo_width, owned.length) for sample, owned in zip(samples, plan.local.samples)],
        dim=sequence_dim,
    ).contiguous()


__all__ = [
    "GdnLosslessRuntimePlan",
    "attach_state_dependency",
    "align_gdn_varlen_chunks",
    "compile_gdn_lossless_runtime_plan",
    "exchange_conv_halo",
    "make_state_participation",
    "make_state_template",
    "owned_to_physical",
    "physical_to_owned",
    "receive_initial_state",
    "send_final_state",
    "trim_conv_halo",
    "unpad_gdn_varlen_output",
]
