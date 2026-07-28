# Copyright (c) 2025 VeOmni Authors.
"""GDN state-passing CP: zigzag↔contiguous repartition, conv halo, serial/KCP scan.

Contract: [[GDN State-Passing CP Implementation Contract 20260727]]

* Softmax keeps zigzag Ring shards from the collator.
* GDN must consume **contiguous** time blocks of length ``S/cp`` without ever
  materializing ``S_full`` (INV-1 / INV-5).
* ``gather_full_replicated`` remains in ``gdn_sp.py`` as the lossless baseline.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from .sharding import balanced_cp_slice


# ---------------------------------------------------------------------------
# Impl identity (lossless / lossy families)
# ---------------------------------------------------------------------------

GDN_CP_IMPL_GATHER_FULL = "gather_full"
GDN_CP_IMPL_GATHER_FULL_REPLICATED = "gather_full_replicated"  # alias used in #969
GDN_CP_IMPL_STATE_PASSING_SERIAL = "state_passing_serial"
GDN_CP_IMPL_KCP = "kcp"
GDN_CP_IMPL_NONE = "none"

_LOSSLESS_IMPLS = frozenset(
    {
        GDN_CP_IMPL_GATHER_FULL,
        GDN_CP_IMPL_GATHER_FULL_REPLICATED,
        GDN_CP_IMPL_STATE_PASSING_SERIAL,
        GDN_CP_IMPL_NONE,
    }
)


def normalize_gdn_cp_impl(name: Optional[str]) -> str:
    if name is None or name == "":
        # Default stays gather_full until serial/kcp pass gates (contract).
        return GDN_CP_IMPL_GATHER_FULL_REPLICATED
    key = str(name).strip().lower()
    aliases = {
        "gather_full": GDN_CP_IMPL_GATHER_FULL_REPLICATED,
        "gather_full_replicated": GDN_CP_IMPL_GATHER_FULL_REPLICATED,
        "gather-full": GDN_CP_IMPL_GATHER_FULL_REPLICATED,
        "serial": GDN_CP_IMPL_STATE_PASSING_SERIAL,
        "state_passing": GDN_CP_IMPL_STATE_PASSING_SERIAL,
        "state_passing_serial": GDN_CP_IMPL_STATE_PASSING_SERIAL,
        "state-passing-serial": GDN_CP_IMPL_STATE_PASSING_SERIAL,
        "kcp": GDN_CP_IMPL_KCP,
        "none": GDN_CP_IMPL_NONE,
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown gdn_cp_impl={name!r}. "
            f"Expected one of {sorted(set(aliases.values()))}."
        )
    return aliases[key]


def resolve_gdn_cp_impl_from_env(*, cp_size: int) -> str:
    if int(cp_size) <= 1:
        return GDN_CP_IMPL_NONE
    return normalize_gdn_cp_impl(os.environ.get("VEOMNI_GDN_CP_IMPL"))


def gdn_cp_impl_is_lossless(name: str) -> bool:
    return normalize_gdn_cp_impl(name) in _LOSSLESS_IMPLS or normalize_gdn_cp_impl(name) == GDN_CP_IMPL_NONE


def gdn_replication_factor_for_impl(name: str, *, cp_size: int) -> int:
    impl = normalize_gdn_cp_impl(name) if int(cp_size) > 1 else GDN_CP_IMPL_NONE
    if impl in (GDN_CP_IMPL_GATHER_FULL, GDN_CP_IMPL_GATHER_FULL_REPLICATED):
        return max(int(cp_size), 1)
    return 1


# ---------------------------------------------------------------------------
# Zigzag ↔ contiguous chunk index math (INV-5: permutation, not gather)
# ---------------------------------------------------------------------------

def _validate_cp(cp_size: int, cp_rank: int) -> None:
    if cp_size < 1:
        raise ValueError(f"cp_size must be positive, got {cp_size}.")
    if not 0 <= cp_rank < cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}.")


def zigzag_owner_of_chunk(chunk_idx: int, cp_size: int) -> Tuple[int, int]:
    """Return ``(zigzag_rank, half_idx)`` that holds canonical chunk ``chunk_idx``.

    Balanced CP: rank ``r`` holds ``(chunk_r, chunk_{2C-1-r})`` as halves 0/1.
    """
    _validate_cp(cp_size, 0)
    n = 2 * cp_size
    if not 0 <= chunk_idx < n:
        raise ValueError(f"chunk_idx must be in [0, {n}), got {chunk_idx}.")
    if chunk_idx < cp_size:
        return chunk_idx, 0
    return n - 1 - chunk_idx, 1


def contiguous_block_chunk_indices(cp_rank: int, cp_size: int) -> Tuple[int, int]:
    """Canonical chunk indices owned by contiguous block rank ``cp_rank``."""
    _validate_cp(cp_size, cp_rank)
    return 2 * cp_rank, 2 * cp_rank + 1


def zigzag_half_destination(cp_rank: int, half_idx: int, cp_size: int) -> int:
    """Contiguous-block rank that should receive zigzag rank's half ``half_idx``."""
    _validate_cp(cp_size, cp_rank)
    if half_idx not in (0, 1):
        raise ValueError(f"half_idx must be 0 or 1, got {half_idx}.")
    n = 2 * cp_size
    chunk_idx = cp_rank if half_idx == 0 else n - 1 - cp_rank
    return chunk_idx // 2


def _split_two_halves(tensor: Tensor, seq_dim: int) -> Tuple[Tensor, Tensor]:
    seq_dim = seq_dim % tensor.ndim
    length = tensor.size(seq_dim)
    if length % 2 != 0:
        raise ValueError(f"Zigzag local length ({length}) must be even.")
    half = length // 2
    return tensor.narrow(seq_dim, 0, half).contiguous(), tensor.narrow(seq_dim, half, half).contiguous()


def _cat_two(a: Tensor, b: Tensor, seq_dim: int) -> Tensor:
    return torch.cat((a, b), dim=seq_dim).contiguous()


def simulate_zigzag_to_block(
    zigzag_shards: Sequence[Tensor],
    *,
    cp_size: int,
    seq_dim: int = 1,
) -> List[Tensor]:
    """Single-process reference for zigzag→contiguous (no full-sequence gather)."""
    if len(zigzag_shards) != cp_size:
        raise ValueError(f"Need {cp_size} zigzag shards, got {len(zigzag_shards)}.")
    # Collect every canonical chunk without building S_full as one tensor first:
    # we only keep per-chunk pieces then cat two per block.
    chunks: dict[int, Tensor] = {}
    for z_rank, shard in enumerate(zigzag_shards):
        early, late = _split_two_halves(shard, seq_dim)
        chunks[z_rank] = early
        chunks[2 * cp_size - 1 - z_rank] = late
    blocks = []
    for c_rank in range(cp_size):
        i0, i1 = contiguous_block_chunk_indices(c_rank, cp_size)
        blocks.append(_cat_two(chunks[i0], chunks[i1], seq_dim))
    return blocks


def simulate_block_to_zigzag(
    contig_blocks: Sequence[Tensor],
    *,
    cp_size: int,
    seq_dim: int = 1,
) -> List[Tensor]:
    """Inverse of :func:`simulate_zigzag_to_block`."""
    if len(contig_blocks) != cp_size:
        raise ValueError(f"Need {cp_size} contiguous blocks, got {len(contig_blocks)}.")
    chunks: dict[int, Tensor] = {}
    for c_rank, block in enumerate(contig_blocks):
        left, right = _split_two_halves(block, seq_dim)
        i0, i1 = contiguous_block_chunk_indices(c_rank, cp_size)
        chunks[i0] = left
        chunks[i1] = right
    zig = []
    for z_rank in range(cp_size):
        zig.append(_cat_two(chunks[z_rank], chunks[2 * cp_size - 1 - z_rank], seq_dim))
    return zig


def assert_no_full_sequence(tensor: Tensor, *, cp_size: int, seq_dim: int, full_length: int) -> None:
    """INV-1 helper: local tensor must be ``full_length / cp_size``."""
    if cp_size <= 1:
        return
    local = tensor.size(seq_dim % tensor.ndim)
    expected = full_length // cp_size
    if local != expected:
        raise RuntimeError(
            f"INV-1 violated: GDN local seq={local}, expected S/cp={expected} "
            f"(S={full_length}, cp={cp_size}). Full-sequence materialization is forbidden."
        )
    if local == full_length and cp_size > 1:
        raise RuntimeError("INV-1 violated: local seq equals S_full under cp_size>1.")


# ---------------------------------------------------------------------------
# Distributed permutation via all_to_all (INV-5)
# ---------------------------------------------------------------------------

def _all_to_all_halves(
    local_zigzag: Tensor,
    *,
    group: ProcessGroup,
    cp_size: int,
    cp_rank: int,
    seq_dim: int,
) -> Tensor:
    """Zigzag local → contiguous block via uniform all_to_all of half-chunks.

    Each rank sends ``cp_size`` tensors of shape ``half`` (zeros if not destined
    to that rank). Receiver concatenates the two nonzero halves for its block
    in canonical chunk order. Never materializes ``S_full``.
    """
    seq_dim = seq_dim % local_zigzag.ndim
    early, late = _split_two_halves(local_zigzag, seq_dim)
    half_len = early.size(seq_dim)
    zero = torch.zeros_like(early)

    if cp_size == 1:
        return local_zigzag

    send_list: List[Tensor] = []
    for dst in range(cp_size):
        if zigzag_half_destination(cp_rank, 0, cp_size) == dst:
            send_list.append(early)
        elif zigzag_half_destination(cp_rank, 1, cp_size) == dst:
            send_list.append(late)
        else:
            send_list.append(zero.clone())

    for i, t in enumerate(send_list):
        if t.size(seq_dim) != half_len:
            raise RuntimeError(
                f"Uniform half all_to_all broken: send[{i}] seq={t.size(seq_dim)} half={half_len}"
            )

    recv_list = [torch.empty_like(early) for _ in range(cp_size)]
    dist.all_to_all(recv_list, send_list, group=group)

    i0, i1 = contiguous_block_chunk_indices(cp_rank, cp_size)
    src0, half0 = zigzag_owner_of_chunk(i0, cp_size)
    src1, half1 = zigzag_owner_of_chunk(i1, cp_size)
    left = recv_list[src0]
    right = recv_list[src1]
    if zigzag_half_destination(src0, half0, cp_size) != cp_rank:
        raise RuntimeError("routing inconsistency for left chunk")
    if zigzag_half_destination(src1, half1, cp_size) != cp_rank:
        raise RuntimeError("routing inconsistency for right chunk")
    out = _cat_two(left, right, seq_dim)
    assert_no_full_sequence(
        out,
        cp_size=cp_size,
        seq_dim=seq_dim,
        full_length=local_zigzag.size(seq_dim) * cp_size,
    )
    return out


def _all_to_all_halves_inverse(
    local_block: Tensor,
    *,
    group: ProcessGroup,
    cp_size: int,
    cp_rank: int,
    seq_dim: int,
) -> Tensor:
    """Contiguous block → zigzag local (inverse permutation)."""
    seq_dim = seq_dim % local_block.ndim
    left, right = _split_two_halves(local_block, seq_dim)
    i0, i1 = contiguous_block_chunk_indices(cp_rank, cp_size)
    # chunk i0 must go to zigzag owner as the correct half
    # Build send_list indexed by zigzag destination rank.
    early_like = left
    zero = torch.zeros_like(early_like)
    send_list: List[Tensor] = [zero.clone() for _ in range(cp_size)]

    def _place(chunk: Tensor, chunk_idx: int) -> None:
        z_rank, _half = zigzag_owner_of_chunk(chunk_idx, cp_size)
        send_list[z_rank] = chunk

    _place(left, i0)
    _place(right, i1)

    if cp_size == 1:
        return local_block

    recv_list = [torch.empty_like(early_like) for _ in range(cp_size)]
    dist.all_to_all(recv_list, send_list, group=group)

    early_chunk_idx = cp_rank
    late_chunk_idx = 2 * cp_size - 1 - cp_rank
    src_early = early_chunk_idx // 2
    src_late = late_chunk_idx // 2
    early = recv_list[src_early]
    late = recv_list[src_late]
    out = _cat_two(early, late, seq_dim)
    assert_no_full_sequence(
        out,
        cp_size=cp_size,
        seq_dim=seq_dim,
        full_length=local_block.size(seq_dim) * cp_size,
    )
    return out


class _RepartitionZigzagToBlock(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        tensor: Tensor,
        group: ProcessGroup,
        cp_size: int,
        cp_rank: int,
        seq_dim: int,
    ) -> Tensor:
        ctx.group = group
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        ctx.seq_dim = seq_dim
        return _all_to_all_halves(
            tensor, group=group, cp_size=cp_size, cp_rank=cp_rank, seq_dim=seq_dim
        )

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None, None, None, None]:
        grad_input = _all_to_all_halves_inverse(
            grad_output.contiguous(),
            group=ctx.group,
            cp_size=ctx.cp_size,
            cp_rank=ctx.cp_rank,
            seq_dim=ctx.seq_dim,
        )
        return grad_input, None, None, None, None


class _RepartitionBlockToZigzag(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        tensor: Tensor,
        group: ProcessGroup,
        cp_size: int,
        cp_rank: int,
        seq_dim: int,
    ) -> Tensor:
        ctx.group = group
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        ctx.seq_dim = seq_dim
        return _all_to_all_halves_inverse(
            tensor, group=group, cp_size=cp_size, cp_rank=cp_rank, seq_dim=seq_dim
        )

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None, None, None, None]:
        grad_input = _all_to_all_halves(
            grad_output.contiguous(),
            group=ctx.group,
            cp_size=ctx.cp_size,
            cp_rank=ctx.cp_rank,
            seq_dim=ctx.seq_dim,
        )
        return grad_input, None, None, None, None


def repartition_zigzag_to_block(
    tensor: Tensor,
    *,
    cp_group: Optional[ProcessGroup],
    cp_size: int,
    cp_rank: int,
    seq_dim: int = 1,
) -> Tensor:
    """Permute zigzag CP shard → contiguous time block (length still ``S/cp``)."""
    if cp_size <= 1:
        return tensor
    if cp_group is None:
        raise RuntimeError("cp_group is required for zigzag→block repartition.")
    return _RepartitionZigzagToBlock.apply(tensor, cp_group, cp_size, cp_rank, seq_dim)


def repartition_block_to_zigzag(
    tensor: Tensor,
    *,
    cp_group: Optional[ProcessGroup],
    cp_size: int,
    cp_rank: int,
    seq_dim: int = 1,
) -> Tensor:
    """Inverse of :func:`repartition_zigzag_to_block`."""
    if cp_size <= 1:
        return tensor
    if cp_group is None:
        raise RuntimeError("cp_group is required for block→zigzag repartition.")
    return _RepartitionBlockToZigzag.apply(tensor, cp_group, cp_size, cp_rank, seq_dim)


def _cu_lengths(cu_seqlens: Tensor) -> List[int]:
    points = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    if not points or points[0] != 0:
        raise ValueError(f"cu_seqlens must start at 0, got {points[:3]}")
    return [end - start for start, end in zip(points[:-1], points[1:])]


def repartition_zigzag_to_block_packed(
    tensor: Tensor,
    *,
    cp_local_cu_seqlens: Tensor,
    cp_group: Optional[ProcessGroup],
    cp_size: int,
    cp_rank: int,
    seq_dim: int = 1,
) -> Tensor:
    """Per-sample zigzag→block (packed). Each sample stays length ``L_s/cp``."""
    if cp_size <= 1:
        return tensor
    lengths = _cu_lengths(cp_local_cu_seqlens)
    if sum(lengths) != tensor.size(seq_dim % tensor.ndim):
        raise ValueError(
            f"Packed local length mismatch: sum(cu)={sum(lengths)} vs seq={tensor.size(seq_dim)}"
        )
    parts = tensor.split(lengths, dim=seq_dim % tensor.ndim)
    out = [
        repartition_zigzag_to_block(
            p.contiguous(),
            cp_group=cp_group,
            cp_size=cp_size,
            cp_rank=cp_rank,
            seq_dim=seq_dim,
        )
        for p in parts
    ]
    return torch.cat(out, dim=seq_dim % tensor.ndim).contiguous()


def repartition_block_to_zigzag_packed(
    tensor: Tensor,
    *,
    cp_local_cu_seqlens: Tensor,
    cp_group: Optional[ProcessGroup],
    cp_size: int,
    cp_rank: int,
    seq_dim: int = 1,
) -> Tensor:
    if cp_size <= 1:
        return tensor
    lengths = _cu_lengths(cp_local_cu_seqlens)
    parts = tensor.split(lengths, dim=seq_dim % tensor.ndim)
    out = [
        repartition_block_to_zigzag(
            p.contiguous(),
            cp_group=cp_group,
            cp_size=cp_size,
            cp_rank=cp_rank,
            seq_dim=seq_dim,
        )
        for p in parts
    ]
    return torch.cat(out, dim=seq_dim % tensor.ndim).contiguous()


# ---------------------------------------------------------------------------
# Conv1d halo (after repartition)
# ---------------------------------------------------------------------------

class _Conv1dHaloExchange(torch.autograd.Function):
    """Dense causal halo with gradient exchange on the trailing/leading edges."""

    @staticmethod
    def forward(
        ctx: Any,
        x_block: Tensor,
        kernel_size: int,
        group: ProcessGroup,
        cp_size: int,
        cp_rank: int,
        seq_dim: int,
    ) -> Tensor:
        seq_dim = seq_dim % x_block.ndim
        halo = int(kernel_size) - 1
        if halo <= 0 or cp_size <= 1:
            ctx.noop = True
            return x_block
        if x_block.size(seq_dim) < halo:
            raise ValueError(
                f"Contiguous block length ({x_block.size(seq_dim)}) < conv halo ({halo})."
            )
        send = x_block.narrow(seq_dim, x_block.size(seq_dim) - halo, halo).contiguous()
        recv = torch.zeros_like(send)
        group_ranks = dist.get_process_group_ranks(group)
        ops = []
        if cp_rank < cp_size - 1:
            ops.append(dist.isend(send, group_ranks[cp_rank + 1], group=group))
        if cp_rank > 0:
            ops.append(dist.irecv(recv, group_ranks[cp_rank - 1], group=group))
        for op in ops:
            op.wait()
        left = torch.zeros_like(send) if cp_rank == 0 else recv
        ctx.noop = False
        ctx.group = group
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        ctx.seq_dim = seq_dim
        ctx.halo = halo
        return torch.cat((left, x_block), dim=seq_dim).contiguous()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None, None, None, None, None]:
        if getattr(ctx, "noop", True):
            return grad_output, None, None, None, None, None
        seq_dim = ctx.seq_dim
        halo = ctx.halo
        grad_left = grad_output.narrow(seq_dim, 0, halo).contiguous()
        grad_local = grad_output.narrow(
            seq_dim, halo, grad_output.size(seq_dim) - halo
        ).contiguous()
        group_ranks = dist.get_process_group_ranks(ctx.group)
        send_shape = list(grad_local.shape)
        send_shape[seq_dim] = halo
        grad_from_next = torch.zeros(send_shape, dtype=grad_local.dtype, device=grad_local.device)
        ops = []
        if ctx.cp_rank > 0:
            ops.append(dist.isend(grad_left, group_ranks[ctx.cp_rank - 1], group=ctx.group))
        if ctx.cp_rank < ctx.cp_size - 1:
            ops.append(dist.irecv(grad_from_next, group_ranks[ctx.cp_rank + 1], group=ctx.group))
        for op in ops:
            op.wait()
        if ctx.cp_rank < ctx.cp_size - 1:
            # Add grads that the next rank attributed to our trailing halo tokens.
            trailing = grad_local.narrow(seq_dim, grad_local.size(seq_dim) - halo, halo)
            trailing.add_(grad_from_next)
        return grad_local, None, None, None, None, None


def conv1d_halo_exchange(
    x_block: Tensor,
    *,
    kernel_size: int,
    cp_group: Optional[ProcessGroup],
    cp_size: int,
    cp_rank: int,
    seq_dim: int = 1,
) -> Tensor:
    """Prepend ``(kernel_size-1)`` tokens from rank ``r-1`` (zeros on rank 0).

    Must run **after** zigzag→block repartition. Does not gather ``S_full``.
    Dense (single-segment) helper; prefer :func:`conv1d_halo_exchange_packed` for varlen.
    """
    if kernel_size <= 1 or cp_size <= 1:
        return x_block
    if cp_group is None:
        raise RuntimeError("cp_group is required for conv1d halo exchange.")
    return _Conv1dHaloExchange.apply(
        x_block, int(kernel_size), cp_group, int(cp_size), int(cp_rank), int(seq_dim)
    )


class _Conv1dHaloExchangePacked(torch.autograd.Function):
    """Packed per-sample causal halo with autograd-safe edge exchange."""

    @staticmethod
    def forward(
        ctx: Any,
        x_block: Tensor,
        cp_local_cu_seqlens: Tensor,
        kernel_size: int,
        group: ProcessGroup,
        cp_size: int,
        cp_rank: int,
        seq_dim: int,
    ) -> Tensor:
        seq_dim = seq_dim % x_block.ndim
        halo = int(kernel_size) - 1
        lengths = _cu_lengths(cp_local_cu_seqlens)
        if halo <= 0 or cp_size <= 1:
            ctx.noop = True
            ctx.lengths = lengths
            ctx.halo = 0
            ctx.seq_dim = seq_dim
            return x_block
        parts = list(x_block.split(lengths, dim=seq_dim))
        for i, p in enumerate(parts):
            if p.size(seq_dim) < halo:
                raise ValueError(f"Sample {i} local length {p.size(seq_dim)} < conv halo {halo}.")
        send = torch.cat(
            [p.narrow(seq_dim, p.size(seq_dim) - halo, halo).contiguous() for p in parts],
            dim=seq_dim,
        )
        recv = torch.zeros_like(send)
        group_ranks = dist.get_process_group_ranks(group)
        ops = []
        if cp_rank < cp_size - 1:
            ops.append(dist.isend(send, group_ranks[cp_rank + 1], group=group))
        if cp_rank > 0:
            ops.append(dist.irecv(recv, group_ranks[cp_rank - 1], group=group))
        for op in ops:
            op.wait()
        left_parts = (
            list(torch.zeros_like(send).split([halo] * len(parts), dim=seq_dim))
            if cp_rank == 0
            else list(recv.split([halo] * len(parts), dim=seq_dim))
        )
        out_parts = [
            torch.cat((left_parts[i], parts[i]), dim=seq_dim).contiguous() for i in range(len(parts))
        ]
        ctx.noop = False
        ctx.group = group
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        ctx.seq_dim = seq_dim
        ctx.halo = halo
        ctx.lengths = lengths
        return torch.cat(out_parts, dim=seq_dim).contiguous()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None, None, None, None, None, None]:
        if getattr(ctx, "noop", True):
            return grad_output, None, None, None, None, None, None
        seq_dim = ctx.seq_dim
        halo = ctx.halo
        lengths = ctx.lengths
        halo_lengths = [length + halo for length in lengths]
        parts = list(grad_output.split(halo_lengths, dim=seq_dim))
        grad_left = torch.cat(
            [p.narrow(seq_dim, 0, halo).contiguous() for p in parts],
            dim=seq_dim,
        )
        grad_local_parts = [
            p.narrow(seq_dim, halo, p.size(seq_dim) - halo).contiguous() for p in parts
        ]
        group_ranks = dist.get_process_group_ranks(ctx.group)
        grad_from_next = torch.zeros_like(grad_left)
        ops = []
        if ctx.cp_rank > 0:
            ops.append(dist.isend(grad_left, group_ranks[ctx.cp_rank - 1], group=ctx.group))
        if ctx.cp_rank < ctx.cp_size - 1:
            ops.append(dist.irecv(grad_from_next, group_ranks[ctx.cp_rank + 1], group=ctx.group))
        for op in ops:
            op.wait()
        if ctx.cp_rank < ctx.cp_size - 1:
            next_parts = list(grad_from_next.split([halo] * len(grad_local_parts), dim=seq_dim))
            for i, local in enumerate(grad_local_parts):
                local.narrow(seq_dim, local.size(seq_dim) - halo, halo).add_(next_parts[i])
        grad_local = torch.cat(grad_local_parts, dim=seq_dim).contiguous()
        return grad_local, None, None, None, None, None, None


def conv1d_halo_exchange_packed(
    x_block: Tensor,
    *,
    cp_local_cu_seqlens: Tensor,
    kernel_size: int,
    cp_group: Optional[ProcessGroup],
    cp_size: int,
    cp_rank: int,
    seq_dim: int = 1,
) -> Tuple[Tensor, Tensor]:
    """Per-sample causal halo. Returns ``(x_with_halo, cu_seqlens_with_halo)``."""
    if kernel_size <= 1 or cp_size <= 1:
        return x_block, cp_local_cu_seqlens
    if cp_group is None:
        raise RuntimeError("cp_group is required for packed conv1d halo.")
    out = _Conv1dHaloExchangePacked.apply(
        x_block,
        cp_local_cu_seqlens,
        int(kernel_size),
        cp_group,
        int(cp_size),
        int(cp_rank),
        int(seq_dim),
    )
    halo = int(kernel_size) - 1
    lengths = _cu_lengths(cp_local_cu_seqlens)
    new_lengths = [length + halo for length in lengths]
    cu = torch.tensor([0] + new_lengths, dtype=torch.int32, device=x_block.device)
    cu = cu.cumsum(0).to(dtype=torch.int32)
    return out, cu


def trim_conv_halo_packed(
    x_with_halo: Tensor,
    *,
    cp_local_cu_seqlens_with_halo: Tensor,
    kernel_size: int,
    seq_dim: int = 1,
) -> Tensor:
    """Drop the leading ``(kernel_size-1)`` halo tokens from each packed sample."""
    if kernel_size <= 1:
        return x_with_halo
    seq_dim = seq_dim % x_with_halo.ndim
    halo = kernel_size - 1
    lengths = _cu_lengths(cp_local_cu_seqlens_with_halo)
    parts = x_with_halo.split(lengths, dim=seq_dim)
    trimmed = [p.narrow(seq_dim, halo, p.size(seq_dim) - halo).contiguous() for p in parts]
    return torch.cat(trimmed, dim=seq_dim).contiguous()


class _SerialRecvInitialState(torch.autograd.Function):
    """Forward: recv ``S_init`` from r-1 (zeros on rank 0). Backward: send ``dS_init`` to r-1."""

    @staticmethod
    def forward(
        ctx: Any,
        state_template: Tensor,
        group: ProcessGroup,
        cp_size: int,
        cp_rank: int,
    ) -> Tensor:
        ctx.group = group
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        if cp_size <= 1 or cp_rank == 0:
            return torch.zeros_like(state_template)
        group_ranks = dist.get_process_group_ranks(group)
        s_init = torch.empty_like(state_template)
        dist.recv(s_init, group_ranks[cp_rank - 1], group=group)
        return s_init

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[None, None, None, None]:
        if ctx.cp_size > 1 and ctx.cp_rank > 0:
            group_ranks = dist.get_process_group_ranks(ctx.group)
            dist.send(grad_output.contiguous(), group_ranks[ctx.cp_rank - 1], group=ctx.group)
        return None, None, None, None


class _SerialSendFinalState(torch.autograd.Function):
    """Forward: send ``S_final`` to r+1. Backward: recv ``dS_final`` from r+1 and add."""

    @staticmethod
    def forward(
        ctx: Any,
        final_state: Tensor,
        group: ProcessGroup,
        cp_size: int,
        cp_rank: int,
    ) -> Tensor:
        ctx.group = group
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        if cp_size > 1 and cp_rank < cp_size - 1:
            group_ranks = dist.get_process_group_ranks(group)
            # Peer recv uses empty_like(state_template); Mojo S_final is fp32.
            # Fail closed on non-finite / wrong dtype rather than silently corrupt.
            payload = final_state.detach().contiguous()
            if payload.dtype != torch.float32:
                payload = payload.float().contiguous()
            if not torch.isfinite(payload).all():
                raise RuntimeError(
                    "state_passing_serial: refusing to send non-finite S_final "
                    f"(cp_rank={cp_rank})"
                )
            dist.send(payload, group_ranks[cp_rank + 1], group=group)
        return final_state

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None, None, None]:
        grad = grad_output.contiguous()
        if ctx.cp_size > 1 and ctx.cp_rank < ctx.cp_size - 1:
            group_ranks = dist.get_process_group_ranks(ctx.group)
            grad_from_next = torch.empty_like(grad)
            dist.recv(grad_from_next, group_ranks[ctx.cp_rank + 1], group=ctx.group)
            grad = grad + grad_from_next
        return grad, None, None, None


def recv_serial_initial_state(
    *,
    cp_group: ProcessGroup,
    cp_size: int,
    cp_rank: int,
    state_template: Tensor,
) -> Tensor:
    """Receive ``S_init`` from rank r-1 (zeros on rank 0). Autograd-safe."""
    return _SerialRecvInitialState.apply(
        state_template, cp_group, int(cp_size), int(cp_rank)
    )


def send_serial_final_state(
    final_state: Tensor,
    *,
    cp_group: ProcessGroup,
    cp_size: int,
    cp_rank: int,
) -> Tensor:
    """Send ``S_final`` to rank r+1; returns ``final_state`` (identity) for autograd."""
    return _SerialSendFinalState.apply(
        final_state, cp_group, int(cp_size), int(cp_rank)
    )


def apply_sample_bos_zero_s_init(
    s_init: Tensor,
    *,
    sample_bos_on_rank: Sequence[bool],
) -> Tensor:
    """Force ``S_init[i]=0`` when this rank's block starts a new packed sample.

    Under per-sample balanced CP every sample BOS lives on ``cp_rank==0``, but
    callers that ever repartition a *global* packed stream must mark BOS per
    sample explicitly — serial state must not cross sample boundaries.
    """
    if s_init.ndim < 1:
        raise ValueError(f"s_init must be at least 1D, got shape={tuple(s_init.shape)}")
    n = int(s_init.shape[0])
    if len(sample_bos_on_rank) != n:
        raise ValueError(
            f"sample_bos_on_rank length ({len(sample_bos_on_rank)}) != num_seqs ({n})"
        )
    if not any(sample_bos_on_rank):
        return s_init
    out = s_init
    wrote = False
    for i, is_bos in enumerate(sample_bos_on_rank):
        if not is_bos:
            continue
        if not wrote:
            out = s_init.clone()
            wrote = True
        out[i].zero_()
    return out


def assert_serial_initial_state_contract(
    s_init: Tensor,
    *,
    cp_rank: int,
    sample_bos_on_rank: Sequence[bool],
    cu_seqlens: Optional[Tensor] = None,
    atol: float = 0.0,
) -> None:
    """Fail-closed serial S_init checks (INV + sample-BOS).

    * ``s_init`` must be finite.
    * ``cp_rank==0`` ⇒ entire ``S_init`` is zero (no left neighbor).
    * every sample marked BOS on this rank ⇒ that sample's ``S_init`` is zero.
    * optional ``cu_seqlens``: ``cu[0]==0``, monotone, ``cu[-1]`` matches when known.
    """
    if not torch.isfinite(s_init.detach()).all():
        raise RuntimeError("state_passing_serial: S_init has non-finite values")
    absmax = float(s_init.detach().float().abs().max().item()) if s_init.numel() else 0.0
    if int(cp_rank) == 0 and absmax > atol:
        raise RuntimeError(
            f"state_passing_serial: cp_rank0 S_init must be zero, absmax={absmax}"
        )
    n = int(s_init.shape[0])
    if len(sample_bos_on_rank) != n:
        raise RuntimeError(
            f"state_passing_serial: sample_bos_on_rank length "
            f"{len(sample_bos_on_rank)} != num_seqs {n}"
        )
    for i, is_bos in enumerate(sample_bos_on_rank):
        if not is_bos:
            continue
        sample_abs = float(s_init[i].detach().float().abs().max().item())
        if sample_abs > atol:
            raise RuntimeError(
                f"state_passing_serial: sample {i} BOS on cp_rank={cp_rank} "
                f"but S_init absmax={sample_abs} (must be 0; serial must not "
                f"carry previous sample state across packed boundary)"
            )
    if cu_seqlens is not None:
        points = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
        if not points or points[0] != 0:
            raise RuntimeError(f"state_passing_serial: cu_seqlens must start at 0, got {points[:3]}")
        if any(end < start for start, end in zip(points[:-1], points[1:])):
            raise RuntimeError(f"state_passing_serial: cu_seqlens not monotone: {points}")


def sample_bos_flags_for_per_sample_cp(*, num_seqs: int, cp_rank: int) -> List[bool]:
    """Per-sample balanced CP: every sample's time origin is on ``cp_rank==0``."""
    bos = int(cp_rank) == 0
    return [bos] * int(num_seqs)


def make_gdn_state_template(
    query: Tensor,
    value: Tensor,
    *,
    cu_seqlens: Optional[Tensor],
) -> Tensor:
    """Allocate zero state matching chunk_gated_delta_rule layout.

    Dense: ``[B, H, K, V]``. Varlen (``cu_seqlens``): ``[N, H, K, V]``.
    ``query``/``value`` must already be head-expanded (GQA repeat done).

    Dtype is **float32**: Mojo GDR returns fp32 ``final_state``. Serial CP
    send/recv uses ``empty_like(template)`` byte buffers — a bf16 template with
    an fp32 send silently corrupts ``S_init`` on the next rank (observed NaN on
    U2×CP2 NPU seam).
    """
    num_heads = int(query.shape[2])
    k_dim = int(query.shape[3])
    v_dim = int(value.shape[3])
    if cu_seqlens is None:
        batch = int(query.shape[0])
        return torch.zeros(
            batch, num_heads, k_dim, v_dim, device=query.device, dtype=torch.float32
        )
    num_seqs = int(cu_seqlens.numel() - 1)
    return torch.zeros(
        num_seqs, num_heads, k_dim, v_dim, device=query.device, dtype=torch.float32
    )


# 910B4 Mojo GDR varlen wrapper (MR912) pads each segment to this chunk size and
# refuse ``output_final_state=True`` when it must pad. Pre-align here so the
# wrapper takes the raw path and state-passing can read ``S_final``.
MOJO_GDR_CHUNK_SIZE = 32


def _ceil_to_chunk(length: int, chunk_size: int) -> int:
    return ((int(length) + chunk_size - 1) // chunk_size) * chunk_size


def align_gdn_varlen_for_mojo_gdr(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    cu_seqlens: Tensor,
    *,
    chunk_size: int = MOJO_GDR_CHUNK_SIZE,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Pad each varlen segment to ``chunk_size`` for Mojo GDR + final_state.

    Pad slots use ``v=g=beta=0`` so recurrent state is preserved through the
    tail (``S' = exp(0)*S + 0``). ``q``/``k`` copy the segment's last real
    token to avoid ``l2norm(0)`` NaNs when ``use_qk_l2norm_in_kernel=True``.

    Returns ``(q, k, v, g, beta, padded_cu, unpad_index)``. ``unpad_index`` is
    ``None`` when no padding was required.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            "align_gdn_varlen_for_mojo_gdr expects q/k/v rank 4, "
            f"got {(query.ndim, key.ndim, value.ndim)}"
        )
    if g.ndim != 3 or beta.ndim != 3:
        raise ValueError(
            f"align_gdn_varlen_for_mojo_gdr expects g/beta rank 3, got {(g.ndim, beta.ndim)}"
        )
    if int(query.shape[0]) != 1:
        raise ValueError("align_gdn_varlen_for_mojo_gdr requires batch size 1")

    lengths = _cu_lengths(cu_seqlens)
    padded_lengths = [_ceil_to_chunk(length, chunk_size) for length in lengths]
    if padded_lengths == lengths:
        return query, key, value, g, beta, cu_seqlens, None

    token_count = int(query.shape[1])
    padded_cursor = sum(padded_lengths)
    original_indices: List[int] = []
    padded_boundaries = [0]
    src_cursor = 0
    dst_cursor = 0
    for length, padded_length in zip(lengths, padded_lengths):
        original_indices.extend(range(dst_cursor, dst_cursor + length))
        src_cursor += length
        dst_cursor += padded_length
        padded_boundaries.append(dst_cursor)
    if src_cursor != token_count:
        raise RuntimeError(
            f"align_gdn_varlen_for_mojo_gdr token mismatch: walked={src_cursor} dim={token_count}"
        )

    index = torch.tensor(original_indices, dtype=torch.long, device=query.device)

    def _zeros_like_padded(tensor: Tensor) -> Tensor:
        shape = list(tensor.shape)
        shape[1] = padded_cursor
        return tensor.new_zeros(shape)

    # Zero-fill v/g/beta (state-preserving pads). Seed q/k from last real token.
    q_pad = _zeros_like_padded(query)
    k_pad = _zeros_like_padded(key)
    v_pad = _zeros_like_padded(value)
    g_pad = _zeros_like_padded(g)
    beta_pad = _zeros_like_padded(beta)

    q_pad.index_copy_(1, index, query)
    k_pad.index_copy_(1, index, key)
    v_pad.index_copy_(1, index, value)
    g_pad.index_copy_(1, index, g)
    beta_pad.index_copy_(1, index, beta)

    dst_cursor = 0
    src_cursor = 0
    for length, padded_length in zip(lengths, padded_lengths):
        pad = padded_length - length
        if pad > 0:
            last = src_cursor + length - 1
            fill_q = query[:, last : last + 1].expand(-1, pad, *([-1] * (query.ndim - 2)))
            fill_k = key[:, last : last + 1].expand(-1, pad, *([-1] * (key.ndim - 2)))
            q_pad[:, dst_cursor + length : dst_cursor + padded_length] = fill_q
            k_pad[:, dst_cursor + length : dst_cursor + padded_length] = fill_k
        src_cursor += length
        dst_cursor += padded_length

    padded_cu = torch.tensor(
        padded_boundaries,
        dtype=cu_seqlens.dtype,
        device=cu_seqlens.device,
    )
    return q_pad, k_pad, v_pad, g_pad, beta_pad, padded_cu, index


def unpad_gdn_varlen_output(output: Tensor, unpad_index: Optional[Tensor]) -> Tensor:
    """Drop Mojo chunk-alignment pad tokens from GDN output."""
    if unpad_index is None:
        return output
    return output.index_select(1, unpad_index)
