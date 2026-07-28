# Copyright (c) 2025 VeOmni Authors.
# Load-balancing helpers adapted from MindSpeed-LLM gdn_context_parallel (BSD-3-Clause).
"""GDN sequence-parallel helpers for hybrid Ulysses and context parallelism.

Gated DeltaNet has two independent distributed dimensions:

* Ulysses shards heads and gathers sequence with all-to-all.
* Context parallelism shards sequence while leaving the Ulysses-local heads
  unchanged.

The current correctness path gathers the CP sequence for each Ulysses head
shard, runs the recurrent core redundantly on every CP rank, and scatters the
output sequence again. The paired gather/scatter autograd functions preserve
the input-activation gradient. Only parameters consumed inside that repeated
core need a ``1 / cp_size`` gradient factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from ..parallel_state import get_parallel_state
from .sharding import balanced_cp_restore, balanced_cp_to_rank_major


# GDN CP impl family (see gdn_scan_cp + implementation contract):
#   gather_full_replicated — lossless redundant full-seq (legacy / golden)
#   state_passing_serial   — lossless S/cp + serial state pass (P0)
#   kcp                    — lossy parallel prefix (P1, opt-in only)
GDN_CP_IMPL_GATHER_FULL_REPLICATED = "gather_full_replicated"
GDN_CP_IMPL_STATE_PASSING_SERIAL = "state_passing_serial"
GDN_CP_IMPL_KCP = "kcp"
GDN_CP_IMPL_NONE = "none"
_GDN_CP_IMPL_LOGGED = False


def gdn_cp_impl_name(*, cp_size: int) -> str:
    if int(cp_size) <= 1:
        return GDN_CP_IMPL_NONE
    from .gdn_scan_cp import resolve_gdn_cp_impl_from_env

    return resolve_gdn_cp_impl_from_env(cp_size=cp_size)


def gdn_replication_factor(*, cp_size: int) -> int:
    """How many times the GDN core runs per global token under the active impl."""
    if int(cp_size) <= 1:
        return 1
    from .gdn_scan_cp import gdn_replication_factor_for_impl

    return gdn_replication_factor_for_impl(gdn_cp_impl_name(cp_size=cp_size), cp_size=cp_size)


def gdn_cp_identity(*, cp_size: int) -> dict[str, object]:
    """Stable identity dict for logs / wandb."""
    cp = max(int(cp_size), 1)
    impl = gdn_cp_impl_name(cp_size=cp)
    return {
        "gdn_cp_impl": impl,
        "gdn_replication_factor": gdn_replication_factor(cp_size=cp),
    }


def log_gdn_cp_impl_once(*, cp_size: int, logger: Any = None) -> dict[str, object]:
    """Rank-0 once: label active GDN CP impl (lossless/lossy)."""
    global _GDN_CP_IMPL_LOGGED
    identity = gdn_cp_identity(cp_size=cp_size)
    if _GDN_CP_IMPL_LOGGED:
        return identity
    _GDN_CP_IMPL_LOGGED = True
    impl = str(identity["gdn_cp_impl"])
    if int(cp_size) <= 1:
        note = "CP1: gdn_cp_impl=none"
    elif impl in (GDN_CP_IMPL_GATHER_FULL_REPLICATED, "gather_full"):
        note = "lossless gather→full GDN→scatter; NOT state-passing"
    elif impl == GDN_CP_IMPL_STATE_PASSING_SERIAL:
        note = "lossless zigzag→block + serial state pass; S/cp only"
    elif impl == GDN_CP_IMPL_KCP:
        note = "lossy KCP local+AG+merge; opt-in"
    else:
        note = f"impl={impl}"
    msg = (
        "AI4SE_GDN_CP_IMPL "
        f"gdn_cp_impl={identity['gdn_cp_impl']} "
        f"gdn_replication_factor={identity['gdn_replication_factor']} "
        f"({note})"
    )
    try:
        if dist.is_initialized() and dist.get_rank() != 0:
            return identity
    except Exception:
        pass
    if logger is not None:
        try:
            logger.warning(msg)
        except Exception:
            print(msg, flush=True)
    else:
        print(msg, flush=True)
    return identity


@dataclass(frozen=True)
class GdnParallelPlan:
    """Two-dimensional GDN plan without collapsing CP into head parallelism."""

    ulysses_group: Optional[ProcessGroup]
    ulysses_size: int
    ulysses_rank: int
    cp_group: Optional[ProcessGroup]
    cp_size: int
    cp_rank: int

    @property
    def head_parallel_enabled(self) -> bool:
        return self.ulysses_size > 1

    @property
    def cp_sequence_enabled(self) -> bool:
        return self.cp_size > 1

    @property
    def cp_seq_enabled(self) -> bool:
        """Compatibility alias used by the Ring-CP generated model patches."""
        return self.cp_sequence_enabled


def resolve_gdn_parallel_plan() -> GdnParallelPlan:
    """Resolve Ulysses head parallelism and CP sequence parallelism separately."""
    state = get_parallel_state()

    ulysses_group: Optional[ProcessGroup] = None
    ulysses_size = 1
    ulysses_rank = 0
    if getattr(state, "ulysses_enabled", False):
        ulysses_group = state.ulysses_group
        if ulysses_group is None:
            raise RuntimeError("Ulysses-enabled GDN requires ulysses_group.")
        ulysses_size = int(state.ulysses_size)
        ulysses_rank = int(state.ulysses_rank)

    cp_group: Optional[ProcessGroup] = None
    cp_size = 1
    cp_rank = 0
    if getattr(state, "cp_enabled", False):
        cp_group = state.cp_group
        if cp_group is None:
            raise RuntimeError("CP-enabled GDN requires cp_group.")
        cp_size = int(state.cp_size)
        cp_rank = int(state.cp_rank)

    return GdnParallelPlan(
        ulysses_group=ulysses_group,
        ulysses_size=ulysses_size,
        ulysses_rank=ulysses_rank,
        cp_group=cp_group,
        cp_size=cp_size,
        cp_rank=cp_rank,
    )


def resolve_gdn_sp_group() -> tuple[Optional[ProcessGroup], int, int, bool]:
    """Return the Ulysses-only group used for GDN head all-to-all.

    The final boolean is retained for compatibility with the original Ring-CP
    helper API. CP ordering is now handled by ``cp_gather_sequence`` and
    ``cp_scatter_sequence``, so no Ulysses tensor needs an inline zigzag
    transform.
    """
    plan = resolve_gdn_parallel_plan()
    if plan.head_parallel_enabled:
        return plan.ulysses_group, plan.ulysses_size, plan.ulysses_rank, False
    return None, 1, 0, False


def _cu_lengths(cu_seqlens: Tensor, *, expected_total: Optional[int] = None) -> list[int]:
    points = [int(point) for point in cu_seqlens.detach().cpu().tolist()]
    if not points or points[0] != 0:
        raise ValueError(f"cu_seqlens must start at 0, got {points[:3]}.")
    if any(end < start for start, end in zip(points[:-1], points[1:])):
        raise ValueError("cu_seqlens must be monotonically non-decreasing.")
    if expected_total is not None and points[-1] != expected_total:
        raise ValueError(f"cu_seqlens end ({points[-1]}) does not match sequence length ({expected_total}).")
    return [end - start for start, end in zip(points[:-1], points[1:])]


def _cu_from_lengths(lengths: list[int], *, reference: Tensor) -> Tensor:
    cu = torch.tensor([0] + lengths, dtype=torch.int32, device=reference.device)
    return cu.cumsum(0).to(dtype=torch.int32)


def derive_gdn_local_cu_seqlens(
    global_cu_seqlens: Tensor,
    *,
    ulysses_size: int,
    cp_size: int,
) -> tuple[Tensor, Tensor]:
    """Derive pre-Ulysses-local and post-Ulysses CP-local packed metadata."""
    if ulysses_size < 1 or cp_size < 1:
        raise ValueError(f"ulysses_size and cp_size must be positive, got {ulysses_size}, {cp_size}.")
    global_lengths = _cu_lengths(global_cu_seqlens)
    local_divisor = ulysses_size * cp_size
    for sample_idx, length in enumerate(global_lengths):
        if length % local_divisor != 0:
            raise ValueError(
                f"Packed GDN sample {sample_idx} length ({length}) must be divisible by "
                f"ulysses_size * cp_size ({local_divisor})."
            )
    ulysses_local_lengths = [length // local_divisor for length in global_lengths]
    cp_local_lengths = [length // cp_size for length in global_lengths]
    return (
        _cu_from_lengths(ulysses_local_lengths, reference=global_cu_seqlens),
        _cu_from_lengths(cp_local_lengths, reference=global_cu_seqlens),
    )


def _packed_cp_restore(
    rank_tensors: list[Tensor],
    *,
    local_lengths: list[int],
    cp_size: int,
    seq_dim: int,
) -> Tensor:
    offsets = [0]
    for length in local_lengths:
        offsets.append(offsets[-1] + length)
    restored_samples = []
    for sample_idx, local_length in enumerate(local_lengths):
        rank_major = torch.cat(
            [rank_tensor.narrow(seq_dim, offsets[sample_idx], local_length) for rank_tensor in rank_tensors],
            dim=seq_dim,
        )
        restored_samples.append(balanced_cp_restore(rank_major, cp_size=cp_size, dim=seq_dim))
    if restored_samples:
        return torch.cat(restored_samples, dim=seq_dim).contiguous()
    shape = list(rank_tensors[0].shape)
    shape[seq_dim] = 0
    return rank_tensors[0].new_empty(shape)


def _packed_cp_rank_shard(
    tensor: Tensor,
    *,
    local_lengths: list[int],
    cp_size: int,
    cp_rank: int,
    seq_dim: int,
) -> Tensor:
    full_lengths = [length * cp_size for length in local_lengths]
    samples = tensor.split(full_lengths, dim=seq_dim)
    local_samples = []
    for sample, local_length in zip(samples, local_lengths):
        rank_major = balanced_cp_to_rank_major(sample.contiguous(), cp_size=cp_size, dim=seq_dim)
        local_samples.append(rank_major.narrow(seq_dim, cp_rank * local_length, local_length))
    if local_samples:
        return torch.cat(local_samples, dim=seq_dim).contiguous()
    shape = list(tensor.shape)
    shape[seq_dim] = 0
    return tensor.new_empty(shape)


class _CpGatherSequence(torch.autograd.Function):
    """Gather balanced CP shards and restore canonical sequence order.

    This operation is paired with ``_CpScatterSequence``. Scatter backward
    reconstructs the complete output gradient on every CP rank, so gather
    backward only selects the current rank's input shard. It intentionally
    does not scale or reduce the input-activation gradient.
    """

    @staticmethod
    def forward(
        ctx: Any,
        tensor: Tensor,
        group: ProcessGroup,
        cp_size: int,
        seq_dim: int,
        cp_local_cu_seqlens: Tensor,
    ) -> Tensor:
        world_size = dist.get_world_size(group)
        if world_size != cp_size:
            raise RuntimeError(f"cp_group world size ({world_size}) does not match cp_size ({cp_size}).")

        ctx.group = group
        ctx.cp_size = cp_size
        ctx.seq_dim = seq_dim
        ctx.local_lengths = _cu_lengths(
            cp_local_cu_seqlens,
            expected_total=tensor.size(seq_dim),
        )

        try:
            from .gdn_mem_probe import emit_comm_layer, mem_probe_enabled

            if mem_probe_enabled():
                local_bytes = int(tensor.numel()) * int(tensor.element_size())
                emit_comm_layer(
                    impl="gather_full_replicated",
                    op="all_gather_sequence",
                    payload_bytes_local=local_bytes,
                    payload_bytes_total=local_bytes * int(cp_size),
                    shape=list(tensor.shape),
                    seq_len_local=int(tensor.size(seq_dim)),
                    depends_on_s=True,
                    extra={"seq_dim": int(seq_dim)},
                )
        except Exception:
            pass
        gathered = [torch.empty_like(tensor) for _ in range(cp_size)]
        dist.all_gather(gathered, tensor.contiguous(), group=group)
        return _packed_cp_restore(
            gathered,
            local_lengths=ctx.local_lengths,
            cp_size=cp_size,
            seq_dim=seq_dim,
        )

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None, None, None, None]:
        local_grad = _packed_cp_rank_shard(
            grad_output.contiguous(),
            local_lengths=ctx.local_lengths,
            cp_size=ctx.cp_size,
            cp_rank=dist.get_rank(ctx.group),
            seq_dim=ctx.seq_dim,
        )
        return local_grad, None, None, None, None


class _CpScatterSequence(torch.autograd.Function):
    """Select one balanced CP output shard from a canonical full sequence."""

    @staticmethod
    def forward(
        ctx: Any,
        tensor: Tensor,
        group: ProcessGroup,
        cp_size: int,
        cp_rank: int,
        seq_dim: int,
        cp_local_cu_seqlens: Tensor,
    ) -> Tensor:
        world_size = dist.get_world_size(group)
        if world_size != cp_size:
            raise RuntimeError(f"cp_group world size ({world_size}) does not match cp_size ({cp_size}).")
        if dist.get_rank(group) != cp_rank:
            raise RuntimeError(f"cp_rank ({cp_rank}) does not match group-local rank ({dist.get_rank(group)}).")
        local_lengths = _cu_lengths(cp_local_cu_seqlens)
        expected_full_length = sum(local_lengths) * cp_size
        if tensor.size(seq_dim) != expected_full_length:
            raise ValueError(
                f"GDN full sequence length ({tensor.size(seq_dim)}) does not match packed CP metadata "
                f"({expected_full_length})."
            )

        ctx.group = group
        ctx.cp_size = cp_size
        ctx.seq_dim = seq_dim
        ctx.local_lengths = local_lengths
        return _packed_cp_rank_shard(
            tensor.contiguous(),
            local_lengths=local_lengths,
            cp_size=cp_size,
            cp_rank=cp_rank,
            seq_dim=seq_dim,
        )

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None, None, None, None, None]:
        # The recurrent core couples tokens across CP shards. Every rank needs
        # the complete dOut to obtain the full gradient for its local dInput.
        gathered = [torch.empty_like(grad_output) for _ in range(ctx.cp_size)]
        dist.all_gather(gathered, grad_output.contiguous(), group=ctx.group)
        canonical = _packed_cp_restore(
            gathered,
            local_lengths=ctx.local_lengths,
            cp_size=ctx.cp_size,
            seq_dim=ctx.seq_dim,
        )
        return canonical, None, None, None, None, None


class _ScaleRepeatedParameterGrad(torch.autograd.Function):
    """Identity in forward; scale only this tensor's backward gradient."""

    @staticmethod
    def forward(ctx: Any, tensor: Tensor, cp_size: int) -> Tensor:
        ctx.cp_size = cp_size
        return tensor

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None]:
        return grad_output / ctx.cp_size, None


def cp_gather_sequence(
    tensor: Tensor,
    *,
    cp_group: Optional[ProcessGroup],
    cp_size: int,
    seq_dim: int = 1,
    cp_local_cu_seqlens: Optional[Tensor] = None,
) -> Tensor:
    """Gather balanced CP sequence shards into canonical sequence order.

    P0-C0: paired with full-sequence GDN on every CP rank (replicated compute).
    """
    if cp_size <= 1:
        return tensor
    log_gdn_cp_impl_once(cp_size=cp_size)
    if cp_group is None:
        raise RuntimeError("cp_group is required when cp_size > 1.")
    if cp_local_cu_seqlens is None:
        cp_local_cu_seqlens = torch.tensor(
            [0, tensor.size(seq_dim)],
            dtype=torch.int32,
            device=tensor.device,
        )
    return _CpGatherSequence.apply(tensor, cp_group, cp_size, seq_dim, cp_local_cu_seqlens)


def cp_scatter_sequence(
    tensor: Tensor,
    *,
    cp_group: Optional[ProcessGroup],
    cp_size: int,
    cp_rank: int,
    seq_dim: int = 1,
    cp_local_cu_seqlens: Optional[Tensor] = None,
) -> Tensor:
    """Scatter a canonical sequence back to the current balanced CP shard."""
    if cp_size <= 1:
        return tensor
    if cp_group is None:
        raise RuntimeError("cp_group is required when cp_size > 1.")
    if cp_local_cu_seqlens is None:
        if tensor.size(seq_dim) % cp_size != 0:
            raise ValueError(
                f"GDN full sequence length ({tensor.size(seq_dim)}) must be divisible by cp_size ({cp_size})."
            )
        cp_local_cu_seqlens = torch.tensor(
            [0, tensor.size(seq_dim) // cp_size],
            dtype=torch.int32,
            device=tensor.device,
        )
    return _CpScatterSequence.apply(
        tensor,
        cp_group,
        cp_size,
        cp_rank,
        seq_dim,
        cp_local_cu_seqlens,
    )


def scale_cp_repeated_parameter_grad(tensor: Tensor, *, cp_size: int) -> Tensor:
    """Scale gradients for a parameter used by every redundant CP core.

    Apply this only to parameters consumed between ``cp_gather_sequence`` and
    ``cp_scatter_sequence``. Do not apply it to input activations or to
    parameters used only before the gather or after the scatter.
    """
    if cp_size <= 1:
        return tensor
    return _ScaleRepeatedParameterGrad.apply(tensor, cp_size)


def maybe_restore_canonical(tensor: Tensor, *, enabled: bool, cp_size: int, dim: int = 1) -> Tensor:
    if not enabled or cp_size <= 1:
        return tensor
    return balanced_cp_restore(tensor, cp_size=cp_size, dim=dim)


def maybe_to_rank_major(tensor: Tensor, *, enabled: bool, cp_size: int, dim: int = 1) -> Tensor:
    if not enabled or cp_size <= 1:
        return tensor
    return balanced_cp_to_rank_major(tensor, cp_size=cp_size, dim=dim)


def gdn_sp_world_size(group: Optional[ProcessGroup]) -> int:
    if group is None:
        return 1
    return dist.get_world_size(group)
