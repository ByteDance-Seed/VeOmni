from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from ...distributed.parallel_state import get_parallel_state
from .comm import get_ulysses_sequence_parallel_group, get_unified_sequence_parallel_group
from .ulysses import _Gather, _Slice
from .utils import pad_tensor, unpadding_tensor_for_seqeunce_parallel


def sp_pad(
    tensor: torch.Tensor,
    dim: int = -1,
    pad_value: int = 0,
) -> torch.Tensor:
    """Pad ``tensor`` along ``dim`` so its length is divisible by ``sp_size``.

    Pad-only counterpart of :func:`sp_pad_and_slice` (which also slices). Used by
    SeedOmni V2 modules that need the FULL padded sequence (e.g. to compute
    FlashAttention ``cu_seqlens`` over the whole sequence) BEFORE slicing the
    per-rank chunk. Returns the tensor unchanged when already divisible or SP is
    disabled.
    """
    sp_size = get_parallel_state().sp_size
    if sp_size <= 1:
        return tensor
    seq_length = tensor.size(dim)
    pad_size = (sp_size - seq_length % sp_size) % sp_size
    if pad_size == 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_size
    pad = torch.full(pad_shape, fill_value=pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, pad), dim=dim)


def sp_pad_and_slice(
    tensor: torch.Tensor,
    dim: int = -1,
    pad_value: int = 0,
    pad_scale: int = 1,
) -> torch.Tensor:
    """
    Pads and slices a tensor for sequence parallelism (SP) distribution.
    This function ensures the tensor can be evenly distributed across SP ranks by:
    1. Padding the tensor to make its length divisible by (sp_size * pad_scale)
    2. Slicing the padded tensor to extract the chunk for the current SP rank
    Args:
        tensor: Input tensor to pad and slice
        dim: Dimension along which to pad and slice (default: -1)
        pad_value: Value to use for padding (default: 0)
        pad_scale: Scaling factor for SP size during padding (default: 1).
                   This is needed for some VLMs that perform token merging to ensure
                   padding is handled correctly before the merge operation
    Returns:
        The sliced tensor chunk for the current SP rank
    """
    # Get sequence parallelism configuration
    sp_size = get_parallel_state().sp_size
    sp_rank = get_parallel_state().sp_rank
    # Phase 1: Pad the tensor to align with (sp_size * pad_scale)
    # This ensures the tensor can be evenly split across all SP ranks
    seq_length = tensor.size(dim)
    scale_sp_size = sp_size * pad_scale
    # Calculate the chunk size after scaling, rounding up to ensure full coverage
    sp_chunk_size = (seq_length + scale_sp_size - 1) // scale_sp_size
    # Calculate how much padding is needed to reach the target length
    pad_size = sp_chunk_size * scale_sp_size - seq_length
    if pad_size != 0:
        # Create padding tensor with the same shape except for the target dimension
        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_size
        pad = torch.full(pad_shape, fill_value=pad_value, dtype=tensor.dtype, device=tensor.device)
        # Concatenate padding to the end of the tensor
        tensor = torch.cat((tensor, pad), dim=dim)
    # Phase 2: Slice the padded tensor for the current SP rank
    # After padding, recalculate the chunk size based on the actual sp_size
    seq_length = tensor.size(dim)
    sp_chunk_size = (seq_length + sp_size - 1) // sp_size
    # Extract the chunk for this rank: each rank gets a contiguous slice
    # narrow(dim, start, length) extracts tensor[start:start+length] along dim
    return tensor.narrow(dim, sp_rank * sp_chunk_size, sp_chunk_size)


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
    return x[tuple(slc)].clone()


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
    scale_grad=False,
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


# ── Per-module SP redistribution ────────────────────────────────────────────────
#
# The orchestrator runs with SP disabled, so every rank owns a DISTINCT DP-shard
# sample. A module may declare a larger ``ulysses_size`` (its SP size); the SP
# group of ``sp_size`` ranks then holds ``sp_size`` distinct samples that must be
# processed as ONE packed sequence sharded across the group.
#
# ``sp_gather_seqs`` (pre-forward) all-gathers the group's distinct per-rank
# tensors and concatenates them into the full sequence (identical on every SP
# rank); it is autograd-aware. Its backward sums grads across the SP group (each
# rank consumed a *different* downstream slice) and routes each rank the grad of
# its OWN segment — correct because every rank owns a distinct sample and FSDP
# sums param grads over the ``dp_shard_sp`` mesh. ``sp_take_own_seq``
# (post-forward) narrows the gathered full output back to this rank's own segment.


class _GatherConcatSP(torch.autograd.Function):
    """Autograd-aware all-gather + concat over the SP group.

    Forward all-gathers every SP rank's padded ``local`` tensor, trims each to
    its true length and concatenates along ``dim`` — producing the full sequence,
    identical on every SP rank.

    Backward: ``full`` is replicated and each rank consumed a *different* slice
    of it downstream, so the grad wrt the full sequence is the SUM of per-rank
    grads (an all-reduce over the SP group); each rank then takes the grad of its
    OWN contiguous segment.
    """

    @staticmethod
    def forward(ctx, group, local, dim, lengths):
        sp_size = dist.get_world_size(group)

        max_len = max(lengths)
        pad_size = max_len - local.size(dim)
        padded = pad_tensor(local, dim, pad_size, 0) if pad_size > 0 else local
        padded = padded.contiguous()
        gathered = [torch.empty_like(padded) for _ in range(sp_size)]
        dist.all_gather(gathered, padded, group=group)

        segments = [gathered[i].narrow(dim, 0, lengths[i]) for i in range(sp_size)]
        full = torch.cat(segments, dim=dim)

        ctx.group = group
        ctx.dim = dim
        ctx.rank = dist.get_rank(group)
        ctx.lengths = lengths
        return full

    @staticmethod
    def backward(ctx, grad_full):
        # Clone before the in-place all-reduce: ``.contiguous()`` is a no-op (returns
        # the same tensor) when ``grad_full`` is already contiguous, so reducing in
        # place would scribble on the incoming grad buffer owned by autograd.
        grad_full = grad_full.contiguous().clone()
        dist.all_reduce(grad_full, op=dist.ReduceOp.SUM, group=ctx.group)

        offset = sum(ctx.lengths[: ctx.rank])
        grad_local = grad_full.narrow(ctx.dim, offset, ctx.lengths[ctx.rank]).clone()
        return None, grad_local, None, None


def sp_gather_seqs(local: Tensor, dim: int) -> Tuple[Tensor, List[int], int]:
    """Gather the SP group's distinct per-rank sequences into one packed sequence.

    Each rank owns a distinct sample (the orchestrator runs SP-disabled); a module
    with ``sp_size > 1`` concatenates its group's ``sp_size`` sequences into one
    packed sequence so the forward runs at its declared SP. When SP is disabled
    (``sp_size <= 1``) ``local`` already holds the full data and is returned
    unchanged.

    Returns ``(full, seg_lengths, sp_rank)`` where ``seg_lengths`` is the per-rank
    length along ``dim`` and ``sp_rank`` this rank's position in the SP group —
    both needed by :func:`sp_take_own_seq` to restore the per-rank layout after
    the forward.
    """
    ps = get_parallel_state()
    sp_size = ps.sp_size
    dim = dim % local.dim()
    if sp_size <= 1:
        return local, [local.size(dim)], 0

    group = ps.sp_group
    sp_rank = dist.get_rank(group)

    len_t = torch.tensor([local.size(dim)], device=local.device, dtype=torch.long)
    len_list = [torch.empty_like(len_t) for _ in range(sp_size)]
    dist.all_gather(len_list, len_t, group=group)
    lengths = [int(x.item()) for x in len_list]

    # Enforce a single dtype across the SP group before the data all-gather.
    #
    # The SP ranks hold DISTINCT samples, and a rank's packed embeds/hidden
    # promote to a dtype that depends on which modalities its local samples
    # contain — e.g. bf16 text embeds (FSDP2 mixed precision) vs float32 image
    # embeds (a DDP vision tower). ``all_gather`` requires an identical dtype on
    # every rank; a mismatch makes NCCL read mismatched byte counts and SILENTLY
    # corrupts the gathered buffer (NaN garbage), which only surfaces as a loss
    # NaN once it lands on a label token. Promote to the common dtype (matching
    # the dtype a non-SP concat of the same tensors would produce); FSDP2
    # ``cast_forward_inputs`` re-casts to the compute dtype at the module
    # boundary, so this is numerically a no-op relative to the non-SP path.
    local = _sp_unify_dtype(local, group, sp_size)

    full = _GatherConcatSP.apply(group, local, dim, tuple(lengths))
    return full, lengths, sp_rank


# torch dtype <-> stable integer code, for all-gathering dtypes across ranks.
_SP_DTYPE_CODES: Tuple[torch.dtype, ...] = (
    torch.float32,
    torch.float64,
    torch.float16,
    torch.bfloat16,
)


def _sp_unify_dtype(local: Tensor, group: ProcessGroup, sp_size: int) -> Tensor:
    """Cast ``local`` to the dtype promoted across all ranks of ``group``.

    No-op unless the tensor is floating point AND the ranks disagree on dtype.
    """
    if not local.is_floating_point() or local.dtype not in _SP_DTYPE_CODES:
        return local
    code = _SP_DTYPE_CODES.index(local.dtype)
    code_t = torch.tensor([code], device=local.device, dtype=torch.long)
    code_list = [torch.empty_like(code_t) for _ in range(sp_size)]
    dist.all_gather(code_list, code_t, group=group)
    codes = {int(c.item()) for c in code_list}
    if len(codes) <= 1:
        return local
    promoted = local.dtype
    for c in codes:
        promoted = torch.promote_types(promoted, _SP_DTYPE_CODES[c])
    return local.to(promoted)


def sp_take_own_seq(full: Tensor, dim: int, seg_lengths: List[int], sp_rank: int) -> Tensor:
    """Narrow the gathered full output back to this rank's own segment.

    Inverse of :func:`sp_gather_seqs`: restores the per-rank layout (each SP rank
    recovers the output for its own sample). ``seg_lengths`` are the per-rank
    segment lengths and ``sp_rank`` this rank's position in the SP group.
    Differentiable via ``narrow`` — grad flows to this segment of ``full``.
    """
    dim = dim % full.dim()
    if len(seg_lengths) <= 1:
        return full
    offset = sum(seg_lengths[:sp_rank])
    return full.narrow(dim, offset, seg_lengths[sp_rank])


# ── Looped per-sample SP (memory-bounded alternative to gather-concat) ─────────
#
# The gather-concat path (``sp_gather_seqs`` → slice → forward → ``gather_outputs``
# → ``sp_take_own_seq``) packs the SP group's ``sp_size`` DISTINCT samples into ONE
# sequence and runs a single forward. Because the orchestrator pins outer SP to 1
# (so ``dp == world_size`` and every rank owns a distinct sample), that concat
# makes the module's effective micro-batch ``sp_size`` samples and all-gathers an
# ``sp_size``× buffer onto every rank — an OOM that scales with the SP size and
# defeats SP's purpose (splitting ONE large sample).
#
# The looped path keeps the micro-batch at ONE sample: the graph driver
# (``graphs/dispatch.run_sp_looped_endpoint``) loops over the SP group and, per
# iteration, redistributes a single owner's sample across the whole group (standard
# Ulysses on one sample) via the module's ``@sp_pre_forward`` / ``@sp_post_forward``
# hooks, so the peak footprint is ≈ one sample sharded to ``1/sp_size`` rather than
# ``sp_size`` samples.
#
# * ``_BroadcastFromOwnerSP`` (input side) replaces the all-gather-concat: it
#   broadcasts ONE owner's sample to the group (autograd-aware; backward sums the
#   group's grads back to the owner). Only one sample is materialized at a time.
# * ``_GatherToOwnerSP`` (output side) gathers the per-rank output shards to the
#   iteration's OWNER only — the owner reconstructs its full sample, non-owners
#   keep just their ``1/sp_size`` shard — so the output side never holds
#   ``sp_size`` full samples either. Backward scatters the owner's grad back to
#   every rank's shard.
# * A zero-magnitude anchor keeps every iteration reachable from the loss on every
#   rank, so all ranks traverse each iteration's collectives in the same
#   (autograd-deterministic) order — no hang, no grad corruption.


def _sp_global_rank(group: ProcessGroup, group_rank: int) -> int:
    """Translate an SP-group-local rank into its global process rank."""
    return dist.get_global_rank(group, group_rank)


class _BroadcastFromOwnerSP(torch.autograd.Function):
    """Autograd-aware broadcast of ONE owner's tensor to the whole SP group.

    Forward broadcasts the owner's ``tensor`` (shape included, so peers need not
    know the owner's per-sample length) from ``src_group_rank`` to every rank.

    Backward: the broadcast value was consumed by a *different* shard on each rank,
    so the grad wrt the owner's tensor is the SUM of the group's grads (a reduce
    onto the owner). Only the owner rank receives a grad for its input — peers pass
    their own (unused) tensor and get ``None``, so across the ``sp_size`` calls a
    rank makes (one per owner) only the ``i == my_rank`` call carries its grad.
    """

    @staticmethod
    def forward(ctx, group, src_group_rank, tensor):  # noqa: D401
        my_rank = dist.get_rank(group)
        src_global = _sp_global_rank(group, src_group_rank)
        is_src = my_rank == src_group_rank

        # Broadcast the owner's rank/shape so peers can allocate a matching buffer.
        # After each broadcast every rank holds the owner's value.
        ndim = torch.tensor([tensor.dim() if is_src else 0], device=tensor.device, dtype=torch.long)
        dist.broadcast(ndim, src=src_global, group=group)
        if is_src:
            shape = torch.tensor(list(tensor.shape), device=tensor.device, dtype=torch.long)
        else:
            shape = torch.empty(int(ndim.item()), device=tensor.device, dtype=torch.long)
        dist.broadcast(shape, src=src_global, group=group)
        full_shape = torch.Size([int(x) for x in shape.tolist()])

        if is_src:
            buf = tensor.contiguous().clone()
        else:
            buf = torch.empty(full_shape, dtype=tensor.dtype, device=tensor.device)
        dist.broadcast(buf, src=src_global, group=group)

        ctx.group = group
        ctx.src_global = src_global
        ctx.is_src = is_src
        return buf

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = grad_out.contiguous().clone()
        dist.reduce(grad_out, dst=ctx.src_global, op=dist.ReduceOp.SUM, group=ctx.group)
        grad_tensor = grad_out if ctx.is_src else None
        return None, None, grad_tensor


class _GatherToOwnerSP(torch.autograd.Function):
    """Gather the group's per-rank output shards to ONE owner (dim-``dim`` concat).

    Under Ulysses each rank holds an equal ``1/sp_size`` shard of the (padded)
    sample. Forward gathers those shards onto ``dst_group_rank`` only and
    concatenates them into the owner's full (padded) sample; peers keep their own
    shard as a pass-through so it stays in the autograd graph (the caller anchors
    it with a zero-magnitude term).

    Backward: the owner splits its grad into ``sp_size`` shards and scatters them
    back, so every rank recovers the grad of its own shard — peers ignore the
    (zero) grad arriving from the anchor and use the scattered one instead.
    """

    @staticmethod
    def forward(ctx, group, dst_group_rank, dim, chunk):  # noqa: D401
        sp_size = dist.get_world_size(group)
        my_rank = dist.get_rank(group)
        dst_global = _sp_global_rank(group, dst_group_rank)
        chunk = chunk.contiguous()

        ctx.group = group
        ctx.dim = dim
        ctx.dst_global = dst_global
        ctx.is_dst = my_rank == dst_group_rank
        ctx.sp_size = sp_size
        ctx.chunk_shape = chunk.shape
        ctx.chunk_dtype = chunk.dtype

        if my_rank == dst_group_rank:
            gathered = [torch.empty_like(chunk) for _ in range(sp_size)]
            dist.gather(chunk, gather_list=gathered, dst=dst_global, group=group)
            return torch.cat(gathered, dim=dim)
        dist.gather(chunk, gather_list=None, dst=dst_global, group=group)
        return chunk

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.is_dst:
            grads = [g.contiguous() for g in grad_out.contiguous().chunk(ctx.sp_size, dim=ctx.dim)]
            local_grad = torch.empty(ctx.chunk_shape, dtype=grad_out.dtype, device=grad_out.device)
            dist.scatter(local_grad, scatter_list=grads, src=ctx.dst_global, group=ctx.group)
            return None, None, None, local_grad
        local_grad = torch.empty(ctx.chunk_shape, dtype=ctx.chunk_dtype, device=grad_out.device)
        dist.scatter(local_grad, scatter_list=None, src=ctx.dst_global, group=ctx.group)
        return None, None, None, local_grad


def sp_broadcast_from_owner(tensor: Tensor, src_group_rank: int, group: ProcessGroup) -> Tensor:
    """Autograd-aware broadcast of one owner's ``tensor`` to the SP ``group``.

    The owner→all-ranks counterpart of :func:`sp_gather_to_owner` (all-ranks→owner).
    Thin wrapper around :class:`_BroadcastFromOwnerSP` (see it for the backward
    contract). The whole tensor (shape included) is broadcast, so there is no
    axis/``dim`` to specify — every rank ends up holding the owner's exact tensor.
    """
    return _BroadcastFromOwnerSP.apply(group, src_group_rank, tensor)


def sp_gather_to_owner(chunk: Tensor, dim: int, dst_group_rank: int, group: ProcessGroup) -> Tensor:
    """Gather the group's shards of one sample to its owner along ``dim``.

    Thin wrapper around :class:`_GatherToOwnerSP`.
    """
    return _GatherToOwnerSP.apply(group, dst_group_rank, dim, chunk)


# The looped per-sample SP path now lives in the graph:
# ``graphs/dispatch.run_sp_looped_endpoint`` drives the per-owner loop, threads the
# deterministic backward-collective link, and broadcasts each owner's ``pre_forward``
# output to the group with :func:`sp_broadcast_from_owner` (whole-dict, axis-agnostic
# — no ``dim``). Modules then supply only the axis-aware halves: an ``@sp_pre_forward``
# hook that slices the shared data, and an ``@sp_post_forward`` hook that calls
# :func:`sp_gather_to_owner` along its own known axis. The primitives above
# (``_BroadcastFromOwnerSP`` / ``_GatherToOwnerSP`` and their wrappers) are the
# building blocks the driver and those hooks use.
