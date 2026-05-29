# Host-side orchestration for WS PUSH:
#   GEMM epilogue → TMA S2G to peer symm-mem → GpuBarrierAll.
#
# Call chain:
#   ws_push_forward_impl
#     → torch.ops.ulysses.ws_push_gemm_a2a
#         → WS GEMM (PackQKV epilogue, TMA push)
#         → GpuBarrierAll (system-scope fence + cross-rank CAS)
#     → _extract_qkv_views (zero-copy)

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass import Float32, Int32
from quack.cache_utils import jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import get_device_capacity, torch2cute_dtype_map
from quack.gemm_tvm_ffi_utils import make_scheduler_args
from quack.tile_scheduler import TileSchedulerOptions

from veomni.utils.device import get_torch_device, synchronize
from veomni.utils.logging import get_logger

from ._kernels.kernel import quack_patch
from ._kernels.kernel.gemm_ws_packqkv import GemmWsPreAttnPackQKVSm90, PackQKVEpilogueArguments
from ._kernels.kernel.gpu_barrier import _compile_gpu_barrier
from .state_manager import get_active_manager


if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


logger = get_logger(__name__)


# ``torch.distributed._symmetric_memory`` is a private API that is absent on
# some PyTorch builds. Degrade gracefully here so the module imports; the
# error surfaces from ``init_ws_push_state`` instead.
try:
    import torch.distributed._symmetric_memory as _symm_mem  # noqa: SLF001

    _SYMM_MEM_AVAILABLE = True
    _SYMM_MEM_IMPORT_ERROR: Exception | None = None
except ImportError as _exc:  # pragma: no cover - depends on torch build
    _symm_mem = None  # type: ignore[assignment]
    _SYMM_MEM_AVAILABLE = False
    _SYMM_MEM_IMPORT_ERROR = _exc
    logger.debug("symmetric memory unavailable: %s", _exc)


# ============================================================================
# Layout: single source of truth for the [Q | K | V] contiguous-region slice.
# ============================================================================


def _slice_peer_out_buf(
    buf_1d: torch.Tensor,
    rows: int,
    q_cols: int,
    k_cols: int,
    v_cols: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice a flat peer-out buffer into zero-copy ``(rows, *_cols)`` Q/K/V views."""
    q_numel = rows * q_cols
    k_numel = rows * k_cols
    q = buf_1d[:q_numel].view(rows, q_cols)
    k = buf_1d[q_numel : q_numel + k_numel].view(rows, k_cols)
    v = buf_1d[q_numel + k_numel :].view(rows, v_cols)
    return q, k, v


# ============================================================================
# Symmetric blob layout — [peer_out | pad | sync] in one rendezvous'd int8 blob.
# ============================================================================


_SYNC_REGION_ALIGN = 128  # cache-line isolation between TMA region and CAS slots


def _compute_symm_layout(
    *,
    bs: int,
    seq_global: int,
    pack_hidden_local: int,
    world_size: int,
    dtype: torch.dtype,
) -> tuple[int, int, int]:
    """Return ``(total_bytes, sync_byte_offset, peer_out_nbytes)`` for the unified ``[peer_out | pad | sync]`` blob."""
    itemsize = torch.empty(0, dtype=dtype).element_size()
    peer_out_nbytes = bs * seq_global * pack_hidden_local * itemsize
    sync_byte_offset = (peer_out_nbytes + _SYNC_REGION_ALIGN - 1) & ~(_SYNC_REGION_ALIGN - 1)
    total_bytes = sync_byte_offset + world_size * 4
    return total_bytes, sync_byte_offset, peer_out_nbytes


# ============================================================================
# State object — initialized once, reused across forward calls
# ============================================================================


@dataclass(frozen=True)
class WsPushState:
    """Persistent symm-mem state for the WS PUSH fused GEMM + a2a pipeline.

    ``frozen=True`` so config/resource bindings can't drift between init and
    kernel invocation; ``close()`` uses ``object.__setattr__`` to null fields
    without barrier/synchronize side effects. Caller orders teardown::

        torch.cuda.synchronize(); dist.barrier(state.sp_group); state.close()
    """

    # ---- Process group info ----
    sp_group: object
    rank: int
    world_size: int
    device: torch.device

    # ---- Shape metadata (problem dimensions + derived) ----
    bs: int
    local_seq: int
    seq_global: int  # = local_seq * world_size
    nheads_q: int
    nheads_k: int
    nheads_v: int
    nheads_q_per_rank: int  # = nheads_q // world_size
    nheads_k_per_rank: int  # = nheads_k // world_size
    nheads_v_per_rank: int  # = nheads_v // world_size
    head_dim: int
    dtype: torch.dtype
    pack_hidden_local: int  # = (nheads_q + nheads_k + nheads_v) // world_size * head_dim
    q_region_cols: int  # = nheads_q_per_rank * head_dim
    k_region_cols: int  # = nheads_k_per_rank * head_dim
    v_region_cols: int  # = nheads_v_per_rank * head_dim

    # ---- GEMM tile config ----
    tile_m: int
    tile_n: int
    pingpong: bool

    # ---- Symmetric memory: unified [peer_out | pad | sync] blob ----
    # ``symm_blob`` is the raw int8 storage; ``peer_out_buf`` and
    # ``group_sync_buf`` are zero-copy typed views into disjoint slices of it.
    # All three share storage with a single rendezvous'd allocation, so only
    # one handle (``symm_handle``) needs to be kept alive.
    symm_blob: torch.Tensor
    symm_handle: object  # -- KEEP ALIVE
    peer_out_buf: torch.Tensor  # 1D bf16 view of peer_out region
    group_sync_buf: torch.Tensor  # 1D int32 view of sync region
    sync_buf_ptrs_i64: torch.Tensor  # per-rank pointers to peer sync regions

    # ---- Cached per-peer per-section 2D views (built once during init) ----
    peer_out_2d_views: tuple

    # ---- Packed scalar metadata (CPU tensor — no D2H sync on .tolist()) ----
    metadata: torch.Tensor

    # ---- Internal lifecycle flag (flipped by close() via object.__setattr__) ----
    _closed: bool = field(default=False, init=False)

    def close(self) -> None:
        """Drop symm-mem references and mark failed-closed; idempotent."""
        if self._closed:
            return
        for _f in (
            "symm_blob",
            "symm_handle",
            "peer_out_buf",
            "group_sync_buf",
            "sync_buf_ptrs_i64",
            "peer_out_2d_views",
            "metadata",
        ):
            object.__setattr__(self, _f, None)
        object.__setattr__(self, "_closed", True)

    def _check_alive(self) -> None:
        """Raise immediately if this state has been closed."""
        if self._closed or self.peer_out_buf is None:
            raise RuntimeError(
                "WsPushState has been closed — cannot reuse. Re-run init_ws_push_state() to obtain a fresh state."
            )


# ============================================================================
# WsPushDispatch — fused-path invocation token
# ============================================================================


@dataclass(frozen=True)
class WsPushDispatch:
    """Fused-path invocation token bundling ``state``, ``W_qkv_B``, and ``bias_B``.

    ``W_qkv_B`` is a detached, block-per-rank concatenation of the caller's
    ``qkv_weight`` Parameter; the autograd graph still anchors on the original
    Parameter via ``AsyncUlyssesQKVProjection.save_for_backward``.
    """

    state: "WsPushState"
    W_qkv_B: torch.Tensor
    bias_B: torch.Tensor | None

    @classmethod
    def try_resolve_fused(
        cls,
        hidden_states: torch.Tensor,
        qkv_weight: torch.Tensor,
        qkv_bias: torch.Tensor | None = None,
    ) -> "WsPushDispatch | None":
        """Build dispatch from a pre-packed ``qkv_weight`` (no cat, no cache).

        For callers storing QKV as a single fused Parameter (see
        ``FusedQKVLinear``). The kernel requires ``W_qkv_B.requires_grad ==
        False``; the detach below is safe because the autograd graph still
        anchors on the original Parameter via ``AsyncUlyssesQKVProjection``'s
        ``save_for_backward``, which rebuilds ``grad_qkv_weight`` from the
        three a2a'd partial grads.
        """
        mgr = get_active_manager()
        if mgr is None:
            return None
        bs, local_seq = hidden_states.shape[0], hidden_states.shape[1]
        if not mgr.is_compatible(bs=bs, local_seq=local_seq, dtype=hidden_states.dtype):
            return None
        W_qkv_B = qkv_weight.detach().contiguous()
        bias_B = qkv_bias.detach().contiguous() if qkv_bias is not None else None
        return cls(state=mgr.state, W_qkv_B=W_qkv_B, bias_B=bias_B)

    @classmethod
    def try_resolve_auto(
        cls,
        hidden_states: torch.Tensor,
        *,
        qkv_weight: torch.Tensor | None = None,
        qkv_bias: torch.Tensor | None = None,
    ) -> "WsPushDispatch | None":
        """Resolve dispatch when the caller exposes a fused ``qkv_weight``.

        Returns ``None`` otherwise so callers fall through to eager.
        """
        if qkv_weight is not None:
            return cls.try_resolve_fused(hidden_states, qkv_weight, qkv_bias)
        return None


def validate_ws_push_dispatch(
    dispatch: WsPushDispatch,
    hidden_states: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    seq_dimension: int,
    head_dimension: int,
    unpadded_dim_size: int,
    q_bias: torch.Tensor | None,
    k_bias: torch.Tensor | None,
    v_bias: torch.Tensor | None,
    group: "ProcessGroup | None",
) -> None:
    """Validate that ``dispatch`` is compatible with the forward inputs.

    All 14 system-boundary preconditions for the fused path live here so
    they stay out of the autograd Function body. Raises ``ValueError`` on
    any mismatch.
    """
    state = dispatch.state
    W_qkv_B = dispatch.W_qkv_B
    bias_B = dispatch.bias_B
    ws = state.world_size

    # Stricter than `not need_repeat_kv`: rejects ws<num_kv_heads with non-divisible kv too.
    if num_kv_heads % ws != 0:
        raise ValueError(f"fused path requires num_kv_heads ({num_kv_heads}) % ws ({ws}) == 0")
    if num_q_heads % ws != 0:
        raise ValueError(f"fused path requires num_q_heads ({num_q_heads}) % ws ({ws}) == 0")
    # Fused kernel hard-codes seq=1, head=2 in its output layout.
    if not (seq_dimension == 1 and head_dimension == 2):
        raise ValueError(
            "fused path requires seq_dimension=1, head_dimension=2; "
            f"got seq_dimension={seq_dimension}, head_dimension={head_dimension}"
        )
    if group is not None and group is not state.sp_group:
        raise ValueError("forward `group` must be the same ProcessGroup as `dispatch.state.sp_group`")
    # ``group=None`` resolves via ``get_ulysses_sequence_parallel_group``;
    # verify ``state.sp_group`` still matches to prevent silent reuse of a
    # state bound to a previously-destroyed process group.
    if group is None:
        from veomni.distributed.sequence_parallel.comm import get_ulysses_sequence_parallel_group

        if state.sp_group is not get_ulysses_sequence_parallel_group():
            raise ValueError(
                "dispatch.state.sp_group does not match the currently-active "
                "ulysses sequence-parallel group; rebuild the state after "
                "re-initializing the process group"
            )
    if hidden_states.shape[0] != state.bs:
        raise ValueError(f"bs mismatch: hidden_states={hidden_states.shape[0]} vs state.bs={state.bs}")
    if hidden_states.shape[1] != state.local_seq:
        raise ValueError(
            f"local_seq mismatch: hidden_states={hidden_states.shape[1]} vs state.local_seq={state.local_seq}"
        )
    if hidden_states.dtype != state.dtype:
        raise ValueError(f"dtype mismatch: hidden_states={hidden_states.dtype} vs state.dtype={state.dtype}")
    if state.local_seq * state.world_size < unpadded_dim_size:
        raise ValueError(
            f"seq_global ({state.local_seq * state.world_size}) < unpadded_dim_size ({unpadded_dim_size})"
        )
    if W_qkv_B.dim() != 2:
        raise ValueError(f"W_qkv_B must be 2D, got {W_qkv_B.dim()}D")
    if not W_qkv_B.is_contiguous():
        raise ValueError("W_qkv_B must be C-contiguous (row-major)")
    if W_qkv_B.dtype != state.dtype:
        raise ValueError("W_qkv_B dtype must match state.dtype")
    if W_qkv_B.requires_grad:
        raise ValueError(
            "W_qkv_B must be detached; gradients flow through the original q/k/v_weight "
            "params via the hand-written backward"
        )
    any_bias = (q_bias is not None) or (k_bias is not None) or (v_bias is not None)
    if any_bias:
        if not ((q_bias is not None) and (k_bias is not None) and (v_bias is not None)):
            raise ValueError("fused path requires either all of q/k/v_bias provided, or none")
        if bias_B is None:
            raise ValueError("dispatch.bias_B must be provided when any of q/k/v_bias is set")
        if not bias_B.is_contiguous():
            raise ValueError("bias_B must be C-contiguous")
        if bias_B.dtype != state.dtype:
            raise ValueError("bias_B dtype must match state.dtype")
        if bias_B.requires_grad:
            raise ValueError("bias_B must be detached")


# ============================================================================
# Metadata packing — compact scalar transport through custom op boundary
# ============================================================================


class WsPushMetadata(NamedTuple):
    rank: int
    world_size: int
    bs: int
    local_seq: int
    seq_global: int
    nheads_q: int
    nheads_k: int
    nheads_v: int
    head_dim: int
    pack_hidden_local: int
    tile_m: int
    tile_n: int
    pingpong: bool


def _pack_metadata(nt: WsPushMetadata) -> torch.Tensor:
    """Encode a ``WsPushMetadata`` for the custom op boundary.

    CPU-resident int64 tensor — ``.tolist()`` inside the op is a host-side
    memcpy, no implicit ``cudaStreamSynchronize``.
    """
    return torch.tensor([int(v) for v in nt], dtype=torch.int64, device="cpu")


def _unpack_metadata(metadata: torch.Tensor) -> WsPushMetadata:
    vals = metadata.tolist()
    vals[-1] = bool(vals[-1])  # pingpong: int → bool
    return WsPushMetadata(*vals)


# ============================================================================
# Heuristic Configuration Adaptive tile-config selection for different local seq_len
# ============================================================================

# At/above this *global* seq length, use ``(tile_n=256, pingpong=False)``
# (epi_tile_M=128, ~4× fewer TMA-PUSH stores per rank)
_TILE_CONFIG_LONGSEQ_THRESHOLD = 128 * 1024


def choose_tile_config(local_seq: int, world_size: int) -> tuple[int, bool]:
    """Return ``(tile_n, pingpong)`` for the given workload.

    Shared between ``WsPushStateManager`` (production) and the perf sweep
    test so the threshold lives in exactly one place. ``tile_m`` always
    stays at the ``init_ws_push_state`` default (128) — the two regimes
    differ only in ``tile_n`` and ``pingpong``.
    """
    if local_seq * world_size >= _TILE_CONFIG_LONGSEQ_THRESHOLD:
        return 256, False
    return 128, True


# ============================================================================
# Initialization — collective call on all ranks
# ============================================================================
def init_ws_push_state(
    *,
    sp_group,
    device: torch.device,
    bs: int,
    local_seq: int,
    nheads_q: int,
    nheads_k: int,
    nheads_v: int,
    head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    tile_m: int = 128,
    tile_n: int = 128,
    pingpong: bool = False,
    symm_backend: str = "NVSHMEM",
) -> WsPushState:
    """Initialize persistent state for the WS PUSH fused GEMM + All-to-All.

    Collective: must be called on all ranks of ``sp_group`` together.  The
    init body performs two implicit cross-rank synchronizations: a single
    ``symm_mem.rendezvous`` (over the unified ``[peer_out | pad | sync]`` int8
    blob) and a final ``dist.barrier`` to guarantee every rank has zeroed its
    CAS-barrier sync slots before any forward call begins.  Hot path performs
    none of these.
    """
    if not _SYMM_MEM_AVAILABLE:
        raise RuntimeError(
            "torch.distributed._symmetric_memory is unavailable on this "
            "PyTorch build — Ulysses pre-attention fused kernel cannot run."
        ) from _SYMM_MEM_IMPORT_ERROR

    rank = dist.get_rank(sp_group)
    world_size = dist.get_world_size(sp_group)
    assert nheads_q % world_size == 0
    assert nheads_k % world_size == 0
    assert nheads_v % world_size == 0

    # PackQKV's epi_begin_loop assumes one M-subtile lives inside a single
    # batch, i.e. ``epi_tile_M <= local_seq``. Violating this silently emits
    # corrupted Q/K/V (even-batch garbage, odd-batch zeros) with no crash.
    # ``epi_tile_M`` is derived inside GemmSm90 from ``(tile_m, tile_n, pingpong)``;
    # mirror quack 0.3.4's full ladder (gemm_sm90.py:188-209, 1689-1704) so
    # config errors fail fast.
    if pingpong:
        if tile_m not in (64, 128, 192):
            raise ValueError(f"CTA tile shape M must be 64/128/192 if pingpong, got tile_m={tile_m}.")
        _atom_layout_m = 1
    elif tile_m == 320:
        _atom_layout_m = 1
    elif tile_m == 192:
        _atom_layout_m = 3 if tile_n <= 128 else 1
    else:
        _atom_layout_m = tile_m // 64 if tile_m < 256 else 2
    if tile_m % 128 == 0 and _atom_layout_m > 1:
        _epi_tile_m = math.gcd(128, tile_m)
    elif tile_m % 192 == 0 and _atom_layout_m > 1:
        _epi_tile_m = math.gcd(192, tile_m)
    else:
        _epi_tile_m = math.gcd(64, tile_m)
    if _epi_tile_m > local_seq:
        raise ValueError(
            f"PackQKV requires epi_tile_M <= local_seq, "
            f"got epi_tile_M={_epi_tile_m} > local_seq={local_seq} "
            f"(tile_m={tile_m}, tile_n={tile_n}, pingpong={pingpong}). "
            f"Set pingpong=True (forces atom_layout_m=1 → epi_tile_M=64) "
            f"or reduce tile_m so the resolved epi_tile_M fits in local_seq."
        )

    nheads_q_per_rank = nheads_q // world_size
    nheads_k_per_rank = nheads_k // world_size
    nheads_v_per_rank = nheads_v // world_size
    seq_global = local_seq * world_size
    pack_hidden_local = (nheads_q_per_rank + nheads_k_per_rank + nheads_v_per_rank) * head_dim

    if _symm_mem.get_backend(device) is None:
        _symm_mem.set_backend(symm_backend)

    # Unified symmetric blob — one rendezvous, single KEEP-ALIVE handle, zero-copy
    # typed views over [peer_out | pad | sync].
    total_bytes, sync_byte_offset, peer_out_nbytes = _compute_symm_layout(
        bs=bs,
        seq_global=seq_global,
        pack_hidden_local=pack_hidden_local,
        world_size=world_size,
        dtype=dtype,
    )
    symm_blob = _symm_mem.empty([total_bytes], dtype=torch.int8, device=device)
    symm_handle = _symm_mem.rendezvous(symm_blob, sp_group)
    peer_out_buf = symm_blob[:peer_out_nbytes].view(dtype)
    group_sync_buf = symm_blob[sync_byte_offset : sync_byte_offset + world_size * 4].view(torch.int32)

    # ``GpuBarrierAll`` is self-clearing per iteration but the *initial* state
    # must be all-zero on every rank. ``cuda.synchronize`` must precede
    # ``dist.barrier`` so the zero write reaches DRAM before peers read it.
    symm_blob.zero_()
    synchronize()
    dist.barrier(sp_group)

    # Per-rank pointers into each peer's sync sub-region. ``GpuBarrierAll``
    # only reads int64 pointers, so a slice into the unified blob works the
    # same as a dedicated allocation.
    sync_buf_ptrs_i64 = torch.tensor(
        [int(bp) + sync_byte_offset for bp in symm_handle.buffer_ptrs],
        dtype=torch.int64,
        device=device,
    )

    q_region_cols = nheads_q_per_rank * head_dim
    k_region_cols = nheads_k_per_rank * head_dim
    v_region_cols = nheads_v_per_rank * head_dim

    peer_out_2d_views = _build_peer_out_2d_views(
        symm_handle=symm_handle,
        peer_out_nbytes=peer_out_nbytes,
        world_size=world_size,
        bs=bs,
        seq_global=seq_global,
        pack_hidden_local=pack_hidden_local,
        q_region_cols=q_region_cols,
        k_region_cols=k_region_cols,
        v_region_cols=v_region_cols,
        dtype=dtype,
    )
    metadata = _pack_metadata(
        WsPushMetadata(
            rank=rank,
            world_size=world_size,
            bs=bs,
            local_seq=local_seq,
            seq_global=seq_global,
            nheads_q=nheads_q,
            nheads_k=nheads_k,
            nheads_v=nheads_v,
            head_dim=head_dim,
            pack_hidden_local=pack_hidden_local,
            tile_m=tile_m,
            tile_n=tile_n,
            pingpong=pingpong,
        )
    )

    return WsPushState(
        sp_group=sp_group,
        rank=rank,
        world_size=world_size,
        device=device,
        bs=bs,
        local_seq=local_seq,
        seq_global=seq_global,
        nheads_q=nheads_q,
        nheads_k=nheads_k,
        nheads_v=nheads_v,
        nheads_q_per_rank=nheads_q_per_rank,
        nheads_k_per_rank=nheads_k_per_rank,
        nheads_v_per_rank=nheads_v_per_rank,
        head_dim=head_dim,
        dtype=dtype,
        pack_hidden_local=pack_hidden_local,
        tile_m=tile_m,
        tile_n=tile_n,
        pingpong=pingpong,
        symm_blob=symm_blob,
        symm_handle=symm_handle,
        peer_out_buf=peer_out_buf,
        group_sync_buf=group_sync_buf,
        sync_buf_ptrs_i64=sync_buf_ptrs_i64,
        q_region_cols=q_region_cols,
        k_region_cols=k_region_cols,
        v_region_cols=v_region_cols,
        peer_out_2d_views=peer_out_2d_views,
        metadata=metadata,
    )


# ============================================================================
# PyTorch Custom Op — torch.compile-friendly graph node
# ============================================================================


@torch.library.custom_op(
    "ulysses::ws_push_gemm_a2a",
    mutates_args=("peer_out_buf", "peer_out_views"),
    device_types="cuda",
)
def ws_push_gemm_a2a(
    A: torch.Tensor,
    W_qkv_B: torch.Tensor,
    peer_out_buf: torch.Tensor,
    peer_out_views: Sequence[torch.Tensor],
    sync_buf_ptrs_i64: torch.Tensor,
    metadata: torch.Tensor,
    bias_B: torch.Tensor | None = None,
) -> None:
    """Fused GEMM + All-to-All (WS PUSH) registered as a PyTorch custom op.

    In-place op covering:
      Phase 1: WS GEMM with PackQKV epilogue TMA S2G push to all peers
      Phase 2: GpuBarrierAll (system-scope fence + cross-rank CAS sync.

    ``bias_B`` (optional, shape ``[1, N_total]``) is the round-robin-interleaved
    per-N bias added to the GEMM accumulator before the peer TMA store.  Its
    presence is the cache key for the compiled kernel — passing ``None`` and a
    real tensor compile to two distinct binaries.
    """
    meta = _unpack_metadata(metadata)

    device = A.device
    # ``torch.cuda.current_stream`` would tie this op to CUDA; route through the
    # device-agnostic helper so NPU / future-accelerator builds still resolve a
    # stream when the kernel itself is gated behind a Hopper preflight upstream.
    gemm_stream = get_torch_device().current_stream(device)

    # Phase 1: Launch WS GEMM with PackQKV epilogue
    _launch_ws_gemm_packqkv_from_args(
        A,
        W_qkv_B,
        tuple(peer_out_views),
        meta.rank,
        meta.world_size,
        meta.bs,
        meta.local_seq,
        meta.seq_global,
        meta.nheads_q,
        meta.nheads_k,
        meta.nheads_v,
        meta.head_dim,
        meta.pack_hidden_local,
        meta.tile_m,
        meta.tile_n,
        meta.pingpong,
        bias_B,
        gemm_stream,
    )

    # Phase 2: Cross-rank barrier (system-scope fence + sync)
    _launch_gpu_barrier_from_args(
        sync_buf_ptrs_i64,
        meta.rank,
        meta.world_size,
        device,
        gemm_stream,
    )


@ws_push_gemm_a2a.register_fake
def _ws_push_gemm_a2a_fake(
    A: torch.Tensor,
    W_qkv_B: torch.Tensor,
    peer_out_buf: torch.Tensor,
    peer_out_views: Sequence[torch.Tensor],
    sync_buf_ptrs_i64: torch.Tensor,
    metadata: torch.Tensor,
    bias_B: torch.Tensor | None = None,
) -> None:
    """Fake implementation for torch.compile tracing — no GEMM/barrier."""
    pass


# ============================================================================
# Forward — single-stream pipeline
# ============================================================================


def ws_push_forward_impl(
    A: torch.Tensor,
    W_qkv: torch.Tensor,
    state: WsPushState,
    *,
    W_qkv_B: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    bias_B: torch.Tensor | None = None,
    _return_views: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute fused GEMM + All-to-All via WS single-kernel PUSH.

    ``W_qkv`` / ``W_qkv_B`` are both the **raw** block-per-rank concatenation
    of Q/K/V projection weights (shape ``[N_total, hidden]``); NVLink balance
    is achieved device-side by ``PackQKVTileScheduler``. ``bias`` is the same
    raw ``[N_total]`` concat.

    Returns:
        (q_out, k_out, v_out) — owned C-contiguous tensors cloned out of
        ``state.peer_out_buf`` so callers can hold them across the next
        forward (which overwrites the buffer). Set ``_return_views=True`` to
        skip the clone and get zero-copy views; **bench/ablation only** —
        views are invalidated by the next forward.
    """
    state._check_alive()
    A_2d = A.reshape(-1, A.shape[-1])

    if W_qkv_B is None:
        W_qkv_B = W_qkv.contiguous()

    # Normalize bias to [1, N_total] (RowVecLoad's 2D ABI inside the epilogue).
    if bias_B is None and bias is not None:
        bias_B = bias.contiguous().unsqueeze(0)
    elif bias_B is not None and bias_B.dim() == 1:
        bias_B = bias_B.unsqueeze(0)

    torch.ops.ulysses.ws_push_gemm_a2a(
        A_2d,
        W_qkv_B,
        state.peer_out_buf,
        state.peer_out_2d_views,
        state.sync_buf_ptrs_i64,
        state.metadata,
        bias_B,
    )

    q, k, v = _extract_qkv_views(state)
    if _return_views:
        return q, k, v
    return q.clone(), k.clone(), v.clone()


def _extract_qkv_views(
    state: WsPushState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-copy ``[bs, seq_global, nheads_pr, head_dim]`` Q/K/V views over ``state.peer_out_buf``."""
    rows = state.bs * state.seq_global
    q_2d, k_2d, v_2d = _slice_peer_out_buf(
        state.peer_out_buf,
        rows,
        state.q_region_cols,
        state.k_region_cols,
        state.v_region_cols,
    )
    q_out = q_2d.view(state.bs, state.seq_global, state.nheads_q_per_rank, state.head_dim)
    k_out = k_2d.view(state.bs, state.seq_global, state.nheads_k_per_rank, state.head_dim)
    v_out = v_2d.view(state.bs, state.seq_global, state.nheads_v_per_rank, state.head_dim)
    return q_out, k_out, v_out


# ============================================================================
# Compilation (cached) + Launch helpers
# make cuda binary .so cache
# ============================================================================


@jit_cache
def _compile_ws_gemm_packqkv(
    a_dtype,
    b_dtype,
    d_dtype,
    world_size,
    tile_m,
    tile_n,
    pingpong,
    has_bias,
    device_capacity,
    nheads_q,
    nheads_k,
    nheads_v,
    head_dim,
):
    """Compile WS GEMM with PackQKV epilogue (cached via @jit_cache).

    ``has_bias`` is part of the cache key — bias-present and bias-absent paths
    have different smem layouts and epi_begin control flow, so they must
    produce distinct compiled binaries.

    ``nheads_q/k/v`` and ``head_dim`` are part of the cache key because the
    PackQKVTileScheduler needs them as compile-time constants (to compute
    per-segment cluster counts for its round-robin remap). They're attached
    to the GEMM instance as Python ints; ``_packqkv_get_scheduler_arguments``
    in quack_patch.py reads them from there.
    """
    # Both installs are idempotent; calling per-compile keeps the install
    # local to the kernel's first real use (no import-time side effects).
    quack_patch.install()
    quack_patch._install_packqkv_scheduler_overrides()
    # Defensive: detect re-install drift early instead of much later as a
    # silent round-robin perf regression to quack's default serpentine
    # scheduler.
    assert getattr(GemmWsPreAttnPackQKVSm90, quack_patch._PACKQKV_SCHED_FLAG, False), (
        "PackQKVTileScheduler override missing on GemmWsPreAttnPackQKVSm90 — "
        "_install_packqkv_scheduler_overrides() did not install the scheduler. "
        "Fused-Ulysses NVLink balance would silently degrade to quack's default "
        "serpentine scheduler."
    )

    m, n, k = cute.sym_int(), cute.sym_int(), cute.sym_int()
    div_a = 128 // a_dtype.width
    div_d = 128 // d_dtype.width

    # Upstream ``quack.gemm_sm90`` is a batched GEMM and accesses
    # ``mA_mkl[None, None, batch_idx]`` etc.; we feed a unit batch (L=1) so the
    # PackQKV launcher remains dense-only.
    mA = fake_tensor(a_dtype, (m, k, 1), divisibility=div_a, leading_dim=1)
    mB = fake_tensor(b_dtype, (n, k, 1), divisibility=div_a, leading_dim=1)

    # Build per-peer per-section fake output tensors as 2D [rows, section_cols].
    # Contiguous layout: each peer's buffer has 3 regions (Q, K, V) with
    # potentially different column widths. Total 3*world_size fake tensors.
    #
    # epi_tile_n must divide each section_cols. Guaranteed since head_dim >= 64
    # and section_cols = nheads_X_per_rank * head_dim.
    epi_tile_n = math.gcd(32, tile_n)
    s_rows = cute.sym_int(divisibility=tile_m)
    s_pack_q = cute.sym_int(divisibility=epi_tile_n)
    s_pack_k = cute.sym_int(divisibility=epi_tile_n)
    s_pack_v = cute.sym_int(divisibility=epi_tile_n)
    fake_peer_outs = tuple(
        fake_tensor(d_dtype, dims, divisibility=div_d, leading_dim=1)
        for _ in range(world_size)
        for dims in ((s_rows, s_pack_q), (s_rows, s_pack_k), (s_rows, s_pack_v))
    )

    # Fake bias tensor [1, N_total]; symbolic N_total divisible by tile_n so
    # epilogue cp.async vectorization assumptions hold.
    fake_bias = None
    if has_bias:
        s_n_total = cute.sym_int(divisibility=tile_n)
        fake_bias = fake_tensor(d_dtype, (1, s_n_total), divisibility=div_d, leading_dim=1)

    fake_epi = PackQKVEpilogueArguments(
        mPeerOuts_mnl=fake_peer_outs,
        mBias=fake_bias,
        rank=Int32(0),
        world_size=Int32(world_size),
        bs=Int32(1),
        local_seq=Int32(1),
        seq_global=Int32(1),
        nheads_q=Int32(1),
        nheads_k=Int32(1),
        nheads_v=Int32(1),
        nheads_q_per_rank=Int32(1),
        nheads_k_per_rank=Int32(1),
        nheads_v_per_rank=Int32(1),
        head_dim=Int32(1),
        pack_hidden_local=Int32(1),
    )
    fake_sched = TileSchedulerOptions(
        max_active_clusters=Int32(1),
        max_swizzle_size=Int32(8),
    )

    gemm_obj = GemmWsPreAttnPackQKVSm90(
        Float32,
        a_dtype,
        (tile_m, tile_n),
        (2, 1, 1),
        pingpong=pingpong,
        is_persistent=True,
    )
    # Side-channel: PackQKVTileScheduler reads these as Python ints to
    # compute its per-segment N-cluster counts at trace time. Kept off the
    # GEMM constructor signature (which is quack-defined) and off the
    # epilogue args (which trace through MLIR). Type-asserted (not int-cast)
    # so a numpy/torch scalar slipping in here surfaces as a clear failure —
    # such values would otherwise hash differently from a Python ``int`` in
    # the @jit_cache key and trigger silent recompiles.
    for _name, _val in (
        ("nheads_q", nheads_q),
        ("nheads_k", nheads_k),
        ("nheads_v", nheads_v),
        ("head_dim", head_dim),
        ("world_size", world_size),
    ):
        assert isinstance(_val, int) and not isinstance(_val, bool), (
            f"_compile_ws_gemm_packqkv: packqkv_{_name} must be a Python int, got {type(_val).__name__}={_val!r}"
        )
    gemm_obj.packqkv_nheads_q = nheads_q
    gemm_obj.packqkv_nheads_k = nheads_k
    gemm_obj.packqkv_nheads_v = nheads_v
    gemm_obj.packqkv_head_dim = head_dim
    gemm_obj.packqkv_world_size = world_size
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        gemm_obj,
        mA,
        mB,
        None,  # mD: peer output is supplied via the PackQKV epilogue
        None,  # mC: no bias/skip-connection
        fake_epi,
        fake_sched,
        None,  # varlen_args: dense, non-varlen
        stream,
        options="--enable-tvm-ffi",
    )


def _launch_ws_gemm_packqkv_from_args(
    A_2d: torch.Tensor,
    W_qkv_B: torch.Tensor,
    peer_out_views: tuple[torch.Tensor, ...],
    rank: int,
    world_size: int,
    bs: int,
    local_seq: int,
    seq_global: int,
    nheads_q: int,
    nheads_k: int,
    nheads_v: int,
    head_dim: int,
    pack_hidden_local: int,
    tile_m: int,
    tile_n: int,
    pingpong: bool,
    bias_B: torch.Tensor | None,
    stream: torch.cuda.Stream,
) -> None:
    """Compile (if needed) + launch WS GEMM with PackQKV epilogue.

    Accepts decomposed scalar args instead of ``WsPushState`` so the function
    can be called from inside a ``torch.library.custom_op`` (which cannot
    accept arbitrary Python objects). See ``ws_push_forward_impl`` for the
    weight/bias layout contract; ``bias_B`` gates a separate cached kernel
    via the ``has_bias`` cache key.
    """
    nheads_q_per_rank = nheads_q // world_size
    nheads_k_per_rank = nheads_k // world_size
    nheads_v_per_rank = nheads_v // world_size

    dtype = A_2d.dtype
    a_dtype = torch2cute_dtype_map[dtype]
    b_dtype = a_dtype
    d_dtype = torch2cute_dtype_map[dtype]
    device_cap = get_device_capacity(A_2d.device)
    has_bias = bias_B is not None

    compiled_fn = _compile_ws_gemm_packqkv(
        a_dtype,
        b_dtype,
        d_dtype,
        world_size,
        tile_m,
        tile_n,
        pingpong,
        has_bias,
        device_cap,
        nheads_q,
        nheads_k,
        nheads_v,
        head_dim,
    )

    real_epi = PackQKVEpilogueArguments(
        mPeerOuts_mnl=tuple(peer_out_views),
        mBias=bias_B,
        rank=Int32(rank),
        world_size=Int32(world_size),
        bs=Int32(bs),
        local_seq=Int32(local_seq),
        seq_global=Int32(seq_global),
        nheads_q=Int32(nheads_q),
        nheads_k=Int32(nheads_k),
        nheads_v=Int32(nheads_v),
        nheads_q_per_rank=Int32(nheads_q_per_rank),
        nheads_k_per_rank=Int32(nheads_k_per_rank),
        nheads_v_per_rank=Int32(nheads_v_per_rank),
        head_dim=Int32(head_dim),
        pack_hidden_local=Int32(pack_hidden_local),
        use_seq_offsets=None,
        peer_store_enabled=None,
    )
    real_sched = make_scheduler_args(
        132,  # max_active_clusters
        8,  # max_swizzle_size
        None,  # no tile_count_semaphore
    )

    # Rank-3 with a unit batch dim matches the upstream batched-GEMM
    # signature. ``cute.compile`` strips the ``stream`` arg of _patched_call
    # (set via ``with torch.cuda.stream(...)`` below), leaving 7 positional
    # args: (mA, mB, mD, mC, epi_args, sched_args, varlen_args).
    A_3d = A_2d.unsqueeze(-1)
    B_3d = W_qkv_B.unsqueeze(-1)
    with torch.cuda.stream(stream):
        compiled_fn(A_3d, B_3d, None, None, real_epi, real_sched, None)


def _build_peer_out_2d_views(
    *,
    symm_handle,
    peer_out_nbytes: int,
    world_size: int,
    bs: int,
    seq_global: int,
    pack_hidden_local: int,
    q_region_cols: int,
    k_region_cols: int,
    v_region_cols: int,
    dtype: torch.dtype,
) -> tuple:
    """Flat tuple of ``3 * world_size`` 2D ``[bs * seq_global, section_cols]`` TMA-descriptor views, in ``[q_r0, k_r0, v_r0, q_r1, ...]`` order."""
    rows = bs * seq_global

    views: list[torch.Tensor] = []
    for r in range(world_size):
        peer_int8 = symm_handle.get_buffer(r, (peer_out_nbytes,), torch.int8)
        peer_1d = peer_int8.view(dtype)
        q, k, v = _slice_peer_out_buf(peer_1d, rows, q_region_cols, k_region_cols, v_region_cols)
        views.append(q)
        views.append(k)
        views.append(v)
    return tuple(views)


# ============================================================================
# Barrier lifecycle helpers
# ============================================================================


def _launch_gpu_barrier_from_args(
    sync_buf_ptrs_i64: torch.Tensor,
    rank: int,
    world_size: int,
    device: torch.device,
    stream: torch.cuda.Stream,
) -> None:
    """Launch GPU group barrier (system-scope fence + cross-rank sync).

    Accepts decomposed args instead of WsPushState, so this function can be
    called from inside a torch.library.custom_op.
    """
    device_cap = get_device_capacity(device)
    compiled_fn = _compile_gpu_barrier(world_size, device_cap)
    with torch.cuda.stream(stream):
        compiled_fn(
            sync_buf_ptrs_i64,
            Int32(rank),
        )
